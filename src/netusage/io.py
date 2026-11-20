"""File I/O for the pipeline.

Transforms are pure: they take and return DataFrames. Every read of landing
files and every Delta read/write goes through this module.
"""

from datetime import date
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from netusage import schemas

CORRUPT_COLUMN = "_corrupt_record"
SOURCE_COLUMN = "source_file"
FILE_PATH_COLUMN = "_metadata.file_path"

# Landing layout: one sub-directory per entity under the landing root. Traffic logs are further
# split into Hive-style daily folders, traffic_logs/batch_date=YYYY-MM-DD/.
LANDING_DIRS = {
    "users": "users",
    "apps": "apps",
    "vpn_rules": "vpn_rules",
    "ad_group": "ad_group",
    "ad_group_membership": "ad_group_membership",
    "resource_resolution": "resource_resolution",
    "traffic_logs": "traffic_logs",
}
BATCH_COLUMN = "batch_date"


def landing_path(root: str | Path, entity: str) -> str:
    return f"{str(root).rstrip('/')}/{LANDING_DIRS[entity]}"


def with_corrupt_column(schema: StructType) -> StructType:
    """Return a copy of ``schema`` with Spark's corrupt-record column added.

    ``StructType.add()`` mutates in place and returns the same object, so build
    a new StructType: landing schemas are shared module-level constants.
    """
    if CORRUPT_COLUMN in schema.fieldNames():
        return schema
    return StructType([*schema.fields, StructField(CORRUPT_COLUMN, StringType(), True)])


def _with_source_file(df: DataFrame) -> DataFrame:
    """Capture the file path from Spark's hidden metadata column.

    ``_metadata.file_path`` is evaluated in the file scan, works on Databricks
    serverless, and stays populated after a cache (unlike ``input_file_name()``).
    """
    return df.withColumn(SOURCE_COLUMN, F.col(FILE_PATH_COLUMN))


def read_csv(spark: SparkSession, path: str | Path, schema: StructType) -> DataFrame:
    """Read a landing CSV, validating the header and retaining malformed rows.

    ``enforceSchema=false`` makes Spark check the header against the schema and
    fail loudly on a mismatch — the default positional matching would silently
    shift a reordered export into the wrong columns. ``columnNameOfCorruptRecord``
    keeps the raw text of rows with the wrong number of fields.
    """
    return _with_source_file(
        spark.read.schema(with_corrupt_column(schema))
        .option("header", "true")
        .option("enforceSchema", "false")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_COLUMN)
        .csv(str(path))
    )


def read_json_lines(spark: SparkSession, path: str | Path, schema: StructType) -> DataFrame:
    """Read JSON Lines, retaining malformed lines and their source file.

    One JSON object per line, no page envelope or top-level array. In PERMISSIVE
    mode Spark puts an unparseable line — including a field with the wrong type
    — into ``_corrupt_record`` instead of silently turning it into nulls.
    """
    return _with_source_file(
        spark.read.schema(with_corrupt_column(schema))
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_COLUMN)
        .json(str(path))
    )


def read_parquet(spark: SparkSession, path: str | Path, schema: StructType) -> DataFrame:
    """Read Parquet with the source file attached.

    Parquet is schema-enforced, so there are no malformed rows to capture.
    """
    return _with_source_file(spark.read.schema(schema).parquet(str(path)))


# Entities that arrive as full snapshots (every export holds the whole table), and the event
# entity that arrives in daily batch folders.
SNAPSHOT_ENTITIES = (
    "users",
    "apps",
    "vpn_rules",
    "ad_group",
    "ad_group_membership",
    "resource_resolution",
)
EVENT_ENTITY = "traffic_logs"

_LANDING_READERS = {
    "users": (read_json_lines, schemas.USERS),
    "apps": (read_json_lines, schemas.APPS),
    "vpn_rules": (read_json_lines, schemas.VPN_RULES),
    "ad_group": (read_csv, schemas.AD_GROUP),
    "ad_group_membership": (read_csv, schemas.AD_GROUP_MEMBERSHIP),
    "resource_resolution": (read_parquet, schemas.RESOURCE_RESOLUTION),
    "traffic_logs": (read_csv, schemas.TRAFFIC_LOGS),
}


def read_landing(spark: SparkSession, root: str | Path, entity: str) -> DataFrame:
    """Read one landing entity with its format's reader and its landing schema.

    For ``traffic_logs`` this reads the root folder, so partition discovery adds ``batch_date``
    (a ``DateType`` column, parsed from the ``batch_date=YYYY-MM-DD`` folder names).
    """
    reader, schema = _LANDING_READERS[entity]
    return reader(spark, landing_path(root, entity), schema)


def read_traffic_batches(
    spark: SparkSession, root: str | Path, start: date, end: date
) -> DataFrame:
    """Traffic log batches with ``start <= batch_date <= end``.

    The filter is on the partition column, so Spark prunes folders outside the range before
    reading a single file (partition pruning), and the remaining CSVs are still parsed in full.
    """
    return read_landing(spark, root, EVENT_ENTITY).where(
        F.col(BATCH_COLUMN).between(F.lit(start), F.lit(end))
    )


# --- Delta tables --------------------------------------------------------------------------------
# A "target" is either a Unity Catalog name (``netusage.bronze.users``, on Databricks) or a folder
# path (``/…/data/bronze/users``, locally, where there is no catalog). Paths contain "/", table
# names never do.


def table_target(base: str | Path, name: str) -> str:
    """Join a schema (``netusage.bronze``) or a folder (``/…/data/bronze``) with a table name."""
    base = str(base)
    return f"{base.rstrip('/')}/{name}" if "/" in base else f"{base}.{name}"


def _is_path(target: str) -> bool:
    return "/" in target


def read_delta(spark: SparkSession, target: str) -> DataFrame:
    if _is_path(target):
        return spark.read.format("delta").load(target)
    return spark.read.table(target)


def replace_delta(df: DataFrame, target: str, condition: str) -> None:
    """Atomically replace the rows matching ``condition`` with the rows of ``df``.

    Delta's ``replaceWhere``: in one commit, remove every existing row matching ``condition`` and
    add all of ``df``. Rows outside the condition are untouched, so re-running the same slice is
    idempotent. Once the table exists, Delta also checks that every row of ``df`` matches
    ``condition`` and fails the whole write otherwise (``DELTA_REPLACE_WHERE_MISMATCH``). The first
    write creates the table and does *not* check, so callers must still filter ``df`` correctly.

    Writes every column of ``df``: for a CSV landing read, that forces a full parse of each line,
    which is what makes ``_corrupt_record`` reliable in bronze (see ``corrupt_rows``).
    """
    writer = df.write.format("delta").mode("overwrite").option("replaceWhere", condition)
    if _is_path(target):
        writer.save(target)
    else:
        writer.saveAsTable(target)


def corrupt_rows(df: DataFrame) -> DataFrame:
    """Filter a landing read down to rows Spark could not parse.

    Only reliable when every column gets parsed. On a raw CSV read, Spark's CSV column pruning
    parses only the columns a query needs, and a row with an extra field then goes unflagged.
    So count corrupt rows after a cache or in bronze, never with an aggregation on the raw files.

    This only filters; it deliberately does not cache. Serverless compute does
    not support ``cache()``/``persist()``, and hiding that memory decision inside
    a helper prevents callers from unpersisting it. Locally, cache the DataFrame
    in the caller to work around Spark's "only the corrupt column" restriction.
    In production, write the parsed DataFrame to bronze Delta and query corrupt
    rows there — Delta has no such restriction.
    """
    if CORRUPT_COLUMN not in df.columns:
        raise ValueError(f"{CORRUPT_COLUMN} is missing; use a landing reader from this module")
    return df.where(F.col(CORRUPT_COLUMN).isNotNull())


# --- Landing writers (used by the synthetic generator) -------------------------------------------


def write_csv_lines(df: DataFrame, path: str | Path, partition_by: str | None = None) -> None:
    """Write ``df`` as header + comma-joined lines, without any quoting.

    Why not ``df.write.csv``: Spark's CSV writer quotes any value containing the delimiter, so it
    can never produce a row with the wrong field count, and the generator needs exactly that to
    plant malformed rows. Here the caller owns the format: values must not contain commas or
    newlines unless a broken row is intended. NULL is written as an empty field.

    Every output file starts with the header line (``enforceSchema=false`` checks the first line
    of each file):

    - ``partition_by`` set: one folder and one file per value (``batch_date=2025-01-01/``).
      ``repartition`` puts all rows of a value in one task, and ``sortWithinPartitions`` with the
      header's ``_order = 0`` puts the header first. The writer only re-sorts when the existing
      order doesn't already start with the partition column, so this order survives.
    - otherwise: a single file (fine up to the ~20M-row memberships at full scale).

    Spark's text writer names files ``part-*.txt``; Spark's CSV reader doesn't care.
    """
    columns = [c for c in df.columns if c != partition_by]
    line = F.concat_ws(",", *[F.coalesce(F.col(c).cast("string"), F.lit("")) for c in columns])
    keys = [partition_by] if partition_by else []
    body = df.select(*keys, line.alias("value"), F.lit(1).alias("_order"))

    header_value = F.lit(",".join(columns)).alias("value")
    if partition_by:
        header = df.select(partition_by).distinct().select(partition_by, header_value)
        out = (
            body.unionByName(header.withColumn("_order", F.lit(0)))
            .repartition(partition_by)
            .sortWithinPartitions(partition_by, "_order")
        )
    else:
        header = df.sparkSession.range(1).select(header_value, F.lit(0).alias("_order"))
        out = body.unionByName(header).coalesce(1).sortWithinPartitions("_order")

    writer = out.drop("_order").write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.text(str(path))


def write_json_lines(df: DataFrame, path: str | Path, num_files: int = 1) -> None:
    """One JSON object per line. NULL fields are omitted (Spark's default)."""
    df.coalesce(num_files).write.mode("overwrite").json(str(path))


def write_parquet(df: DataFrame, path: str | Path, num_files: int = 1) -> None:
    df.coalesce(num_files).write.mode("overwrite").parquet(str(path))
