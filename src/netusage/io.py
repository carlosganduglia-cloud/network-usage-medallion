"""File I/O for the pipeline.

Transforms are pure: they take and return DataFrames. Every read of landing
files (and later every Delta read/write) goes through this module.
"""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

CORRUPT_COLUMN = "_corrupt_record"
SOURCE_COLUMN = "source_file"
FILE_PATH_COLUMN = "_metadata.file_path"


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


def corrupt_rows(df: DataFrame) -> DataFrame:
    """Filter a landing read down to rows Spark could not parse.

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
