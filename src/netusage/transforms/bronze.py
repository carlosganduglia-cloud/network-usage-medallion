"""Bronze transforms: raw landing rows plus ingest metadata, and per-batch audit counts.

Pure functions, no I/O. Values that vary per run (``ingest_ts``, ``batch_date``) are passed in
rather than computed here, so a run stamps one ``ingest_ts`` on every table it writes, and tests
are deterministic.
"""

from datetime import date, datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from netusage.io import BATCH_COLUMN, CORRUPT_COLUMN, SOURCE_COLUMN

INGEST_TS_COLUMN = "ingest_ts"
# Metadata goes after the landing columns, in this order, in every bronze table.
METADATA_COLUMNS = (CORRUPT_COLUMN, SOURCE_COLUMN, BATCH_COLUMN, INGEST_TS_COLUMN)


def add_ingest_metadata(
    df: DataFrame, ingest_ts: datetime, batch_date: date | None = None
) -> DataFrame:
    """Add ``batch_date`` and ``ingest_ts``; move all metadata columns to the end.

    - Snapshot entities have no batch in landing: pass the run's ``batch_date``.
    - Traffic logs already carry ``batch_date`` from their landing folder: leave it ``None``.

    ``ingest_ts`` must be timezone-aware so it means the same instant in any session time zone.
    """
    if ingest_ts.tzinfo is None:
        raise ValueError("ingest_ts must be timezone-aware (e.g. datetime.now(UTC))")
    if batch_date is not None:
        df = df.withColumn(BATCH_COLUMN, F.lit(batch_date))
    elif BATCH_COLUMN not in df.columns:
        raise ValueError(f"{BATCH_COLUMN} is missing: pass batch_date for snapshot entities")

    df = df.withColumn(INGEST_TS_COLUMN, F.lit(ingest_ts))
    data = [c for c in df.columns if c not in METADATA_COLUMNS]
    meta = [c for c in METADATA_COLUMNS if c in df.columns]  # Parquet has no _corrupt_record
    return df.select(*data, *meta)


def audit_counts(bronze: DataFrame, entity: str) -> DataFrame:
    """One row per ``batch_date``: total rows and rows Spark could not parse.

    Run this on rows read back from the bronze Delta table, not on a raw landing read: there,
    every column has been parsed and stored, so CSV column pruning can't hide malformed rows.
    """
    corrupt = (
        F.sum(F.col(CORRUPT_COLUMN).isNotNull().cast("long"))
        if CORRUPT_COLUMN in bronze.columns
        else F.lit(0).cast("long")
    )
    return (
        bronze.groupBy(BATCH_COLUMN)
        .agg(
            F.lit(entity).alias("entity"),
            F.max(INGEST_TS_COLUMN).alias(INGEST_TS_COLUMN),
            F.count(F.lit(1)).alias("rows"),
            corrupt.alias("corrupt_rows"),
        )
        .select("entity", BATCH_COLUMN, INGEST_TS_COLUMN, "rows", "corrupt_rows")
    )
