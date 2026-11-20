"""Bronze ingest: landing files -> bronze Delta tables, plus an audit row per entity and batch.

Re-runs are safe: each write replaces its slice of the table (Delta ``replaceWhere``) instead of
appending.

- ``traffic_logs``: the batches with ``start <= batch_date <= end``, replaced as one range. A daily
  run is ``start == end``; a backfill is one read and one write for the whole range.
- Snapshot entities (users, apps, rules, groups, memberships, resolutions): the current landing
  snapshot, stamped ``batch_date = end`` and replacing that date only. Bronze keeps one snapshot
  per run date; silver takes the latest. A backfill loads them once, not once per day.
- ``ingest_audit``: rows and ``_corrupt_record`` counts per entity and batch, counted from the
  bronze tables (where every column has been parsed) and replaced by the same slice.

Locally: ``python -m netusage.jobs.bronze --start 2025-01-01 --end 2025-01-31``.
"""

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from netusage import config as C
from netusage import io
from netusage.transforms.bronze import add_ingest_metadata, audit_counts

AUDIT_TABLE = "ingest_audit"


def _date_range(start: date, end: date) -> str:
    """A replaceWhere condition. Dates are ``date`` objects, so the literal is always ISO."""
    col = io.BATCH_COLUMN
    return f"{col} >= '{start.isoformat()}' AND {col} <= '{end.isoformat()}'"


def ingest_traffic(
    spark: SparkSession,
    landing_root: str | Path,
    bronze: str | Path,
    start: date,
    end: date,
    ingest_ts: datetime,
) -> str:
    """Replace the traffic batches in ``[start, end]``. Returns the replaced condition."""
    df = add_ingest_metadata(io.read_traffic_batches(spark, landing_root, start, end), ingest_ts)
    condition = _date_range(start, end)
    io.replace_delta(df, io.table_target(bronze, io.EVENT_ENTITY), condition)
    return condition


def ingest_snapshot(
    spark: SparkSession,
    landing_root: str | Path,
    bronze: str | Path,
    entity: str,
    batch_date: date,
    ingest_ts: datetime,
) -> str:
    """Replace the ``batch_date`` snapshot of one entity. Returns the replaced condition."""
    df = add_ingest_metadata(io.read_landing(spark, landing_root, entity), ingest_ts, batch_date)
    condition = _date_range(batch_date, batch_date)
    io.replace_delta(df, io.table_target(bronze, entity), condition)
    return condition


def write_audit(spark: SparkSession, bronze: str | Path, entity: str, condition: str) -> None:
    """Count the slice just written, reading it back from bronze, and replace its audit rows."""
    written = io.read_delta(spark, io.table_target(bronze, entity)).where(condition)
    io.replace_delta(
        audit_counts(written, entity),
        io.table_target(bronze, AUDIT_TABLE),
        f"entity = '{entity}' AND ({condition})",
    )


def run(
    spark: SparkSession,
    landing_root: str | Path,
    bronze: str | Path,
    start: date,
    end: date,
    load_snapshots: bool = True,
    ingest_ts: datetime | None = None,
) -> DataFrame:
    """Ingest one date range into bronze. Returns this run's audit rows."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    ingest_ts = ingest_ts or datetime.now(UTC)

    written = {io.EVENT_ENTITY: ingest_traffic(spark, landing_root, bronze, start, end, ingest_ts)}
    if load_snapshots:
        for entity in io.SNAPSHOT_ENTITIES:
            written[entity] = ingest_snapshot(spark, landing_root, bronze, entity, end, ingest_ts)

    for entity, condition in written.items():
        write_audit(spark, bronze, entity, condition)

    return (
        io.read_delta(spark, io.table_target(bronze, AUDIT_TABLE))
        .where(F.col("entity").isin(list(written)) & F.col("ingest_ts").eqNullSafe(ingest_ts))
        .orderBy("entity", io.BATCH_COLUMN)
    )


def main(argv: list[str] | None = None) -> None:
    from netusage.session import local_spark

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, help="default: same as --start")
    parser.add_argument("--landing", default=str(C.LOCAL_LANDING_PATH))
    parser.add_argument("--bronze", default=str(C.LOCAL_BRONZE_PATH))
    parser.add_argument("--no-snapshots", action="store_true", help="traffic logs only")
    args = parser.parse_args(argv)

    spark = local_spark("netusage-bronze")
    audit = run(
        spark,
        args.landing,
        args.bronze,
        args.start,
        args.end or args.start,
        load_snapshots=not args.no_snapshots,
    )
    audit.show(truncate=False)


if __name__ == "__main__":
    main()
