"""Bronze ingest: metadata transform, Delta replace semantics, audit counts, idempotent reruns."""

from datetime import UTC, date, datetime, timedelta

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from netusage import io
from netusage.jobs import bronze as job
from netusage.transforms.bronze import INGEST_TS_COLUMN, add_ingest_metadata, audit_counts

TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
DAY = date(2025, 3, 1)
SCHEMA = StructType(
    [StructField("event_id", StringType()), StructField("user_email", StringType())]
)


# --- Transform: add_ingest_metadata --------------------------------------------------------------


def test_snapshot_gets_run_batch_date_and_metadata_at_the_end(spark):
    df = spark.createDataFrame([Row(**{"id": "u1", io.SOURCE_COLUMN: "f.json", "name": "a"})])

    out = add_ingest_metadata(df, TS, batch_date=DAY)

    assert out.columns == ["id", "name", io.SOURCE_COLUMN, io.BATCH_COLUMN, INGEST_TS_COLUMN]
    row = out.first()
    assert row[io.BATCH_COLUMN] == DAY
    assert row[INGEST_TS_COLUMN].astimezone(UTC) == TS


def test_traffic_keeps_batch_date_from_its_landing_folder(spark):
    df = spark.createDataFrame([Row(**{io.BATCH_COLUMN: DAY, "event_id": "e1"})])

    out = add_ingest_metadata(df, TS)

    assert out.first()[io.BATCH_COLUMN] == DAY
    assert out.columns == ["event_id", io.BATCH_COLUMN, INGEST_TS_COLUMN]


def test_missing_batch_date_and_naive_timestamp_are_rejected(spark):
    df = spark.createDataFrame([Row(event_id="e1")])
    with pytest.raises(ValueError, match="batch_date is missing"):
        add_ingest_metadata(df, TS)
    with pytest.raises(ValueError, match="timezone-aware"):
        add_ingest_metadata(df, TS.replace(tzinfo=None), batch_date=DAY)


# --- Transform: audit_counts ---------------------------------------------------------------------


def test_audit_counts_rows_and_corrupt_rows_per_batch(spark):
    rows = [
        (DAY, TS, None),
        (DAY, TS, "e2,broken"),
        (DAY + timedelta(days=1), TS, None),
    ]
    df = spark.createDataFrame(
        rows, f"{io.BATCH_COLUMN} date, {INGEST_TS_COLUMN} timestamp, {io.CORRUPT_COLUMN} string"
    )

    got = {r[io.BATCH_COLUMN]: r for r in audit_counts(df, "traffic_logs").collect()}

    assert (got[DAY].rows, got[DAY].corrupt_rows) == (2, 1)
    assert got[DAY + timedelta(days=1)].corrupt_rows == 0
    assert got[DAY].entity == "traffic_logs"


def test_audit_counts_zero_corrupt_rows_for_parquet_sources(spark):
    df = spark.createDataFrame([(DAY, TS)], f"{io.BATCH_COLUMN} date, {INGEST_TS_COLUMN} timestamp")
    assert audit_counts(df, "resource_resolution").first().corrupt_rows == 0


# --- io: Delta targets and replaceWhere ----------------------------------------------------------


def test_table_target_is_a_catalog_name_or_a_path():
    assert io.table_target("netusage.bronze", "users") == "netusage.bronze.users"
    assert io.table_target("/data/bronze/", "users") == "/data/bronze/users"


def test_replace_delta_replaces_only_the_matching_slice(spark, tmp_path):
    target = str(tmp_path / "t")
    schema = f"{io.BATCH_COLUMN} date, v string"
    d1, d2 = DAY, DAY + timedelta(days=1)
    io.replace_delta(
        spark.createDataFrame([(d1, "old1"), (d2, "old2")], schema), target, "v IS NOT NULL"
    )

    io.replace_delta(
        spark.createDataFrame([(d2, "new2a"), (d2, "new2b")], schema),
        target,
        f"{io.BATCH_COLUMN} = '{d2}'",
    )

    got = sorted((r[0], r[1]) for r in io.read_delta(spark, target).collect())
    assert got == [(d1, "old1"), (d2, "new2a"), (d2, "new2b")]


def test_replace_delta_rejects_rows_outside_the_condition(spark, tmp_path):
    """Delta's safety net: a buggy job can't write outside the slice it declared.

    Only once the table exists: the first write creates it without checking the condition.
    """
    target = str(tmp_path / "t")
    schema = f"{io.BATCH_COLUMN} date, v string"
    io.replace_delta(spark.createDataFrame([(DAY, "first")], schema), target, "v IS NOT NULL")

    outside = spark.createDataFrame([(DAY, "x")], schema)
    with pytest.raises(Exception, match="DELTA_REPLACE_WHERE_MISMATCH"):
        io.replace_delta(outside, target, f"{io.BATCH_COLUMN} = '{DAY + timedelta(1)}'")
    assert [r.v for r in io.read_delta(spark, target).collect()] == ["first"]  # unchanged


def test_corrupt_rows_are_reliable_after_the_bronze_write(spark, tmp_path):
    """Counterpart of test_io's column-pruning gotcha: once written to Delta, an aggregation
    sees the extra-field row, because the write parsed every column (no cache needed)."""
    path = tmp_path / "extra.csv"
    path.write_text(
        "event_id,user_email\ne1,u1@corp.example\ne2,u2@corp.example,EXTRA\n", encoding="utf-8"
    )
    target = str(tmp_path / "bronze")
    df = add_ingest_metadata(io.read_csv(spark, path, SCHEMA), TS, batch_date=DAY)
    io.replace_delta(df, target, f"{io.BATCH_COLUMN} = '{DAY}'")

    flagged = F.sum(F.col(io.CORRUPT_COLUMN).isNotNull().cast("int"))
    assert io.read_delta(spark, target).agg(flagged).first()[0] == 1


# --- Job: end to end on TINY landing data --------------------------------------------------------


def _counts(spark, bronze, entity):
    df = io.read_delta(spark, io.table_target(bronze, entity))
    return {r[0]: r[1] for r in df.groupBy(io.BATCH_COLUMN).count().collect()}


def _audit(spark, bronze):
    df = io.read_delta(spark, io.table_target(bronze, job.AUDIT_TABLE))
    return {(r.entity, r.batch_date): (r.rows, r.corrupt_rows) for r in df.collect()}


def test_bronze_matches_landing_and_audit_matches_bronze(spark, tiny_landing, tmp_path):
    root, start, end = tiny_landing
    bronze = str(tmp_path / "bronze")

    job.run(spark, root, bronze, start, end, ingest_ts=TS)

    # Landing also has batches past `end` (late events): compare the same range. cache = full parse
    landing = io.read_traffic_batches(spark, root, start, end).cache()
    traffic = _counts(spark, bronze, io.EVENT_ENTITY)
    assert sum(traffic.values()) == landing.count()
    assert set(traffic) == {start + timedelta(d) for d in range((end - start).days + 1)}

    audit = _audit(spark, bronze)
    assert {d: audit[("traffic_logs", d)][0] for d in traffic} == traffic
    landing_corrupt = io.corrupt_rows(landing).count()
    assert landing_corrupt > 0  # the generator plants some
    assert sum(audit[("traffic_logs", d)][1] for d in traffic) == landing_corrupt

    # Snapshots: one copy each, stamped with the run's end date.
    for entity in io.SNAPSHOT_ENTITIES:
        assert set(_counts(spark, bronze, entity)) == {end}
        assert (entity, end) in audit


def test_rerunning_is_idempotent(spark, tiny_landing, tmp_path):
    root, start, end = tiny_landing
    bronze = str(tmp_path / "bronze")

    job.run(spark, root, bronze, start, end, ingest_ts=TS)
    before = {e: _counts(spark, bronze, e) for e in (io.EVENT_ENTITY, *io.SNAPSHOT_ENTITIES)}
    audit_before = _audit(spark, bronze)

    rerun_ts = TS + timedelta(hours=1)
    job.run(spark, root, bronze, start, end, ingest_ts=rerun_ts)

    after = {e: _counts(spark, bronze, e) for e in before}
    assert after == before
    assert _audit(spark, bronze) == audit_before
    stamps = io.read_delta(spark, io.table_target(bronze, io.EVENT_ENTITY)).select(INGEST_TS_COLUMN)
    assert {r[0].astimezone(UTC) for r in stamps.distinct().collect()} == {rerun_ts}


def test_daily_run_replaces_only_its_day(spark, tiny_landing, tmp_path):
    root, start, end = tiny_landing
    bronze = str(tmp_path / "bronze")
    job.run(spark, root, bronze, start, end, ingest_ts=TS)
    day = start + timedelta(days=2)
    daily_ts = TS + timedelta(days=1)

    audit = job.run(spark, root, bronze, day, day, load_snapshots=False, ingest_ts=daily_ts)

    traffic = io.read_delta(spark, io.table_target(bronze, io.EVENT_ENTITY))
    stamps = {
        r[0]: r[1].astimezone(UTC)
        for r in traffic.select(io.BATCH_COLUMN, INGEST_TS_COLUMN).distinct().collect()
    }
    assert stamps[day] == daily_ts
    assert all(ts == TS for d, ts in stamps.items() if d != day)
    assert [(r.entity, r.batch_date) for r in audit.collect()] == [("traffic_logs", day)]
    # Snapshots untouched: still only the first run's copy.
    assert set(_counts(spark, bronze, "users")) == {end}
