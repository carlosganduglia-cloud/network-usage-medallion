"""Generator integration tests: generate TINY landing data once, then validate it.

This is also the first end-to-end test of the landing readers on generated files: header
checks per daily batch file, _corrupt_record capture, _metadata provenance, and partition
discovery of batch_date folders.
"""

import pytest
from pyspark.sql import functions as F

from netusage import config as C
from netusage import generator, io, schemas
from netusage.validation import report, validate


@pytest.fixture
def landing(tiny_landing):
    return tiny_landing.root


def test_generated_landing_passes_validation(spark, tiny_landing):
    root, start, end = tiny_landing
    checks = validate(spark, root, C.TINY, start, end)
    assert all(c.ok for c in checks), "\n" + report(checks)


def test_every_traffic_batch_file_starts_with_the_header(landing):
    expected = ",".join(schemas.TRAFFIC_LOGS.fieldNames())
    batch_dirs = sorted((landing / "traffic_logs").glob(f"{io.BATCH_COLUMN}=*"))
    assert batch_dirs
    for batch in batch_dirs:
        files = list(batch.glob("part-*"))
        assert len(files) == 1, batch
        assert files[0].read_text().splitlines()[0] == expected


def test_readers_capture_planted_malformed_rows(spark, landing):
    traffic = io.read_csv(spark, io.landing_path(landing, "traffic_logs"), schemas.TRAFFIC_LOGS)
    bad = io.corrupt_rows(traffic.cache()).collect()

    assert bad, "the generator plants malformed rows; the reader should capture some"
    assert all(r[io.CORRUPT_COLUMN].endswith(generator.MALFORMED_SUFFIX) for r in bad)
    assert all(f"{io.BATCH_COLUMN}=" in r[io.SOURCE_COLUMN] for r in bad)


def test_generation_is_deterministic(spark):
    """Recomputing a shuffled DataFrame gives identical rows (hash-based randomness)."""
    first = generator.clean_memberships(spark, C.TINY).repartition(3)
    second = generator.clean_memberships(spark, C.TINY).repartition(7)
    assert first.exceptAll(second).count() == 0
    assert second.exceptAll(first).count() == 0


def test_resolution_ips_match_traffic_ips_at_event_time(spark, landing):
    """The generator computes traffic IPs arithmetically. Every clean ALLOW event's destination
    must fall in *some* resolution range that's valid at that instant. This is a coarse
    sanity check, not the matching transform: it ignores rules, ports and groups."""
    traffic = io.read_csv(spark, io.landing_path(landing, "traffic_logs"), schemas.TRAFFIC_LOGS)
    res = io.read_parquet(
        spark, io.landing_path(landing, "resource_resolution"), schemas.RESOURCE_RESOLUTION
    )
    octets = F.split("destination_ip", r"\.")
    ip = sum(octets.getItem(i).cast("long") * (256 ** (3 - i)) for i in range(4))
    events = traffic.where(F.col(io.CORRUPT_COLUMN).isNull()).select(
        "event_id",
        ip.alias("ip"),
        F.to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ssX").alias("ts"),
    )
    in_range = (
        (F.col("ip").between(F.col("ip_start"), F.col("ip_end")))
        & (F.col("ts") >= F.col("first_seen"))
        & (F.col("last_seen").isNull() | (F.col("ts") < F.col("last_seen")))
    )
    unmatched = events.join(res, in_range, "left_anti").count()
    assert unmatched == 0
