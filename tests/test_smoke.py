from delta.tables import DeltaTable
from pyspark.sql import functions as F


def test_spark_runs(spark):
    df = spark.createDataFrame(
        [("u1", 100), ("u1", 50), ("u2", 10)],
        "user_id string, bytes long",
    )
    totals = {
        r.user_id: r.total
        for r in df.groupBy("user_id").agg(F.sum("bytes").alias("total")).collect()
    }
    assert totals == {"u1": 150, "u2": 10}


def test_delta_write_and_merge(spark, tmp_path):
    """Delta works locally: the jar loaded and matches this Spark version."""
    path = str(tmp_path / "events")
    schema = "event_id string, action string"
    spark.createDataFrame([("e1", "ALLOW"), ("e2", "DENY")], schema).write.format("delta").save(
        path
    )

    updates = spark.createDataFrame([("e2", "DENY"), ("e3", "ALLOW")], schema)
    (
        DeltaTable.forPath(spark, path)
        .alias("t")
        .merge(updates.alias("s"), "t.event_id = s.event_id")
        .whenNotMatchedInsertAll()
        .execute()
    )

    ids = {r.event_id for r in spark.read.format("delta").load(path).collect()}
    assert ids == {"e1", "e2", "e3"}
