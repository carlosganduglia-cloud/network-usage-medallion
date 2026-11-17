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
