"""Local SparkSession for command-line runs (generator, validation).

On Databricks the session already exists as ``spark``; this module is never imported there.
"""

from pyspark.sql import SparkSession

from netusage import config as C


def local_spark(app_name: str = "netusage") -> SparkSession:
    """All local cores, with the Spark UI on http://localhost:4040 while the job runs."""
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "4g")  # only effective because the JVM starts here
        .config("spark.sql.shuffle.partitions", "16")  # default 200: too many tiny tasks locally
        .config("spark.sql.session.timeZone", C.TIMEZONE)
        .getOrCreate()
    )
