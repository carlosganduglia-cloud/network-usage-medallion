"""Local SparkSessions with Delta Lake, for command-line runs and tests.

On Databricks the session already exists as ``spark`` (Delta built in); this module is never
imported there.

Locally Delta is not part of Spark: ``configure_spark_with_delta_pip`` adds the Delta jar
(``io.delta:delta-spark_2.13``, matching the installed ``delta-spark`` package) to
``spark.jars.packages``. Spark downloads it from Maven Central on the first start and caches it
under ``~/.ivy2.5.2``. Like every JVM setting, this only works if this builder starts the JVM: a
session that already exists is returned as-is by ``getOrCreate()``.
"""

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from netusage import config as C


def delta_builder(
    app_name: str, master: str = "local[*]", partitions: int = 16
) -> SparkSession.Builder:
    """A builder with Delta enabled and partition counts sized for one machine.

    ``partitions`` sets both Spark's shuffle partitions (default 200) and Delta's
    ``snapshotPartitions`` (default 50: the tasks Delta uses to rebuild a table's state from
    ``_delta_log`` on every read/write). The defaults suit clusters; locally they mean dozens of
    near-empty tasks per operation.
    """
    builder = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.databricks.delta.snapshotPartitions", str(partitions))
        .config("spark.sql.session.timeZone", C.TIMEZONE)
    )
    return configure_spark_with_delta_pip(builder)


def local_spark(app_name: str = "netusage") -> SparkSession:
    """All local cores, with the Spark UI on http://localhost:4040 while the job runs."""
    return (
        delta_builder(app_name)
        .config("spark.driver.memory", "4g")  # only effective because the JVM starts here
        .getOrCreate()
    )
