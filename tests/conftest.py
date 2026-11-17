import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """One small local SparkSession shared by all tests (startup is the slow part)."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("netusage-tests")
        .config("spark.sql.shuffle.partitions", "2")  # default 200 is overkill for tiny data
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()
