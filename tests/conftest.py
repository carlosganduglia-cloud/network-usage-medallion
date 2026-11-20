import pytest
from pyspark.sql import SparkSession

from netusage.session import delta_builder


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """One small local SparkSession with Delta, shared by all tests (startup is the slow part)."""
    session = (
        delta_builder("netusage-tests", master="local[2]", partitions=2)
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
