from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import pytest
from pyspark.sql import SparkSession

from netusage import config as C
from netusage import generator
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


class Landing(NamedTuple):
    root: Path
    start: date
    end: date


@pytest.fixture(scope="session")
def tiny_landing(spark, tmp_path_factory) -> Landing:
    """TINY landing data over 5 days, generated once per test run (~1 min) and shared.

    Read-only for tests: write anything derived (bronze tables) under your own tmp_path.
    """
    start = date(2025, 1, 1)
    end = start + timedelta(days=4)
    root = tmp_path_factory.mktemp("landing")
    generator.generate(spark, C.TINY, root, start, end)
    return Landing(root, start, end)
