from pyspark.sql import Row

from netusage.transforms.silver import normalize_port


def test_single_port(spark):
    """A single port should produce identical start and end values."""

    df = spark.createDataFrame([("49",), ("80",), ("443",)], ["port"])

    result = normalize_port(df)

    expected = [
        Row(port_start=49, port_end=49),
        Row(port_start=80, port_end=80),
        Row(port_start=443, port_end=443),
    ]

    assert result.collect() == expected


def test_port_range(spark):
    """A port range should produce the correct start and end values."""

    df = spark.createDataFrame([("1-20",), ("100-200",), ("500-1000",)], ["port"])

    result = normalize_port(df)

    expected = [
        Row(port_start=1, port_end=20),
        Row(port_start=100, port_end=200),
        Row(port_start=500, port_end=1000),
    ]

    assert result.collect() == expected


def test_boundary_ports(spark):
    """Port values at the valid boundaries (1 and 65535) should be accepted."""

    df = spark.createDataFrame([("1",), ("65535",), ("1-65535",)], ["port"])

    result = normalize_port(df)

    expected = [
        Row(port_start=1, port_end=1),
        Row(port_start=65535, port_end=65535),
        Row(port_start=1, port_end=65535),
    ]

    assert result.collect() == expected


def test_padded_ports_are_trimmed(spark):
    """Whitespace-padded ports should be trimmed and accepted."""

    df = spark.createDataFrame([(" 443 ",), (" 1-20 ",)], ["port"])

    result = normalize_port(df)

    expected = [
        Row(port_start=443, port_end=443),
        Row(port_start=1, port_end=20),
    ]

    assert result.collect() == expected


def test_invalid_port_values_are_filtered(spark):
    """Malformed port values should be filtered out."""

    df = spark.createDataFrame(
        [
            ("abc",),
            ("80-abc",),
            ("-20",),
            ("80-",),
            ("80-90-100",),
            ("",),
            ("   ",),
        ],
        ["port"],
    )

    result = normalize_port(df)

    assert result.count() == 0


def test_integer_overflow_values_are_filtered(spark):
    """Values that overflow INT should be filtered, not crash the cast (ANSI mode)."""

    df = spark.createDataFrame(
        [
            ("99999999999",),
            ("1-99999999999",),
            ("99999999999-1",),
        ],
        ["port"],
    )

    result = normalize_port(df)

    assert result.count() == 0


def test_out_of_range_ports_are_filtered(spark):
    """Ports outside 1-65535 should be filtered out (port 0 is reserved)."""

    df = spark.createDataFrame(
        [
            ("0",),
            ("0-65535",),
            ("-1",),
            ("65536",),
            ("1-65536",),
            ("65536-65536",),
        ],
        ["port"],
    )

    result = normalize_port(df)

    assert result.count() == 0


def test_invalid_range_order_is_filtered(spark):
    """A range where start is greater than end should be filtered."""

    df = spark.createDataFrame(
        [
            ("20-10",),
            ("100-50",),
        ],
        ["port"],
    )

    result = normalize_port(df)

    assert result.count() == 0


def test_mixed_valid_and_invalid_ports(spark):
    """Only valid rows should remain after normalization."""

    df = spark.createDataFrame(
        [
            ("49",),
            ("1-20",),
            ("invalid",),
            ("65536",),
            ("100-200",),
            ("99999999999",),
        ],
        ["port"],
    )

    result = normalize_port(df)

    expected = [
        Row(port_start=49, port_end=49),
        Row(port_start=1, port_end=20),
        Row(port_start=100, port_end=200),
    ]

    assert result.collect() == expected


def test_original_port_column_is_removed(spark):
    """The original port column should not exist in the output."""

    df = spark.createDataFrame([("49",)], ["port"])

    result = normalize_port(df)

    assert "port" not in result.columns
    assert "port_start" in result.columns
    assert "port_end" in result.columns


def test_output_columns_are_integers(spark):
    """Normalized columns should have IntegerType."""

    df = spark.createDataFrame([("1-20",)], ["port"])

    result = normalize_port(df)

    assert result.schema["port_start"].dataType.simpleString() == "int"
    assert result.schema["port_end"].dataType.simpleString() == "int"
