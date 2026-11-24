from pyspark.sql import Row
from pyspark.sql import functions as F

from netusage import io
from netusage.transforms.silver import ip_to_long, normalize_port


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


def test_ip_to_long_known_values(spark):
    """A dotted-decimal IP converts to a*256**3 + b*256**2 + c*256 + d."""

    df = spark.createDataFrame([("10.0.0.1",), ("1.2.3.4",)], ["destination_ip"])

    result = df.select(ip_to_long("destination_ip").alias("ip_long"))

    expected = [
        Row(ip_long=167772161),
        Row(ip_long=16909060),
    ]

    assert result.collect() == expected


def test_ip_to_long_boundaries(spark):
    """0.0.0.0 and 255.255.255.255 are the valid boundaries."""

    df = spark.createDataFrame([("0.0.0.0",), ("255.255.255.255",)], ["destination_ip"])

    result = df.select(ip_to_long("destination_ip").alias("ip_long"))

    expected = [
        Row(ip_long=0),
        Row(ip_long=4294967295),
    ]

    assert result.collect() == expected


def test_ip_to_long_above_int_range(spark):
    """Addresses above 127.255.255.255 need long arithmetic: int math would overflow."""

    df = spark.createDataFrame([("192.168.1.1",)], ["destination_ip"])

    result = df.select(ip_to_long("destination_ip").alias("ip_long"))

    assert result.first() == Row(ip_long=3232235777)


def test_ip_to_long_trims_whitespace(spark):
    """Whitespace-padded IPs are trimmed, same decision as the port transform."""

    df = spark.createDataFrame([(" 10.0.0.1 ",)], ["destination_ip"])

    result = df.select(ip_to_long("destination_ip").alias("ip_long"))

    assert result.first() == Row(ip_long=167772161)


def test_ip_to_long_invalid_returns_null(spark):
    """Malformed or unparseable IPs return null instead of raising; the caller filters."""

    df = spark.createDataFrame(
        [
            ("256.0.0.1",),
            ("10.0.0",),
            ("10.0.0.",),
            ("10..0.1",),
            ("10.0.0.1.5",),
            ("a.b.c.d",),
            ("",),
            ("   ",),
            (None,),
            ("99999999999999999999.1.1.1",),  # digits that overflow a long: try_cast -> null
        ],
        ["destination_ip"],
    )

    result = df.select(ip_to_long("destination_ip").alias("ip_long"))

    assert result.collect() == [Row(ip_long=None) for _ in range(10)]


def test_ip_to_long_rejects_leading_zeros(spark):
    """Leading zeros are rejected as ambiguous (octal); see the function docstring."""

    df = spark.createDataFrame(
        [
            ("010.0.0.1",),
            ("10.01.0.1",),
            ("00.0.0.0",),
        ],
        ["destination_ip"],
    )

    result = df.select(ip_to_long("destination_ip").alias("ip_long"))

    assert result.collect() == [Row(ip_long=None) for _ in range(3)]


def test_ip_to_long_returns_long_type(spark):
    """The converted column is LongType, matching resource_resolution.ip_start/ip_end."""

    df = spark.createDataFrame([("10.0.0.1",)], ["destination_ip"])

    result = df.select(ip_to_long("destination_ip").alias("ip_long"))

    assert result.schema["ip_long"].dataType.simpleString() == "bigint"


def test_traffic_ips_fall_in_resolution_ranges(spark, tiny_landing):
    """Confidence check on real pipeline data: every traffic IP matches a resolution range.

    The generator picks each event's destination IP from the resource's IP range at the
    event's instant, so ip_to_long(traffic_logs.destination_ip) must land inside some
    (ip_start, ip_end) in resource_resolution. The planted malformed rows only append to the
    last column (action), so their destination_ip values are unaffected and the check can run
    on the whole read without the corrupt-record caveat from tests/test_io.py.
    """
    root = tiny_landing.root

    ips = (
        io.read_landing(spark, root, io.EVENT_ENTITY)
        .select(ip_to_long("destination_ip").alias("ip"))
        .distinct()
    )
    # The generator writes canonical dotted-decimal, so the conversion must never yield null.
    assert ips.where(F.col("ip").isNull()).count() == 0

    ranges = io.read_landing(spark, root, "resource_resolution").select("ip_start", "ip_end")
    unmatched = ips.join(
        ranges,
        (ips["ip"] >= ranges["ip_start"]) & (ips["ip"] <= ranges["ip_end"]),
        "left_anti",
    )

    assert ips.count() > 0
    assert unmatched.count() == 0
