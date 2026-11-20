"""Tiny unit tests for normalize_ports."""

from netusage.transforms.silver import normalize_ports


def _normalize(spark, ports):
    """Build a one-column DataFrame from raw port strings and normalize it."""
    return normalize_ports(spark.createDataFrame([(p,) for p in ports], "port string"))


def _as_dict(df):
    """Collect as {raw_port: (start, end, flag)} so row order never matters."""
    return {
        r.port: (r.port_start, r.port_end, r.port_spec_valid)
        for r in df.select("port", "port_start", "port_end", "port_spec_valid").collect()
    }


def test_valid_single_port_and_range(spark):
    result = _as_dict(_normalize(spark, ["99", "1-9"]))

    assert result == {
        "99": (99, 99, True),
        "1-9": (1, 9, True),
    }


def test_invalid_values_do_not_throw_and_are_flagged(spark):
    # "99999999999" overflows try_cast; the flag must be False, not NULL.
    ports = [
        # malformed
        "abc",
        "80-abc",
        "-20",
        "80-",
        "1-2-3",
        "",
        "   ",
        None,
        # out of range (port 0 is reserved)
        "0",
        "0-65535",
        "65536",
        "1-65536",
        "65536-65536",
        # start > end
        "100-1",
        # overflow try_cast to NULL
        "99999999999",
        "1-99999999999",
        "99999999999-1",
    ]
    result = _as_dict(_normalize(spark, ports))

    assert result == {port: (None, None, False) for port in ports}


def test_surrounding_whitespace_is_trimmed(spark):
    result = _as_dict(_normalize(spark, ["  1-9  "]))

    assert result == {"  1-9  ": (1, 9, True)}


def test_boundaries_are_inclusive(spark):
    result = _as_dict(_normalize(spark, ["1", "65535", "1-65535"]))

    assert result == {
        "1": (1, 1, True),
        "65535": (65535, 65535, True),
        "1-65535": (1, 65535, True),
    }


def test_output_columns_are_integers_and_raw_port_is_kept(spark):
    schema = _normalize(spark, ["1-20"]).schema

    assert schema["port"].dataType.simpleString() == "string"
    assert schema["port_start"].dataType.simpleString() == "int"
    assert schema["port_end"].dataType.simpleString() == "int"
