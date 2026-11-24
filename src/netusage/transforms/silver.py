from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


def normalize_port(df: DataFrame) -> DataFrame:
    """
    Normalize the port field into port_start and port_end.

    Supported formats (surrounding whitespace is trimmed):
        - "49"    -> port_start=49, port_end=49
        - "1-20"  -> port_start=1, port_end=20

    Valid ports are 1-65535 (port 0 is reserved in TCP/UDP and rejected).
    Rows with anything else are dropped. Values are try_cast rather than
    cast so out-of-INT-range input (e.g. "99999999999") yields null and
    gets filtered instead of raising CAST_INVALID_INPUT under ANSI mode
    (Spark 4 default).
    """
    port = F.trim(F.col("port"))
    ports = F.split(port, "-")
    start = ports.getItem(0).try_cast("int")
    end = F.when(port.contains("-"), ports.getItem(1).try_cast("int")).otherwise(start)

    is_valid_port = (
        port.rlike(r"^\d+(-\d+)?$")
        & (start >= 1)
        & (start <= 65535)
        & (end >= 1)
        & (end <= 65535)
        & (start <= end)
    )

    return (
        df.filter(is_valid_port)
        .withColumn("port_start", start)
        .withColumn("port_end", end)
        .drop("port")
    )


# An octet with no leading zeros: "0" and "10" match, "00" and "010" do not. Leading zeros are
# ambiguous ("010" is 10 in decimal but 8 in octal in some parsers), so they are rejected
# outright; the generator writes canonical dotted-decimal.
_IP_OCTET = r"(0|[1-9]\d*)"
_IP_REGEX = rf"^{_IP_OCTET}\.{_IP_OCTET}\.{_IP_OCTET}\.{_IP_OCTET}$"


def ip_to_long(ip: Column | str) -> Column:
    """
    Convert a dotted-decimal IPv4 string column into its LongType integer value.

    "a.b.c.d" -> a*256**3 + b*256**2 + c*256 + d, e.g. "10.0.0.1" -> 167772161. The result is
    LongType because addresses above 127.255.255.255 overflow a signed int ("192.168.1.1" ->
    3232235777), which is why resource_resolution.ip_start/ip_end are longs too. Each octet is
    try_cast to long BEFORE multiplying, so int arithmetic can never overflow.

    Decisions (documented so callers and tests can rely on them):
    - Surrounding whitespace is trimmed, same as the port transform (" 10.0.0.1 " is valid).
    - Leading zeros ("010.0.0.1") are rejected: canonical dotted-decimal has none, they are
      ambiguous with octal, and the generator never writes them.
    - Anything invalid (wrong shape, octet > 255, digits overflowing a long) returns null
      instead of raising: try_cast returns null on failure and the null comparisons reject the
      row even under ANSI mode. Returning null (rather than filtering) keeps this composable
      with the other silver expressions in one select, and leaves the quarantine decision to
      the caller.

    Unlike normalize_port, this is not tied to a column name.
    """
    ip = F.col(ip) if isinstance(ip, str) else ip
    trimmed = F.trim(ip)
    parts = F.split(trimmed, r"\.")
    octets = [parts.getItem(i).try_cast("long") for i in range(4)]

    is_valid_ip = (
        trimmed.rlike(_IP_REGEX)
        & (octets[0] >= 0)
        & (octets[0] <= 255)
        & (octets[1] >= 0)
        & (octets[1] <= 255)
        & (octets[2] >= 0)
        & (octets[2] <= 255)
        & (octets[3] >= 0)
        & (octets[3] <= 255)
    )
    value = (
        octets[0] * F.lit(256**3) + octets[1] * F.lit(256**2) + octets[2] * F.lit(256) + octets[3]
    )
    # No otherwise: invalid or null input yields null.
    return F.when(is_valid_ip, value)
