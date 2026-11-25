"""Silver transforms: pure functions that normalize bronze columns.

Bad values are flagged in an output column, not dropped here; callers decide
whether to quarantine them or let downstream filters exclude them.
"""

from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as F

PORT_SPEC = r"^[0-9]+(?:-[0-9]+)?\z"


def normalize_ports(df: DataFrame) -> DataFrame:
    """Normalize a raw ``port`` string into integer start/end columns.

    Valid formats are ``N`` and ``N-M`` with 1 <= N <= M <= 65535. The original
    ``port`` column is preserved. Invalid rows get null start/end values and
    ``port_spec_valid = false``; they are flagged, not dropped. ``coalesce``
    keeps the flag strictly boolean when an all-digit value overflows
    ``try_cast("int")`` to null.
    """
    port = F.trim(F.col("port"))
    parts = F.split(port, "-")
    start = F.get(parts, 0).try_cast("int")
    # Branch on structure, not on a NULL: try_cast also returns NULL when the second part exists
    # but overflows, and coalesce(end, start) would turn "1-99999999999" into the valid port 1.
    end = F.when(F.size(parts) > 1, F.get(parts, 1).try_cast("int")).otherwise(start)

    port_spec_valid = F.coalesce(
        (
            F.col("port").isNotNull()
            & port.rlike(PORT_SPEC)
            & start.between(1, 65535)
            & end.between(1, 65535)
            & (start <= end)
        ),
        F.lit(False),
    )
    return (
        df.withColumn("port_start", F.when(port_spec_valid, start))
        .withColumn("port_end", F.when(port_spec_valid, end))
        .withColumn("port_spec_valid", port_spec_valid)
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
      instead of raising: try_cast and F.get return null on failure (cast overflow, missing
      array index), and the null comparisons reject the row even under ANSI mode. Returning
      null (rather than filtering) keeps this composable with the other silver expressions in
      one select, and leaves the quarantine decision to the caller.

    Unlike normalize_port, this is not tied to a column name.
    """
    ip = F.col(ip) if isinstance(ip, str) else ip
    trimmed = F.trim(ip)
    # F.get (not getItem): under ANSI mode, getItem throws INVALID_ARRAY_INDEX on a missing
    # index — inputs like "" or "10.0.0" split into fewer than 4 parts — and Spark may hoist it
    # out of the AND chain via common subexpression elimination, so a guard beside it is not
    # enough (the port function only survives getItem because it sits inside F.when's branch).
    # F.get returns null instead. Rule of thumb under ANSI: anything that can throw (cast,
    # getItem, element_at, division, ...) needs its try_/null-safe version.
    parts = F.split(trimmed, r"\.")
    octets = [F.get(parts, i).try_cast("long") for i in range(4)]

    # The regex only allows digits, so octets can never be negative; only the 255 cap matters.
    is_valid_ip = trimmed.rlike(_IP_REGEX)
    for octet in octets:
        is_valid_ip &= octet <= 255
    value = (
        octets[0] * F.lit(256**3) + octets[1] * F.lit(256**2) + octets[2] * F.lit(256) + octets[3]
    )
    # No otherwise: invalid or null input yields null.
    return F.when(is_valid_ip, value)
