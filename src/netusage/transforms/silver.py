"""Silver transforms: pure functions that normalize bronze columns.

Bad values are flagged in an output column, not dropped here; callers decide
whether to quarantine them or let downstream filters exclude them.
"""

from pyspark.sql import DataFrame
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
