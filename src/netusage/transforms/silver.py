from pyspark.sql import DataFrame
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
