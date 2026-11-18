"""Schemas for the raw landing files read by the bronze layer.

Layer convention
----------------
These are LANDING schemas only. They describe the files exactly as they arrive,
so most fields are strings (bronze captures malformed records rather than
silently dropping them). Silver schemas are produced by the transforms and
asserted in their tests; they are not defined here.

Two consequences worth remembering:
- File sources (CSV/JSON/Parquet) ignore ``nullable`` and report every column
  as nullable. Required-value checks belong in silver or in Delta constraints.
- Foreign keys are recorded in the field ``metadata["comment"]`` below and,
  together with the actual join conditions, in ``docs/architecture.md``. The
  contract tests in ``tests/test_schemas.py`` fail if columns or types drift.

All data in this project is synthetic.
"""

from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# --- JSON Lines (API exports, one object per line) ---------------------------------------------

USERS = StructType(
    [
        StructField(
            "sys_id",
            StringType(),
            True,
            metadata={"comment": "PK; same identifier as ad_group_membership.member_id"},
        ),
        StructField(
            "email",
            StringType(),
            True,
            metadata={"comment": "Join key for traffic_logs.user_email (silver: lower/strip)"},
        ),
        StructField("department", StringType(), True),
        StructField("site", StringType(), True),
        StructField(
            "status",
            IntegerType(),
            True,
            metadata={"comment": "1=ACTIVE (only active code); others filtered in silver"},
        ),
    ]
)

APPS = StructType(
    [
        StructField("sys_id", StringType(), True, metadata={"comment": "PK"}),
        StructField("name", StringType(), True),
        StructField("department", StringType(), True),
    ]
)

VPN_RULES = StructType(
    [
        StructField("sys_id", StringType(), True, metadata={"comment": "PK"}),
        StructField(
            "app_sys_id",
            StringType(),
            True,
            metadata={"comment": "FK -> apps.sys_id; exactly one app per rule (required)"},
        ),
        StructField("gateway", StringType(), True),
        StructField(
            "resource",
            StringType(),
            True,
            metadata={"comment": "FK -> resource_resolution.resource_name"},
        ),
        StructField("port", StringType(), True),  # "443" or "1-99"; silver parses to start/end
        StructField("protocol", StringType(), True),
        StructField(
            "ad_group_name",
            StringType(),
            True,
            metadata={"comment": "FK 1/2 -> ad_group.name; unique only with domain"},
        ),
        StructField(
            "ad_group_domain",
            StringType(),
            True,
            metadata={"comment": "FK 2/2 -> ad_group.domain"},
        ),
        StructField(
            "operational_status",
            IntegerType(),
            True,
            metadata={"comment": "1=OPERATIONAL (only active code); others filtered in silver"},
        ),
    ]
)

# --- CSV ---------------------------------------------------------------------------------------

AD_GROUP = StructType(
    [
        StructField(
            "group_id", StringType(), True, metadata={"comment": "PK; FK from memberships"}
        ),
        StructField("name", StringType(), True),  # unique only together with domain
        StructField("domain", StringType(), True),
    ]
)

# One row per membership: a member in N groups produces N rows sharing the same member_id.
AD_GROUP_MEMBERSHIP = StructType(
    [
        StructField(
            "group_id", StringType(), True, metadata={"comment": "FK -> ad_group.group_id"}
        ),
        StructField(
            "member_id",
            StringType(),
            True,
            metadata={"comment": "FK -> users.sys_id (Option A); silver keeps member_type='user'"},
        ),
        StructField(
            "group_domain",
            StringType(),
            True,
            metadata={"comment": "Duplicates ad_group.domain by design (flat upstream CSV)"},
        ),
        StructField("member_type", StringType(), True),
    ]
)

TRAFFIC_LOGS = StructType(
    [
        StructField("event_id", StringType(), True, metadata={"comment": "Dedup key in silver"}),
        StructField("event_time", StringType(), True),  # string in bronze; silver parses it
        StructField("user_email", StringType(), True, metadata={"comment": "FK -> users.email"}),
        StructField("destination_ip", StringType(), True),  # silver converts to long
        StructField("destination_port", StringType(), True),
        StructField("protocol", StringType(), True),
        StructField("action", StringType(), True),  # ALLOW / DENY; silver keeps ALLOW
    ]
)

# --- Parquet -----------------------------------------------------------------------------------

# One state = (resource_name, state_id, first_seen, last_seen) and may hold several IP ranges.
# Rows with the same state_id share the timestamps; disjointness is checked between states.
RESOURCE_RESOLUTION = StructType(
    [
        StructField(
            "resource_name",
            StringType(),
            True,
            metadata={"comment": "Joined from vpn_rules.resource"},
        ),
        StructField(
            "state_id",
            StringType(),
            True,
            metadata={"comment": "One state may span several rows (one row per IP range)"},
        ),
        # LongType, not IntegerType: IPv4 addresses above 127.255.255.255 overflow signed int.
        StructField("ip_start", LongType(), True),
        StructField("ip_end", LongType(), True),
        # Half-open [first_seen, last_seen); NULL last_seen = current state. UTC.
        StructField("first_seen", TimestampType(), True),
        StructField("last_seen", TimestampType(), True),
    ]
)
