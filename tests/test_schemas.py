"""Contract tests for the landing schemas.

No SparkSession needed — these inspect StructType objects only, so they run in
milliseconds. JOIN_KEYS mirrors the "Matching join path" section of
docs/architecture.md; if a join column is renamed or retyped, these fail.
"""

from pyspark.sql.types import LongType, StringType, StructType, TimestampType

from netusage.config import OPERATIONAL_STATUS_ACTIVE, USER_STATUS_ACTIVE
from netusage.schemas import (
    AD_GROUP,
    AD_GROUP_MEMBERSHIP,
    APPS,
    RESOURCE_RESOLUTION,
    TRAFFIC_LOGS,
    USERS,
    VPN_RULES,
)

ALL_SCHEMAS = (
    USERS,
    APPS,
    VPN_RULES,
    AD_GROUP,
    AD_GROUP_MEMBERSHIP,
    RESOURCE_RESOLUTION,
    TRAFFIC_LOGS,
)


def fields(schema, names, context=""):
    """Look up columns by name; accept a tuple for composite keys."""
    if isinstance(names, str):
        names = (names,)
    by_name = {f.name: f for f in schema.fields}
    missing = [n for n in names if n not in by_name]
    assert not missing, f"{context or 'schema'}: missing columns {missing}"
    return [by_name[n] for n in names]


JOIN_KEYS = (
    ("traffic_logs.user_email -> users.email", TRAFFIC_LOGS, "user_email", USERS, "email"),
    (
        "users.sys_id -> ad_group_membership.member_id",
        USERS,
        "sys_id",
        AD_GROUP_MEMBERSHIP,
        "member_id",
    ),
    (
        "ad_group_membership.group_id -> ad_group.group_id",
        AD_GROUP_MEMBERSHIP,
        "group_id",
        AD_GROUP,
        "group_id",
    ),
    (
        "vpn_rules.(ad_group_name, ad_group_domain) -> ad_group.(name, domain)",
        VPN_RULES,
        ("ad_group_name", "ad_group_domain"),
        AD_GROUP,
        ("name", "domain"),
    ),
    ("vpn_rules.app_sys_id -> apps.sys_id", VPN_RULES, "app_sys_id", APPS, "sys_id"),
    (
        "vpn_rules.resource -> resource_resolution.resource_name",
        VPN_RULES,
        "resource",
        RESOURCE_RESOLUTION,
        "resource_name",
    ),
)


def test_join_keys_exist_and_have_matching_types():
    for description, left, left_names, right, right_names in JOIN_KEYS:
        lhs = fields(left, left_names, description)
        rhs = fields(right, right_names, description)
        assert [f.dataType for f in lhs] == [f.dataType for f in rhs], description


def test_operational_status_documents_the_active_code():
    field = fields(VPN_RULES, "operational_status")[0]
    assert field.metadata["comment"].startswith(f"{OPERATIONAL_STATUS_ACTIVE}=")


def test_user_status_documents_the_active_code():
    field = fields(USERS, "status")[0]
    assert field.metadata["comment"].startswith(f"{USER_STATUS_ACTIVE}=")


def test_membership_can_be_filtered_to_users():
    assert "member_type" in {f.name for f in AD_GROUP_MEMBERSHIP.fields}


def test_ip_ranges_are_long():
    # 192.0.2.10 = 3221225994 > 2**31 - 1; IntegerType would overflow in ANSI mode.
    for name in ("ip_start", "ip_end"):
        assert fields(RESOURCE_RESOLUTION, name)[0].dataType == LongType()


def test_resolution_validity_is_timestamp_typed():
    # Matching compares these directly to the parsed traffic event_time (no date cast).
    for name in ("first_seen", "last_seen"):
        assert fields(RESOURCE_RESOLUTION, name)[0].dataType == TimestampType()


def test_resolution_has_state_key():
    # A state may hold several IP ranges; identify it independently of the range rows.
    assert fields(RESOURCE_RESOLUTION, "state_id")[0].dataType == StringType()


def test_no_duplicate_column_names():
    for schema in ALL_SCHEMAS:
        assert isinstance(schema, StructType)
        names = [f.name for f in schema.fields]
        assert len(names) == len(set(names))
