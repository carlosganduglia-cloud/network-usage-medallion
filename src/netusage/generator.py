"""Synthetic landing-data generator (all 7 entities, with planted dirty data).

Run locally:
    uv run python -m netusage.generator --profile small            # -> data/landing
    uv run python -m netusage.generator --profile tiny --days 7

Everything is built with Spark from ``spark.range`` so the same code scales from the tiny test
profile to 100M events on Databricks.

Randomness: hashing instead of ``F.rand(seed)``
-----------------------------------------------
``F.rand(seed)`` is only reproducible for the same partitioning: each task seeds its generator
with ``seed + partition_index`` and draws in row order. After a shuffle (``dropDuplicates``,
joins), a row's partition and position can change between runs, or between two computations of
the same DataFrame within one run, and with them its "random" value. That matters here because
some DataFrames are computed twice. For example, the clean memberships are written to landing
and also indexed to choose traffic users, and both computations must agree.

So every random value is ``uniform(purpose, *row_keys)``: ``xxhash64(SEED, purpose, keys)``
scaled to [0, 1). It's a pure function of the row, so it gives the same answer on any cluster
and in any partition layout. Each ``purpose`` string is a separate random stream.

Determinism also lets entities agree without joining. Group names and domains, user emails and
the IP a resource resolves to at time t are all arithmetic on an index, so the traffic generator
computes a destination IP from (resource, event time) and gets the same value the resolution
table holds.

Shapes the governance metrics should find (all ratios in ``config.py``)
-------------------------------------------------------------------------
- Unused rules: ``UNUSED_RULE_RATIO`` of rule templates point at a "dead" pool of resources
  that no traffic ever hits. Picking an unused rule at random wouldn't work: with huge groups,
  any rule sharing a resource and port with a used one would match by accident.
- Duplicate rules: the last ``DUPLICATE_RULE_RATIO`` of rules copy an earlier rule's resource,
  port, protocol and group, but get their own app and gateway.
- Over-provisioning: huge groups (users), wide port ranges with skewed actual ports (ports),
  subnets where traffic hits a few low host addresses (IPs).
- Unstable resources: a new resolution state every ``UNSTABLE_STATE_HOURS`` vs every
  ``STABLE_STATE_DAYS``. Some single-IP resources resolve to 2 IPs per state.
- Popularity is skewed (``u**3``): a few rules carry most traffic.

Planted dirty data: late and duplicate events, DENY events, unknown and noisy emails, inactive
users and rules, non-user/orphan/duplicate memberships, malformed CSV rows (traffic logs and
memberships only, so that ad_group stays a clean target for the rule foreign-key checks).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from netusage import config as C
from netusage import io
from netusage.config import ScaleProfile

# --- Fake domain values (synthetic: reserved example domains, 10.0.0.0/8 IPs only) ---------------
EMAIL_DOMAIN = "corp.example"
DEPARTMENTS = ("eng", "fin", "hr", "ops", "sales", "it", "legal", "mkt")
SITES = tuple(f"site-{i:02d}" for i in range(1, 13))
AD_DOMAINS = ("corp.example", "emea.corp.example", "amer.corp.example")
GATEWAYS = tuple(f"gw-{i:02d}" for i in range(1, 9))
ROLES = ("db", "app", "web", "file", "mq", "cache", "api", "auth")
COMMON_PORTS = (443, 22, 3389, 1433, 5432, 8080, 80, 636, 3306, 8443)
PORT_RANGES = ((8000, 8100), (5000, 5099), (9000, 9010), (1024, 2048))
INACTIVE_USER_CODES = (0, 2, 3, 4, 5, 6)
INACTIVE_RULE_CODES = (0, 3)
NON_USER_MEMBER_TYPES = ("computer", "group", "contact")
MALFORMED_SUFFIX = ",EXTRA"  # an extra field -> wrong field count -> _corrupt_record

# --- IP plan (all inside 10.0.0.0/8) -------------------------------------------------------------
# Subnet resources:    10.0.0.0  + r * 1024 -> 4 alternate /24-aligned positions per resource.
# Single-IP resources: 10.128.0.0 + i * 8   -> 4 alternates x 2 ranges (multi-IP resources).
# A resolution state uses alternate (state_idx mod 4), so "changing IPs" really change.
SUBNET_BASE = 10 << 24
SINGLE_BASE = (10 << 24) + (128 << 16)
IP_SPACE_END = 11 << 24
SUBNET_SLOT, SUBNET_ALT_STEP = 1024, 256
SINGLE_SLOT, SINGLE_ALT_STEP = 8, 2
IP_ALTERNATES = 4

_MASK_53 = (1 << 53) - 1
DAY_SECONDS = 86_400


# --- Random helpers ------------------------------------------------------------------------------
def uniform(purpose: str, *keys: str | Column) -> Column:
    """Deterministic uniform double in [0, 1) for this row and purpose (see module docstring)."""
    h = F.xxhash64(F.lit(C.SEED), F.lit(purpose), *keys)
    return h.bitwiseAND(F.lit(_MASK_53)) / F.lit(float(1 << 53))


def pick(values: tuple, u: Column) -> Column:
    """Choose one of ``values`` uniformly using the uniform column ``u``."""
    return F.element_at(
        F.array(*[F.lit(v) for v in values]), (F.floor(u * len(values)) + 1).cast("int")
    )


def padded(prefix: str, idx: Column, width: int) -> Column:
    return F.concat(F.lit(prefix), F.lpad(idx.cast("string"), width, "0"))


# --- Time window ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class DateWindow:
    """Event dates [start, end] inclusive; batches run LATE_EVENT_MAX_DAYS past ``end``."""

    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def start_ts(self) -> int:
        return int(
            datetime(self.start.year, self.start.month, self.start.day, tzinfo=UTC).timestamp()
        )

    @property
    def end_ts(self) -> int:
        """Exclusive: midnight UTC after ``end``."""
        return self.start_ts + self.days * DAY_SECONDS

    @property
    def landing_end(self) -> date:
        return self.end + timedelta(days=C.LATE_EVENT_MAX_DAYS)


# --- Shared entity expressions (pure functions of an index) --------------------------------------
def user_sys_id(idx: Column) -> Column:
    return padded("usr", idx, 8)


def user_email(idx: Column) -> Column:
    return F.concat(padded("user", idx, 6), F.lit(f"@{EMAIL_DOMAIN}"))


def group_id(idx: Column) -> Column:
    return padded("grp", idx, 8)


def group_domain(idx: Column) -> Column:
    return F.element_at(
        F.array(*[F.lit(d) for d in AD_DOMAINS]), (F.pmod(idx, F.lit(3)) + 1).cast("int")
    )


def group_name(idx: Column) -> Column:
    """Regular groups share a name across the 3 domains (idx // 3): names are unique only with
    the domain, which is exactly why rules reference groups by (name, domain)."""
    stem = F.floor(idx / 3)
    regular = F.concat(
        F.lit("GG-"),
        pick(DEPARTMENTS, uniform("group_dept", stem)),
        F.lit("-"),
        F.lpad(stem.cast("string"), 6, "0"),
    )
    return F.when(
        idx < C.HUGE_GROUP_COUNT, F.concat(F.lit("All-Staff-"), idx.cast("string"))
    ).otherwise(regular)


def _plant_malformed(df: DataFrame, column: str, purpose: str, *keys: str) -> DataFrame:
    broken = F.concat(F.col(column), F.lit(MALFORMED_SUFFIX))
    is_bad = uniform(purpose, *keys) < C.MALFORMED_CSV_ROW_RATIO
    return df.withColumn(column, F.when(is_bad, broken).otherwise(F.col(column)))


# --- Users, apps, groups -------------------------------------------------------------------------
def generate_users(spark: SparkSession, p: ScaleProfile) -> DataFrame:
    idx = F.col("id")
    inactive = uniform("user_inactive", idx) < C.DISABLED_USER_RATIO
    status = F.when(inactive, pick(INACTIVE_USER_CODES, uniform("user_status", idx))).otherwise(
        F.lit(C.USER_STATUS_ACTIVE)
    )
    return spark.range(p.users).select(
        user_sys_id(idx).alias("sys_id"),
        user_email(idx).alias("email"),
        pick(DEPARTMENTS, uniform("user_dept", idx)).alias("department"),
        pick(SITES, uniform("user_site", idx)).alias("site"),
        status.cast("int").alias("status"),
    )


def generate_apps(spark: SparkSession, p: ScaleProfile) -> DataFrame:
    idx = F.col("id")
    dept = pick(DEPARTMENTS, uniform("app_dept", idx))
    return spark.range(p.apps).select(
        padded("app", idx, 6).alias("sys_id"),
        F.concat(F.lit("app-"), dept, F.lit("-"), F.lpad(idx.cast("string"), 4, "0")).alias("name"),
        dept.alias("department"),
    )


def generate_ad_groups(spark: SparkSession, p: ScaleProfile) -> DataFrame:
    idx = F.col("id")
    return spark.range(p.ad_groups).select(
        group_id(idx).alias("group_id"),
        group_name(idx).alias("name"),
        group_domain(idx).alias("domain"),
    )


def clean_memberships(spark: SparkSession, p: ScaleProfile) -> DataFrame:
    """Valid user memberships as (group_idx, user_idx), deduplicated.

    - Huge groups: each holds ~HUGE_GROUP_MEMBER_SHARE of all users (a cross join, filtered).
    - Every regular group gets one seed member, so no group is empty.
    - The remaining rows land on group ``floor(regular * u**2)``: heavy-tailed sizes (the
      first regular groups hold thousands, the median a few dozen).
    """
    huge = C.HUGE_GROUP_COUNT
    regular = p.ad_groups - huge
    huge_rows = (
        spark.range(huge)
        .withColumnRenamed("id", "group_idx")
        .crossJoin(spark.range(p.users).withColumnRenamed("id", "user_idx"))
        .where(uniform("huge_member", "group_idx", "user_idx") < C.HUGE_GROUP_MEMBER_SHARE)
    )
    seed_rows = spark.range(huge, p.ad_groups).select(
        F.col("id").alias("group_idx"),
        F.floor(uniform("seed_member", "id") * p.users).alias("user_idx"),
    )
    expected_huge = round(huge * p.users * C.HUGE_GROUP_MEMBER_SHARE)
    extra = max(p.ad_groups * p.avg_members_per_group - expected_huge - regular, 0)
    extra_rows = spark.range(extra).select(
        (F.lit(huge) + F.floor(F.pow(uniform("member_group", "id"), 2) * regular)).alias(
            "group_idx"
        ),
        F.floor(uniform("member_user", "id") * p.users).alias("user_idx"),
    )
    return (
        huge_rows.unionByName(seed_rows)
        .unionByName(extra_rows)
        .dropDuplicates(["group_idx", "user_idx"])
    )


def generate_memberships(spark: SparkSession, p: ScaleProfile, clean: DataFrame) -> DataFrame:
    gidx, uidx = F.col("group_idx"), F.col("user_idx")
    users = clean.select(
        group_id(gidx).alias("group_id"),
        user_sys_id(uidx).alias("member_id"),
        group_domain(gidx).alias("group_domain"),
        F.lit("user").alias("member_type"),
    )
    base = p.ad_groups * p.avg_members_per_group
    rid = F.col("id")
    rand_group = F.floor(uniform("dirty_member_group", rid) * p.ad_groups)

    non_user = spark.range(round(base * C.NON_USER_MEMBER_RATIO)).select(
        group_id(rand_group).alias("group_id"),
        padded("cmp", rid, 8).alias("member_id"),
        group_domain(rand_group).alias("group_domain"),
        pick(NON_USER_MEMBER_TYPES, uniform("non_user_type", rid)).alias("member_type"),
    )
    # Even ids: unknown group. Odd ids: unknown member.
    orphan_user = F.floor(uniform("orphan_user", rid) * p.users)
    orphans = spark.range(round(base * C.ORPHAN_MEMBERSHIP_RATIO)).select(
        F.when(rid % 2 == 0, padded("grp-missing-", rid, 6))
        .otherwise(group_id(rand_group))
        .alias("group_id"),
        F.when(rid % 2 == 1, padded("usr-missing-", rid, 6))
        .otherwise(user_sys_id(orphan_user))
        .alias("member_id"),
        group_domain(rand_group).alias("group_domain"),
        F.lit("user").alias("member_type"),
    )
    dups = users.where(
        uniform("membership_dup", "group_id", "member_id") < C.DUPLICATE_MEMBERSHIP_RATIO
    )

    out = users.unionByName(non_user).unionByName(orphans).unionByName(dups)
    return _plant_malformed(
        out, "member_type", "membership_malformed", "group_id", "member_id", "member_type"
    )


# --- Resources and resolutions -------------------------------------------------------------------
def resources(spark: SparkSession, p: ScaleProfile) -> DataFrame:
    """Internal resource table (one row per resource) with everything needed to compute its
    resolution states. Subnets are the first ``n_subnets`` ids so the IP plan stays dense."""
    n_subnets = round(p.resources * C.SUBNET_RESOURCE_RATIO)
    n_single = p.resources - n_subnets
    if (
        SUBNET_BASE + n_subnets * SUBNET_SLOT > SINGLE_BASE
        or SINGLE_BASE + n_single * SINGLE_SLOT > IP_SPACE_END
    ):
        raise ValueError(f"profile {p.name!r}: too many resources for the 10.0.0.0/8 IP plan")

    r = F.col("id")
    is_subnet = r < n_subnets
    prefix = pick(C.SUBNET_PREFIX_LENGTHS, uniform("subnet_prefix", r))
    unstable = uniform("unstable_resource", r) < C.UNSTABLE_RESOURCE_RATIO
    period = F.when(unstable, F.lit(C.UNSTABLE_STATE_HOURS * 3600)).otherwise(
        F.lit(C.STABLE_STATE_DAYS * DAY_SECONDS)
    )
    return spark.range(p.resources).select(
        r.alias("resource_idx"),
        F.concat(
            F.lit("srv-"),
            pick(ROLES, uniform("resource_role", r)),
            F.lit("-"),
            F.lpad(r.cast("string"), 5, "0"),
        ).alias("resource_name"),
        is_subnet.alias("is_subnet"),
        F.when(is_subnet, F.pow(F.lit(2), 32 - prefix)).otherwise(1).cast("long").alias("width"),
        F.when(~is_subnet & (uniform("multi_range", r) < C.MULTI_RANGE_RESOURCE_RATIO), 2)
        .otherwise(1)
        .alias("n_ranges"),
        period.cast("long").alias("period"),
        F.floor(uniform("state_phase", r) * period).alias("phase"),
        F.when(is_subnet, F.lit(SUBNET_BASE) + r * SUBNET_SLOT)
        .otherwise(F.lit(SINGLE_BASE) + (r - n_subnets) * SINGLE_SLOT)
        .alias("slot_base"),
        F.when(is_subnet, SUBNET_ALT_STEP).otherwise(SINGLE_ALT_STEP).alias("alt_step"),
        (uniform("dead_resource", r) < C.UNUSED_RULE_RATIO).alias("dead"),
    )


def _state_index(ts: Column, win: DateWindow) -> Column:
    """State i covers [start + i*period - phase, start + (i+1)*period - phase), clipped at start."""
    return F.floor((ts - F.lit(win.start_ts) + F.col("phase")) / F.col("period"))


def _range_start(state_idx: Column, range_idx: Column) -> Column:
    return (
        F.col("slot_base") + F.pmod(state_idx, F.lit(IP_ALTERNATES)) * F.col("alt_step") + range_idx
    )


def generate_resolutions(res: DataFrame, win: DateWindow) -> DataFrame:
    n_states = _state_index(F.lit(win.end_ts), win) + 1
    i, k = F.col("state_idx"), F.col("range_idx")
    first = F.when(i == 0, F.lit(win.start_ts)).otherwise(
        F.lit(win.start_ts) + i * F.col("period") - F.col("phase")
    )
    last = F.when(i == F.col("n_states") - 1, F.lit(None)).otherwise(
        F.lit(win.start_ts) + (i + 1) * F.col("period") - F.col("phase")
    )
    ip_start = _range_start(i, k)
    return (
        res.withColumn("n_states", n_states)
        .withColumn(
            "state_idx", F.explode(F.sequence(F.lit(0).cast("long"), F.col("n_states") - 1))
        )
        .withColumn("range_idx", F.explode(F.sequence(F.lit(0), F.col("n_ranges") - 1)))
        .select(
            "resource_name",
            F.concat(F.col("resource_name"), F.lit("#"), F.lpad(i.cast("string"), 5, "0")).alias(
                "state_id"
            ),
            ip_start.cast("long").alias("ip_start"),
            (ip_start + F.col("width") - 1).cast("long").alias("ip_end"),
            F.timestamp_seconds(first).alias("first_seen"),
            F.timestamp_seconds(last).alias("last_seen"),
        )
    )


# --- Rules ---------------------------------------------------------------------------------------
def rules(spark: SparkSession, p: ScaleProfile, res: DataFrame) -> DataFrame:
    """Internal rule table: landing columns plus the indexes the traffic generator needs."""
    n_dup = round(p.vpn_rules * C.DUPLICATE_RULE_RATIO)
    n_base = p.vpn_rules - n_dup
    rid = F.col("id")
    t = F.col("template")
    huge, regular = C.HUGE_GROUP_COUNT, p.ad_groups - C.HUGE_GROUP_COUNT

    # Dense index within the live and the dead pool, so a rule can pick "the n-th live resource".
    pool = res.withColumn(
        "res_pool_idx", F.row_number().over(Window.partitionBy("dead").orderBy("resource_idx")) - 1
    )
    sizes = {row["dead"]: row["count"] for row in res.groupBy("dead").count().collect()}
    if not sizes.get(True) or not sizes.get(False):
        raise ValueError(f"profile {p.name!r}: too few resources to fill both live and dead pools")

    unused = uniform("unused_rule", t) < C.UNUSED_RULE_RATIO
    on_huge = uniform("rule_on_huge", t) < C.RULES_ON_HUGE_GROUP_RATIO
    group_idx = F.when(on_huge, F.floor(uniform("rule_huge_group", t) * huge)).otherwise(
        F.lit(huge) + F.floor(F.pow(uniform("rule_group", t), 2) * regular)
    )
    port_kind = uniform("port_kind", t)
    range_i = (F.floor(uniform("port_range", t) * len(PORT_RANGES)) + 1).cast("int")
    range_starts = F.array(*[F.lit(a) for a, _ in PORT_RANGES])
    range_ends = F.array(*[F.lit(b) for _, b in PORT_RANGES])
    single_port = pick(COMMON_PORTS, uniform("port_single", t))
    is_wide = port_kind < C.WIDE_PORT_RANGE_RATIO
    is_range = port_kind < C.WIDE_PORT_RANGE_RATIO + C.RANGE_PORT_RULE_RATIO
    port_start = (
        F.when(is_wide, 1)
        .when(is_range, F.element_at(range_starts, range_i))
        .otherwise(single_port)
    )
    port_end = (
        F.when(is_wide, 65535)
        .when(is_range, F.element_at(range_ends, range_i))
        .otherwise(single_port)
    )
    inactive = uniform("rule_inactive", rid) < C.NON_OPERATIONAL_RULE_RATIO

    base = spark.range(p.vpn_rules).select(
        rid,
        F.when(rid < n_base, rid)
        .otherwise(F.floor(uniform("dup_template", rid) * n_base))
        .alias("template"),
    )
    with_attrs = base.select(
        "id",
        "template",
        unused.alias("unused"),
        F.floor(
            uniform("rule_resource", t) * F.when(unused, sizes[True]).otherwise(sizes[False])
        ).alias("pool_idx"),
        group_idx.alias("group_idx"),
        port_start.cast("int").alias("port_start"),
        port_end.cast("int").alias("port_end"),
        F.when(uniform("rule_udp", t) < C.UDP_RULE_RATIO, "UDP").otherwise("TCP").alias("protocol"),
        padded("app", F.floor(uniform("rule_app", rid) * p.apps), 6).alias("app_sys_id"),
        pick(GATEWAYS, uniform("rule_gateway", rid)).alias("gateway"),
        F.when(inactive, pick(INACTIVE_RULE_CODES, uniform("rule_status", rid)))
        .otherwise(C.OPERATIONAL_STATUS_ACTIVE)
        .cast("int")
        .alias("operational_status"),
    )
    gidx = F.col("group_idx")
    return (
        # The resource tables are small: broadcast them and skip the shuffle.
        with_attrs.join(
            F.broadcast(pool),
            (F.col("unused") == F.col("dead")) & (F.col("pool_idx") == F.col("res_pool_idx")),
        )
        .drop("pool_idx", "res_pool_idx")
        .withColumn("sys_id", padded("rul", F.col("id"), 8))
        .withColumn(
            "port",
            F.when(
                F.col("port_start") == F.col("port_end"), F.col("port_start").cast("string")
            ).otherwise(
                F.concat_ws(
                    "-", F.col("port_start").cast("string"), F.col("port_end").cast("string")
                )
            ),
        )
        .withColumn("ad_group_name", group_name(gidx))
        .withColumn("ad_group_domain", group_domain(gidx))
    )


def landing_rules(rule_table: DataFrame) -> DataFrame:
    return rule_table.select(
        "sys_id",
        "app_sys_id",
        "gateway",
        F.col("resource_name").alias("resource"),
        "port",
        "protocol",
        "ad_group_name",
        "ad_group_domain",
        "operational_status",
    )


# --- Traffic -------------------------------------------------------------------------------------
def generate_traffic(
    spark: SparkSession, p: ScaleProfile, win: DateWindow, rule_table: DataFrame, clean: DataFrame
) -> DataFrame:
    """One row per connection event, with dirty data, plus the ``batch_date`` it lands in.

    Each clean event: pick a used rule (skewed popularity) -> a member of its group -> a time ->
    the resource's IP *at that time* -> a port in the rule's range. Every clean ALLOW event is
    therefore matched by at least the rule it was generated from (and maybe others).
    """
    # Dense index over used rules. An un-partitioned window moves all rows to one task (Spark
    # warns about it); fine for ~100k rules, a red flag for anything big.
    used = rule_table.where(~F.col("unused")).withColumn(
        "used_idx", F.row_number().over(Window.orderBy("id")) - 1
    )
    n_used = used.count()

    member_index = clean.withColumn(
        "member_k", F.row_number().over(Window.partitionBy("group_idx").orderBy("user_idx")) - 1
    )
    group_sizes = clean.groupBy("group_idx").agg(F.count("*").alias("group_size"))

    e = F.col("event_idx")
    day = F.floor(uniform("event_day", e) * win.days)
    second_of_day = F.floor(uniform("event_second", e) * DAY_SECONDS)
    ts = F.lit(win.start_ts) + day * DAY_SECONDS + second_of_day
    event_date = F.date_add(F.lit(win.start), day.cast("int"))
    # Built from parts, so the string is UTC regardless of spark.sql.session.timeZone.
    event_time = F.concat(
        F.date_format(event_date, "yyyy-MM-dd"),
        F.lit("T"),
        F.lpad(F.floor(second_of_day / 3600).cast("string"), 2, "0"),
        F.lit(":"),
        F.lpad(F.floor(second_of_day % 3600 / 60).cast("string"), 2, "0"),
        F.lit(":"),
        F.lpad((second_of_day % 60).cast("string"), 2, "0"),
        F.lit("Z"),
    )

    range_idx = F.when((F.col("n_ranges") == 2) & (uniform("event_range", e) < 0.5), 1).otherwise(0)
    host_offset = F.when(
        F.col("is_subnet"), F.floor(F.pow(uniform("event_host", e), 4) * F.col("width"))
    ).otherwise(0)
    ip = _range_start(_state_index(ts, win), range_idx) + host_offset
    ip_string = F.concat_ws(
        ".", *[F.shiftright(ip, s).bitwiseAND(255).cast("string") for s in (24, 16, 8, 0)]
    )
    port = F.col("port_start") + F.floor(
        F.pow(uniform("event_port", e), 3) * (F.col("port_end") - F.col("port_start") + 1)
    )

    email = user_email(F.col("user_idx"))
    noisy = F.when(uniform("email_noise_kind", e) < 0.5, F.upper(email)).otherwise(
        F.concat(F.lit(" "), email, F.lit(" "))
    )
    email = (
        F.when(
            uniform("unknown_user", e) < C.UNKNOWN_USER_EVENT_RATIO,
            F.concat(padded("ghost", e, 8), F.lit(f"@{EMAIL_DOMAIN}")),
        )
        .when(uniform("email_noise", e) < C.EMAIL_NOISE_RATIO, noisy)
        .otherwise(email)
    )
    late_days = F.when(
        uniform("late_event", e) < C.LATE_EVENT_RATIO,
        F.floor(uniform("late_days", e) * C.LATE_EVENT_MAX_DAYS) + 1,
    ).otherwise(0)

    rule_cols = [
        "used_idx",
        "group_idx",
        "port_start",
        "port_end",
        "protocol",
        "is_subnet",
        "width",
        "n_ranges",
    ]
    rule_cols += ["period", "phase", "slot_base", "alt_step"]
    events = (
        spark.range(p.traffic_events)
        .withColumnRenamed("id", "event_idx")
        .withColumn("used_idx", F.floor(F.pow(uniform("rule_popularity", e), 3) * n_used))
        .join(F.broadcast(used.select(*rule_cols)), "used_idx")
        .join(F.broadcast(group_sizes), "group_idx")
        .withColumn("member_k", F.floor(uniform("event_member", e) * F.col("group_size")))
        .join(member_index, ["group_idx", "member_k"])
        .select(
            padded("evt", e, 10).alias("event_id"),
            event_time.alias("event_time"),
            email.alias("user_email"),
            ip_string.alias("destination_ip"),
            port.cast("string").alias("destination_port"),
            "protocol",
            F.when(uniform("deny", e) < C.DENY_EVENT_RATIO, "DENY")
            .otherwise("ALLOW")
            .alias("action"),
            F.date_add(event_date, late_days.cast("int")).alias(io.BATCH_COLUMN),
        )
    )
    # Retries: the same event (same event_id) again, in the same or the next batch.
    shift = F.floor(uniform("dup_shift", "event_id") * 2).cast("int")
    dups = events.where(uniform("dup_event", "event_id") < C.DUPLICATE_EVENT_RATIO).withColumn(
        io.BATCH_COLUMN, F.least(F.date_add(io.BATCH_COLUMN, shift), F.lit(win.landing_end))
    )
    out = events.unionByName(dups)
    return _plant_malformed(out, "action", "event_malformed", "event_id", io.BATCH_COLUMN)


# --- Orchestration -------------------------------------------------------------------------------
def generate(
    spark: SparkSession,
    profile: ScaleProfile,
    landing_root: str | Path,
    start: date = C.START_DATE,
    end: date = C.END_DATE,
) -> None:
    """Generate all 7 landing entities under ``landing_root`` (overwrites)."""
    win = DateWindow(start, end)
    res = resources(spark, profile)
    rule_table = rules(spark, profile, res)
    clean = clean_memberships(spark, profile)

    def timed(entity: str, write, df: DataFrame, **kwargs) -> None:
        t0 = time.perf_counter()
        write(df, io.landing_path(landing_root, entity), **kwargs)
        print(f"  {entity:<22} {time.perf_counter() - t0:6.1f}s")

    # Each write is one Spark job. DataFrames used twice (rule_table, clean) are recomputed per
    # job; that's cheap here, and safe only because the randomness is deterministic.
    timed("users", io.write_json_lines, generate_users(spark, profile))
    timed("apps", io.write_json_lines, generate_apps(spark, profile))
    timed("vpn_rules", io.write_json_lines, landing_rules(rule_table))
    timed("ad_group", io.write_csv_lines, generate_ad_groups(spark, profile))
    timed("ad_group_membership", io.write_csv_lines, generate_memberships(spark, profile, clean))
    timed("resource_resolution", io.write_parquet, generate_resolutions(res, win))
    traffic = generate_traffic(spark, profile, win, rule_table, clean)
    timed("traffic_logs", io.write_csv_lines, traffic, partition_by=io.BATCH_COLUMN)


def main(argv: list[str] | None = None) -> None:
    from netusage.session import local_spark

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(C.PROFILES), default="small")
    parser.add_argument("--out", default=str(C.LOCAL_LANDING_PATH))
    parser.add_argument(
        "--days", type=int, help="only the first N days from START_DATE (quick runs)"
    )
    args = parser.parse_args(argv)

    end = C.START_DATE + timedelta(days=args.days - 1) if args.days else C.END_DATE
    profile = C.PROFILES[args.profile]
    print(f"Generating profile={profile.name} {C.START_DATE}..{end} -> {args.out}")
    generate(local_spark("netusage-generator"), profile, args.out, C.START_DATE, end)


if __name__ == "__main__":
    main()
