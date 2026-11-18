"""Validate generated landing data: row counts, join-path integrity, dirty-data ratios.

Run locally after the generator:
    uv run python -m netusage.validation --profile small
    uv run python -m netusage.validation --profile tiny --days 7

Everything is read through the landing readers in ``io.py``, so this is also the first test of
those readers on realistic files: header checks, ``_corrupt_record``, ``_metadata``, and
partition discovery on the traffic folders.

What this does NOT check: that an event actually matches a rule (IP range at the event's
instant, port range, protocol, group membership). That is the matching transform, and the
transform's tests verify it.

Ratio checks compare an observed share against the configured one. The tolerance is the larger
of 4 binomial standard deviations (sampling noise, which matters at the tiny profile) and 20% of
the expected value (slack for interactions between kinds of planted data, e.g. a duplicate of a
late event).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from netusage import config as C
from netusage import io, schemas
from netusage.config import ScaleProfile


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def exact(name: str, actual, expected) -> Check:
    return Check(name, actual == expected, f"actual={actual} expected={expected}")


def at_most(name: str, actual: int, limit: int = 0) -> Check:
    return Check(name, actual <= limit, f"actual={actual} limit={limit}")


def near(name: str, actual: float, expected: float, rel_tol: float) -> Check:
    ok = abs(actual - expected) <= rel_tol * expected
    return Check(name, ok, f"actual={actual:,.0f} expected~{expected:,.0f} (±{rel_tol:.0%})")


def ratio(name: str, hits: int, total: int, expected: float) -> Check:
    observed = hits / total if total else 0.0
    sigma = math.sqrt(expected * (1 - expected) / total) if total else 0.0
    tol = max(4 * sigma, 0.2 * expected)
    ok = abs(observed - expected) <= tol
    return Check(
        name, ok, f"{hits:,}/{total:,} = {observed:.4%} expected {expected:.4%} ±{tol:.4%}"
    )


def _norm_email(col: str) -> F.Column:
    return F.lower(F.trim(F.col(col)))


def read_landing(spark: SparkSession, root: str | Path) -> dict[str, DataFrame]:
    def p(entity: str) -> str:
        return io.landing_path(root, entity)

    return {
        "users": io.read_json_lines(spark, p("users"), schemas.USERS),
        "apps": io.read_json_lines(spark, p("apps"), schemas.APPS),
        "vpn_rules": io.read_json_lines(spark, p("vpn_rules"), schemas.VPN_RULES),
        "ad_group": io.read_csv(spark, p("ad_group"), schemas.AD_GROUP),
        # cache(): see _full_parse.
        "ad_group_membership": _full_parse(
            io.read_csv(spark, p("ad_group_membership"), schemas.AD_GROUP_MEMBERSHIP)
        ),
        "resource_resolution": io.read_parquet(
            spark, p("resource_resolution"), schemas.RESOURCE_RESOLUTION
        ),
        # Reading the root (not one batch folder) makes Spark discover batch_date=... folders and
        # add batch_date as a column (partition discovery).
        "traffic_logs": _full_parse(io.read_csv(spark, p("traffic_logs"), schemas.TRAFFIC_LOGS)),
    }


def _full_parse(df: DataFrame) -> DataFrame:
    """Cache a raw CSV read so every column is parsed before any counting.

    With ``spark.sql.csv.parser.columnPruning.enabled`` (the default), a query that needs only
    some columns makes the parser read only those, and a row with an extra field is *not*
    flagged in ``_corrupt_record``. An aggregation over the raw files then silently undercounts
    malformed rows (tests/test_io.py pins this down). The cache materializes all columns. That's
    fine for this local tool; the pipeline gets the same effect by writing every column to bronze.
    """
    return df.cache()


def validate(
    spark: SparkSession,
    root: str | Path,
    profile: ScaleProfile,
    start: date = C.START_DATE,
    end: date = C.END_DATE,
) -> list[Check]:
    d = read_landing(spark, root)
    checks: list[Check] = []
    checks += _check_users_apps_groups(d, profile)
    checks += _check_memberships(d, profile)
    checks += _check_rules(d, profile)
    checks += _check_resolutions(d, profile, start, end)
    checks += _check_traffic(d, profile, start, end)
    return checks


def _check_users_apps_groups(d: dict[str, DataFrame], p: ScaleProfile) -> list[Check]:
    u = (
        d["users"]
        .agg(
            F.count("*").alias("rows"),
            F.countDistinct(_norm_email("email")).alias("emails"),
            F.sum((F.col("status") != C.USER_STATUS_ACTIVE).cast("int")).alias("inactive"),
            F.sum(F.col(io.CORRUPT_COLUMN).isNotNull().cast("int")).alias("corrupt"),
        )
        .first()
    )
    g = (
        d["ad_group"]
        .agg(
            F.count("*").alias("rows"),
            F.countDistinct("name", "domain").alias("name_domain"),
            F.countDistinct("name").alias("names"),
        )
        .first()
    )
    return [
        exact("users: row count", u["rows"], p.users),
        exact("users: malformed JSON lines", u["corrupt"], 0),
        exact("users: emails unique after lower/trim", u["emails"], u["rows"]),
        ratio("users: inactive status", u["inactive"], u["rows"], C.DISABLED_USER_RATIO),
        exact("apps: row count", d["apps"].count(), p.apps),
        exact("ad_group: row count", g["rows"], p.ad_groups),
        exact("ad_group: (name, domain) unique", g["name_domain"], g["rows"]),
        Check(
            "ad_group: names repeat across domains", g["names"] < g["rows"], f"names={g['names']}"
        ),
    ]


def _check_memberships(d: dict[str, DataFrame], p: ScaleProfile) -> list[Check]:
    m = d["ad_group_membership"]
    users = d["users"].select(F.col("sys_id").alias("member_id"))
    groups = d["ad_group"].select("group_id")
    user_rows = m.where(F.col("member_type") == "user")

    agg = m.agg(
        F.count("*").alias("rows"),
        F.countDistinct("group_id", "member_id", "member_type").alias("distinct"),
        F.sum((F.col("member_type") != "user").cast("int")).alias("non_user"),
        F.sum(F.col(io.CORRUPT_COLUMN).isNotNull().cast("int")).alias("corrupt"),
    ).first()
    # Steps 2 and 3 of the join path: membership -> users.sys_id and -> ad_group.group_id.
    unknown_member = user_rows.join(users, "member_id", "left_anti").count()
    unknown_group = m.join(groups, "group_id", "left_anti").count()
    huge_ids = [f"grp{i:08d}" for i in range(C.HUGE_GROUP_COUNT)]
    huge_rows = user_rows.where(F.col("group_id").isin(huge_ids)).count()
    base = p.ad_groups * p.avg_members_per_group
    dirty = C.NON_USER_MEMBER_RATIO + C.ORPHAN_MEMBERSHIP_RATIO + C.DUPLICATE_MEMBERSHIP_RATIO
    return [
        near("membership: row count", agg["rows"], base * (1 + dirty), rel_tol=0.15),
        ratio(
            "membership: non-user members", agg["non_user"], agg["rows"], C.NON_USER_MEMBER_RATIO
        ),
        ratio(
            "membership: orphans (unknown group or member) [steps 2-3]",
            unknown_member + unknown_group,
            agg["rows"],
            C.ORPHAN_MEMBERSHIP_RATIO,
        ),
        ratio(
            "membership: duplicate rows",
            agg["rows"] - agg["distinct"],
            agg["rows"],
            C.DUPLICATE_MEMBERSHIP_RATIO,
        ),
        ratio(
            "membership: malformed CSV rows", agg["corrupt"], agg["rows"], C.MALFORMED_CSV_ROW_RATIO
        ),
        ratio(
            "membership: huge-group share of users",
            huge_rows,
            C.HUGE_GROUP_COUNT * p.users,
            C.HUGE_GROUP_MEMBER_SHARE,
        ),
    ]


def _check_rules(d: dict[str, DataFrame], p: ScaleProfile) -> list[Check]:
    r = d["vpn_rules"]
    agg = r.agg(
        F.count("*").alias("rows"),
        F.sum((F.col("port") == "1-65535").cast("int")).alias("wide"),
        F.sum((F.col("operational_status") != C.OPERATIONAL_STATUS_ACTIVE).cast("int")).alias(
            "inactive"
        ),
        F.sum(F.col(io.CORRUPT_COLUMN).isNotNull().cast("int")).alias("corrupt"),
    ).first()
    # A rule is a duplicate if an earlier-numbered rule grants the same thing to the same group.
    grant = ["resource", "port", "protocol", "ad_group_name", "ad_group_domain"]
    dup_rules = r.groupBy(*grant).count().agg(F.sum(F.col("count") - 1)).first()[0] or 0

    groups = d["ad_group"].select(
        F.col("name").alias("ad_group_name"), F.col("domain").alias("ad_group_domain")
    )
    apps = d["apps"].select(F.col("sys_id").alias("app_sys_id"))
    resources = d["resource_resolution"].select(F.col("resource_name").alias("resource")).distinct()
    return [
        exact("vpn_rules: row count", agg["rows"], p.vpn_rules),
        exact("vpn_rules: malformed JSON lines", agg["corrupt"], 0),
        # Step 4 (composite key), step 5 (resource exists), step 6 (app).
        at_most(
            "vpn_rules: unknown (group name, domain) [step 4]",
            r.join(groups, ["ad_group_name", "ad_group_domain"], "left_anti").count(),
        ),
        at_most(
            "vpn_rules: unknown resource [step 5]",
            r.join(resources, "resource", "left_anti").count(),
        ),
        at_most("vpn_rules: unknown app [step 6]", r.join(apps, "app_sys_id", "left_anti").count()),
        ratio(
            "vpn_rules: wide port range 1-65535", agg["wide"], agg["rows"], C.WIDE_PORT_RANGE_RATIO
        ),
        ratio(
            "vpn_rules: non-operational", agg["inactive"], agg["rows"], C.NON_OPERATIONAL_RULE_RATIO
        ),
        ratio(
            "vpn_rules: duplicates of another rule", dup_rules, agg["rows"], C.DUPLICATE_RULE_RATIO
        ),
    ]


def _check_resolutions(
    d: dict[str, DataFrame], p: ScaleProfile, start: date, end: date
) -> list[Check]:
    rr = d["resource_resolution"]
    far_future = F.lit("9999-12-31").cast("timestamp")
    states = rr.groupBy("resource_name", "state_id").agg(
        F.count("*").alias("ranges"),
        F.countDistinct("first_seen").alias("n_first"),
        F.countDistinct(F.coalesce("last_seen", far_future)).alias("n_last"),
        F.min("first_seen").alias("first_seen"),
        F.min(F.coalesce("last_seen", far_future)).alias("last_seen"),
        F.max((F.col("ip_end") > F.col("ip_start")).cast("int")).alias("is_subnet"),
    )
    # Disjoint and ordered: each state starts no earlier than the previous one ended.
    w = Window.partitionBy("resource_name").orderBy("first_seen")
    ordered = states.withColumn("prev_last", F.lag("last_seen").over(w))
    per_resource = states.groupBy("resource_name").agg(
        F.count("*").alias("n_states"),
        F.sum((F.col("last_seen") == far_future).cast("int")).alias("current"),
        F.max("is_subnet").alias("is_subnet"),
        F.max((F.col("ranges") > 1).cast("int")).alias("multi_range"),
    )
    # Stable resources get at most days/STABLE_STATE_DAYS + 2 states in the window; unstable ones
    # get ~days*24/UNSTABLE_STATE_HOURS. Anything above the stable bound is "unstable-shaped".
    days = (end - start).days + 1
    stable_max = days // C.STABLE_STATE_DAYS + 2
    res = per_resource.agg(
        F.count("*").alias("resources"),
        F.sum((F.col("current") != 1).cast("int")).alias("bad_current"),
        F.sum((F.col("n_states") > stable_max).cast("int")).alias("unstable"),
        F.sum("is_subnet").alias("subnets"),
        F.sum("multi_range").alias("multi_range"),
    ).first()
    single_ip = res["resources"] - res["subnets"]
    return [
        exact("resolutions: distinct resources", res["resources"], p.resources),
        at_most(
            "resolutions: rows of one state disagree on timestamps",
            states.where((F.col("n_first") != 1) | (F.col("n_last") != 1)).count(),
        ),
        at_most(
            "resolutions: last_seen <= first_seen",
            rr.where(F.col("last_seen") <= F.col("first_seen")).count(),
        ),
        at_most(
            "resolutions: states overlap or out of order",
            ordered.where(F.col("prev_last") > F.col("first_seen")).count(),
        ),
        at_most("resolutions: resources without exactly one current state", res["bad_current"]),
        at_most(
            "resolutions: ip_start > ip_end", rr.where(F.col("ip_start") > F.col("ip_end")).count()
        ),
        ratio(
            "resolutions: subnet resources",
            res["subnets"],
            res["resources"],
            C.SUBNET_RESOURCE_RATIO,
        ),
        ratio(
            "resolutions: unstable resources",
            res["unstable"],
            res["resources"],
            C.UNSTABLE_RESOURCE_RATIO,
        ),
        ratio(
            "resolutions: multi-range (single-IP) resources",
            res["multi_range"],
            single_ip,
            C.MULTI_RANGE_RESOURCE_RATIO,
        ),
    ]


def _check_traffic(d: dict[str, DataFrame], p: ScaleProfile, start: date, end: date) -> list[Check]:
    t = d["traffic_logs"]
    users = d["users"].select(_norm_email("email").alias("norm_email"))
    landing_end = end + timedelta(days=C.LATE_EVENT_MAX_DAYS)
    ok_rows = F.col(io.CORRUPT_COLUMN).isNull()
    event_date = F.to_date(F.substring("event_time", 1, 10))

    agg = t.agg(
        F.count("*").alias("rows"),
        F.countDistinct("event_id").alias("events"),
        F.sum((~ok_rows).cast("int")).alias("corrupt"),
        F.sum((F.col("action") == "DENY").cast("int")).alias("deny"),
        F.sum((F.col("user_email") != _norm_email("user_email")).cast("int")).alias("noisy"),
        F.sum((F.col(io.BATCH_COLUMN) < event_date).cast("int")).alias("early"),
        F.sum(
            (
                ok_rows & F.try_to_timestamp("event_time", F.lit("yyyy-MM-dd'T'HH:mm:ssX")).isNull()
            ).cast("int")
        ).alias("bad_time"),
        F.min(io.BATCH_COLUMN).alias("first_batch"),
        F.max(io.BATCH_COLUMN).alias("last_batch"),
        F.countDistinct(io.BATCH_COLUMN).alias("batches"),
    ).first()
    # Late = an event whose FIRST landing is after its event date (retries landing a day later
    # are counted by the duplicate check instead).
    first_landing = t.groupBy("event_id").agg(
        F.min(io.BATCH_COLUMN).alias("first_batch"), F.min(event_date).alias("event_date")
    )
    late = first_landing.where(F.col("first_batch") > F.col("event_date")).count()
    # Step 1 of the join path: traffic_logs.user_email -> users.email, after normalization.
    unknown = t.join(users, _norm_email("user_email") == F.col("norm_email"), "left_anti").count()
    files_per_batch = t.groupBy(io.BATCH_COLUMN).agg(
        F.countDistinct(io.SOURCE_COLUMN).alias("files")
    )
    multi_file_batches = files_per_batch.where(F.col("files") != 1).count()
    rows = agg["rows"]
    return [
        exact("traffic: distinct event_id", agg["events"], p.traffic_events),
        ratio(
            "traffic: duplicate rows (retries)", rows - agg["events"], rows, C.DUPLICATE_EVENT_RATIO
        ),
        ratio("traffic: malformed CSV rows", agg["corrupt"], rows, C.MALFORMED_CSV_ROW_RATIO),
        ratio("traffic: DENY", agg["deny"], rows, C.DENY_EVENT_RATIO),
        ratio("traffic: unknown user email [step 1]", unknown, rows, C.UNKNOWN_USER_EVENT_RATIO),
        ratio("traffic: noisy email (case/whitespace)", agg["noisy"], rows, C.EMAIL_NOISE_RATIO),
        ratio("traffic: late (first batch > event date)", late, agg["events"], C.LATE_EVENT_RATIO),
        at_most("traffic: batch_date before event date", agg["early"]),
        at_most("traffic: unparseable event_time", agg["bad_time"]),
        exact("traffic: first batch", agg["first_batch"], start),
        Check(
            "traffic: last batch within landing window",
            agg["last_batch"] <= landing_end,
            f"{agg['last_batch']} <= {landing_end}",
        ),
        Check(
            "traffic: one batch folder per event day (+ late days)",
            agg["batches"] >= (end - start).days + 1,
            f"{agg['batches']} folders",
        ),
        at_most("traffic: batch folders with more than one file", multi_file_batches),
    ]


def report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"{'OK ' if c.ok else 'FAIL'} {c.name:<{width}}  {c.detail}" for c in checks]
    failed = sum(not c.ok for c in checks)
    lines.append(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from netusage.session import local_spark

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(C.PROFILES), default="small")
    parser.add_argument("--landing", default=str(C.LOCAL_LANDING_PATH))
    parser.add_argument("--days", type=int, help="must match the generator's --days")
    args = parser.parse_args(argv)

    end = C.START_DATE + timedelta(days=args.days - 1) if args.days else C.END_DATE
    checks = validate(
        local_spark("netusage-validation"),
        args.landing,
        C.PROFILES[args.profile],
        C.START_DATE,
        end,
    )
    print(report(checks))
    sys.exit(0 if all(c.ok for c in checks) else 1)


if __name__ == "__main__":
    main()
