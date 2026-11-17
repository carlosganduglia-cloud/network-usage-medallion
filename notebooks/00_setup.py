# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC
# MAGIC 1. Creates the catalog, the medallion schemas, and a Volume for raw "landing" files.
# MAGIC 2. Checks that the repo's `netusage` package is importable from this Git folder.
# MAGIC 3. Runs a hello-Spark query.
# MAGIC 4. Probes what serverless compute allows (cache, Spark conf, SparkContext / Spark UI).
# MAGIC
# MAGIC Safe to re-run: everything is `IF NOT EXISTS`.

# COMMAND ----------

dbutils.widgets.text("catalog", "netusage", "Catalog")
CATALOG = dbutils.widgets.get("catalog")
SCHEMAS = ["landing", "bronze", "silver", "gold"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Catalog, schemas, landing volume
# MAGIC If your workspace doesn't allow creating catalogs, this falls back to the built-in
# MAGIC `workspace` catalog — note which one it used.

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception as e:
    print(f"Could not create catalog '{CATALOG}' ({type(e).__name__}); using 'workspace'.")
    CATALOG = "workspace"

for schema in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")

# Raw synthetic exports (Parquet) are written here, e.g. /Volumes/<catalog>/landing/raw/...
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.landing.raw")

LANDING_PATH = f"/Volumes/{CATALOG}/landing/raw"
print(f"Catalog: {CATALOG}")
print(f"Landing path: {LANDING_PATH}")
display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Import the repo package
# MAGIC In a Git folder, the working directory is this notebook's folder, so `../src` is the package
# MAGIC root.

# COMMAND ----------

import os
import sys

src_path = os.path.abspath("../src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import netusage  # noqa: E402

print(f"netusage imported from: {netusage.__file__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Hello Spark
# MAGIC Look at the `explain()` output: the `Exchange` line is a **shuffle** — rows moving between
# MAGIC partitions so that all rows for the same `user_id` end up together for the `groupBy`.

# COMMAND ----------

from pyspark.sql import functions as F

events = spark.range(1_000_000).select(
    (F.col("id") % 1000).cast("string").alias("user_id"),
    (F.rand(seed=42) * 10_000).cast("long").alias("bytes"),
)
per_user = events.groupBy("user_id").agg(F.sum("bytes").alias("total_bytes"))

per_user.explain()
display(per_user.orderBy(F.desc("total_bytes")).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Serverless capability check
# MAGIC Each probe records OK or the error instead of failing the notebook.

# COMMAND ----------

results = {"spark_version": spark.version}


def probe(name, fn):
    try:
        results[name] = f"OK: {fn()}"
    except Exception as e:
        results[name] = f"NOT AVAILABLE: {type(e).__name__}: {str(e).splitlines()[0][:150]}"


def try_cache():
    df = spark.range(100_000)
    df.cache()
    df.count()
    return "cache() + count() succeeded"


def try_shuffle_conf():
    spark.conf.set("spark.sql.shuffle.partitions", "64")
    return f"shuffle.partitions now = {spark.conf.get('spark.sql.shuffle.partitions')}"


def try_broadcast_conf():
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    return (
        f"autoBroadcastJoinThreshold now = {spark.conf.get('spark.sql.autoBroadcastJoinThreshold')}"
    )


def try_aqe_conf():
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    value = spark.conf.get("spark.sql.adaptive.enabled")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    return f"adaptive.enabled could be set to {value}"


def try_spark_context():
    return f"Spark UI URL = {spark.sparkContext.uiWebUrl}"


probe("cache", try_cache)
probe("conf: shuffle partitions", try_shuffle_conf)
probe("conf: broadcast threshold", try_broadcast_conf)
probe("conf: AQE toggle", try_aqe_conf)
probe("sparkContext / Spark UI", try_spark_context)

for name, outcome in results.items():
    print(f"{name:<28} {outcome}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Manual check
# MAGIC Under the output of the *Hello Spark* cell, look for a **"See performance"** / query
# MAGIC profile link. Open it and note what you can see (stages? shuffle sizes? per-task times?).
# MAGIC Paste the probe results above + what you saw to the agent.
