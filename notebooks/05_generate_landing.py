# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Generate landing data
# MAGIC
# MAGIC Writes all 7 synthetic landing entities into the Volume, using the same
# MAGIC `netusage.generator` code that runs locally.
# MAGIC
# MAGIC The generator's randomness is a hash of `(SEED, purpose, row key)`, not `rand()`, so this
# MAGIC produces **the same rows as a local run** of the same profile. Local validation results
# MAGIC are therefore the expected numbers here.
# MAGIC
# MAGIC Overwrites whatever is in the landing Volume.

# COMMAND ----------

dbutils.widgets.text("catalog", "netusage", "Catalog")
dbutils.widgets.dropdown("profile", "small", ["tiny", "small", "full"], "Scale profile")
dbutils.widgets.text("days", "", "Days from START_DATE (blank = whole year)")

# COMMAND ----------

import os
import sys
from datetime import timedelta

src_path = os.path.abspath("../src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from netusage import config as C  # noqa: E402
from netusage import generator, schemas, validation  # noqa: E402

CATALOG = dbutils.widgets.get("catalog")
PROFILE = C.PROFILES[dbutils.widgets.get("profile")]
DAYS = dbutils.widgets.get("days").strip()
END = C.START_DATE + timedelta(days=int(DAYS) - 1) if DAYS else C.END_DATE
LANDING = f"/Volumes/{CATALOG}/landing/raw"

# Fail early with a clear message if 00_setup fell back to another catalog.
spark.sql(f"DESCRIBE VOLUME {CATALOG}.landing.raw")
print(f"profile={PROFILE.name}  {C.START_DATE}..{END} (+{C.LATE_EVENT_MAX_DAYS} late days)")
print(f"-> {LANDING}")

# COMMAND ----------

generator.generate(spark, PROFILE, LANDING, C.START_DATE, END)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick file-layout check
# MAGIC The traffic writer relies on Spark keeping a sort order so each daily file **starts with its
# MAGIC header**. That has only been proven locally, so check it here before bronze reads the files.
# MAGIC (Bronze would also fail loudly: `enforceSchema=false` checks every file's first line.)

# COMMAND ----------

traffic_dir = f"{LANDING}/traffic_logs"
folders = sorted(
    f.name.rstrip("/") for f in dbutils.fs.ls(traffic_dir) if f.name.startswith("batch_date=")
)
print(f"{len(folders)} batch folders: {folders[0]} .. {folders[-1]}")

expected_header = ",".join(schemas.TRAFFIC_LOGS.fieldNames())
problems = []
for folder in folders:  # ~370 small metadata calls: takes a minute or two
    data_files = [
        f for f in dbutils.fs.ls(f"{traffic_dir}/{folder}") if not f.name.startswith(("_", "."))
    ]
    if len(data_files) != 1:
        problems.append(f"{folder}: {len(data_files)} data files")
        continue
    first_line = dbutils.fs.head(data_files[0].path, 4096).splitlines()[0]
    if first_line != expected_header:
        problems.append(f"{folder}: first line is {first_line[:80]!r}")

print("OK: one file per folder, header first" if not problems else "\n".join(problems[:20]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: run `validation.py` here (an experiment, not a gate)
# MAGIC Local run of the same data: **44/44 checks pass**, traffic malformed rows = **476**,
# MAGIC membership malformed rows = **1,007**.
# MAGIC
# MAGIC Prediction: the two *malformed CSV rows* checks may **fail** here. Validation counts
# MAGIC `_corrupt_record` with an aggregation over the raw CSV, and relies on `cache()` to force
# MAGIC every column to be parsed first. If serverless ignores the cache, CSV column pruning
# MAGIC parses only the columns the aggregation needs, and a row with an extra field is never
# MAGIC flagged. Every planted malformed row has an extra field, so the count could drop to 0.
# MAGIC That's why the trusted corrupt counts are the ones in `bronze.ingest_audit`
# MAGIC (see `10_bronze_ingest`).

# COMMAND ----------

checks = validation.validate(spark, LANDING, PROFILE, C.START_DATE, END)
print(validation.report(checks))
