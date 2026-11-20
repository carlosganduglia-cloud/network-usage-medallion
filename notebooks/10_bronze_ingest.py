# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Bronze ingest
# MAGIC
# MAGIC Landing files (Volume) → bronze Delta tables, with `source_file`, `batch_date`, `ingest_ts`
# MAGIC and an `ingest_audit` row (rows + `_corrupt_record` count) per entity and batch.
# MAGIC
# MAGIC - **Daily run:** `start_date = end_date` = the batch to load.
# MAGIC - **Backfill:** a range. Traffic logs are read and written once for the whole range; the
# MAGIC   snapshot entities (users, apps, rules, groups, memberships, resolutions) are loaded once,
# MAGIC   stamped `batch_date = end_date`.
# MAGIC - **Re-runs are safe:** each write replaces its slice (Delta `replaceWhere`), never appends.
# MAGIC
# MAGIC All logic lives in `netusage.jobs.bronze` (tested locally); this notebook only wires it up.

# COMMAND ----------

dbutils.widgets.text("catalog", "netusage", "Catalog")
dbutils.widgets.text("start_date", "2025-01-01", "Start date (batch_date)")
dbutils.widgets.text("end_date", "2025-01-01", "End date (batch_date, inclusive)")
dbutils.widgets.dropdown("load_snapshots", "true", ["true", "false"], "Load snapshot entities")

# COMMAND ----------

import os
import sys
from datetime import date

src_path = os.path.abspath("../src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from netusage.jobs import bronze  # noqa: E402

CATALOG = dbutils.widgets.get("catalog")
START = date.fromisoformat(dbutils.widgets.get("start_date"))
END = date.fromisoformat(dbutils.widgets.get("end_date"))
LOAD_SNAPSHOTS = dbutils.widgets.get("load_snapshots") == "true"

LANDING = f"/Volumes/{CATALOG}/landing/raw"
BRONZE = f"{CATALOG}.bronze"  # a schema name -> Unity Catalog tables, e.g. netusage.bronze.users
print(f"{START}..{END}  snapshots={LOAD_SNAPSHOTS}  {LANDING} -> {BRONZE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest
# MAGIC The returned audit rows are counted from the bronze tables after the write: every column has
# MAGIC been parsed there, so the `_corrupt_record` counts are exact (unlike a count on raw CSV).

# COMMAND ----------

audit = bronze.run(spark, LANDING, BRONZE, START, END, load_snapshots=LOAD_SNAPSHOTS)
display(audit)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look around
# MAGIC Each `replaceWhere` write is one Delta commit. `DESCRIBE HISTORY` shows them: re-run this
# MAGIC notebook with the same dates and you get a new table version (`operationParameters` shows
# MAGIC the predicate, `operationMetrics` the files removed/added and rows written), while the
# MAGIC table's row count stays the same.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {BRONZE}.traffic_logs").limit(5))

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT entity, batch_date, rows, corrupt_rows, ingest_ts
        FROM {BRONZE}.ingest_audit
        ORDER BY batch_date DESC, entity
        LIMIT 50
    """)
)
