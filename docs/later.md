# Parking lot

Interesting tangents, one line each. Revisit after the core pipeline is done.

- Spark-native transitive AD group expansion (iterative joins / GraphFrames) — upstream in the real system.
- Track how users/apps/rules change over time (SCD2 from daily snapshots) instead of using the latest snapshot.
- (Partly addressed: `bronze.ingest_audit` now counts corrupt rows reliably on Databricks.) Validation on Databricks: CSV column pruning hides extra-field rows from `_corrupt_record`, and `validation.py` relies on `cache()` (which serverless may ignore), so malformed counts could be wrong there.
  Fix: validate malformed rows after bronze, or force a full parse (a query that consumes every column, since `select(*cols).count()` is still pruned, or `spark.sql.csv.parser.columnPruning.enabled=false` if serverless allows it).
