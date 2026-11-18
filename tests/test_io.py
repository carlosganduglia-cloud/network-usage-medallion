"""Landing-reader tests: strict headers, malformed rows, schema safety, provenance."""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from netusage.io import (
    CORRUPT_COLUMN,
    SOURCE_COLUMN,
    corrupt_rows,
    read_csv,
    read_json_lines,
    read_parquet,
    with_corrupt_column,
)

SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("user_email", StringType()),
    ]
)
JSON_SCHEMA = StructType(
    [
        StructField("id", StringType()),
        StructField("status", IntegerType()),
    ]
)
PARQUET_SCHEMA = StructType([StructField("id", StringType())])


def test_csv_header_must_match_schema(spark, tmp_path):
    """enforceSchema=False fails loudly on a reordered header instead of shifting data."""
    path = tmp_path / "swapped.csv"
    path.write_text("user_email,event_id\nu1@corp.example,e1\n", encoding="utf-8")

    with pytest.raises(Exception, match="CSV header does not conform to the schema"):
        read_csv(spark, path, SCHEMA).collect()


def test_csv_with_matching_header_reads_by_name(spark, tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text("event_id,user_email\ne1,u1@corp.example\n", encoding="utf-8")

    row = read_csv(spark, path, SCHEMA).first()

    assert (row.event_id, row.user_email) == ("e1", "u1@corp.example")


def test_readers_do_not_mutate_shared_schemas(spark, tmp_path):
    """StructType.add() mutates in place; the reader must build a copy."""
    csv_path = tmp_path / "ok.csv"
    csv_path.write_text("event_id,user_email\ne1,u1@corp.example\n", encoding="utf-8")
    json_path = tmp_path / "ok.jsonl"
    json_path.write_text('{"id":"a","status":1}\n', encoding="utf-8")

    csv_before = list(SCHEMA.fields)
    json_before = list(JSON_SCHEMA.fields)

    read_csv(spark, csv_path, SCHEMA).count()
    read_json_lines(spark, json_path, JSON_SCHEMA).count()

    assert SCHEMA.fields == csv_before
    assert JSON_SCHEMA.fields == json_before
    assert CORRUPT_COLUMN not in SCHEMA.fieldNames()

    enhanced = with_corrupt_column(SCHEMA)
    assert enhanced is not SCHEMA
    assert CORRUPT_COLUMN in enhanced.fieldNames()
    assert CORRUPT_COLUMN not in SCHEMA.fieldNames()


def test_unparseable_csv_rows_are_captured(spark, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        """event_id,user_email
e1,u1@corp.example
e2
e3,u3@corp.example,EXTRA
""",
        encoding="utf-8",
    )
    df = read_csv(spark, path, SCHEMA).cache()  # local workaround; production writes to Delta
    bad = corrupt_rows(df)

    assert bad.count() == 2
    rows = {r.event_id: r for r in bad.collect()}
    assert rows["e2"][CORRUPT_COLUMN] == "e2"
    assert rows["e3"][CORRUPT_COLUMN] == "e3,u3@corp.example,EXTRA"
    assert all("bad.csv" in r[SOURCE_COLUMN] for r in rows.values())
    assert df.count() - bad.count() == 1


def test_unparseable_json_lines_are_captured(spark, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"a","status":1}\n'
        '{"id":"b","status":"abc"}\n'  # wrong type, not just bad syntax
        "NOT JSON\n",
        encoding="utf-8",
    )
    df = read_json_lines(spark, path, JSON_SCHEMA).cache()
    bad = corrupt_rows(df)

    assert bad.count() == 2
    rows = {r[CORRUPT_COLUMN]: r for r in bad.collect()}
    assert any(line.startswith('{"id":"b"') for line in rows)
    assert any(line == "NOT JSON" for line in rows)
    assert all("bad.jsonl" in r[SOURCE_COLUMN] for r in rows.values())
    assert df.where("id = 'a'").count() == 1


def test_parquet_reads_capture_source_file(spark, tmp_path):
    path = tmp_path / "data.parquet"
    spark.createDataFrame([Row(id="p1")], PARQUET_SCHEMA).write.mode("overwrite").parquet(str(path))

    row = read_parquet(spark, path, PARQUET_SCHEMA).first()

    assert row.id == "p1"
    assert "data.parquet" in row[SOURCE_COLUMN]


def test_corrupt_rows_requires_a_landing_reader(spark):
    with pytest.raises(ValueError, match="use a landing reader"):
        corrupt_rows(spark.createDataFrame([Row(event_id="e1")]))


def test_csv_column_pruning_hides_extra_field_rows(spark, tmp_path):
    """Pins down a Spark gotcha the validation module depends on.

    An aggregation that only needs some columns makes the CSV parser skip the rest
    (spark.sql.csv.parser.columnPruning.enabled), and then a row with an extra field
    is NOT flagged in _corrupt_record. Parsing every column (here via cache) catches it.
    If this test ever fails, Spark changed the behavior; revisit validation._full_parse.
    """
    path = tmp_path / "extra.csv"
    path.write_text(
        "event_id,user_email\ne1,u1@corp.example\ne2,u2@corp.example,EXTRA\n", encoding="utf-8"
    )
    flagged = F.sum(F.col(CORRUPT_COLUMN).isNotNull().cast("int"))

    pruned = read_csv(spark, path, SCHEMA).agg(F.count("event_id"), flagged).first()
    full = read_csv(spark, path, SCHEMA).cache().agg(F.count("event_id"), flagged).first()

    assert pruned[1] == 0
    assert full[1] == 1
