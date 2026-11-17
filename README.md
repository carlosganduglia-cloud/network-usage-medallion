# network-usage-medallion

A medallion-architecture (bronze → silver → gold) pipeline in PySpark for enterprise
network-access analytics, validated at 100M+ synthetic traffic events on Databricks.

> 🚧 Work in progress. All data in this project is synthetic.

## Local development

Requires WSL2/Linux, Java 17, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
```
