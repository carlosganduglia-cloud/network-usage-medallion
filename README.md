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

### Synthetic data

```bash
uv run python -m netusage.generator --profile small     # 1M events, 365 daily batches -> data/landing (~1 min)
uv run python -m netusage.validation --profile small    # row counts, join-path integrity, dirty-data ratios
uv run python -m netusage.generator --profile tiny --days 7   # seconds; add --days to validation too
```
