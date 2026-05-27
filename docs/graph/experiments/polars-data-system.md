---
type: experiment
status: Exploratory
tags: [asset/nq, infra/pipeline, infra/ticks]
---
# Polars Data System Migration

**Question.** Can active code and tests be migrated from pandas-first to Polars-first data handling, with tick parquet accessed only through lazy scans or bounded streaming collects rather than eager full-file loads?

**Scope.** Makes Polars the project dataframe API while preserving file layout and entry points. Core IO centralizes in `utils/minute_bars.py` (Polars) and a new `utils/tick_data.py` exposing `TICK_COLUMNS` (`ts_event, intra_ts_rank, side, price_ticks, size`), `get_tick_schema` (metadata only), `scan_tick_data` (LazyFrame), `collect_tick_window` (requires bounded start/end UTC, streaming collect), and `ticks_to_minute_bars` (price = `price_ticks / 4.0`). Downstream scripts (`session_tagger`, `macro_outcomes`, `pm_3pm`, `pm_macro_interactions`, `macro_fvg_study`, `helper`, `viz/*`) and tests migrate to Polars; pandas/numpy only at matplotlib boundaries on small frames. Adds an explicit `requirements.txt`. Excludes prose-only `docs/plans/` snippets.

**Headline.** see source artifacts

**Concepts.** [[data-pipeline]] · [[tick-data]]

**Artifacts.** [[2026-04-24-polars-data-system-design]] · [[2026-04-24-polars-data-system]]

**Related.** [[minute-base-utc-pipeline-refactor]]
