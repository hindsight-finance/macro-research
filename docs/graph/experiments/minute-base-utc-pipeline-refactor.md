---
type: experiment
status: Exploratory
tags: [asset/nq, infra/time, infra/pipeline]
---
# Minute-Base UTC Pipeline Refactor

**Question.** Can the minute-bar pipeline be reduced to one canonical per-instrument UTC base parquet (durable OHLCV only) while downstream scripts derive New York market time and session/window labels in memory?

**Scope.** Defines a canonical base contract: `outputs/nq_minute_base.parquet` / `es_minute_base.parquet` holding `datetime_utc` (UTC `datetime64`, not string) plus OHLCV, with no persisted `DateTime_ET`/`session`/`window`. Adds shared `utils/minute_bars.py` helpers (`normalize_minute_bars`, `load_minute_bars`, `build_market_time_columns`, `derive_session_window`) using `America/New_York` (DST-safe), accepting legacy `DateTime_UTC`/`DateTime_ET` on ingest. `session_tagger.py` becomes the base writer; `macro_outcomes.py`, `features/macro_fvg_study.py`, `features/pm_3pm.py`, and `viz/macro_analysis.py` derive ET windows in memory. Window labels: H3PM 15:00-15:49, MACRO 15:50-15:59, POST 16:00-16:10. Trend/LRLR fixture conversion is out of scope.

**Headline.** see source artifacts

**Concepts.** [[time-handling]] · [[data-pipeline]]

**Artifacts.** [[2026-04-17-minute-base-utc-pipeline-design]] · [[2026-04-17-minute-base-utc-pipeline-refactor]]

**Related.** [[polars-data-system]]
