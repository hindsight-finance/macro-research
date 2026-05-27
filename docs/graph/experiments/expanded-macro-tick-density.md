---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, feature/tick-density]
support_windows: 15:40-16:10 ET
bucket_size: 1m
---
# Expanded Macro Tick Density Window

**Question.** Can the 1-minute macro tick-density dataset be widened from 15:50–15:59 ET to a broader 15:40–16:10 ET window without changing its schema?

**Scope.** NQ tick data via `tick_density.py`; expands `build_macro_tick_density()` to filter ET timestamps 15:40:00 through 16:10:59 inclusive (31 ET minutes/day), keeping UTC timestamps in parquet and the unchanged schema (`datetime_utc`, `date_utc`, `macro_minute_index`, `tick_count`, `total_size`, `buy_ticks`, `sell_ticks`, `none_ticks`). `macro_minute_index` is redefined as the actual ET minute number across the hour boundary.

**Headline.** see source artifacts

**Concepts.** [[tick-density]] · [[macro-window]] · [[tick-data]]

**Artifacts.** [[2026-04-25-expanded-macro-tick-density-design]] · [[2026-04-25-expanded-macro-tick-density]]

**Related.** [[tick-density-visualization]] · [[one-minute-total-size-bands]]
