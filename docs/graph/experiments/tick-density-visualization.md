---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, feature/tick-density]
bucket_size: 1m, 5s
---
# Tick Density Visualization

**Question.** Are cross-day macro tick-density distributions normally distributed at each minute/bucket, and how do their mean/p25/p75 bands look?

**Scope.** A matplotlib (`Agg`) module under `viz/` reads the macro tick-density parquet datasets, aggregates `tick_count` by index (`macro_minute_index` vs `bucket_index`) across `date_utc`, runs a per-index Shapiro normality test, and writes histogram and mean/p25/p75 band PNGs plus CSV summaries to `outputs/figs/tick_density/`. An overall normality label (e.g. `mixed`) is reported.

**Headline.** see source artifacts

**Concepts.** [[tick-density]] · [[macro-window]]

**Artifacts.** [[2026-04-25-tick-density-visualization]]

**Related.** [[expanded-macro-tick-density]] · [[one-minute-total-size-bands]]
