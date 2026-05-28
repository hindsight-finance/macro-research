---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, feature/tick-density, feature/volume-delta]
bucket_size: 1m
---
# One-Minute Total Size Bands

**Question.** How does traded `total_size` accumulate across the macro window by minute index, alongside the existing `tick_count` bands?

**Scope.** Extends `viz/tick_density_viz.py` to add a second band chart for the 1-minute macro dataset (`outputs/nq_macro_tick_density.parquet`), grouping by `macro_minute_index` and computing cross-day mean / p25 / p75 of `total_size` with full minute coverage `0..9`. Emits `nq_macro_tick_density_total_size_bands.png` and a matching band-stats CSV; leaves the tick-count path and no-histogram behavior intact.

**Headline.** see source artifacts

**Concepts.** [[tick-density]] · [[volume-delta]] · [[macro-window]]

**Artifacts.** [[2026-04-25-one-minute-total-size-bands-design]] · [[2026-04-25-one-minute-total-size-bands]]

**Related.** [[tick-density-visualization]] · [[expanded-macro-tick-density]]
