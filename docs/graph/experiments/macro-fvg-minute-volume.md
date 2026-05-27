---
type: experiment
status: Exploratory
tags:
  - asset/nq
  - window/macro
  - feature/fvg
  - feature/fvg/minute-volume
  - feature/volume-delta
predictors: "assigned_minute_index (15:50->0 .. 15:58->8), raw bar2_volume (assigned candle)"
bins: "creation minute; bar2_volume quartiles (qcut, duplicates=drop)"
targets: "hold/retrace/invalidation/untouched rates by 15:59"
---
# Macro FVG Creation Minute and Bar-2 Volume

**Question.** Are earlier or later macro-created FVGs more likely to hold, and does the assigned candle's raw volume change retrace/hold/invalidation likelihood?

**Scope.** Extends the macro FVG event study in place. Adds `assigned_minute_hhmm`, `assigned_minute_index` (15:50 -> 0 ... 15:58 -> 8), and raw `bar2_volume` (volume of the assigned FVG candle). Volume normalization explicitly rejected because the macro window is only 10 minutes; volume bucketed by simple quantiles (e.g. quartiles) with `duplicates="drop"`. Grouped summaries by creation minute and bar-2 volume bucket.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[macro-window]] · [[volume-delta]]

**Artifacts.** [[2026-03-09-macro-fvg-minute-volume-design]] · [[2026-03-09-macro-fvg-minute-volume]]

**Related.** [[macro-fvg-study]] · [[macro-fvg-alignment]]
