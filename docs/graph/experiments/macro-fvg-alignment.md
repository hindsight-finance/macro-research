---
type: experiment
status: Exploratory
tags:
  - asset/nq
  - window/macro
  - feature/fvg
  - feature/fvg/alignment
candles: "3 pattern candles (bar1/bar2/bar3); body direction from Open/Close, doji=neutral"
bins: "alignment: 3_aligned / 2_aligned_1_opposite / 1_aligned_2_opposite / contains_neutral; gap_size_bucket_225: <2.25 / >=2.25; minute_block"
---
# Macro FVG Candle Alignment Buckets

**Question.** Does candle-body alignment within the 3-bar FVG pattern add explanatory power for macro-window FVG outcomes beyond minute, gap-size, and volume effects?

**Scope.** Extends the existing macro FVG event study (`features/macro_fvg_study.py`, `outputs/nq_macro_fvg_events.parquet`). Uses only the three pattern candles (bar1/bar2/bar3); body direction from Open/Close with dojis kept as neutral. Four public buckets: `3_aligned`, `2_aligned_1_opposite`, `1_aligned_2_opposite`, `contains_neutral`. Controlled by minute block and the `<2.25` vs `>=2.25` gap-size split.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[macro-window]]

**Artifacts.** [[2026-03-09-macro-fvg-alignment-design]] · [[2026-03-09-macro-fvg-alignment]]

**Related.** [[macro-fvg-study]] · [[macro-fvg-success-context]]
