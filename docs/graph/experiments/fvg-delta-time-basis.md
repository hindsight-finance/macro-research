---
type: experiment
status: Exploratory
tags:
  - asset/nq
  - window/macro
  - feature/fvg
  - feature/fvg/delta-dominance
  - feature/volume-delta
  - feature/volume-delta/bucket-5s
predictors: "aligned & absolute delta_imbalance dominance"
bins: "delta quantiles by creation minute (assigned_minute_index/hhmm) and by minute_block"
targets: "successful_share_of_confirmable"
---
# FVG Delta Time-Basis Summaries

**Question.** Does macro FVG volume-delta dominance relate to win rate differently when split by FVG creation minute versus by coarse minute block?

**Scope.** Extends `features/macro_fvg_study.py`, reusing the enriched delta-dominance event columns and `_group_success_context_stats`. Adds four success-context summary scopes: aligned and absolute delta-imbalance quantiles, each grouped by creation minute (`assigned_minute_index`/`assigned_minute_hhmm`) and by minute block. Keeps `nq_macro_fvg_summary.parquet` as the single summary output; primary metric is `successful_share_of_confirmable`.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[volume-delta]]

**Artifacts.** [[2026-05-13-fvg-delta-time-basis]]

**Related.** [[macro-fvg-volume-delta-dominance]] · [[fvg-delta-mae-mfe-profiles]]
