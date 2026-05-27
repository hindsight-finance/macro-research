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
anchors: "join on FVG confirmed_at -> macro_bucket_index (0-119)"
predictors: "aligned & absolute delta_imbalance dominance"
bins: "delta_imbalance quartiles (q1_lowest..q4_highest), rank-based"
bucket_size: "5s"
targets: "successful_share_of_confirmable"
---
# Macro FVG Volume Delta Dominance

**Question.** Is 5-second tick volume-delta dominance around a macro FVG associated with the FVG's expected win rate?

**Scope.** Joins macro FVG events (`outputs/nq_macro_fvg_events.parquet`) to the 5-second volume-delta parquet (`outputs/nq_macro_volume_delta_5s.parquet`) via the FVG `confirmed_at` bucket (`macro_bucket_index` 0-119 over 15:50-15:59 ET, 5s each). Compares win metrics across quartiles of aligned vs. absolute `delta_imbalance` dominance; primary metric is `successful_share_of_confirmable`. Implemented as a helper module consuming the prebuilt delta parquet, never the raw tick file.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[volume-delta]] · [[macro-window]]

**Artifacts.** [[2026-05-13-macro-fvg-volume-delta-dominance-design]] · [[2026-05-13-macro-fvg-volume-delta-dominance]]

**Related.** [[macro-fvg-success-context]] · [[fvg-delta-mae-mfe-profiles]] · [[fvg-delta-time-basis]]
