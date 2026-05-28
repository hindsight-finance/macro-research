---
type: concept
tags: [feature/volume-delta]
---
# Volume delta

Signed traded volume — buy size minus sell size (`side` 2=buy, 1=sell; see [[tick-data]]) —
aggregated into time buckets. Produced by `volume_delta.py` into 5-second and 1-minute grids
(`outputs/nq_macro_volume_delta_5s.parquet`, `..._1m.parquet`, `nq_globex_volume_delta_1m.parquet`).
The core flow primitive for the macro thread.

Accumulated, unresolved delta over a session forms a [[cumulative-delta-imbalance]] predictor.

**Related.** [[tick-data]] · [[cumulative-delta-imbalance]] · [[macro-bucket-path|early conviction]]
**Used by.** [[macro-1550-delta-impulse]] · [[macro-delta-reversal]] · [[macro-bucket-path]] · [[volume-delta-tick]] · [[macro-fvg-volume-delta-dominance]]
