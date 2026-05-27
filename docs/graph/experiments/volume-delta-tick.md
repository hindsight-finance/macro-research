---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, feature/volume-delta, feature/volume-delta/bucket-1m, feature/volume-delta/bucket-5s, infra/ticks]
support_windows: 18:00-17:00 ET (Globex), 15:50-16:00 ET (macro)
bucket_size: 1m, 5s
---
# Tick Volume Delta

**Question.** What is the signed buy/sell volume delta and imbalance across the Globex session and macro window, using the tick aggressor-side field?

**Scope.** New root script `volume_delta.py` reads `input-data/merged_nq_ticks.parquet` (`ts_event`, `intra_ts_rank`, `side`, `price_ticks`, `size`) with lazy Polars scans. `side`: `2`=buy, `1`=sell, `0`=none (excluded from delta, kept as diagnostic). Builds three outputs: full Globex 1-minute (18:00–17:00 ET), macro 1-minute (15:50–16:00 ET, `macro_minute_index` 50–59), and macro 5-second (120 buckets incl. empties). Metrics include `volume_delta`, `delta_imbalance`, `tick_delta`, `classified_share`; UTC stored, ET derived.

**Headline.** see source artifacts

**Concepts.** [[volume-delta]] · [[tick-data]]

**Artifacts.** [[2026-04-26-volume-delta-tick-design]] · [[2026-04-26-volume-delta-tick]]

**Related.** [[macro-vwap-barrier-context]]
