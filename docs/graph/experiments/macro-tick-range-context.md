---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/open-1550, window/macro/close-1559, feature/tick-density, feature/outcome]
candles: k350, k359
bucket_size: 5s
bins: 25/50/75/90 thresholds, deciles
---
# Macro Tick Range Context

**Question.** How much price range forms in the early time slices (especially the first 10 seconds) of the 15:50 and 15:59 ET candles, relative to the candle and the full macro range?

**Scope.** New module `features/macro_tick_range_context.py` scans `input-data/merged_nq_ticks.parquet` (`ts_event`, `intra_ts_rank`, `price_ticks`), filters 15:50:00–15:59:59 ET, and builds per-window ranges for candles `k350`/`k359` over 5-second cumulative windows (`00_04`…`00_59`) and named windows (`first_5s`, `first_10s`, `first_30s`, `last_30s`, `full_candle`). Metrics: raw % of open, % of candle range, % of macro range, and additive high/low extension. Emits long-form study + summary parquet (window baseline, 25/50/75/90 thresholds, deciles). Price-range only; no volume delta or direction.

**Headline.** see source artifacts

**Concepts.** [[tick-density]] · [[macro-window]] · [[macro-outcome]] · [[macro-open-1550]] · [[macro-close-1559]]

**Artifacts.** [[2026-05-14-macro-tick-range-context-design]] · [[2026-05-14-macro-tick-range-context]]

**Related.** [[macro-vwap-barrier-context]] · [[macro-range-forecast]]
