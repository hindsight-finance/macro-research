---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/open-1550, feature/vwap, feature/vwap/anchor-1550, feature/vwap/anchor-1555, feature/barrier]
anchors: 15:50:00 ET (macro-open VWAP), 15:50 barrier extreme
checkpoints: 15:50:10, 15:51:00, 15:55:00 ET
candles: k350
targets: target_1550_1554, target_1555_1559, target_1550_1559, target_1550_10s_1554, target_1550_10s_1559, target_1551_1559
---
# Macro VWAP Barrier Context

**Question.** How does price behave relative to macro-open VWAP after the 15:50 first-10-second barrier forms, and at the 15:55 decision point?

**Scope.** New module `features/macro_vwap_barrier_context.py` joins `outputs/nq_macro_1550_barrier.parquet` and `outputs/nq_macro_vwap_intramacro.parquet` with bounded 15:50–16:00 ET tick metrics from `input-data/merged_nq_ticks.parquet`. Computes direction-aware constructive/wrong-side VWAP classification, post-barrier wrong-side distance/share inside the 15:50 minute, 15:55 transition context, and post-10s/post-15:51 targets. Emits raw + long-form summary parquet (barrier-only, VWAP-10s, barrier+VWAP, wrong-side, 15:55 decision scopes) plus distribution CSVs/figures. UTC internal, ET via `MARKET_TZ`; no eager tick read.

**Headline.** see source artifacts

**Concepts.** [[barrier-context]] · [[anchored-vwap]] · [[macro-window]] · [[macro-open-1550]]

**Artifacts.** [[2026-05-15-macro-vwap-barrier-context-design]] · [[2026-05-15-macro-vwap-barrier-context]]

**Related.** [[0001-macro-vwap-features]] · [[macro-tick-range-context]]
