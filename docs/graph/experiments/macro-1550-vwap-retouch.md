---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/open-1550, window/macro/close-1559, feature/barrier, feature/barrier/break-direction, feature/vwap, feature/vwap/anchor-1550, feature/vwap/retouch-1550, feature/mae-mfe, feature/outcome]
anchors: [macro_1550_break, retouch_frozen, retouch_rolling]
checkpoints: ["15:50:10", "15:55:00", "16:00:00"]
barrier_seconds: 10
side_threshold_pts: 0.25
sample_n: 1252
---
# Macro 15:50 VWAP Retouch

**Question.** After the first-10s barrier of the 15:50 macro open breaks (the first side to break sets a directional bias), does a retouch of the 15:50-anchored VWAP — frozen first-10s, or rolling cumulative — describe a usable forward path into the 15:59 close?

**Scope.** New module `features/macro_1550_vwap_retouch.py`; one row per macro date over 1,252 NQ dates (2020-09→2025-11), range-read in place from the R2 lake on GitHub Actions ([[remote-compute]]). Descriptive event study — no target/stop. Emits a per-date parquet + long-form summary (coverage, signal-validation, retouch frequency/lag, forward-outcome and MFE/MAE by bias × anchor × horizon, and decile cross-tabs).

**Headline.** Barrier breaks on 99.9% of days; frozen-VWAP retouch on 76% of triggered days (rolling 97%); break direction matches the macro close 68.9%; mean forward-to-close +2 to +4 pts favourable-to-bias against ±~20 pt MFE/MAE → a modest, noisy descriptive tilt. Exploratory — see [[0002-macro-1550-vwap-retouch]].

**Concepts.** [[first-10s-break-direction]] · [[anchored-vwap]] · [[barrier-context]] · [[macro-open-1550]] · [[macro-window]] · [[macro-close-1559]] · [[macro-outcome]] · [[mae-mfe]]

**Artifacts.** [[2026-05-29-macro-1550-vwap-retouch-design]] · [[2026-05-29-macro-1550-vwap-retouch]] · [[0002-macro-1550-vwap-retouch]]

**Related.** [[macro-vwap-barrier-context]] · [[0001-macro-vwap-features]]
