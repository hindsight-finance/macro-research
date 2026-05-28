---
type: experiment
status: Exploratory
tags: [asset/nq, feature/trend, feature/trend/regime, feature/trend/trendability, feature/trend/containment, feature/trend/adx, feature/trend/atr, feature/trend/dra, feature/trend/irr, feature/trend/lag-hurst, feature/trend/mss, feature/trend/spd, feature/trend/efficiency-ratio, feature/trend/variance-ratio, feature/trend/state-detector, window/pm]
support_windows: [1pm-3pm, 3pm-3:50pm, 3:50pm-4pm]
targets: [descriptive_target]
predictors: [mss, adx_quality, irr, er, log_vr, dra]
era_filters: [full_dev, pre_covid, post_covid]
model: [ridge, elastic_net]
---
# Trend Modeling Table and Walk-Forward Harness

**Question.** Can a canonical session-level feature table plus a grouped walk-forward training harness calibrate usable, stable feature weights for a descriptive trend-regime filter from realized intraday price action?

**Scope.** New `features/trend/modeling/` package: a descriptive realized-window target (`descriptive_target`, a 0.35/0.25/0.20/0.20 blend of strength/consistency/smoothness/retention), a cached one-row-per `instrument x trade_date x session_name` feature table over sessions `1pm-3pm`, `3pm-3:50pm`, `3:50pm-4pm`, an experiment registry (`core5`, `core5_dra`, `adx_parts`; era filters `full_dev`/`pre_covid`/`post_covid` excluding the 2020-03 to 2020-06 transition), and a Ridge-default walk-forward runner (24m train / 3m validate / 3m step, last 15% held out) with Elastic Net as confirmatory pass. DRA is an experiment toggle, not baseline.

**Headline.** see source artifacts

**Concepts.** [[trend-regime]] · [[state-detector]] · [[trendability]] · [[containment]] · [[adx]] · [[atr-range-ratio]] · [[dra]] · [[irr]] · [[lag-autocorr-hurst]] · [[mss]] · [[swing-point-density]] · [[efficiency-ratio]] · [[variance-ratio]]

**Artifacts.** [[2026-04-12-trend-modeling-table]]

**Related.** [[containment-feature-v2]] · [[post-adx-ablation]] · [[three-scalar-regime]]
