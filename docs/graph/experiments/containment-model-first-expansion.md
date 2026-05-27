---
type: experiment
status: Exploratory
tags: [asset/nq, feature/trend, feature/trend/containment, feature/trend/state-detector, window/pm]
support_windows: [1pm-3pm, 3pm-3:50pm]
targets: [containment_target]
predictors: [containment_ib_extension_ratio, containment_ib_asymmetry, containment_bandwidth_squeeze, containment_vwap_acceptance, containment_excess_rejection]
model: [ridge, hist_gbm, random_forest, extra_trees, logistic_regression]
---
# Containment Model-First Expansion

**Question.** After model family (not features) drove the larger containment gains, does a formal regression/classification model bakeoff plus a few low-duplication OHLCV-native features add real incremental signal for clean-containment labeling?

**Scope.** Three-batch plan on the `containment_v2` table for `1pm-3pm` and `3pm-3:50pm`. Batch A: regression bakeoff (Ridge, HistGBM, RandomForest, ExtraTrees) and binary top-quartile clean-containment classification (LogisticRegression, HistGBM classifier, RandomForest). Batch B: five new features `containment_ib_extension_ratio`, `containment_ib_asymmetry`, `containment_bandwidth_squeeze`, `containment_vwap_acceptance`, `containment_excess_rejection`. Batch C: rerun only winning families. Adds a separate `containment_research.py` runner. Stated current bests cited as context: `1pm-3pm` ridge 0.5198 -> hist_gbm 0.5591; `3pm-3:50pm` ridge 0.4763 -> extra_trees 0.5167; top-decile precision 0.7727.

**Headline.** see source artifacts

**Concepts.** [[containment]] · [[trend-regime]] · [[state-detector]]

**Artifacts.** [[2026-04-16-containment-model-first-expansion-design]] · [[2026-04-16-containment-model-first-expansion]]

**Related.** [[containment-feature-v2]] · [[three-scalar-regime]]
