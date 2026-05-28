---
type: experiment
status: Exploratory
tags: [asset/nq, feature/trend, feature/trend/regime, feature/trend/state-detector, window/pm]
targets: [trend_score, containment_score, chop_score]
model: [ridge]
---
# Three-Scalar Regime System

**Question.** Does splitting regime description into three independent scalars (`trend_score`, `containment_score`, `chop_score`) instead of one containment scalar better separate clean trend, clean bounded rotation, and ugly two-sided chop?

**Scope.** Aliases `trend_score = descriptive_target` and `containment_score = containment_target` (both frozen/unchanged) and adds a new realized-window `chop_score` (0.35 path-waste / 0.30 flip-rate / 0.20 outside-share / 0.15 instability, clipped 0-1) via `build_chop_target(...)`. Scores are deliberately not forced to sum to 1, so mixed sessions stay ambiguous. New table columns include the chop components and `chop_status`. Three-way labels are derived from scalar thresholds in the research runner (not persisted to the table), evaluated by macro-F1, balanced accuracy, and confusion matrix.

**Headline.** see source artifacts

**Concepts.** [[trend-regime]] · [[state-detector]]

**Artifacts.** [[2026-04-17-three-scalar-regime-design]] · [[2026-04-17-three-scalar-regime]]

**Related.** [[containment-model-first-expansion]] · [[historical-regimes-main-switch]]
