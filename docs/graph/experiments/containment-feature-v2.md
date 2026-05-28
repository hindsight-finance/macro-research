---
type: experiment
status: Exploratory
tags: [asset/nq, feature/trend, feature/trend/containment, feature/trend/trendability, window/pm]
support_windows: [1pm-3pm, 3pm-3:50pm]
targets: [containment_target]
predictors: [containment_overshoot_ratio, containment_range_stability, containment_mid_cross_count, containment_swing_symmetry]
model: [ridge]
---
# Containment Feature V2

**Question.** Can a descriptive containment target plus a small set of v2 features better separate clean bounded rotational auctions from one-sided trends and messy two-sided chop, without disturbing the frozen trendability target or walk-forward harness?

**Scope.** Adds a second realized-window target `containment_target` (0.30 displacement / 0.30 edge-balance / 0.25 inside-share / 0.15 path-efficiency, clipped 0-1) independent of trendability. Adds four v2 features to the modeling table: `containment_overshoot_ratio`, `containment_range_stability`, `containment_mid_cross_count`, `containment_swing_symmetry`. Two narrow registry variants (`core5_dra + containment_v2`, `adx_parts + containment_v2`) rerun only for `1pm-3pm` and `3pm-3:50pm`; this is historical labeling/backtesting, not live prediction.

**Headline.** see source artifacts

**Concepts.** [[containment]] · [[trendability]] · [[trend-regime]]

**Artifacts.** [[2026-04-14-containment-feature-v2-design]] · [[2026-04-14-containment-target-design]] · [[2026-04-14-containment-feature-v2]]

**Related.** [[trend-modeling-table]] · [[containment-model-first-expansion]]
