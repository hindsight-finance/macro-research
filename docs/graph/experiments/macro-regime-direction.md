---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/pm, feature/outcome, feature/trend/regime, feature/volume-delta]
prior_windows: ["1pm-3pm", "3pm-3:50pm"]
bins: [regime_score_tertiles, delta_sign]
targets: [macro_direction]
---
# Macro regime → direction

**Question.** Do prior regime windows (1pm–3pm, 3pm–3:50pm) and prior-window [[volume-delta]]
predict [[macro-window]] direction?

**Scope.** Date-level study table joining prior regime windows, [[macro-outcome]], and 1m
volume-delta windows; regime-score tertiles + delta-sign buckets + outcome correlations.

**Headline.** Macro direction baseline ≈ coin-flip (~49.6% bull / 48.7% bear). Prior regime
scores and prior-window delta are **weak** directional predictors; the only strong signal is
delta inside the macro window itself (a sanity check, not an edge). The joined table is reusable
for later modeling.

**Concepts.** [[macro-outcome]] · [[trend-regime]] · [[historical-regimes]] · [[volume-delta]]

**Artifacts.** [[macro_regime_direction_study]]

**Related.** [[three-scalar-regime]] · [[trend-modeling-table]]
