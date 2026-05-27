---
type: experiment
status: Exploratory
tags: [asset/nq, feature/trend, feature/trend/adx, feature/trend/state-detector, window/pm]
support_windows: [1pm-3pm, 3pm-3:50pm, 3:50pm-4pm]
targets: [descriptive_target]
predictors: [mss, adx_strength, adx_persistence, adx_crossover, irr, er, log_vr]
era_filters: [post_covid]
model: [ridge]
---
# Post-COVID ADX Ablation

**Question.** Which components of the winning post-COVID `adx_parts` trendability representation actually carry weight when ablated one at a time?

**Scope.** Adds an additive `post_adx_ablation` experiment group (post-COVID only, all sessions via the existing CLI) built by a focused registry spec builder, reusing the existing walk-forward runner and summary flow unchanged. Variants: base `adx_parts` (`EXP20`), `adx_parts_minus_persistence` (`EXP21`), `adx_parts_minus_crossover` (`EXP22`), `adx_parts_minus_log_vr` (`EXP23`), and optional `adx_parts_minus_irr` (`EXP24`). The main representation sweep is left intact.

**Headline.** see source artifacts

**Concepts.** [[adx]] · [[trend-regime]] · [[state-detector]]

**Artifacts.** [[2026-04-14-post-adx-ablation-design]] · [[2026-04-14-post-adx-ablation]]

**Related.** [[trend-modeling-table]]
