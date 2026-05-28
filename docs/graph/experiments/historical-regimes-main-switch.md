---
type: experiment
status: Exploratory
tags: [asset/nq, feature/trend, feature/trend/regime, feature/trend/state-detector, window/pm]
targets: [trend_score, containment_score, chop_score]
---
# Historical Regimes Main Switch

**Question.** Can the main historical regime classification path be switched from the legacy `StateDetector` ensemble to the three-scalar score stack plus a canonical discrete-label assignment?

**Scope.** Extracts label assignment into `features/trend/modeling/labels.py` (`assign_three_scalar_labels(...)`, `DEFAULT_LABEL_THRESHOLDS`: trend/containment/chop highs 0.70, low cutoff 0.40, `containment_chop_max` 0.55) and adds a `features/trend/historical_regimes.py` entry script that builds scores then assigns labels. Unmatched rows stay in output as `label="uncertain"`; the research probe drops uncertain rows before fitting. Output format chosen by `.parquet`/`.csv` suffix. `state_detector.py` gets docstring-only deprecation notes; indicator wrappers it provides are not removed.

**Headline.** see source artifacts

**Concepts.** [[historical-regimes]] · [[trend-regime]] · [[state-detector]]

**Artifacts.** [[2026-04-17-historical-regimes-main-switch]]

**Related.** [[three-scalar-regime]]
