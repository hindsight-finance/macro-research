---
type: experiment
status: Exploratory
tags:
  - asset/nq
  - window/macro
  - feature/fvg
  - feature/fvg/delta-dominance
  - feature/volume-delta
  - feature/volume-delta/bucket-5s
  - feature/mae-mfe
predictors: "absolute delta_imbalance dominance only"
targets: "entry-triggered & successful-only mfe_pct/mae_pct (mean/median/p75/p90)"
bins: "abs_delta_imbalance quartiles; minute_block; creation minute"
---
# FVG Delta MAE/MFE Profiles

**Question.** What do the MAE/MFE excursion profiles look like across absolute volume-delta dominance quantiles, for both entry-triggered and successful FVGs?

**Scope.** Extends the macro FVG summary machinery in `features/macro_fvg_study.py`. Uses only absolute delta dominance (cleaner than aligned in earlier analysis). Adds entry-excursion scopes by abs-delta quantile, by minute block x quantile, and by creation minute x quantile; success-context scopes already carry MFE/MAE. Exports three CSV profiles and two PNG figures driven from the summary table, combining `entry_triggered` and `successful_only` profile rows.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[volume-delta]] · [[mae-mfe]]

**Artifacts.** [[2026-05-13-fvg-delta-mae-mfe-profiles]]

**Related.** [[macro-fvg-volume-delta-dominance]] · [[fvg-delta-time-basis]] · [[macro-fvg-excursion]]
