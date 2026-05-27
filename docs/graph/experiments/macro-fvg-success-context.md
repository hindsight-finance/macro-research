---
type: experiment
status: Exploratory
tags:
  - asset/nq
  - window/macro
  - feature/fvg
  - feature/fvg/success-context
  - feature/mae-mfe
anchors: "success_reference_price = first-retrace-candle high (bullish) / low (bearish)"
targets: "successful_share_of_confirmable; mae_pct/mfe_pct on successful only (mean/median/p75)"
bins: "alignment_bucket, stacked_continuation_fvg"
---
# Macro FVG Successful Retrace Context

**Question.** For macro FVGs that retrace, how often do they successfully continue past the first retrace candle, and how much adverse excursion do those successful setups tolerate?

**Scope.** Extends the macro FVG event study in place. First retrace candle = first post-confirmation bar trading back into the gap; success = a later bar breaking the first retrace candle's high (bullish) or low (bearish) by 15:59. MAE/MFE reported on successful FVGs only (mean/median/p75), using the existing percent excursion convention. Adds stacked-continuation flag for the aligned 4-bar case, grouped by alignment bucket and stacked flag.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[macro-window]] · [[mae-mfe]]

**Artifacts.** [[2026-03-09-macro-fvg-success-context-design]] · [[2026-03-09-macro-fvg-success-context]] · [[2026-03-09-macro-fvg-success-context-mfe]] · [[2026-03-09-macro-fvg-success-context-figures]] · [[2026-03-09-macro-fvg-success-rate-chart]]

**Related.** [[macro-fvg-study]] · [[macro-fvg-excursion]] · [[macro-fvg-alignment]] · [[macro-fvg-volume-delta-dominance]]
