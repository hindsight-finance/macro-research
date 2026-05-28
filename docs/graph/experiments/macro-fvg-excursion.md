---
type: experiment
status: Exploratory
tags:
  - asset/nq
  - window/macro
  - feature/fvg
  - feature/fvg/excursion
  - feature/mae-mfe
anchors: "entry_price = bar3_high (bullish) / bar3_low (bearish)"
targets: "mfe_pct_to_1559, mae_pct_to_1559 (percent of entry_price)"
bins: "alignment_bucket, minute_block, gap_size_bucket_225"
---
# Macro FVG Conditional Entry Excursion

**Question.** After confirmation, how often does a bar3 breakout entry actually trigger, and what favorable/adverse excursion (MFE/MAE) does it see through the 15:59 close?

**Scope.** Extends the macro FVG event study in place. Entry price = `bar3_high` (bullish) or `bar3_low` (bearish); trigger is the first bar at/after `confirmed_at` trading through entry. MFE/MAE measured from the bar after the trigger to the 15:59 close, normalized as percent of `entry_price`; triggers on 15:59 mark triggered but leave excursions null. Distribution-first, not a barrier study.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[macro-window]] · [[mae-mfe]]

**Artifacts.** [[2026-03-09-macro-fvg-excursion-design]] · [[2026-03-09-macro-fvg-excursion]]

**Related.** [[macro-fvg-study]] · [[macro-fvg-success-context]] · [[fvg-delta-mae-mfe-profiles]]
