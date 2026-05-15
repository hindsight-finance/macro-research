# Macro Regime → Macro Direction Study

## What we built
- Joined prior regime windows (`1pm-3pm`, `3pm-3:50pm`), macro outcomes, and 1m volume-delta windows into one date-level study table.
- Added regime-score tertiles, delta sign buckets, and basic outcome correlations.
- Wrote parquet + CSV summaries + quick figures for cohort review.

## Short result readout
- Macro direction baseline was near coin-flip: ~49.6% bullish, ~48.7% bearish.
- Prior regime scores were weak directional predictors.
- Prior-window delta was modest at best.
- The obvious strong signal was delta inside the macro window itself, which is more sanity check than edge.

## Takeaway
- The study says pre-macro regime context alone is not a strong macro direction signal.
- The joined table is now reusable for later modeling work.
