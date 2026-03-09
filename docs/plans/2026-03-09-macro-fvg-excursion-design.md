# Macro FVG Conditional Entry Excursion Design

**Date:** 2026-03-09

## Goal

Characterize the post-confirmation excursion profile of macro-window FVGs using a simple, tradeable trigger based on the third FVG candle, without yet imposing a fixed barrier or exit rule.

This pass is explicitly distribution-first:
- measure whether the `bar3` breakout actually triggers after confirmation
- measure conditional favorable and adverse excursion after that trigger
- normalize excursion in percent
- keep the existing macro FVG event-study architecture intact

## Existing Repo Fit

The current study already lives in:
- `features/macro_fvg_study.py`
- `outputs/nq_macro_fvg_events.parquet`
- `outputs/nq_macro_fvg_summary.parquet`
- `outputs/figs/fvg/`

This extension should modify that existing study in place.
The event parquet should remain the source of truth.

## Approved Scope

Use the existing detected macro FVG events and add a conditional entry-excursion layer.

Approved first-pass behavior:
- use `bar3_high` as bullish entry and `bar3_low` as bearish entry
- only measure MAE/MFE for events where price trades through that entry after `confirmed_at`
- normalize excursions as percent of `entry_price`
- stop the study at the existing `15:59` close
- keep this as an excursion study, not a barrier study

Do not add in this pass:
- 1:1 RR barrier outcomes
- multi-target barrier trees
- longer holding horizons
- separate standalone scripts

## Definitions

### Entry Price

Use the third candle of the FVG pattern as the reference breakout level:
- bullish FVG: `entry_price = bar3_high`
- bearish FVG: `entry_price = bar3_low`

### Trigger Rule

Only evaluate excursions for events that actually trade through the entry price after confirmation:
- bullish FVG triggers on the first bar at or after `confirmed_at` where `High >= entry_price`
- bearish FVG triggers on the first bar at or after `confirmed_at` where `Low <= entry_price`

Recommended event-level fields:
- `entry_price`
- `entry_triggered_by_1559`
- `first_entry_trigger_at`
- `entry_trigger_minute_hhmm`
- `entry_trigger_minute_index`

### Excursion Window

Use the current study horizon:
- start from the first trigger event
- stop at the `15:59` close

Implementation note:
- because only 1-minute OHLC is available, do not measure MFE/MAE using the same trigger bar extremes
- begin excursion measurement on the bar after `first_entry_trigger_at`
- if the first trigger occurs on `15:59`, mark the event as triggered but leave excursion fields null

### Excursion Metrics

Normalize by `entry_price`, not by stop distance.

Recommended event-level fields:
- `mfe_pct_to_1559`
- `mae_pct_to_1559`

Definitions:
- bullish FVG:
  - `mfe_pct_to_1559 = (max_future_high - entry_price) / entry_price`
  - `mae_pct_to_1559 = (entry_price - min_future_low) / entry_price`
- bearish FVG:
  - `mfe_pct_to_1559 = (entry_price - min_future_low) / entry_price`
  - `mae_pct_to_1559 = (max_future_high - entry_price) / entry_price`

All excursion fields should remain null for non-triggered events.

## Data Additions

Extend each event row with:
- `entry_price`
- `entry_triggered_by_1559`
- `first_entry_trigger_at`
- `entry_trigger_minute_hhmm`
- `entry_trigger_minute_index`
- `mfe_pct_to_1559`
- `mae_pct_to_1559`

These should coexist with the current alignment, minute-block, and gap-bucket annotations.

## Summary Extensions

Append excursion-oriented grouped views to the existing summary parquet.

Recommended new `summary_scope` values:
- `entry_excursion_overall`
- `entry_excursion_alignment_bucket`
- `entry_excursion_minute_block`
- `entry_excursion_gap_bucket`
- `entry_excursion_alignment_bucket_minute_block`

Recommended grouped metrics:
- `n_total`
- `n_confirmable`
- `n_triggered`
- `entry_trigger_rate`
- `mfe_pct_mean`
- `mfe_pct_median`
- `mfe_pct_p75`
- `mfe_pct_p90`
- `mae_pct_mean`
- `mae_pct_median`
- `mae_pct_p75`
- `mae_pct_p90`

Interpretation note:
- `entry_trigger_rate` should use all confirmable events in the denominator
- MAE/MFE aggregates should use only triggered events

## Visualization

Recommended first-pass figures under `outputs/figs/fvg/`:
- `entry_trigger_rate_by_alignment_bucket.png`
- `mfe_mae_pct_by_alignment_bucket.png`
- `mfe_pct_by_minute_block.png`
- `mfe_pct_by_gap_bucket.png`

Recommended interpretation:
- `entry_trigger_rate_by_alignment_bucket.png`
  - how often the conditional `bar3` breakout actually trades
- `mfe_mae_pct_by_alignment_bucket.png`
  - compare favorable and adverse excursion after trigger by alignment bucket
- `mfe_pct_by_minute_block.png`
  - show whether later-created FVGs have stronger post-trigger expansion
- `mfe_pct_by_gap_bucket.png`
  - show whether larger gaps retain better post-trigger excursion

Keep the current figures intact.

## Interpretation Goal

This extension should answer:
- how often the `bar3` breakout actually becomes tradeable before `15:59`
- conditional on trigger, how much favorable excursion exists
- how much adverse excursion is typically paid before `15:59`
- whether alignment improves trigger rate, post-trigger excursion quality, or both

The first-pass read should not be framed as profitability.
This is still a structure and distribution study.

## Implementation Shape

Modify the existing study in place:
- enrich event rows with conditional entry and excursion fields
- scan from `confirmed_at` through `15:59` for the first trigger
- if triggered before the last bar, compute MAE/MFE from subsequent bars only
- derive grouped summary tables from those enriched event rows
- add a small set of excursion-focused figures

Recommended function additions inside `features/macro_fvg_study.py`:
- `assign_entry_price()`
- `scan_entry_excursions_until_1559_close()`
- `build_entry_excursion_summary()`
- `build_entry_excursion_alignment_bucket_summary()`
- `build_entry_excursion_minute_block_summary()`
- `build_entry_excursion_gap_bucket_summary()`
- `build_entry_excursion_alignment_bucket_minute_block_summary()`

Equivalent naming is fine if the responsibilities stay this clear.

## Testing

Add tests for:
- bullish trigger detection
- bearish trigger detection
- no-trigger events keep excursion fields null
- trigger-on-15:59 events mark triggered but keep excursion fields null
- bullish MAE/MFE percent calculations
- bearish MAE/MFE percent calculations
- summary builder computes:
  - `entry_trigger_rate`
  - `n_triggered`
  - triggered-only MAE/MFE aggregates
- end-to-end run writes the new excursion figure files

Focused real-data verification should also confirm:
- event parquet contains the new excursion fields
- summary parquet contains the new entry-excursion scopes
- the new figure files are present under `outputs/figs/fvg/`
- triggered sample counts are large enough to interpret by alignment bucket
