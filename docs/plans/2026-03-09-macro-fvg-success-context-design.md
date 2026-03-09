# Macro FVG Successful Retrace Context Design

**Date:** 2026-03-09

## Goal

Contextualize macro-window FVGs by focusing on the subset that retrace, then successfully continue through the first retrace candle in the FVG direction, and measure how much adverse excursion those successful setups typically tolerate.

This pass is explicitly context-first:
- keep the existing macro FVG event study intact
- identify the first retrace candle for each FVG
- define success from that first retrace candle
- summarize MAE on successful FVGs
- account for stacked continuation FVG structures

## Existing Repo Fit

The current study already lives in:
- `features/macro_fvg_study.py`
- `outputs/nq_macro_fvg_events.parquet`
- `outputs/nq_macro_fvg_summary.parquet`
- `outputs/figs/fvg/`

This extension should modify that existing study in place.
The event parquet should remain the source of truth.

## Approved Scope

Use the already-detected macro FVG events and add a new success-context layer.

Approved first-pass behavior:
- define the first retrace candle as the first post-confirmation candle that trades back into the FVG
- define success from the first retrace candle, not from the original FVG bars
- keep both FVGs in the 4-bar continuation / stacked case
- add a stacked continuation flag and predecessor link
- report MAE statistics on successful FVGs

Do not add in this pass:
- barrier outcomes
- new trigger or trade entry rules
- longer holding horizons
- standalone scripts
- broad extra figure sets unless the implementation needs one small sanity-check chart

## Definitions

### First Retrace Candle

For an FVG that retraces, the first retrace candle is:
- the first bar at or after `confirmed_at`
- whose range trades back into the FVG interval

This should be stored explicitly on the event row:
- `first_retrace_candle_at`
- `first_retrace_candle_open`
- `first_retrace_candle_high`
- `first_retrace_candle_low`
- `first_retrace_candle_close`

If no retrace occurs, these fields remain null.

### Successful FVG

A successful FVG is defined from the first retrace candle:
- bullish FVG: price later breaks above the first retrace candle `High`
- bearish FVG: price later breaks below the first retrace candle `Low`

Recommended event-level fields:
- `success_reference_price`
- `successful_by_1559`
- `success_break_at`

Implementation note:
- only evaluate success after the first retrace candle has occurred
- if a retrace happens but price never breaks the reference extreme by `15:59`, the event is not successful

### MAE Context

The primary first-pass output should be MAE on successful FVGs only.

Use the already-existing percent excursion convention from the current study:
- normalize by the existing `entry_price`
- use `mae_pct_to_1559` as the adverse excursion metric

For this pass, report:
- `mae_pct_mean`
- `mae_pct_median`
- `mae_pct_p75`

Interpretation note:
- `mae_pct_p75` is the upper-tail MAE threshold, meaning 75% of successful FVGs had MAE at or below this value
- it is not the single worst MAE observation, but it is intentionally on the worse side of the distribution

### Stacked Continuation FVG

Account for the aligned 4-bar continuation case where:
- one FVG forms
- its `bar3` is also the later FVG’s `bar2`
- both FVGs are aligned in the same direction

Approved behavior:
- keep both FVGs as separate events
- add a stacked continuation flag to the later FVG
- link the later FVG back to the earlier FVG

Recommended event-level fields:
- `stacked_continuation_fvg`
- `stack_predecessor_assigned_at`

Recommended identification rule:
- same day
- same `fvg_side`
- later FVG `assigned_at` equals earlier FVG `bar3_time`

Equivalent implementation is fine if it captures the same 4-bar continuation structure.

## Data Additions

Extend each event row with:
- `first_retrace_candle_at`
- `first_retrace_candle_open`
- `first_retrace_candle_high`
- `first_retrace_candle_low`
- `first_retrace_candle_close`
- `success_reference_price`
- `successful_by_1559`
- `success_break_at`
- `stacked_continuation_fvg`
- `stack_predecessor_assigned_at`

These should coexist with the current alignment, trigger, and excursion fields.

## Summary Extensions

Append successful-context grouped views to the existing summary parquet.

Recommended new `summary_scope` values:
- `success_context_overall`
- `success_context_alignment_bucket`
- `success_context_stacked_flag`
- `success_context_alignment_bucket_stacked_flag`

Recommended grouped metrics:
- `n_total`
- `n_confirmable`
- `n_retraced`
- `n_successful`
- `retrace_rate`
- `success_after_retrace_rate`
- `successful_share_of_confirmable`
- `mae_pct_mean`
- `mae_pct_median`
- `mae_pct_p75`

Recommended grouping fields:
- `alignment_bucket`
- `stacked_continuation_fvg`

Interpretation note:
- `retrace_rate` should use confirmable FVGs in the denominator
- `success_after_retrace_rate` should use retraced FVGs in the denominator
- `successful_share_of_confirmable` should use confirmable FVGs in the denominator
- MAE aggregates should use successful FVGs only

## Visualization

This pass does not require a large new figure set.

Recommended optional first-pass sanity figures:
- `successful_fvg_mae_by_alignment_bucket.png`
- `successful_fvg_mae_by_stacked_flag.png`

These should only be added if the implementation benefits from a direct visual check.
The event table and summary table are the primary outputs.

## Interpretation Goal

This extension should answer:
- how often confirmable FVGs retrace
- conditional on retrace, how often they successfully continue through the first retrace candle
- how much MAE successful FVGs usually tolerate
- whether alignment bucket changes that MAE tolerance
- whether stacked continuation FVGs are structurally cleaner or noisier

The first read should be:
- descriptive
- conditional on success
- framed as structure/context, not profitability

## Implementation Shape

Modify the existing study in place:
- retain enough event-level FVG bar metadata to identify stacked continuation structures
- enrich each event with first-retrace candle metadata during the current scan
- derive success from the first retrace candle
- reuse the existing percent MAE field for successful-FVG summaries
- append successful-context summary scopes to the existing summary parquet

Recommended function additions inside `features/macro_fvg_study.py`:
- `mark_stacked_continuation_fvgs()`
- `extract_first_retrace_candle()`
- `scan_success_after_first_retrace_until_1559_close()`
- `build_success_context_summary()`
- `build_success_context_alignment_bucket_summary()`
- `build_success_context_stacked_flag_summary()`
- `build_success_context_alignment_bucket_stacked_flag_summary()`

Equivalent naming is fine if the responsibilities stay clear.

## Testing

Add tests for:
- first retrace candle OHLC/time capture
- bullish success after first retrace candle high break
- bearish success after first retrace candle low break
- retraced but unsuccessful case
- no-retrace case keeps success context fields null / false
- stacked continuation flagging for the 4-bar same-direction continuation structure
- summary builders for:
  - overall success context
  - alignment bucket
  - stacked flag
  - alignment bucket by stacked flag

Focused real-data verification should also confirm:
- event parquet contains the new retrace/success/stacking fields
- summary parquet contains the new `success_context_*` scopes
- MAE summary rows are populated only for successful FVGs
- stacked continuation counts are non-zero if the pattern exists in real data
