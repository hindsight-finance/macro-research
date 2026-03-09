# Macro FVG Minute and Bar-2 Volume Extension Design

**Date:** 2026-03-09

## Goal

Extend the existing macro FVG study so the event-level output can be analyzed by:
- FVG creation minute
- Raw volume on bar 2, where bar 2 is the assigned FVG candle

The purpose is to make the current FVG results easier to interpret in trading terms:
- Are earlier or later macro-created FVGs more likely to hold?
- Does the assigned candle's raw volume change how likely the FVG is to retrace, hold, or invalidate?

## Existing Repo Fit

The repo already has:
- `features/macro_fvg_study.py`
- `outputs/nq_macro_fvg_events.parquet`
- `outputs/nq_macro_fvg_summary.parquet`
- FVG figures under `outputs/figs/fvg/`

This extension should build on the current event table rather than creating a separate volume study.

## Approved Scope

Use only:
- `assigned_at` minute
- raw `bar2_volume`

Do not add:
- volume normalization
- macro-direction alignment logic

The user explicitly rejected normalized volume for this pass because the macro window is only 10 minutes long.

## Data Additions

Extend each FVG event row with:
- `assigned_minute_hhmm`
- `assigned_minute_index`
- `bar2_volume`

Definitions:
- `assigned_minute_hhmm`: clock label from `assigned_at`, for example `15:50`
- `assigned_minute_index`: integer index inside the macro sequence
  - `15:50 -> 0`
  - `15:51 -> 1`
  - ...
  - `15:58 -> 8`
- `bar2_volume`: raw `Volume` from the bar at `assigned_at`

These fields should be stored directly in `outputs/nq_macro_fvg_events.parquet`.

## Summary Extensions

Extend the summary analysis to include grouped outcomes by:
- creation minute
- bar-2 volume bucket

Recommended grouped metrics:
- `n_total`
- `n_confirmable`
- `hold_rate`
- `invalidation_rate`
- `retrace_rate`
- `untouched_rate`

Recommended grouping outputs:
- `assigned_minute_hhmm`
- `bar2_volume_bucket`

The current stage-level and stage-1-to-stage-2 summary table should remain intact.
The new grouped views can be appended to the existing summary parquet or saved as a second summary section inside the same output.

## Visualization

Add new Matplotlib figures under `outputs/figs/fvg/`:
- `creation_minute_outcome_bars.png`
- `bar2_volume_bucket_outcomes.png`
- `creation_minute_avg_bar2_volume.png`
- `creation_minute_volume_heatmap.png`

Recommended interpretation of each:
- `creation_minute_outcome_bars.png`
  - show hold, retrace, and invalidation rates by creation minute
- `bar2_volume_bucket_outcomes.png`
  - show how outcomes change across raw bar-2 volume buckets
- `creation_minute_avg_bar2_volume.png`
  - show whether creation minute and raw volume are mechanically linked
- `creation_minute_volume_heatmap.png`
  - show combined density or outcome rates for minute x volume bucket

These should complement, not replace, the existing four FVG charts.

## Volume Bucket Recommendation

Use simple quantile buckets on `bar2_volume`, for example quartiles.

Reason:
- fast to implement
- robust across long sample windows
- easy to read in the charts

If quantile boundaries collapse because of repeated values, allow bucket count to drop cleanly using `duplicates="drop"`.

## Interpretation Goal

This extension is meant to answer:
- whether stage effects are really minute effects
- whether later-minute FVG behavior is simply because of less time left, or also linked to the assigned candle's volume
- whether high-volume assigned candles produce more durable FVGs than low-volume assigned candles

The combined minute x volume view should be treated as exploratory, not the primary output.
Primary interpretation should still come from simpler grouped summaries first.

## Implementation Shape

Modify the existing study rather than creating a new script.

Recommended changes inside `features/macro_fvg_study.py`:
- enrich FVG event rows with minute and volume fields during detection
- add grouped summary builders for minute and volume buckets
- add four new plotting functions or plotting branches
- keep the current runner and output locations

## Testing

Add tests for:
- event rows include correct `assigned_minute_hhmm`
- event rows include correct `assigned_minute_index`
- event rows include correct `bar2_volume`
- grouped summaries by minute and volume bucket are built correctly
- end-to-end run writes the four new figure files

Smoke coverage should continue to verify:
- event parquet exists
- summary parquet exists
- all expected FVG figure files exist
