# Macro FVG Alignment Bucket Evaluation Design

**Date:** 2026-03-09

## Goal

Evaluate whether candle-body alignment inside the 3-bar FVG pattern adds explanatory power for macro-window FVG outcomes, beyond the minute, gap-size, and volume effects already observed.

This pass is explicitly research-first:
- compute the alignment labels at event level
- summarize their relationship to hold/retrace/invalidation outcomes
- visualize the results
- reassess whether the feature is worth keeping based on the statistics

## Existing Repo Fit

The repo already has an event-study pipeline at:
- `features/macro_fvg_study.py`
- `outputs/nq_macro_fvg_events.parquet`
- `outputs/nq_macro_fvg_summary.parquet`
- `outputs/figs/fvg/`

This extension should build on that existing study rather than create a new sidecar script.
The event parquet should remain the source of truth.

## Approved Scope

Use only the 3 candles that define the FVG pattern:
- `bar1`
- `bar2`
- `bar3`

Approved first-pass behavior:
- evaluate alignment as a research label
- keep dojis as `neutral`
- report only four public buckets
- include new figures in the first pass

Do not add:
- bar-order sequence buckets
- bar-specific weights
- extra pattern families
- strategy logic

## Definitions

### Candle Direction

Define body direction from `Open` and `Close`:
- `bullish` if `Close > Open`
- `bearish` if `Close < Open`
- `neutral` if `Close == Open`

### Alignment

Use `fvg_side` as the reference direction:
- for a `bullish` FVG, a bullish candle is `aligned`
- for a `bearish` FVG, a bearish candle is `aligned`
- a directional candle against the FVG side is `opposite`
- a doji is `neutral`

### Public Buckets

The public evaluation buckets should be exactly:
- `3_aligned`
- `2_aligned_1_opposite`
- `1_aligned_2_opposite`
- `contains_neutral`

Rationale:
- preserves the main directional hypothesis
- keeps the cell count manageable
- avoids a separate `3_opposite` research bucket

Implementation note:
- keep exact counts (`aligned_count`, `opposite_count`, `neutral_count`) on each event row for auditability
- if a non-neutral zero-aligned edge case appears in real data, surface it as an implementation sanity check rather than promoting it to a named research bucket

## Data Additions

Extend each event row with:
- `bar1_direction`
- `bar2_direction`
- `bar3_direction`
- `aligned_count`
- `opposite_count`
- `neutral_count`
- `alignment_bucket`

Recommended control-group helpers on the event row:
- `minute_block`
- `gap_size_bucket_225`

Definitions:
- `minute_block`
  - `15:50-15:52`
  - `15:53-15:57`
  - `15:58_unconfirmable`
- `gap_size_bucket_225`
  - `<2.25`
  - `>=2.25`

## Summary Extensions

Append grouped alignment views to the existing summary parquet.

Recommended new `summary_scope` values:
- `alignment_bucket`
- `alignment_bucket_minute_block`
- `alignment_bucket_gap_bucket`

Recommended grouped metrics:
- `n_total`
- `n_confirmable`
- `hold_rate`
- `invalidation_rate`
- `retrace_rate`
- `untouched_rate`

Recommended grouping fields:
- `alignment_bucket`
- `minute_block`
- `gap_size_bucket_225`

The current stage, minute, volume, and gap outputs should remain intact.

## Visualization

Add new Matplotlib figures under `outputs/figs/fvg/`:
- `alignment_bucket_outcomes.png`
- `alignment_bucket_by_minute_block.png`
- `alignment_bucket_by_gap_bucket.png`
- `alignment_bucket_counts.png`

Recommended interpretation:
- `alignment_bucket_outcomes.png`
  - grouped bars for hold / retrace / invalidation by alignment bucket
- `alignment_bucket_by_minute_block.png`
  - grouped bars or heatmap showing how alignment behaves across early vs later confirmable macro minutes
- `alignment_bucket_by_gap_bucket.png`
  - grouped bars or heatmap showing whether alignment survives the `<2.25` vs `>=2.25` gap split
- `alignment_bucket_counts.png`
  - sample sizes by bucket so the statistical read is visible

These figures should complement the current FVG charts rather than replace them.

## Interpretation Goal

This extension should answer:
- whether alignment carries signal at all
- whether alignment is just a proxy for the already-observed minute effect
- whether alignment is just a proxy for the already-observed gap-size effect
- whether the signal is concentrated only in one narrow bucket or is stable enough to keep

The threshold for keeping this feature should be:
- enough sample size per bucket
- stable direction of effect after conditioning on minute block and gap-size bucket
- no obvious collapse once controls are added

## Implementation Shape

Modify the existing study in place:
- enrich event rows during FVG detection with bar directions and alignment fields
- derive grouped summary tables from those enriched events
- add four new figures driven by the event table / summary table
- keep the current runner and output locations

Recommended function additions inside `features/macro_fvg_study.py`:
- `classify_candle_direction()`
- `assign_alignment_bucket()`
- `build_alignment_bucket_summary()`
- `build_alignment_bucket_minute_block_summary()`
- `build_alignment_bucket_gap_bucket_summary()`
- one plotting helper per new figure, or equivalent plotting branches

## Testing

Add tests for:
- `bar1_direction`, `bar2_direction`, and `bar3_direction`
- correct `aligned_count`, `opposite_count`, and `neutral_count`
- `contains_neutral` for doji-containing patterns
- correct directional bucket assignment for:
  - `3_aligned`
  - `2_aligned_1_opposite`
  - `1_aligned_2_opposite`
- grouped summary builders for:
  - alignment bucket
  - alignment bucket by minute block
  - alignment bucket by gap-size bucket
- end-to-end run writes the new alignment figure files

Focused real-data verification should also confirm:
- no unexpected zero-aligned non-neutral rows are silently hidden
- the summary parquet contains the new alignment scopes
- the new figure files are present under `outputs/figs/fvg/`
