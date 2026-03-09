# Macro FVG Study Design

**Date:** 2026-03-09

## Goal

Study how 1-minute fair value gaps created during the close macro window behave before the `15:59` close, with each FVG treated as an independent observation.

The initial study should answer:
- How often a macro-created FVG holds versus gets invalidated before `15:59`.
- How often price retraces into a macro-created FVG versus never touches it.
- How stage-1 FVGs behave during stage 2.
- Stage-level stats for stage 1, stage 2, and stage-1-to-stage-2 transitions.

## Existing Repo Fit

The existing pipeline already provides:
- `session_tagger.py` to label `window == "MACRO"` for `15:50-15:59`.
- `macro_outcomes.py` to compute daily macro-level features.

This feature should extend the research stack with a per-FVG event study rather than changing the current tagging or daily macro outcomes flow.

## Definitions

Use the standard 3-candle definition with wick-based creation rules:
- Bullish FVG: `bar3.low > bar1.high`
- Bearish FVG: `bar3.high < bar1.low`

Store two timestamps per FVG:
- `assigned_at`: timestamp of bar 2
- `confirmed_at`: first minute after bar 3 closes

Example:
- Bearish FVG with `bar1 = 15:49`, `bar2 = 15:50`, `bar3 = 15:51`
- `assigned_at = 15:50`
- `confirmed_at = 15:52`

No new FVG detection should start at `15:59`.

## Observation Window

Only use the close macro window and stop the study at the `15:59` close.

Rules:
- FVG detection should be limited to patterns whose `assigned_at` falls inside the macro window.
- Outcome scanning starts only after `confirmed_at`.
- No cross-window carry in the initial version.
- If `confirmed_at > 15:59`, exclude the FVG from hold/retrace/invalidation denominators and count it separately as not confirmable inside the study window.

## Stage Definitions

Stage labels should use `assigned_at`:
- `stage_1`: `15:50-15:54`
- `stage_2`: `15:55-15:58`

Deliverables should include:
- Stage-1 stats
- Stage-2 stats
- Stage-1-to-stage-2 transition stats

The stage-1-to-stage-2 transition view should evaluate only FVGs assigned in stage 1 and only what happens during `15:55-15:59`.

## Event Lifecycle

Each FVG is an independent observation, even if multiple same-side or overlapping FVGs exist on the same day.

Outcome rules:
- `retrace`: any later macro bar wick enters the gap range
- `invalidate`: any later macro bar closes through the far side of the gap
- `held_to_1559_close`: not invalidated by the `15:59` close
- `untouched_to_1559_close`: never retraced by the `15:59` close

Additional rules:
- Only bars after `confirmed_at` can retrace or invalidate an FVG.
- Retrace and invalidation should both be tracked because an FVG may retrace first and invalidate later.
- If retrace and invalidation both happen on the same bar, mark both true and assign the same first-event timestamp to both fields.

## Data Model

Create an event-level parquet with one row per FVG instance.

Recommended columns:
- `date`
- `symbol`
- `fvg_side`
- `assigned_at`
- `confirmed_at`
- `assigned_stage`
- `gap_top`
- `gap_bottom`
- `gap_size`
- `is_confirmable_by_1559`
- `first_retrace_at`
- `first_invalidation_at`
- `retraced_by_1559`
- `invalidated_by_1559`
- `held_to_1559_close`
- `untouched_to_1559_close`
- `retraced_in_stage_2`
- `invalidated_in_stage_2`
- `held_through_stage_2`
- `untouched_through_stage_2`
- `last_observed_at`

This event table should be the source of truth for all summary tables and charts.

## Outputs

Primary outputs:
- `outputs/nq_macro_fvg_events.parquet`
- `outputs/nq_macro_fvg_summary.parquet`
- `outputs/figs/fvg/`

The summary parquet should contain pre-aggregated stage-level and transition-level metrics built from the event table.

## Visualization

Use Matplotlib and write figures to `outputs/figs/fvg/`.

Recommended figures:
- `hold_vs_invalidate_by_side.png`
- `stage1_to_stage2_outcomes.png`
- `creation_minute_outcome_heatmap.png`
- `gap_size_vs_outcome.png`

These charts should be generated from the saved event parquet rather than from duplicated detection logic.

## Implementation Shape

Recommended new script:
- `features/macro_fvg_study.py`

Recommended function layout:
- `load_tagged_bars()`
- `extract_macro_bars()`
- `detect_macro_fvgs()`
- `scan_fvg_outcomes_until_1559_close()`
- `build_stage_summary_tables()`
- `plot_fvg_summary_figures()`

Data flow:
1. Read tagged 1-minute parquet.
2. Filter or extract macro bars per day.
3. Detect FVG instances with `assigned_at` and `confirmed_at`.
4. Scan each confirmable FVG forward until the `15:59` close.
5. Save the event parquet.
6. Build summary tables for stage 1, stage 2, and stage-1-to-stage-2 transitions.
7. Save summary parquet and Matplotlib figures.

## Testing

Add small synthetic tests for:
- Bullish FVG creation
- Bearish FVG creation
- Retrace-only outcome
- Invalidate-only outcome
- Retrace-then-invalidate outcome
- Untouched-through-end outcome
- Same-bar retrace plus invalidation
- Stage assignment correctness
- Excluding unconfirmable late FVGs

Also include a smoke path that verifies the full script can create the parquet outputs and figure files under `outputs/` and `outputs/figs/fvg/`.
