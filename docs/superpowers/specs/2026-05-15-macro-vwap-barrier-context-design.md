# Macro VWAP Barrier Context Design

## Purpose

Extend the macro VWAP study into actionable barrier-context research. The goal is to compare the existing 15:50 first-10-second barrier behavior against VWAP context, then describe how price behaves around macro-open VWAP after the barrier forms and at the 15:55 EOII/imbalance-open decision point.

This remains exploratory. The feature should expose continuous distributions and bucketed diagnostics, not impose a hard trade rule.

## Scope

### In scope

- Compare straight 15:50 first-10-second barrier outcomes to VWAP-aware barrier context.
- Measure price-vs-VWAP behavior after the barrier extreme forms inside the 15:50 minute.
- Build 15:55 decision-context features using only information available before `15:55:00 ET`.
- Emit raw daily/event features plus long-form summaries.
- Create distribution visualization script and CSV tables.
- Use tick-derived data for intraminute metrics.
- Preserve UTC internally and derive ET with `utils.minute_bars.MARKET_TZ`.
- Use Polars; no pandas except tiny plotting boundary if needed.

### Out of scope

- Live trading rules.
- Model training.
- Execution/slippage modeling.
- Rebuilding existing VWAP feature outputs.
- Any source mutation under `input-data/`.

## Inputs

Required:

- `input-data/merged_nq_ticks.parquet`
- `outputs/nq_macro_1550_barrier.parquet`
- `outputs/nq_macro_vwap_intramacro.parquet`

Optional but useful for cross-checks:

- `outputs/nq_macro_vwap_premacro.parquet`

Required tick columns:

- `ts_event`
- `intra_ts_rank`
- `price_ticks`
- `size`

Required barrier columns:

- `date`
- `macro_trend_state`
- `barrier_extreme`
- `barrier_price`
- `barrier_time`
- `barrier_first10`
- `barrier_is_macro_extreme`
- `barrier_holds`
- `edge_case`

Required intramacro VWAP columns:

- `date`
- `macro_1550_at_1550_10s_vwap_side`
- `macro_1550_at_1550_10s_vwap_dist_points`
- `macro_1550_at_1550_10s_vwap_dist_bps`
- `macro_1550_at_1555_vwap_side`
- `macro_1550_at_1555_vwap_dist_points`
- `macro_1550_at_1555_vwap_dist_bps`
- `target_1550_1554_points/sign/state`
- `target_1555_1559_points/sign/state`
- `target_1550_1559_points/sign/state`

## Time semantics

All intraminute metrics use ET windows but keep UTC timestamps in raw calculations.

Boundary rules:

- Barrier timestamp = `15:50:00 ET + barrier_time seconds` from existing barrier study.
- 10-second checkpoint = last tick `<15:50:10 ET`.
- 15:50 close checkpoint = last tick `<15:51:00 ET`.
- 15:55 decision checkpoint = last tick `<15:55:00 ET`.
- Post-10s target = first tick `>=15:50:10 ET` to last tick `<16:00:00 ET`.
- Post-15:50-minute target = first tick `>=15:51:00 ET` to last tick `<16:00:00 ET`.
- 15:55 target = first tick `>=15:55:00 ET` to last tick `<16:00:00 ET`.

## Core definitions

### Direction-aware barrier side

Existing barrier direction controls “constructive” VWAP side:

- Bullish macro/barrier context:
  - barrier extreme = low
  - constructive side = `above` or `touch` macro VWAP
  - wrong side = below macro VWAP
- Bearish macro/barrier context:
  - barrier extreme = high
  - constructive side = `below` or `touch` macro VWAP
  - wrong side = above macro VWAP

Do not discard neutral/edge cases. Preserve them and summarize separately where sample size permits.

### Macro-open VWAP

For post-barrier 15:50 metrics, use VWAP anchored at `15:50:00 ET` and updated tick-by-tick inside the 15:50 minute.

Per tick:

- `price = price_ticks / 4.0`
- `vwap = cumulative sum(price * size) / cumulative sum(size)` from `15:50:00 ET`
- if cumulative size is zero, VWAP is null.

### Wrong-side distance

For each tick after barrier timestamp and before `15:51:00 ET`:

Bullish barrier context:

- `wrong_side_dist_points = max(vwap - price, 0)`

Bearish barrier context:

- `wrong_side_dist_points = max(price - vwap, 0)`

Also compute signed normal distance:

- bullish: `price - vwap`
- bearish: `vwap - price`

Positive signed distance = constructive side. Negative = wrong side.

## Output dataset

Create module:

- `features/macro_vwap_barrier_context.py`

Raw output:

- `outputs/nq_macro_vwap_barrier_context.parquet`

Summary output:

- `outputs/nq_macro_vwap_barrier_context_summary.parquet`

One row per eligible date, joined from barrier + VWAP + tick-derived intraminute metrics.

### Raw columns

Identity/context:

- `date`
- `macro_trend_state`
- `barrier_extreme`
- `barrier_price`
- `barrier_time`
- `barrier_first10`
- `barrier_is_macro_extreme`
- `barrier_holds`
- `edge_case`

10-second VWAP diff:

- `vwap_10s_side`
- `vwap_10s_dist_points`
- `vwap_10s_dist_bps`
- `vwap_10s_constructive`
- `barrier_first10_and_vwap_constructive`

15:50 post-barrier VWAP metrics:

- `barrier_ts_utc`
- `post_barrier_tick_count_1550`
- `vwap_side_at_barrier`
- `vwap_dist_at_barrier_points`
- `vwap_side_at_1550_close`
- `vwap_dist_at_1550_close_points`
- `closed_wrong_side_1550`
- `closed_wrong_side_more_than_1tick`
- `closed_wrong_side_more_than_2pts`
- `closed_wrong_side_more_than_5pts`
- `worst_wrong_side_dist_points`
- `worst_wrong_side_dist_bps`
- `seconds_wrong_side_vwap`
- `wrong_side_share_1550`

15:55 decision context:

- `vwap_1555_side`
- `vwap_1555_dist_points`
- `vwap_1555_dist_bps`
- `vwap_1555_constructive`
- `vwap_context_10s_to_1555`
  - values: `constructive_to_constructive`, `wrong_to_constructive`, `constructive_to_wrong`, `wrong_to_wrong`, `touch_mixed`, `unknown`
- `barrier_holds_and_1555_constructive`
- `barrier_first10_and_1555_constructive`

Targets:

- existing `target_1550_1554_*`
- existing `target_1555_1559_*`
- existing `target_1550_1559_*`
- new `target_1550_10s_1554_points/sign/state`
- new `target_1550_10s_1559_points/sign/state`
- new `target_1551_1559_points/sign/state`

## Summary design

Summary rows use long format:

- `scope`
- `bucket`
- `target_name`
- `sample_size`
- `bullish_count`
- `bearish_count`
- `neutral_count`
- `bullish_pct`
- `bearish_pct`
- `neutral_pct`
- `avg_target_points`
- `median_target_points`
- `p10_target_points`
- `p25_target_points`
- `p75_target_points`
- `p90_target_points`

Scopes:

1. `barrier_only`
   - buckets: `first10_true`, `first10_false`, `holds_true`, `holds_false`
2. `vwap_10s_only`
   - buckets: `constructive`, `wrong`, `touch`, `unknown`
3. `barrier_vwap_10s`
   - buckets combine barrier first10/holds with VWAP constructive state.
4. `wrong_side_close_bucket`
   - buckets:
     - `no_wrong_side_close`
     - `wrong_le_1tick`
     - `wrong_1tick_to_2pts`
     - `wrong_2_to_5pts`
     - `wrong_gt_5pts`
5. `wrong_side_share_decile`
6. `worst_wrong_side_dist_decile`
7. `vwap_1555_decision`
   - buckets from `vwap_context_10s_to_1555`
8. `barrier_1555_context`
   - buckets combine `barrier_holds` and `vwap_1555_constructive`.

Targets summarized:

- `target_1550_10s_1554`
- `target_1550_10s_1559`
- `target_1551_1559`
- `target_1555_1559`

## Visualization design

Create:

- `viz/macro_vwap_barrier_context_viz.py`

Read:

- `outputs/nq_macro_vwap_barrier_context.parquet`
- `outputs/nq_macro_vwap_barrier_context_summary.parquet`

Write CSVs under:

- `outputs/figs/macro_vwap_barrier_context/`

CSV outputs:

- `summary_by_scope.csv`
- `wrong_side_quantiles.csv`
- `target_quantiles_by_bucket.csv`
- `vwap_1555_decision_summary.csv`

Figures:

- `vwap_10s_dist_hist.png`
- `vwap_1555_dist_hist.png`
- `wrong_side_dist_ecdf.png`
- `wrong_side_share_ecdf.png`
- `target_by_wrong_side_bucket_violin.png`
- `target_by_1555_context_violin.png`
- `barrier_vwap_heatmap.png`
- `vwap_1555_scatter_target.png`

Matplotlib `Agg`; no GUI.

## Implementation notes

Tick parquet is large. Process bounded windows only:

- `15:50:00 <= ET < 16:00:00` for post-barrier metrics and targets.
- Select only required columns.
- Prefer PyArrow batch streaming or Polars lazy bounded filters. Do not eager-read full tick parquet.
- The previous macro VWAP implementation needed one-pass batch streaming for runtime stability; reuse that approach if full Polars lazy plan becomes memory-heavy.

## Tests

Create:

- `test/test_macro_vwap_barrier_context.py`
- `test/test_macro_vwap_barrier_context_viz.py`

Test cases:

1. Direction-aware constructive/wrong side classification for bullish and bearish contexts.
2. 10s VWAP vs straight barrier summary differences.
3. Post-barrier wrong-side close thresholds.
4. Worst wrong-side distance and wrong-side share inside 15:50 minute.
5. `15:55` context transition buckets.
6. Post-10s and post-15:51 target construction.
7. Missing tick/barrier/VWAP rows produce nulls, not crashes.
8. DST ET window handling.
9. Writer creates raw + summary parquet.
10. Viz writes expected CSVs/figures from tiny fixture.

Commands:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py -q
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context_viz.py -q
.venv/bin/python -m features.macro_vwap_barrier_context
.venv/bin/python viz/macro_vwap_barrier_context_viz.py
```

## Acceptance criteria

- Raw output has one row per date with barrier/VWAP/tick context.
- Summary output exposes barrier-only, VWAP-only, barrier+VWAP, wrong-side, and 15:55 decision scopes.
- Distribution CSVs and figures are generated.
- No full raw tick eager read.
- Tests cover calculations, DST, null behavior, writer outputs, and viz outputs.
