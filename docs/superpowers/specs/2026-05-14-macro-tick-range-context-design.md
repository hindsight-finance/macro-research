# Macro Tick Range Context Design

## Goal

Build a tick-level price range context study for the 15:50 ET and 15:59 ET macro candles. The study should quantify how much range forms in early time slices, especially the first 10 seconds, and express that range as raw price percentage, percentage of the candle range, and percentage of the full 15:50-15:59 macro range.

## Research Questions

1. How much raw price range is formed in the first 10 seconds of the 15:50 candle?
2. What percentage of the full 15:50 candle range is already formed by the first 10 seconds?
3. What percentage of the full 15:50-15:59 macro range is formed by the 15:50 first-10-second range?
4. How much of the full macro range is formed by the 15:59 candle?
5. Within 15:59, how does range accumulate by 5-second time slices?
6. After a given window forms a range, how much additional range is added outside that window on the high side, low side, and total?

The study is price-range only. It does not use volume delta or produce trade recommendations.

## Input

Use raw tick data:

- `input-data/merged_nq_ticks.parquet`

Read with `pl.scan_parquet`; never eager-read the full file. Filter bounded ET/UTC macro windows before collecting or sinking.

Required tick columns:

- `ts_event`
- `intra_ts_rank`
- `price_ticks`

Optional columns can remain unused.

Price conversion:

- Use the same tick-price convention as existing tick studies: `price = price_ticks / TICK_PRICE_DENOMINATOR`.
- Reuse `TICK_PRICE_DENOMINATOR` and `MARKET_TZ` patterns from `features/macro_extreme_timing.py` / `utils.minute_bars`.

## Time Windows

All timestamps are UTC internally and converted to ET for filtering/grouping.

Macro session:

- `macro`: 15:50:00-15:59:59 ET.

Candles:

- `k350`: 15:50:00-15:50:59 ET.
- `k359`: 15:59:00-15:59:59 ET.

Relative 5-second windows for each candle:

- `00_04`: seconds `0..4`
- `00_09`: seconds `0..9`
- `00_14`: seconds `0..14`
- Continue every 5 seconds through `00_59`: seconds `0..59`

Also include named windows:

- `first_5s` = seconds `0..4`
- `first_10s` = seconds `0..9`
- `first_30s` = seconds `0..29`
- `last_30s` = seconds `30..59`
- `full_candle` = seconds `0..59`

## Range Definitions

For any window:

- `window_high` = max tick price inside window.
- `window_low` = min tick price inside window.
- `window_range_points` = `window_high - window_low`.

For each candle:

- `candle_open` = first tick price in candle by `ts_event`, then `intra_ts_rank`.
- `candle_close` = last tick price in candle by `ts_event`, then `intra_ts_rank`.
- `candle_high` = max tick price in candle.
- `candle_low` = min tick price in candle.
- `candle_range_points` = `candle_high - candle_low`.

For the macro session:

- `macro_open` = first tick price from 15:50:00.
- `macro_close` = last tick price through 15:59:59.
- `macro_high` = max tick price from 15:50:00-15:59:59.
- `macro_low` = min tick price from 15:50:00-15:59:59.
- `macro_range_points` = `macro_high - macro_low`.

## Percent Metrics

For each candle/window row:

- `range_raw_pct_of_open` = `window_range_points / candle_open * 100`.
- `range_pct_of_candle` = `window_range_points / candle_range_points * 100`.
- `range_pct_of_macro` = `window_range_points / macro_range_points * 100`.

Null when denominator is zero or missing.

For `k359`, add candle contribution to macro:

- `k359_range_pct_of_macro` = `k359.candle_range_points / macro_range_points * 100`.

This value may exceed the incremental contribution to macro if the 15:59 range overlaps already-established macro range. Therefore also compute additive extensions.

## Additive Range Extension Metrics

For any candle/window range inside a larger range context:

Candle context:

- `candle_additive_high_extension_points` = `max(0, candle_high - window_high)`.
- `candle_additive_low_extension_points` = `max(0, window_low - candle_low)`.
- `candle_additive_total_extension_points` = high extension + low extension.
- `candle_additive_total_extension_pct_of_candle` = additive total / candle range * 100.

Macro context:

- `macro_additive_high_extension_points` = `max(0, macro_high - window_high)`.
- `macro_additive_low_extension_points` = `max(0, window_low - macro_low)`.
- `macro_additive_total_extension_points` = high extension + low extension.
- `macro_additive_total_extension_pct_of_macro` = additive total / macro range * 100.

These additive metrics preserve the user's intent that everything outside the window range is added to that initial price range, instead of double-counting movement inside the initial range.

For `k359` as a full candle, also compute its additive contribution relative to the macro range:

- `k359_macro_additive_high_extension_from_pre359_points` = high-side extension beyond the macro high already formed before 15:59.
- `k359_macro_additive_low_extension_from_pre359_points` = low-side extension beyond the macro low already formed before 15:59.
- `k359_macro_additive_total_extension_from_pre359_points` = sum of high/low extensions.
- `k359_macro_additive_total_extension_from_pre359_pct_of_macro` = additive total / macro range * 100.

## Output Table

Create a long-form table:

- `outputs/nq_macro_tick_range_context.parquet`

One row per:

- `date`
- `candle` (`k350`, `k359`)
- `window` (`00_04`, `00_09`, ..., `00_59`, named windows if not aliases)

Core columns:

- `date`
- `candle`
- `window`
- `window_start_second`
- `window_end_second`
- `window_tick_count`
- `window_open`
- `window_high`
- `window_low`
- `window_close`
- `window_range_points`
- `candle_open`
- `candle_high`
- `candle_low`
- `candle_close`
- `candle_range_points`
- `macro_open`
- `macro_high`
- `macro_low`
- `macro_close`
- `macro_range_points`
- `range_raw_pct_of_open`
- `range_pct_of_candle`
- `range_pct_of_macro`
- candle additive extension columns
- macro additive extension columns
- `k359_range_pct_of_macro`
- k359 additive-from-pre359 columns

## Summary Table

Create:

- `outputs/nq_macro_tick_range_context_summary.parquet`

Long-form summary rows keyed by:

- `summary_type`
- `candle`
- optional `window`
- optional `threshold`
- optional `decile_metric`
- optional `decile`

Summary types:

1. `window_baseline`
   - One row per candle/window.
   - Reports n days, median/mean range points, median/mean raw pct, median/mean pct of candle, median/mean pct of macro, median additive extension metrics.

2. `threshold_pct_of_candle`
   - For thresholds 25%, 50%, 75%, 90%.
   - Example: first 10s forms at least 50% of 15:50 candle range on X% of days.

3. `threshold_pct_of_macro`
   - Same thresholds against full macro range.

4. `threshold_k359_range_pct_of_macro`
   - For thresholds 25%, 50%, 75%, 90%.
   - Measures full 15:59 candle range as percentage of macro range.

5. `decile_range_raw_pct_of_open`
   - Deciles of raw price-normalized range.

6. `decile_range_pct_of_candle`
   - Deciles of window percentage of candle range.

7. `decile_range_pct_of_macro`
   - Deciles of window percentage of macro range.

For decile rows, rank separately by candle and window.

## Runtime API

Create a standalone module:

- `features/macro_tick_range_context.py`

Public functions:

- `build_macro_tick_range_context(ticks: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame`
- `summarize_macro_tick_range_context(study: pl.DataFrame) -> pl.DataFrame`
- `write_macro_tick_range_context(input_path, output_path, summary_output_path) -> tuple[Path, Path]`
- `main()` for `.venv/bin/python -m features.macro_tick_range_context`

Constants:

- `TICK_INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")`
- `OUTPUT_PATH = Path("outputs/nq_macro_tick_range_context.parquet")`
- `SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_tick_range_context_summary.parquet")`

## Tick Data Safety

- Use `pl.scan_parquet` for runtime input.
- Select only required columns.
- Convert `ts_event` to UTC datetime and derive ET date/hour/minute/second.
- Filter to 15:50:00-15:59:59 ET before collecting.
- Sort only the filtered macro-window ticks by date/time/rank when deriving opens/closes.
- Tests may use eager in-memory fixtures.

## Testing Strategy

Create:

- `test/test_macro_tick_range_context.py`

Required tests:

1. Schema validation rejects missing tick columns.
2. Tick fixture maps ET windows correctly for `k350`, `k359`, and macro range.
3. First/last tick open/close uses `ts_event`, then `intra_ts_rank` tie-breaker.
4. `first_10s` range is calculated from ticks in seconds `0..9` only.
5. Raw percentage equals `first_10s_range / candle_open * 100`.
6. Percentage of candle range equals `first_10s_range / candle_range * 100`.
7. Percentage of macro range equals `first_10s_range / macro_range * 100`.
8. Additive candle extension equals high-side plus low-side expansion outside the window.
9. Additive macro extension equals high-side plus low-side expansion outside the window.
10. `k359_range_pct_of_macro` and `k359` additive contribution beyond pre-15:59 macro range are correct.
11. Summary threshold rows report correct percentages for 25/50/75/90 levels.
12. Summary decile rows are generated per candle/window and skipped when fewer than 10 unique non-null values exist.
13. Writer persists daily context and summary parquet outputs.

Focused command:

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py -q
```

Related regression command:

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py test/test_macro_bucket_path.py test/test_macro_1550_delta_impulse.py test/test_macro_delta_reversal.py -q
```

## Reporting

Create after runtime:

- `docs/reports/2026-05-14-macro-tick-range-context-findings.md`

Report should cover:

1. First-10-second 15:50 raw price percentage.
2. First-10-second 15:50 percentage of 15:50 range.
3. First-10-second 15:50 percentage of full macro range.
4. Additive range extension after 15:50 first 10 seconds.
5. 15:59 candle range as a percentage of macro range.
6. 15:59 additive contribution beyond pre-15:59 macro range.
7. 15:59 range accumulation by 5-second windows.

## Non-Goals

- No volume-delta analysis in this first version.
- No price direction prediction model.
- No raw tick eager reads.
- No visualization script unless requested later.
- No writes to `input-data/`.
