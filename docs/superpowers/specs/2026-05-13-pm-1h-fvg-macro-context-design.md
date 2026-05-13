# PM 1H FVG Macro Context Design

## Goal

Add a descriptive research script that tests whether no-leak 1H context before 15:00 ET has a visible relationship with the 15:50-15:59 ET closing macro direction. This is not a model-training workflow. It produces per-day context rows and cohort summaries.

## Research Question

Given 1H bars completed before 15:00 ET, does a 12:00-15:00 no-leak FVG setup and 13:00-15:00 imbalance context line up with closing macro direction?

Primary target:

- `macro_dir_sign = sign(macro_close - macro_open)` from `outputs/nq_macro_outcomes.parquet`.

Primary predictors:

- No-leak 1H FVG using 12:00-13:00 and 14:00-15:00 candles.
- 13:00-15:00 imbalance and displacement metrics.

## Inputs

- `outputs/nq_1m.parquet`
  - canonical minute bars with `datetime_utc`, `Open`, `High`, `Low`, `Close`, `Volume`.
- `outputs/nq_macro_outcomes.parquet`
  - macro target columns such as `macro_open`, `macro_close`, `macro_dir_points`, `macro_range_points`.

The script must derive ET time columns from UTC with `utils.minute_bars.MARKET_TZ` helpers. It must not rely on persisted ET/session/window fields.

## Output Files

- `outputs/nq_pm_1h_macro_context.parquet`
  - one row per date with predictors + macro outcome.
- `outputs/nq_pm_1h_macro_summary.csv`
  - compact cohort summary tables.

## New Script

Create:

- `features/pm_1h_macro_context.py`

CLI defaults:

```bash
.venv/bin/python -m features.pm_1h_macro_context
```

Optional args:

- `--minute-input outputs/nq_1m.parquet`
- `--macro-input outputs/nq_macro_outcomes.parquet`
- `--context-output outputs/nq_pm_1h_macro_context.parquet`
- `--summary-output outputs/nq_pm_1h_macro_summary.csv`
- `--include-flat-macro` default false

## Data Flow

1. Load canonical minute bars.
2. Normalize schema via existing minute-bar helpers.
3. Derive ET date/time/hour fields.
4. Aggregate 1H candles for each date:
   - `h12`: 12:00-13:00 ET
   - `h13`: 13:00-14:00 ET
   - `h14`: 14:00-15:00 ET
5. Keep only dates where all three 1H candles are complete enough for the study.
6. Compute no-leak FVG fields.
7. Compute 13:00-15:00 imbalance fields.
8. Join macro outcomes by `date`.
9. Write context parquet.
10. Build summary cohort CSV.

## FVG Definition

No-leak 1H FVG uses candles fully known by 15:00 ET:

- `h12`: 12:00-13:00
- `h13`: 13:00-14:00
- `h14`: 14:00-15:00

Fields:

- `fvg_direction`
  - `bullish` if `h12_high < h14_low`
  - `bearish` if `h12_low > h14_high`
  - `none` otherwise
- `fvg_size_points`
  - bullish: `h14_low - h12_high`
  - bearish: `h12_low - h14_high`
  - none: `0.0`
- `fvg_size_pct_of_pm_range`
  - `fvg_size_points / pm_13_15_range`
- `has_fvg`
  - boolean

This avoids using the 15:00-16:00 candle because that would overlap the macro target.

## 13:00-15:00 Imbalance Metrics

Use combined range/open/close from `h13` and `h14`:

- `pm_13_15_open = h13_open`
- `pm_13_15_close = h14_close`
- `pm_13_15_high = max(h13_high, h14_high)`
- `pm_13_15_low = min(h13_low, h14_low)`
- `pm_13_15_range = high - low`
- `pm_13_15_dir_points = close - open`
- `pm_13_15_dir_sign = sign(dir_points)`
- `pm_13_15_close_pos = (close - low) / range`
- `pm_13_15_body_to_range = abs(close - open) / range`
- `pm_13_15_upper_wick_share = (high - max(open, close)) / range`
- `pm_13_15_lower_wick_share = (min(open, close) - low) / range`

Imbalance direction:

- `bullish` if `pm_13_15_dir_points > 0` and `pm_13_15_close_pos >= 0.60`
- `bearish` if `pm_13_15_dir_points < 0` and `pm_13_15_close_pos <= 0.40`
- `neutral` otherwise

## Macro Target Fields

Join these macro outcome columns when present:

- `macro_open`
- `macro_close`
- `macro_high`
- `macro_low`
- `macro_dir_points`
- `macro_range_points`
- `macro_dir_pct`
- `macro_range_pct`
- `close_in_range`
- `macro_high_time`
- `macro_low_time`

Add:

- `macro_dir_sign = sign(macro_dir_points)`
- `macro_direction`
  - `bullish`, `bearish`, `flat`

By default, summary tables exclude flat macro rows. The context parquet keeps all rows.

## Summary Tables

Write a CSV with a `cohort` and `bucket` structure so multiple summaries can live in one file.

Cohorts:

1. `fvg_direction`
2. `imbalance_direction`
3. `fvg_x_imbalance`
4. `fvg_size_bucket`

Metrics per bucket:

- `n`
- `macro_bull_n`
- `macro_bear_n`
- `macro_bull_rate`
- `macro_bear_rate`
- `avg_macro_dir_points`
- `median_macro_dir_points`
- `avg_macro_range_points`
- `median_macro_range_points`

FVG size buckets:

- `none`
- `small`
- `medium`
- `large`

Use quantile buckets among non-zero FVG rows. If too few FVG rows exist, collapse to `has_fvg` vs `none` without failing.

## Error Handling

- Validate required input columns early.
- Raise clear `ValueError` for missing columns.
- If no complete 12:00-15:00 dates exist, raise clear `ValueError`.
- If macro join produces zero rows, raise clear `ValueError`.
- Do not overwrite source files in `input-data/`.

## Testing

Add focused pytest coverage:

- `test/test_pm_1h_macro_context.py`

Test cases:

1. Builds 1H candles from UTC minute bars with DST-safe ET conversion.
2. Detects bullish FVG: `h12_high < h14_low`.
3. Detects bearish FVG: `h12_low > h14_high`.
4. Computes bullish/bearish/neutral imbalance buckets.
5. Joins macro outcomes and computes `macro_dir_sign`.
6. Summary output contains expected cohort rows and rates.

Use small synthetic Polars fixtures. No raw tick data.

## Non-Goals

- No ML training.
- No walk-forward harness.
- No tick-level processing.
- No charts in first pass.
- No 15:00-16:00 candle in predictor construction.
