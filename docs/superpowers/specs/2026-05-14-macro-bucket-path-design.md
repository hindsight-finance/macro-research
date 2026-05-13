# Macro Bucket Path Design

## Goal

Build a 5-second bucket path study for the 15:50 ET and 15:59 ET macro candles. The study should classify early volume-delta conviction and test whether that early flow continues, fades, or churns through the rest of the candle.

## Research Questions

1. Does high-conviction volume delta in the first 10 seconds of the candle predict same-signed full-candle volume delta?
2. Does first-10-second flow continue into the first 30 seconds, last 30 seconds, or full 60-second candle?
3. Do 15:50 and 15:59 behave differently in continuation/fade tendencies?
4. Are strong positive and strong negative early impulses symmetric, or does one side continue/fade more reliably?

The study is volume-flow only. It does not add price returns or trade recommendations.

## Input

Use the existing macro 5-second volume-delta output:

- `outputs/nq_macro_volume_delta_5s.parquet`

Required columns:

- `trade_date_et`
- `macro_bucket_index`
- `volume_delta`
- `classified_size`
- `total_size`

Optional pass-through columns may be used for diagnostics if present:

- `delta_imbalance`
- `buy_size`
- `sell_size`
- `tick_delta`
- `classified_share`
- `is_empty`

## Candle Definitions

Use `macro_bucket_index`, where bucket `0` starts at 15:50:00 ET.

1. `k350`
   - Buckets `0..11`.
   - Represents 15:50:00-15:50:59 ET.

2. `k359`
   - Buckets `108..119`.
   - Represents 15:59:00-15:59:59 ET.

In the output table, normalize both candles to relative bucket offsets `0..11` so the same path logic applies to both.

## Daily Candle Path Output

Create one row per `date` and `candle`.

Output file:

- `outputs/nq_macro_bucket_path.parquet`

Core columns:

- `date`
- `candle` (`k350` or `k359`)
- `bucket_count`
- `complete_candle` boolean, true when all 12 relative buckets are present

Individual bucket columns:

- `b0_volume_delta` through `b11_volume_delta`
- `b0_classified_size` through `b11_classified_size`
- `b0_total_size` through `b11_total_size`
- `b0_delta_imbalance` through `b11_delta_imbalance`
- `b0_sign` through `b11_sign`

Cumulative path columns:

- `cum_00_04_volume_delta` = bucket 0
- `cum_00_09_volume_delta` = buckets 0..1
- `cum_00_14_volume_delta` = buckets 0..2
- Continue every 5 seconds through `cum_00_59_volume_delta` = buckets 0..11
- Matching `*_classified_size`, `*_total_size`, `*_delta_imbalance`, and `*_sign` columns for each cumulative window

Named window columns:

- `early_5s_volume_delta` = bucket 0
- `early_10s_volume_delta` = buckets 0..1
- `early_30s_volume_delta` = buckets 0..5
- `late_30s_volume_delta` = buckets 6..11
- `full_volume_delta` = buckets 0..11
- Matching classified size, total size, delta imbalance, and sign columns

Path diagnostics:

- `sum_abs_bucket_delta`
- `path_efficiency` = `full_volume_delta / sum_abs_bucket_delta`, null when denominator is zero
- `early_10s_abs_flow_share` = `abs(early_10s_volume_delta) / sum_abs_bucket_delta`, null when denominator is zero
- `max_abs_bucket_delta`
- `max_abs_bucket_index`
- `peak_abs_cum_delta`
- `peak_abs_cum_bucket_index`
- `max_favorable_cum_delta` relative to early-10s sign
- `max_adverse_cum_delta` relative to early-10s sign
- `cum_sign_flip_count` across cumulative 5-second path

Continuation/fade flags:

- `early_10s_continues_to_30s`
- `early_10s_fades_to_30s`
- `early_10s_continues_to_late30`
- `early_10s_fades_to_late30`
- `early_10s_continues_to_full`
- `early_10s_fades_to_full`

A continuation/fade flag is true only when both compared signs are non-zero.

## Conviction Categories

Compute conviction per candle independently, so 15:50 and 15:59 distributions are not mixed.

Primary categorization uses `early_10s_volume_delta`:

- `early_10s_raw_decile`: signed decile rank from lowest to highest early delta, `1..10`; null if fewer than 10 unique values.
- `early_10s_imbalance_decile`: decile rank of `early_10s_delta_imbalance`, `1..10`; null if fewer than 10 unique values.
- `early_10s_abs_decile`: decile rank of `abs(early_10s_volume_delta)`, `1..10`; null if fewer than 10 unique values.

Human-readable signed categories:

- `strong_negative`: raw decile `1` or `2`
- `weak_negative`: raw decile `3`, `4`, or signed negative outside strong buckets
- `neutral`: zero sign or raw decile `5`/`6` near center
- `weak_positive`: raw decile `7`, `8`, or signed positive outside strong buckets
- `strong_positive`: raw decile `9` or `10`

Also include an absolute-conviction category:

- `high_abs_conviction`: abs decile `9` or `10`
- `mid_abs_conviction`: abs decile `4..8`
- `low_abs_conviction`: abs decile `1..3`

## Summary Output

Output file:

- `outputs/nq_macro_bucket_path_summary.parquet`

Long-form summary rows keyed by:

- `summary_type`
- `candle`
- optional `early_10s_category`
- optional `early_10s_abs_category`
- optional `early_10s_raw_decile`
- optional `early_10s_imbalance_decile`
- optional `early_10s_abs_decile`

Primary summary types:

1. `candle_baseline`
   - One row per candle.
   - Reports full-candle and early-window means/medians, positive/negative/zero counts, and sign rates.

2. `early_10s_category`
   - One row per candle and signed early category.
   - Reports n days, continuation/fade rates, full/late delta means and medians, path efficiency stats, and sign flip stats.

3. `early_10s_raw_decile`
   - One row per candle and early raw decile.
   - Same continuation/fade and full/late outcome fields.

4. `early_10s_imbalance_decile`
   - One row per candle and early imbalance decile.
   - Same outcome fields.

5. `early_10s_abs_decile`
   - One row per candle and early absolute-conviction decile.
   - Same outcome fields.

6. `early_10s_abs_category`
   - One row per candle and low/mid/high absolute-conviction category.
   - Same outcome fields.

Recommended fields per summary row:

- `n_days`
- `n_signal_days`
- `continue_to_30s_count`
- `continue_to_30s_rate`
- `fade_to_30s_count`
- `fade_to_30s_rate`
- `continue_to_late30_count`
- `continue_to_late30_rate`
- `fade_to_late30_count`
- `fade_to_late30_rate`
- `continue_to_full_count`
- `continue_to_full_rate`
- `fade_to_full_count`
- `fade_to_full_rate`
- `mean_early_10s_delta`
- `median_early_10s_delta`
- `mean_late_30s_delta`
- `median_late_30s_delta`
- `mean_full_delta`
- `median_full_delta`
- `full_p25`
- `full_p75`
- `mean_path_efficiency`
- `median_path_efficiency`
- `mean_early_10s_abs_flow_share`
- `median_early_10s_abs_flow_share`
- `mean_cum_sign_flip_count`
- `median_cum_sign_flip_count`

## Runtime API

Create a standalone module:

- `features/macro_bucket_path.py`

Public functions:

- `build_macro_bucket_path(macro_5s: pl.DataFrame) -> pl.DataFrame`
- `summarize_macro_bucket_path(study: pl.DataFrame) -> pl.DataFrame`
- `load_macro_5s_input(path=MACRO_5S_INPUT_PATH) -> pl.DataFrame`
- `write_macro_bucket_path(input_path, output_path, summary_output_path) -> tuple[Path, Path]`
- `main()` for `.venv/bin/python -m features.macro_bucket_path`

Constants:

- `MACRO_5S_INPUT_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")`
- `OUTPUT_PATH = Path("outputs/nq_macro_bucket_path.parquet")`
- `SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_bucket_path_summary.parquet")`

## Testing Strategy

Create:

- `test/test_macro_bucket_path.py`

Use in-memory Polars fixtures.

Required tests:

1. Schema validation rejects missing required 5-second columns.
2. Candle mapping:
   - `k350` uses buckets `0..11`.
   - `k359` uses buckets `108..119`.
   - Output normalizes both to relative buckets `0..11`.
3. Individual bucket columns are correct for both candles.
4. Cumulative columns are correct through 60 seconds.
5. Named windows (`early_10s`, `early_30s`, `late_30s`, `full`) are correct.
6. Path diagnostics are correct:
   - `sum_abs_bucket_delta`
   - `path_efficiency`
   - `early_10s_abs_flow_share`
   - peak bucket/cumulative fields
   - cumulative sign flip count
7. Continuation/fade flags handle continuation, fade, zero signs, and missing buckets.
8. Decile/category generation is per candle and skips deciles when fewer than 10 unique values exist.
9. Summary rows include baseline, category, raw decile, imbalance decile, abs decile, and abs category rows.
10. Writer persists daily and summary parquet outputs.

Focused test command:

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py -q
```

Related regression command:

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py test/test_macro_1550_delta_impulse.py test/test_macro_delta_reversal.py -q
```

## Reporting

Create a findings report after runtime generation:

- `docs/reports/2026-05-14-macro-bucket-path-findings.md`

The report should compare 15:50 and 15:59:

- baseline continuation/fade rates
- first-10-second strong positive vs strong negative outcomes
- high absolute-conviction vs low absolute-conviction outcomes
- whether early impulse contributes most of final candle flow or fades/churns

## Non-Goals

- No price-return target in this first version.
- No raw tick scan; use existing 5-second macro delta parquet only.
- No visualization script unless requested after reviewing findings.
- No changes to `input-data/`.
