# Macro Delta Reversal Seconds Extension Design

## Goal

Extend the macro delta reversal study so the primary reports focus on clean unresolved-imbalance definitions and so the 15:59 ET target can be analyzed by sub-minute buckets, especially the final seconds.

## Primary Predictor Definitions

The study should elevate these three cumulative delta definitions as the main predictors:

1. `eth_rth_pre59`
   - ETH + RTH cumulative delta through 15:49 ET.
   - Existing equivalent: `day_pre_macro`.
   - Window: Globex session minute index `0..1309`.

2. `eth_rth_macro_pre59`
   - ETH + RTH + 15:50–15:58 cumulative delta.
   - Existing equivalent: `day_plus_macro_pre59`.
   - Window: Globex session minute index `0..1309` plus macro minute index `50..58`.

3. `rth_macro_pre59`
   - RTH + 15:50–15:58 cumulative delta.
   - Existing equivalent: `rth_plus_macro_pre59`.
   - Window: Globex session minute index `930..1309` plus macro minute index `50..58`.

The existing columns can remain for compatibility, but summaries and documentation should present these clearer names as primary aliases.

## 15:59 Seconds-Level Targets

Use `outputs/nq_macro_volume_delta_5s.parquet` to decompose the 15:59 candle into 5-second target buckets. The macro 5-second bucket index maps from 15:50:00 ET as bucket `0`, so 15:59:00–15:59:59 ET maps to buckets `108..119`.

Create target windows:

- `k359_00_59`: buckets `108..119`, full 15:59 minute. This should match or closely reconcile with current 1-minute `k359`.
- `k359_00_29`: buckets `108..113`, first 30 seconds.
- `k359_30_59`: buckets `114..119`, last 30 seconds.
- `k359_45_59`: buckets `117..119`, last 15 seconds.
- `k359_50_59`: buckets `118..119`, last 10 seconds.
- `k359_bucket_108` through `k359_bucket_119`: individual 5-second buckets.

For each target window, include:

- `*_volume_delta`
- `*_classified_size`
- `*_total_size`
- `*_delta_imbalance`
- `*_sign`

## Outputs

Keep the existing daily and summary outputs:

- `outputs/nq_macro_delta_reversal.parquet`
- `outputs/nq_macro_delta_reversal_summary.parquet`

Add a focused seconds output if keeping all second-level columns in the daily table becomes unwieldy:

- `outputs/nq_macro_delta_reversal_359_5s.parquet`

Preferred first implementation: add the target-window columns to the daily table and add long-form summary rows keyed by `target_window`. Only add the separate parquet if the daily table becomes hard to inspect.

## Summary Extensions

The summary should support predictor/target pairs. For each primary predictor and each 15:59 target window, report:

- `predictor`
- `target_window`
- `summary_type`
- `n_days`
- `n_signal_days`
- `opposite_count`
- `opposite_rate`
- `same_count`
- `same_rate`
- `zero_predictor_count`
- `zero_target_count`
- `mean_target_delta_when_predictor_positive`
- `mean_target_delta_when_predictor_negative`
- `median_target_delta_when_predictor_positive`
- `median_target_delta_when_predictor_negative`
- `target_p25_when_predictor_positive`
- `target_p75_when_predictor_positive`
- `target_p25_when_predictor_negative`
- `target_p75_when_predictor_negative`
- `pearson_corr_predictor_vs_target_delta`

## Robust Distribution Summaries

For each primary predictor and target pair, add:

1. Raw delta deciles
   - Rank by `*_volume_delta`.
   - Include median target delta and target quartiles, not only means.

2. Delta-imbalance deciles
   - Rank by `*_delta_imbalance`.
   - This controls for differences in total classified volume.

3. Positive and negative tail summaries
   - Positive top 20%, positive top 10%.
   - Negative bottom 20%, negative bottom 10%.
   - Include `n_days`, `opposite_rate`, median target delta, and target quartiles.

These rows should make it easy to answer whether the effect only appears away from the mean.

## Macro Pre-59 Context Rows

Add conditional summaries that describe how 15:50–15:58 interacted with prior imbalance before 15:59:

- `macro_pre59_same_as_eth_rth_pre59`
- `macro_pre59_opposes_eth_rth_pre59`
- `macro_pre59_same_as_rth_pre_macro`
- `macro_pre59_opposes_rth_pre_macro`

For each condition, summarize whether each 15:59 target window opposes the combined unresolved imbalance. This is intended to distinguish:

- early macro already processed prior imbalance
- early macro compounded prior imbalance
- 15:59 performed the primary resolution

## Data Flow

1. Load existing 1-minute Globex and macro volume-delta parquet files.
2. Load existing 5-second macro volume-delta parquet file.
3. Build the existing daily study table.
4. Add primary predictor aliases.
5. Aggregate 15:59 5-second buckets into target windows and join by date.
6. Add sign columns and predictor/target relationship flags.
7. Build pairwise summary rows for primary predictors and target windows.
8. Add robust raw-delta, imbalance-decile, and tail summaries.
9. Write parquet outputs.

## Testing Strategy

Extend `test/test_macro_delta_reversal.py` with small 5-second fixture frames.

Tests should cover:

1. Required 5-second schema validation.
2. Correct 15:59 bucket boundaries: `108..119`, `108..113`, `114..119`, `117..119`, and `118..119`.
3. Correct aggregation of volume delta, classified size, total size, imbalance ratio, and sign for each target window.
4. Correct primary predictor alias columns.
5. Correct predictor/target pair summaries for at least two target windows.
6. Correct raw-delta decile and delta-imbalance decile rows.
7. Correct positive/negative tail rows.
8. Correct conditional macro pre-59 context rows.

Tests should use in-memory Polars DataFrames and temporary parquet files. They should not read raw tick data.

## Non-Goals

- Do not rebuild from raw tick data in this extension.
- Do not split NOII and EOII yet.
- Do not add charting yet.
- Do not add price-prediction targets.

## Notes

The current 5-second data can test the final 10 seconds using buckets `118..119`. If a later study needs true last-second or order-type-specific behavior, it should use raw ticks and possibly order metadata if available.
