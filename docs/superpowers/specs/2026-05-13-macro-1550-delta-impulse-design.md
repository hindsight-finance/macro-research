# Macro 15:50 Delta Impulse Design

## Goal

Build a 15:50 ET volume-delta impulse study that mirrors the existing 15:59 delta reversal analysis. The study should test whether accumulated delta before 15:50 predicts the sign and distribution of the initial 15:50 volume-delta impulse, especially the first 10 seconds.

## Research Question

Does pre-15:50 accumulated volume delta predict same-signed or opposite-signed volume delta during the 15:50:00-15:50:09 ET impulse?

Primary predictor groups:

1. ETH-only delta before RTH.
2. RTH-only delta through 15:49 ET.
3. ETH+RTH delta through 15:49 ET.

Primary target:

- 15:50:00-15:50:09 ET volume delta.

The study is volume-flow only. It does not add price impulse, returns, or trade recommendations.

## Inputs

Use existing volume-delta outputs:

- `outputs/nq_globex_volume_delta_1m.parquet`
- `outputs/nq_macro_volume_delta_5s.parquet`

Required Globex/RTH 1-minute columns:

- `trade_date_et`
- `session_minute_index`
- `volume_delta`
- `classified_size`
- `total_size`

Required macro 5-second columns:

- `trade_date_et`
- `macro_bucket_index`
- `volume_delta`
- `classified_size`
- `total_size`

## Predictor Definitions

Use `session_minute_index` from the Globex/RTH 1-minute delta table:

1. `eth_only_pre350`
   - Window: `0..929`
   - Meaning: ETH/session flow before RTH.

2. `rth_only_pre350`
   - Window: `930..1309`
   - Meaning: RTH flow through 15:49 ET.

3. `eth_rth_pre350`
   - Window: `0..1309`
   - Meaning: full pre-macro accumulated flow through 15:49 ET.

For each predictor, include:

- `*_volume_delta`
- `*_classified_size`
- `*_total_size`
- `*_delta_imbalance`
- `*_sign`

`*_delta_imbalance` is `volume_delta / classified_size`, null when classified size is zero.

## Target Definitions

Use `macro_bucket_index` from `outputs/nq_macro_volume_delta_5s.parquet`. Bucket `0` starts at 15:50:00 ET.

Primary target:

- `k350_00_09`: buckets `0..1`, 15:50:00-15:50:09 ET.

Support targets:

- `k350_00_04`: bucket `0`, 15:50:00-15:50:04 ET.
- `k350_05_09`: bucket `1`, 15:50:05-15:50:09 ET.
- `k350_00_29`: buckets `0..5`, 15:50:00-15:50:29 ET.
- `k350_00_59`: buckets `0..11`, full 15:50 minute sanity window.
- `k350_bucket_0` through `k350_bucket_11`: individual 5-second buckets for diagnostics.

For each target, include:

- `*_volume_delta`
- `*_classified_size`
- `*_total_size`
- `*_delta_imbalance`
- `*_sign`

## Output Files

Create a separate focused study instead of extending the 15:59 module:

- `features/macro_1550_delta_impulse.py`
- `test/test_macro_1550_delta_impulse.py`
- `outputs/nq_macro_1550_delta_impulse.parquet`
- `outputs/nq_macro_1550_delta_impulse_summary.parquet`

The separate module keeps the 15:59 reversal study focused and avoids mixing target semantics inside `macro_delta_reversal.py`.

## Daily Study Table

One row per `date` (`trade_date_et`).

Include predictor columns, target columns, sign columns, and pairwise relationship flags for each primary predictor vs each available target:

- `*_has_signal_<target>` or equivalent target-aware summary logic.
- Same/opposite relationships should be computed in summaries even if wide relationship flags are limited to primary target.

A day has signal for a predictor-target pair only when both signs are non-zero.

## Summary Table

Use long-form rows keyed by:

- `summary_type`
- `predictor`
- `target_window`
- optional `predictor_decile`
- optional `tail`

For each predictor/target pair, emit sign-summary fields:

- `n_days`
- `n_signal_days`
- `opposite_count`
- `opposite_rate`
- `same_count`
- `same_rate`
- `zero_predictor_count`
- `zero_target_count`
- `mean_predictor_delta`
- `median_predictor_delta`
- `mean_target_delta`
- `median_target_delta`
- `mean_target_delta_when_predictor_positive`
- `mean_target_delta_when_predictor_negative`
- `median_target_delta_when_predictor_positive`
- `median_target_delta_when_predictor_negative`
- `target_p25_when_predictor_positive`
- `target_p75_when_predictor_positive`
- `target_p25_when_predictor_negative`
- `target_p75_when_predictor_negative`
- `pearson_corr_predictor_vs_target_delta`

Robust summaries should mirror the 15:59 study:

1. Raw predictor delta deciles.
2. Predictor delta-imbalance deciles.
3. Positive top 20% and 10% tails.
4. Negative bottom 20% and 10% tails.

The summary should make it easy to compare ETH-only, RTH-only, and ETH+RTH predictors for the primary first-10-second target.

## Runtime Behavior

The script should expose:

- `build_macro_1550_delta_impulse(globex_1m, macro_5s)`
- `summarize_macro_1550_delta_impulse(study)`
- `load_volume_delta_inputs(...)`
- `write_macro_1550_delta_impulse(...)`
- `main()` for `python -m features.macro_1550_delta_impulse`

Runtime should validate required columns before computation and write parquet outputs under `outputs/`.

## Testing Strategy

Use TDD with in-memory Polars fixtures and temporary parquet files.

Required tests:

1. Schema validation for Globex/RTH 1-minute input.
2. Schema validation for macro 5-second input.
3. Predictor aggregation boundaries:
   - ETH-only includes `0..929`.
   - RTH-only includes `930..1309`.
   - ETH+RTH includes `0..1309`.
   - Rows after `1309` are excluded.
4. Target aggregation boundaries:
   - `k350_00_09` includes buckets `0..1` only.
   - `k350_00_04` includes bucket `0` only.
   - `k350_05_09` includes bucket `1` only.
   - `k350_00_29` includes `0..5`.
   - `k350_00_59` includes `0..11`.
   - Buckets outside the target ranges are excluded.
5. Sign and same/opposite behavior for non-zero and zero target/predictor cases.
6. Target-aware sign summary fields.
7. Raw decile, imbalance decile, and tail summary rows on enough fixture days.
8. Writer persists daily and summary parquet outputs.

Run focused tests with:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py -q
```

Run broader related checks with:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py test/test_macro_delta_reversal.py -q
```

## Non-Goals

- No price impulse or return target.
- No tick-level raw scan; this study uses existing 5-second macro delta output.
- No changes to source data in `input-data/`.
- No visualization script in the first implementation unless requested after reviewing results.
