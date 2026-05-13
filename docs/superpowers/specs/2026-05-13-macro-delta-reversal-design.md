# Macro Delta Reversal Study Design

## Goal

Build an exploratory study that tests whether cumulative volume delta before 15:59 ET predicts an opposite-signed 15:59 ET volume-delta candle. The primary hypothesis is that institutional closeout or resolving flow may cause the 15:59 candle to reverse earlier accumulated imbalance.

## Scope

This study focuses on NQ tick-derived volume delta and treats the 15:59 ET candle as the primary target. Post-close price behavior is out of scope for the first version. Price macro outcomes may be joined later as context, but they are not required to validate the initial volume-delta reversal hypothesis.

## Existing Inputs

The study builds on existing parquet outputs from `volume_delta.py`:

- `outputs/nq_globex_volume_delta_1m.parquet`
- `outputs/nq_macro_volume_delta_1m.parquet`
- `outputs/nq_macro_volume_delta_5s.parquet`

The existing Globex 1-minute table uses `trade_date_et` assigned from the 18:00 ET session start through 16:59 ET. The macro 1-minute table covers 15:50–15:59 ET and includes `macro_minute_index` values 50 through 59.

## Output Files

Create a new feature module:

- `features/macro_delta_reversal.py`

Create two derived output files:

- `outputs/nq_macro_delta_reversal.parquet`
- `outputs/nq_macro_delta_reversal_summary.parquet`

The first file is a daily analysis table. The second file contains aggregate statistics for each tested predictor window.

## Daily Study Table

Each row represents one ET trade date with a valid 15:59 volume-delta candle. The target is the 15:59 ET candle volume delta.

Core predictor windows:

1. `eth_pre_rth`
   - 18:00 prior calendar day through 09:29 ET for the current trade date.
   - Uses `session_minute_index` 0 through 929 from Globex 1-minute volume delta.

2. `rth_pre_macro`
   - 09:30 through 15:49 ET.
   - Uses `session_minute_index` 930 through 1309 from Globex 1-minute volume delta.

3. `day_pre_macro`
   - 18:00 prior calendar day through 15:49 ET.
   - Uses `session_minute_index` 0 through 1309.

4. `macro_pre59`
   - 15:50 through 15:58 ET.
   - Uses `macro_minute_index` 50 through 58 from macro 1-minute volume delta.
   - This is intentionally distinct because early macro flow may already resolve prior imbalance.

5. `rth_plus_macro_pre59`
   - `rth_pre_macro + macro_pre59`.
   - Tests the unresolved RTH-plus-early-macro imbalance immediately before the 15:59 candle.

6. `day_plus_macro_pre59`
   - `day_pre_macro + macro_pre59`.
   - Tests the unresolved full-session-plus-early-macro imbalance immediately before the 15:59 candle.

Target window:

- `k359`
  - 15:59 through 15:59:59 ET.
  - Uses `macro_minute_index == 59` from macro 1-minute volume delta.

For each predictor and the target, include:

- `*_volume_delta`
- `*_classified_size`
- `*_total_size`
- `*_delta_imbalance`, computed as cumulative `volume_delta / classified_size` when classified size is non-zero
- `*_sign`, where positive is `1`, negative is `-1`, and zero is `0`

For predictor-to-target relationships, include:

- `*_opposes_k359`: predictor and 15:59 have opposite non-zero signs
- `*_same_as_k359`: predictor and 15:59 have same non-zero signs
- `*_has_signal`: both signs are non-zero

For imbalance-resolution relationships, include:

- `macro_pre59_opposes_rth_pre_macro`
- `macro_pre59_opposes_day_pre_macro`
- `k359_opposes_rth_plus_macro_pre59`
- `k359_opposes_day_plus_macro_pre59`

## Summary Table

The summary table contains one row per predictor window:

- `predictor`
- `n_days`
- `n_signal_days`
- `opposite_count`
- `opposite_rate`
- `same_count`
- `same_rate`
- `zero_predictor_count`
- `zero_k359_count`
- `mean_predictor_delta`
- `median_predictor_delta`
- `mean_k359_delta_when_predictor_positive`
- `mean_k359_delta_when_predictor_negative`
- `median_k359_delta_when_predictor_positive`
- `median_k359_delta_when_predictor_negative`
- `pearson_corr_predictor_vs_k359_delta`

Add decile/bin analysis in a compact long-form output inside the same summary parquet when practical:

- `summary_type = "sign"` for the sign statistics above.
- `summary_type = "decile"` for binned predictor magnitude statistics.
- Decile rows include `predictor_decile`, `n_days`, `mean_predictor_delta`, `mean_k359_delta`, and `opposite_rate`.

If mixing sign and decile rows makes the schema awkward, create an additional file later. The first implementation should prefer a simple, readable summary over an over-general schema.

## Data Flow

1. Load the existing Globex and macro 1-minute volume-delta parquet files with Polars.
2. Validate required columns before computing outputs.
3. Aggregate each predictor window by `trade_date_et`.
4. Extract the 15:59 target candle by `trade_date_et` and `macro_minute_index == 59`.
5. Join predictors and target into one daily table.
6. Compute sign and relationship flags.
7. Build aggregate summary statistics from the daily table.
8. Write parquet outputs under `outputs/`.

The module should expose pure functions for testing and CLI-style `write_*` / `main()` entry points matching existing project style.

## Required Functions

`features/macro_delta_reversal.py` should provide:

- `build_macro_delta_reversal(globex_1m: pl.DataFrame, macro_1m: pl.DataFrame) -> pl.DataFrame`
- `summarize_macro_delta_reversal(study: pl.DataFrame) -> pl.DataFrame`
- `load_volume_delta_inputs(globex_path: str | Path, macro_path: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]`
- `write_macro_delta_reversal(...) -> tuple[Path, Path]`
- `main() -> None`

The builder accepts in-memory DataFrames so tests can use compact fixtures without touching large tick data.

## Testing Strategy

Create `test/test_macro_delta_reversal.py`.

Tests should cover:

1. Required schema validation for Globex and macro 1-minute inputs.
2. Correct ETH, RTH, day-pre-macro, macro-pre59, and 15:59 window boundaries.
3. Correct cumulative `volume_delta`, `classified_size`, `total_size`, and cumulative `delta_imbalance` calculations.
4. Correct sign encoding for positive, negative, and zero deltas.
5. Correct opposite/same relationship flags.
6. Correct handling of macro pre-59 as a separate resolving window.
7. Summary statistics for a small multi-day fixture with known opposite rates.
8. Persistence behavior for the writer function.

Tests should not read raw tick data. They should construct small Polars DataFrames that mimic the already-generated volume-delta parquet schema.

## Non-Goals

- Do not eager-read `input-data/merged_nq_ticks.parquet`.
- Do not add price-prediction modeling in the first version.
- Do not make post-close behavior a primary target.
- Do not add charts until the base study table and summary are validated.

## Open Follow-Up Ideas

After the first version is validated, useful extensions include:

- 5-second decomposition of the 15:59 candle, especially last 10 seconds.
- Conditioning on macro event type, day type, or prior PM trend.
- Joining price outcomes to see whether delta reversal coincides with price continuation or price reversal.
- Separate ES/NQ comparison if ES tick data is available.
