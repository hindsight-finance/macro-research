# Tick Volume Delta Design

## Goal

Build volume-delta datasets from sanitized NQ tick data using the existing aggressor-side field. The primary signal is signed trade size, not inferred quote-side flow. No BBO reconstruction is required for the first version.

## Inputs

Input parquet: `input-data/merged_nq_ticks.parquet`

Required columns:
- `ts_event`: UTC exchange event timestamp
- `intra_ts_rank`: order for trades sharing the same timestamp
- `side`: aggressor side, where `0 = none/unspecified`, `1 = sell aggressor`, `2 = buy aggressor`
- `price_ticks`: trade price in NQ tick units
- `size`: trade quantity

## Scope

Create three outputs:

1. Full Globex-session 1-minute volume delta.
2. Macro-window 1-minute volume delta for 15:50:00-16:00:00 ET.
3. Macro-window 5-second volume delta for 15:50:00-16:00:00 ET.

Globex session definition:
- Session starts at 18:00:00 ET on the prior calendar day.
- Session ends at 17:00:00 ET on the trade date.
- `trade_date_et` is the ET date of the 17:00 close.
- Session end is exclusive.

Macro window definition:
- 15:50:00 ET inclusive through 16:00:00 ET exclusive.
- 1-minute macro buckets cover 15:50 through 15:59 ET.
- 5-second macro buckets cover 120 fixed buckets.

All stored bucket timestamps remain UTC. ET logic is derived inside transforms.

## Delta Definition

Per bucket:

- `buy_size`: sum `size` where `side == 2`
- `sell_size`: sum `size` where `side == 1`
- `none_size`: sum `size` where `side == 0`
- `classified_size`: `buy_size + sell_size`
- `total_size`: `buy_size + sell_size + none_size`
- `volume_delta`: `buy_size - sell_size`
- `delta_imbalance`: `volume_delta / classified_size`, null when `classified_size == 0`
- `buy_ticks`: count where `side == 2`
- `sell_ticks`: count where `side == 1`
- `none_ticks`: count where `side == 0`
- `tick_delta`: `buy_ticks - sell_ticks`
- `classified_share`: `classified_size / total_size`, null when `total_size == 0`

`side == 0` trades are excluded from delta and retained as diagnostics. No tick-rule fallback is used in version 1.

## Output Schemas

### `outputs/nq_globex_volume_delta_1m.parquet`

Columns:
- `datetime_utc`
- `trade_date_et`
- `session_minute_index`
- metric columns listed above

`session_minute_index` starts at `0` for 18:00 ET and runs through the final minute before 17:00 ET when data exists.

### `outputs/nq_macro_volume_delta_1m.parquet`

Columns:
- `datetime_utc`
- `trade_date_et`
- `macro_minute_index`
- metric columns listed above

`macro_minute_index` is actual ET minute number: `50` through `59`.

### `outputs/nq_macro_volume_delta_5s.parquet`

Columns:
- `datetime_utc`
- `trade_date_et`
- `macro_bucket_index`
- metric columns listed above
- `is_empty`

`macro_bucket_index` is `0` through `119`. Empty macro 5-second buckets are emitted with zero counts/sizes and null ratios.

## Architecture

Add a root script, `volume_delta.py`, mirroring the style of `tick_density.py`.

Main functions:
- validate required tick schema
- scan required tick columns lazily
- compute reusable delta aggregations
- build full-session 1-minute lazy dataset
- build macro 1-minute lazy dataset
- build macro 5-second lazy dataset with complete bucket grid
- write all outputs

Keep `utils/tick_data.py` unchanged unless a shared helper is clearly reusable.

## Error Handling

- Missing required tick columns raises `ValueError` with missing column names.
- Invalid macro window constants are not runtime-configurable in v1.
- Ratio columns return null when denominators are zero.
- Empty 5-second macro buckets are explicit rows with zero metrics and `is_empty = true`.

## Testing

Add pytest coverage in `test/test_volume_delta.py`:

- schema validation failure
- delta aggregation from mixed buy/sell/none rows
- full Globex session assigns correct `trade_date_et` and `session_minute_index`
- macro 1-minute filters 15:50-16:00 ET only
- macro 5-second emits all 120 buckets per trade date, including empty buckets
- ratio columns are null on zero denominator
- write functions create parquet outputs

Use tiny synthetic parquet fixtures; do not read the large source parquet in tests.

## Out of Scope

- BBO reconstruction
- tick-rule inference for `side == 0`
- plots or visual reports
- committing generated parquet outputs
- modifying source data under `input-data/`
