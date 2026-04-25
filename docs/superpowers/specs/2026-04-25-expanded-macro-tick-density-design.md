# Expanded Macro Tick Density Design

## Goal
Expand `outputs/nq_macro_tick_density.parquet` from the current 15:50–15:59 ET window to a broader 15:40–16:10 ET window, while keeping the same schema and storing timestamps in UTC.

## Scope
- In scope:
  - change the 1-minute tick-density build window in `tick_density.py`
  - keep output schema unchanged: `datetime_utc`, `date_utc`, `macro_minute_index`, `tick_count`, `total_size`, `buy_ticks`, `sell_ticks`, `none_ticks`
  - keep UTC timestamps in parquet
  - update tests to reflect broader ET minute coverage
- Out of scope:
  - changing 5-second parquet logic
  - renaming output files
  - changing chart script behavior in this step

## Approaches Considered

### A. Add a new output file for the broader window
Safest for backward compatibility, but user explicitly wants to expand the existing parquet.

### B. Add config options and support both old + new windows
More flexible, but more code + more decisions now than needed.

### C. Replace the current 1-minute macro window with the broader 15:40–16:10 ET range **(chosen)**
Simplest path. Keeps schema identical, updates only the 1-minute dataset semantics, preserves UTC storage.

## Recommended Design
Use approach C.

`build_macro_tick_density()` should:
- filter ET timestamps where time is between 15:40 and 16:10 inclusive
- truncate event time to UTC minute as today
- compute `macro_minute_index` as the actual ET minute number
  - 15:40 -> 40
  - 15:50 -> 50
  - 15:59 -> 59
  - 16:00 -> 0
  - 16:10 -> 10
- preserve identical aggregation columns and output order

This means `macro_minute_index` now means the actual ET minute number within the hour across the expanded window, while schema stays unchanged as requested.

## Data Flow
- Input: sanitized tick parquet with UTC timestamps
- ET filter: 15:40:00 through 16:10:59 inclusive
- Stored datetime: UTC minute in `datetime_utc`
- Derived index: actual ET minute number
- Output path unchanged: `outputs/nq_macro_tick_density.parquet`

## Error Handling
- Keep existing schema validation unchanged
- Keep DST handling via ET timezone conversion unchanged
- Ensure 16:00–16:10 ET rows are included even though they cross the ET hour boundary

## Testing
Update / add tests for:
- inclusion of 15:40, 15:50, 15:59, 16:00, 16:10 ET rows
- exclusion of 15:39 and 16:11 ET rows
- correct `macro_minute_index` ET minute values across the hour boundary
- DST-safe UTC hour handling for both winter and summer dates

## Success Criteria
- `python3 tick_density.py` rewrites `outputs/nq_macro_tick_density.parquet` with 31 ET minutes per day when data exists
- schema unchanged
- timestamps stored in UTC
- tests pass
