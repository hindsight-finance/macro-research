# One-Minute Total Size Bands Design

## Goal
Extend `viz/tick_density_viz.py` so the 1-minute macro dataset (`outputs/nq_macro_tick_density.parquet`) also produces a second band chart for `total_size`, using cross-day mean / p25 / p75 by `macro_minute_index`.

## Scope
- In scope:
  - Keep work inside `viz/tick_density_viz.py`
  - Reuse the existing band-chart style
  - Use only the 1-minute macro dataset for this addition
  - Produce a separate PNG for `total_size`
  - Optionally emit matching band-stat CSV for `total_size` so the chart is auditable
- Out of scope:
  - 5-second datasets
  - Histogram revival
  - New browser/UI work
  - Reworking the existing tick-count normality pipeline beyond what is needed for metric reuse

## Approaches Considered

### 1. Duplicate the current tick-count plotting path for `total_size`  **(chosen)**
Fastest patch. Keep the existing tick-count path intact, add a second parallel band-stat + chart path for `total_size`, avoid broader refactor for now.

### 2. Parameterize the metric pipeline for `tick_count` and `total_size`
Cleaner long-term, but unnecessary for this immediate validation pass.

### 3. Split total-size work into a second script
Clear separation, but unnecessary file sprawl for a very small extension.

## Recommended Design
Use approach 1.

`viz/tick_density_viz.py` should keep the current tick-count implementation, then add a second explicit `total_size` banded flow for the 1-minute dataset:
- `tick_count` -> existing minute-index band PNG remains
- `total_size` -> new minute-index band PNG added

Each metric flow should:
1. load the same parquet file
2. group by `macro_minute_index`
3. compute cross-day `mean`, `p25`, `p75`
4. force full minute coverage `0..9`
5. render a separate PNG with explicit x-ticks for each minute index
6. write a CSV with the underlying band stats

## Data Flow
- Input: `outputs/nq_macro_tick_density.parquet`
- Grouping key: `macro_minute_index`
- Metric columns:
  - `tick_count`
  - `total_size`
- Outputs:
  - `outputs/figs/tick_density/nq_macro_tick_density_bands.png`
  - `outputs/figs/tick_density/nq_macro_tick_density_band_stats.csv`
  - `outputs/figs/tick_density/nq_macro_tick_density_total_size_bands.png`
  - `outputs/figs/tick_density/nq_macro_tick_density_total_size_band_stats.csv`

## Error Handling
- Fail fast if `macro_minute_index` or requested metric column missing
- Preserve explicit minute index range even if some minutes absent in a test fixture
- Do not recreate histograms

## Testing
Add pytest coverage for:
- metric-agnostic band summary helper using `total_size`
- full minute index coverage `0..9`
- artifact writing for the new total-size PNG and CSV without creating histogram files

## Success Criteria
- Running `python3 viz/tick_density_viz.py` writes a second 1-minute macro PNG for `total_size`
- X-axis shows every minute index `0..9`
- Existing no-hist behavior stays intact
- Tests pass
