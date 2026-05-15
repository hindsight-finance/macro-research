# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-first Python research workspace for macro and intraday market analysis. Core entry scripts live at the root, including `session_tagger.py`, `macro_outcomes.py`, `tick_density.py`, and `volume_delta.py`. Feature research modules live under `features/`, with deeper subpackages such as `features/lrlr/` and `features/trend/`; current macro microstructure studies include `features/macro_extreme_timing.py`. Visualization scripts live in `viz/`. Source datasets are stored in `input-data/`; generated parquet files and charts belong in `outputs/`. Keep small test fixtures next to the code they validate, for example `features/lrlr/test/NQ.csv` or focused fixtures inside `test/`.

## Project Context & Data Flow
The project studies NQ/ES intraday behavior around the market-close macro window, especially 15:50-16:00 ET, with supporting research on session context, tick microstructure, volume delta, fair-value-gap/event behavior, liquidity, and trend/regime modeling. The high-level flow is:

1. `input-data/nq_1m.csv` and `input-data/es_1m.csv` are normalized by `session_tagger.py` into canonical UTC minute parquet files such as `outputs/nq_1m.parquet` and `outputs/es_1m.parquet`.
2. Canonical minute bars feed `macro_outcomes.py`, which computes daily 15:50-15:59 ET macro range, direction, skew, close position, first high/low minute, post-close range, and optional regime joins into `outputs/nq_macro_outcomes.parquet`.
3. `input-data/merged_nq_ticks.parquet` feeds tick studies. `tick_density.py` creates 1-minute 15:40-16:10 ET density and selected 5-second macro-minute buckets. `volume_delta.py` creates macro and Globex signed volume/tick delta outputs.
4. `features/` scripts produce derived research datasets such as macro extreme timing, 15:50 barrier behavior, macro FVG events, LRLR liquidity patterns, and trend/regime modeling tables.
5. `viz/` scripts convert derived parquet outputs into CSV summaries and figures under `outputs/figs/`.

Canonical minute bars use `datetime_utc`, `Open`, `High`, `Low`, `Close`, `Volume`, and optional `instrument`. Timezone-sensitive logic should keep UTC internally and derive ET with `utils.minute_bars.MARKET_TZ`. Window labels from `utils.minute_bars.derive_session_window` include `H3PM` for 15:00-15:49 ET, `MACRO` for 15:50-15:59 ET, and `POST` for 16:00-16:10 ET; sessions include `ASIA`, `LONDON`, `NYAM`, `LUNCH`, `PM`, and `OTHER`.

Note current naming drift before running full pipelines: `session_tagger.py` writes `outputs/nq_1m.parquet`, while `macro_outcomes.py` may default to `outputs/nq_minute_base.parquet`; tests or callers may override paths.

## Build, Test, and Development Commands
Run commands from the repository root and always use the project virtualenv: `.venv/bin/python ...`. Do not rely on system `python`.

- `.venv/bin/python session_tagger.py`: tag minute bars by session/window and write parquet outputs.
- `.venv/bin/python macro_outcomes.py`: compute macro-window outcome features from tagged parquet data.
- `.venv/bin/python -m features.macro_extreme_timing`: build tick-level high/low first-touch timing for 15:50, 15:54, 15:55, and 15:59 ET key candles.
- `.venv/bin/python tick_density.py`: regenerate 1-minute and 5-second macro tick-density parquet outputs.
- `.venv/bin/python volume_delta.py`: regenerate macro and Globex volume-delta parquet outputs.
- `.venv/bin/python viz/tick_density_viz.py`: regenerate tick-density CSV summaries and figures.
- `.venv/bin/python viz/macro_extreme_timing_viz.py`: regenerate macro extreme timing CSV summaries and figures.
- `.venv/bin/python viz/macro_analysis.py`: regenerate macro-analysis figures in `outputs/figs/`.
- `.venv/bin/python -m pytest test -q`: run the main pytest suite.
- `.venv/bin/python -m pytest test/test_tick_data.py test/test_tick_density.py test/test_volume_delta.py -q`: run tick-data, tick-density, and volume-delta tests.
- `.venv/bin/python -m pytest test/test_macro_extreme_timing.py test/test_macro_extreme_timing_viz.py -q`: run macro extreme timing tests.
- `.venv/bin/python -m pytest features/lrlr/test features/trend -q`: run deeper feature package tests.
- `.venv/bin/python -m features.lrlr.test.test_lrlr --no-viz`: run the LRLR detector check without generating charts.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, `snake_case` for modules/functions/variables, and `UPPER_CASE` for configuration constants such as file paths. Prefer small, composable functions over long notebook-style blocks. Use Polars for data processing by default. Keep Pandas out of new processing paths unless there is a strong visualization-only reason. Validate required columns early, as in `macro_outcomes.py`, `tick_density.py`, and `features/macro_extreme_timing.py`. Follow the existing timing-column naming pattern: `*_high_time`, `*_low_time`, `*_high_first`, `datetime_utc` for UTC timestamps, `macro_minute_index` for ET minute markers, and `bucket_index` or `macro_bucket_index` for 5-second buckets. Use output names like `nq_macro_<feature>.parquet`, `nq_macro_<feature>_summary.parquet`, `nq_macro_<feature>_5s.parquet`, and `nq_globex_<feature>_1m.parquet`.

## Tick Data & Memory Safety
Tick parquet files can be hundreds of millions of rows. Process them with `pl.scan_parquet` and lazy filters before collecting or sinking. Select only required columns, validate schema from metadata first with helpers such as `get_tick_schema()`, filter bounded ET/UTC windows early, and prefer `collect(engine="streaming")` or `sink_parquet` for full-output pipelines. Never eager-read the full `input-data/merged_nq_ticks.parquet`. Preserve UTC timestamps internally and derive ET market-time columns with `utils.minute_bars.MARKET_TZ` to handle DST correctly. Plotting may convert only small final frames to NumPy/Pandas at the matplotlib boundary.

Tick studies generally require `ts_event`, `intra_ts_rank`, `side`, `price_ticks`, and `size`; `side` uses `2` for buy, `1` for sell, and `0` for none. Preserve empty 5-second buckets in density/delta outputs when those buckets are part of the expected analysis grid.

## Feature Research Areas
`features/macro_extreme_timing.py` is a tick-level first-touch study for the 15:50, 15:54, 15:55, and 15:59 ET candles. It uses lazy Polars scans and metadata schema validation, keeps only dates with a complete key-minute set, and writes `outputs/nq_macro_extreme_timing.parquet` plus `outputs/nq_macro_extreme_timing_summary.parquet`. Main output columns include `date`, `datetime_utc`, `macro_minute_index`, candle OHLCV fields, `candle_high_time`, `candle_low_time`, `candle_high_ts_utc`, `candle_low_ts_utc`, `candle_high_first`, `candle_extreme_gap_seconds`, and macro direction/state fields. First touch means earliest UTC timestamp at the candle high/low; ties use `high_ts <= low_ts`.

`features/lrlr/` implements Low Resistance Liquidity pattern detection. Important types include `LRLRType`, `EqualStrength`, `SwingPoint`, `LRLRPattern`, and `LRLRDetector`. The detector covers swing highs/lows, trendline validation, tick-swing patterns, equal highs/lows, and all tick/equal pattern detection. The LRLR test harness can also generate visualizations; use `--no-viz` for quick checks.

`features/trend/` contains trend/regime feature research. Indicator families include ADX, ATR/session range ratios, DRA, IRR, lag autocorrelation, MSS tangent slope, swing point density, efficiency ratio, variance ratio, and the legacy `state_detector.py` ensemble. The modeling stack under `features/trend/modeling/` builds canonical session feature tables for `1pm-3pm`, `3pm-3:50pm`, and `3:50pm-4pm`, with status columns such as `feature_status`, `target_status`, `containment_status`, `chop_status`, and `regime_status`. Outputs for modeling experiments belong under `outputs/trend_modeling/`.

## Visualization & Reporting
Main visualization scripts should be run from the repository root. `viz/tick_density_viz.py` reads the tick-density parquet outputs and writes band plots, band-stat CSVs, normality CSVs, and total-size figures under `outputs/figs/tick_density/`. `viz/macro_extreme_timing_viz.py` reads `outputs/nq_macro_extreme_timing.parquet` and writes frequency, quantile, directional-stat CSVs plus heatmaps/histograms/ECDF/violin/scatter/sequence-rate plots under `outputs/figs/macro_extreme_timing/`. `viz/macro_1550_barrier_viz.py` reads `outputs/nq_macro_1550_barrier.parquet` and writes summary CSVs and rate/depth figures under `outputs/figs/macro_1550_barrier/`. `viz/macro_analysis.py` writes macro-analysis CSVs and figures under `outputs/figs/ma/`.

Most current visualization scripts use headless Matplotlib `Agg`; older/manual scripts such as `viz/macro_high.py`, `viz/viz_outcome.py`, and `viz/pm_macro_viz.py` may assume `viz/`-relative paths or call `plt.show()`. Eager Pandas reads are acceptable for small summarized parquet outputs at the visualization boundary, not for raw tick inputs.

## Testing Guidelines
Prefer `pytest`-compatible `test_*.py` files even when a test also acts as a runnable analysis script. Store fixture CSVs or generated fixture parquet files beside or inside the tests that use them. Cover both happy-path calculations and schema failures for new data-processing functions. For tick studies, include DST coverage, bounded-window filtering, and first-touch/tie behavior. If a change produces plots, add a non-visual assertion path so the logic can run in CI or headless shells.


## Experiment Log Protocol
Use Markdown-only experiment logs for completed/requested research findings. Create or update logs when the user explicitly asks for an experiment log, research log entry, findings log, or asks to keep an ongoing experiment log for a research thread.

Recommended locations:

- `docs/research_log.md` for a human-maintained index of experiment logs.
- `docs/experiments/NNNN-short-name.md` for individual experiment logs, where `NNNN` is the next sequential number.

When creating an experiment log, use this schema:

```markdown
# Experiment: <name>

## Status
Exploratory / Validated / Superseded / Deprecated

## Question
What was this experiment trying to answer?

## Dataset
- Asset(s):
- Input minute file(s):
- Input tick file(s):
- Date range:
- Timezone/session definition:

## Study Universe
- Dates/events included:
- Window(s):
- Completeness filters:
- Regime/context joins:
- Alignment notes:

## Parameters
- Anchors/windows:
- Bucket size:
- Thresholds/bins:
- Tie rules:
- Other filters:

## Results
Headline metrics, sample sizes, and comparison table.

## Findings
Plain-English conclusions and what changed versus the baseline.

## Caveats
Small samples, in-sample tuning, stale inputs, lookahead/alignment risks, missing-score behavior, or unresolved discrepancies.

## Output Files
- Generated parquet/CSV:
- Reports/charts:
- Source scripts/specs:

## Follow-ups
Specific next tests or documentation updates.
```

Experiment log rules:

- Prefer concise tables over raw dumps.
- Always include sample sizes with rates.
- Always identify whether the result is exploratory or validated.
- If an experiment supersedes older work, state what it supersedes and why.
- If two results disagree, first check anchor definitions, target windows, date ranges, session definitions, completeness filters, and regime/context joins before treating it as a model failure.
- Do not hand-edit generated CSV/parquet artifacts to make an experiment log; regenerate outputs from scripts or cite existing generated files.
- For tick studies, record whether the implementation uses bounded lazy scans, streaming collection, or PyArrow batch streaming to avoid full tick-file eager reads.

## Data, Commits, and Pull Requests
Do not overwrite source files in `input-data/`; write derived artifacts to `outputs/` and avoid committing large generated files unless they are intentional research deliverables. Virtualenvs, including `.venv-codex/`, and generated outputs should remain untracked. Use short imperative Conventional Commit subjects such as `feat: add CPI cohort summary`. Pull requests should describe the dataset touched, commands run, and any output files or figures reviewers should inspect.
