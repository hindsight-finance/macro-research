# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-first Python research workspace for macro and intraday market analysis. Core entry scripts live at the root, including `session_tagger.py`, `macro_outcomes.py`, `tick_density.py`, and `volume_delta.py`. Feature research modules live under `features/`, with deeper subpackages such as `features/lrlr/` and `features/trend/`; current macro microstructure studies include `features/macro_extreme_timing.py`. Visualization scripts live in `viz/`. Source datasets are stored in `input-data/`; generated parquet files and charts belong in `outputs/`. Keep small test fixtures next to the code they validate, for example `features/lrlr/test/NQ.csv` or focused fixtures inside `test/`.

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

## Testing Guidelines
Prefer `pytest`-compatible `test_*.py` files even when a test also acts as a runnable analysis script. Store fixture CSVs or generated fixture parquet files beside or inside the tests that use them. Cover both happy-path calculations and schema failures for new data-processing functions. For tick studies, include DST coverage, bounded-window filtering, and first-touch/tie behavior. If a change produces plots, add a non-visual assertion path so the logic can run in CI or headless shells.

## Data, Commits, and Pull Requests
Do not overwrite source files in `input-data/`; write derived artifacts to `outputs/` and avoid committing large generated files unless they are intentional research deliverables. Virtualenvs, including `.venv-codex/`, and generated outputs should remain untracked. Use short imperative Conventional Commit subjects such as `feat: add CPI cohort summary`. Pull requests should describe the dataset touched, commands run, and any output files or figures reviewers should inspect.
