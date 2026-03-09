# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-first Python research workspace for macro and intraday market analysis. Core entry scripts live at the root, including `session_tagger.py` and `macro_outcomes.py`. Feature research modules live under `features/`, with deeper subpackages such as `features/lrlr/` and `features/trend/`. Visualization scripts live in `viz/`. Source datasets are stored in `input-data/`; generated parquet files and charts belong in `outputs/`. Keep small test fixtures next to the code they validate, for example `features/lrlr/test/NQ.csv`.

## Build, Test, and Development Commands
Run commands from the repository root.

- `python session_tagger.py`: tag minute bars by session/window and write parquet outputs.
- `python macro_outcomes.py`: compute macro-window outcome features from tagged parquet data.
- `python -m pytest features/lrlr/test features/trend -q`: run the pytest-discoverable test suites.
- `python -m features.lrlr.test.test_lrlr --no-viz`: run the LRLR detector check without generating charts.
- `python viz/macro_analysis.py`: regenerate macro-analysis figures in `outputs/figs/`.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, `snake_case` for modules/functions/variables, and `UPPER_CASE` for configuration constants such as file paths. Prefer small, composable functions over long notebook-style blocks. Keep Pandas or Polars transforms explicit, and validate required columns early, as in `macro_outcomes.py`. When adding a new feature family, place it under `features/<topic>/` with matching `test/` or `testing/` coverage nearby.

## Testing Guidelines
Prefer `pytest`-compatible `test_*.py` files even when a test also acts as a runnable analysis script. Store fixture CSVs beside the tests that use them. Cover both happy-path calculations and schema failures for new data-processing functions. If a change produces plots, add a non-visual assertion path such as `--no-viz` so the logic can run in CI or headless shells.

## Data, Commits, and Pull Requests
Do not overwrite source files in `input-data/`; write derived artifacts to `outputs/` and avoid committing large generated files unless they are intentional research deliverables. This workspace snapshot does not include `.git` metadata, so commit conventions cannot be inferred locally; use short imperative Conventional Commit subjects such as `feat: add CPI cohort summary`. Pull requests should describe the dataset touched, commands run, and any output files or figures reviewers should inspect.
