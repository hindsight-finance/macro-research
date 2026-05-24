# Coding Conventions

**Analysis Date:** 2026-05-24

These conventions are documented authoritatively in `AGENTS.md` and consistently
reflected in the code. When in doubt, prefer the patterns observed in
`utils/minute_bars.py`, `macro_outcomes.py`, `tick_density.py`, and
`features/macro_extreme_timing.py`.

## Naming Patterns

**Files / modules:**
- `snake_case.py` for all modules. Entry scripts live at repo root
  (`session_tagger.py`, `macro_outcomes.py`, `tick_density.py`, `volume_delta.py`).
- Feature research modules under `features/` (e.g. `features/macro_extreme_timing.py`),
  with deeper subpackages `features/lrlr/`, `features/trend/`,
  `features/trend/modeling/`.
- Visualization scripts under `viz/` named `<feature>_viz.py`
  (e.g. `viz/tick_density_viz.py`, `viz/macro_extreme_timing_viz.py`).

**Functions / variables:**
- `snake_case` for all functions and variables.
- Private/internal helpers are prefixed with a single underscore:
  `_read_any`, `_to_utc_expr`, `_validate_tick_schema`,
  `_scan_required_tick_columns` (see `utils/minute_bars.py`,
  `features/macro_extreme_timing.py`).
- Build/compute/write verb prefixes are the dominant function-naming style:
  `build_macro_tick_density`, `compute_macro_outcomes`, `write_macro_extreme_timing`,
  `summarize_macro_extreme_timing`, `normalize_minute_bars`, `derive_session_window`,
  `join_regime_context`.

**Constants:**
- `UPPER_CASE` for configuration constants, file paths, and column-list schemas:
  `MARKET_TZ = "America/New_York"`, `UTC = "UTC"`,
  `BASE_COLUMNS`, `INPUT_PATH`, `OUTPUT_PATH`, `DEFAULT_KEY_MINUTES`,
  `MACRO_EXTREME_TIMING_COLUMNS`, `MACRO_TICK_DENSITY_COLUMNS`,
  `TICK_PRICE_DENOMINATOR`.
- Output column ordering is pinned in a module-level `*_COLUMNS` list so the
  written schema is explicit and testable (see
  `features/macro_extreme_timing.py:26`).

**Domain column naming (enforced by `AGENTS.md`):**
- `datetime_utc` for UTC timestamps (canonical, kept internally).
- `datetime_et`, `date_et`, `time_et` for derived ET market-time columns.
- `macro_minute_index` for ET minute markers.
- `bucket_index` or `macro_bucket_index` for 5-second buckets.
- Timing columns: `*_high_time`, `*_low_time`, `*_high_first`,
  `candle_high_ts_utc`, `candle_extreme_gap_seconds`.
- Status columns in the modeling stack: `feature_status`, `target_status`,
  `containment_status`, `chop_status`, `regime_status`.

**Output file naming:**
- `nq_macro_<feature>.parquet`, `nq_macro_<feature>_summary.parquet`,
  `nq_macro_<feature>_5s.parquet`, `nq_globex_<feature>_1m.parquet`.
- Canonical minute bars: `outputs/nq_1m.parquet`, `outputs/es_1m.parquet`.

## Code Style

**Formatting:**
- Python, 4-space indentation.
- No autoformatter config is present (no `.ruff.toml`, `pyproject.toml`,
  `.flake8`, `.pre-commit-config.yaml` detected). Style is maintained by
  convention, not tooling. Match surrounding code.
- Lines are long and unwrapped in many files; do not aggressively rewrap to a
  narrow column limit.

**Linting:**
- No linter configured. Follow `AGENTS.md` "Coding Style & Naming Conventions".

**Type hints:**
- Source modules open with `from __future__ import annotations` and use type
  hints on public functions: `def load_minute_bars(path: str | Path) -> pl.DataFrame:`,
  `def _scan_required_tick_columns(path: str | Path) -> pl.LazyFrame:`.
- `Iterable`, `pathlib.Path`, and `str | Path` unions are the common parameter
  types for path inputs.

## Import Organization

Observed order (no enforced tool, but consistent):
1. Standard library (`sys`, `from pathlib import Path`, `from datetime import date`,
   `from typing import Iterable`).
2. Third-party (`import polars as pl`, `import numpy as np`,
   `import pandas as pd`, `import pytest`).
3. First-party / local (`from utils.minute_bars import ...`,
   `from features.macro_extreme_timing import ...`,
   `from macro_outcomes import compute_macro_outcomes`).

- `from __future__ import annotations` is the first import in source modules.
- Multi-symbol imports use parenthesized multi-line form (see
  `test/test_minute_bars.py:4`, `test/test_tick_density.py:6`).
- Imports run from the repository root; entry scripts are run as
  `.venv/bin/python -m features.macro_extreme_timing` or
  `.venv/bin/python tick_density.py`. There are no path-alias shims.

## Data Processing (the most important convention)

**Use Polars by default. Keep Pandas out of new processing paths.**
- Data pipelines use `polars` (`import polars as pl`) for all transforms.
- Pandas (`import pandas as pd`) is permitted **only at the visualization
  boundary** for small, already-summarized frames (see `viz/tick_density_viz.py`
  and its test `test/test_tick_density_viz.py`, which use pandas/numpy).
- NumPy is used for array-level numeric feature math in the trend modeling stack
  (`features/trend/modeling/target.py` functions take/return `np.ndarray`).

**Polars idioms in use:**
- Expression API with `.with_columns(...)`, `.select(...)`, `.filter(...)`,
  `.group_by(...)`, `pl.when(...).then(...).otherwise(...)`.
- Timestamp parsing via `pl.col(col).str.to_datetime(time_zone="UTC")` and
  conversion with `.dt.convert_time_zone(MARKET_TZ)` /
  `.dt.replace_time_zone(None)`.
- Build named output columns with keyword form:
  `.with_columns(datetime_et=pl.col("datetime_utc").dt.convert_time_zone(MARKET_TZ)...)`.
- Frame inspection in code/tests uses `.item(row, col)`, `.height`,
  `.columns`, `.row(idx, named=True)`, `.to_series().to_list()`.

**Tick-data memory safety (mandatory — see `AGENTS.md`):**
- Never eager-read `input-data/merged_nq_ticks.parquet`.
- Use `pl.scan_parquet(path)` lazily, select only required columns, filter
  bounded ET/UTC windows early, then `collect(engine="streaming")` or
  `sink_parquet`. See `features/macro_extreme_timing.py:78`.
- Validate tick schema from parquet metadata first with `get_tick_schema()`
  before scanning (`utils/tick_data.py`).
- Tick inputs require `ts_event`, `intra_ts_rank`, `side`, `price_ticks`, `size`;
  `side` is `2`=buy, `1`=sell, `0`=none. Prices are integer ticks scaled by
  `TICK_PRICE_DENOMINATOR`.
- Preserve UTC internally; derive ET with `utils.minute_bars.MARKET_TZ` to
  handle DST correctly. Preserve empty 5-second buckets when they are part of
  the expected analysis grid.

## Error Handling

**Validate required columns / schema early, raise `ValueError` with a clear
message.** This is the dominant pattern:
- `normalize_minute_bars` raises `ValueError(f"Missing required columns: {missing}")`,
  `ValueError("Duplicate datetime_utc values after normalization")`,
  and rejects ambiguous DST-fallback ET input
  (`utils/minute_bars.py:38-75`).
- `_validate_tick_schema` raises `ValueError(f"Missing tick columns: {missing}")`
  (`features/macro_extreme_timing.py:71`).
- Numeric feature builders reject bad input:
  `ValueError("close must contain only positive values")`,
  `ValueError("must contain only finite values")`,
  `ValueError("volume must contain only finite non-negative values")`
  (`features/trend/modeling/target.py`).
- Re-raise wrapped errors with context using `raise ValueError(...) from exc`
  (`utils/minute_bars.py:48-51`).
- Tests assert these messages with `pytest.raises(ValueError, match="...")`.

No custom exception classes; `ValueError` is the standard signal for
invalid data/schema.

## Logging

- No logging framework configured. Scripts are run-and-write; status output (if
  any) goes to stdout via `print`. There is no project-wide logger.

## Comments & Docstrings

- Module-level docstrings describe purpose and any memory/processing
  constraints. Example (`features/macro_extreme_timing.py:2-7`) documents that
  the tick input is processed with lazy Polars scans and bounded to the
  requested ET macro minutes.
- Inline comments are sparse and used to clarify timezone/DST math
  (e.g. `# September 2020 is UTC-4 in New York. 13:30 UTC = 09:30 ET.` in
  `test/test_macro_range_forecast.py:42`).
- No JSDoc/Sphinx-style structured docstrings; prose docstrings only.

## Function & Module Design

- Prefer small, composable functions over long notebook-style blocks
  (`AGENTS.md`). Pipelines decompose into `_validate_*`, `_scan_*`, `build_*`,
  `summarize_*`, `write_*` units that are individually importable and testable.
- Functions are pure where possible: `build_*` returns a `pl.LazyFrame` /
  `pl.DataFrame`; only `write_*` functions touch disk, and they return the
  written `Path` (or tuple/list of paths) so callers and tests can assert.
- Output paths are parameterized (default to module constants like
  `OUTPUT_PATH`) so tests can redirect to `tmp_path`.
- Entry scripts expose a `def main() -> None:` guarded by
  `if __name__ == "__main__":` (e.g. `features/macro_extreme_timing.py:253-263`),
  keeping importable logic separate from the CLI runner.

**Exports / packaging:**
- No barrel `__init__` re-export pattern; import directly from the defining
  module. Subpackages under `features/trend/modeling/` use real package
  imports (`from features.trend.modeling.target import build_chop_target`).

## Commits

- Short imperative Conventional Commit subjects (`AGENTS.md`):
  `feat: add CPI cohort summary`, `fix: bound macro vwap barrier tick scans`.
- Do not overwrite source files in `input-data/`; write derived artifacts to
  `outputs/`. Keep `.venv/`, `.venv-codex/`, and generated outputs untracked.

---

*Convention analysis: 2026-05-24*
