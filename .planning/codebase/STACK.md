# Technology Stack

**Analysis Date:** 2026-05-24

## Languages

**Primary:**
- Python 3.12 - All source code (scripts, feature modules, viz, tests). Pinned by the virtualenv at `.venv/pyvenv.cfg` (`version = 3.12.3`).

**Secondary:**
- Markdown - Documentation and experiment logs in `docs/`, `AGENTS.md`, `README.md`, `roadmap.md`, `ideadump.md`.

Note: `README.md` (line 117) states "Python 3.10+" and references `pandas`-centric tooling, but the actual installed/declared baseline is Python 3.12 with a Polars-first stack. Treat `AGENTS.md` and `requirements.txt` as authoritative over `README.md`.

## Runtime

**Environment:**
- CPython 3.12.3 (system interpreter at `/usr/bin/python3.12`)
- Virtualenv at `.venv/` with `include-system-site-packages = true`
- Convention (from `AGENTS.md`): always invoke via `.venv/bin/python ...`, never system `python`

**Package Manager:**
- pip (no Poetry/uv/PDM lockfile present)
- Lockfile: missing - only `requirements.txt` (loose `>=` version floors, no pinned lock)

## Frameworks

This is a script-first research workspace, not an application framework project. There is no web framework, ORM, or service layer.

**Data processing:**
- Polars >= 1.40 (installed 1.40.1) - Primary dataframe/lazy-query engine; default for all processing paths
- PyArrow >= 24 (installed 24.0.0) - Parquet I/O and schema introspection (`utils/tick_data.py:get_tick_schema`)
- NumPy >= 2 (installed 2.4.4) - Numerics at the modeling/plotting boundary
- pandas >= 3 (installed 3.0.2) - Limited use; only at visualization boundary and in `features/macro_range_forecast.py`. `AGENTS.md` explicitly says keep pandas out of new processing paths

**Scientific / modeling:**
- SciPy >= 1.17 (installed 1.17.1) - Statistics
- statsmodels >= 0.14 (installed 0.14.6) - Statistical models
- scikit-learn >= 1.8 (installed 1.8.0) - `Ridge`, `make_pipeline`, `StandardScaler` in `features/macro_range_forecast.py`
- xgboost >= 2.0 - Declared in `requirements.txt`; no current import found in source (reserved for planned modeling)
- joblib (installed 1.5.3, transitive via scikit-learn)

**Testing:**
- pytest >= 9 (installed 9.0.2) - Test runner; config via `test/conftest.py`

**Visualization:**
- matplotlib >= 3.10 (installed 3.10.9) - Headless `Agg` backend in current viz scripts (`viz/tick_density_viz.py`, `viz/macro_extreme_timing_viz.py`, `viz/macro_1550_barrier_viz.py`, `viz/macro_vwap_barrier_context_viz.py`); older scripts (`viz/macro_high.py`, `viz/viz_outcome.py`, `viz/pm_macro_viz.py`) may call `plt.show()`

## Key Dependencies

**Critical:**
- `polars` 1.40.1 - Core engine for all data transforms and tick streaming (`pl.scan_parquet`, lazy filters, `collect(engine="streaming")`)
- `pyarrow` 24.0.0 - Parquet read/write and metadata-only schema validation before large tick reads
- `numpy` 2.4.4 - Underlying numerics

**External editable package:**
- `smt` (editable install, `smt-0.1.0`) - Mapped via `.venv/lib/python3.12/site-packages/__editable__.smt-0.1.0.pth` to `/mnt/e/backup/code/Finance/Misc/SMT/smt` (OUTSIDE this repo). No `import smt` / `from smt` statements found in this repo's source, so it is installed in the venv but not currently referenced here. Treat as an external sibling project linked into the shared environment.

**Infrastructure:**
- None - No database driver, message queue, or cloud SDK installed

## Configuration

**Environment:**
- No `os.environ` / `getenv` / dotenv usage found anywhere in source
- No `.env` files present (`.gitignore` ignores `.env` / `.env.*` defensively, but none exist)
- All paths and parameters are passed via `argparse` CLI flags or hardcoded `UPPER_CASE` path constants in each script

**Build:**
- No build system. Scripts run directly: `.venv/bin/python <script>.py` or `.venv/bin/python -m features.<module>`
- No `pyproject.toml`, `setup.py`, `setup.cfg`, `tox.ini`, or `Makefile` in this repo
- `requirements.txt` is the only manifest

**Test config:**
- `test/conftest.py` - pytest fixtures/configuration
- `.pytest_cache/` present (gitignored)
- Run suites per `AGENTS.md`: `.venv/bin/python -m pytest test -q`, plus targeted runs for `features/lrlr/test` and `features/trend`

## Platform Requirements

**Development:**
- Linux (observed: WSL2). Path constants are repo-relative; the venv `command` line references a case-variant path (`.../Finance/Research/macro/.venv`) indicating the project moved/renamed
- Memory awareness required: tick parquet files (`input-data/merged_nq_ticks.parquet`) can be hundreds of millions of rows; must use lazy scans, never eager full reads (`AGENTS.md` "Tick Data & Memory Safety")

**Production:**
- Not applicable - this is an offline research/analysis workspace, not a deployed service. Outputs are parquet/CSV/PNG artifacts under `outputs/`

---

*Stack analysis: 2026-05-24*
