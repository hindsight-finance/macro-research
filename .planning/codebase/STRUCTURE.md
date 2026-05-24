# Codebase Structure

**Analysis Date:** 2026-05-24

## Directory Layout

```
macro/
├── session_tagger.py          # Normalize raw minute CSV → canonical UTC parquet
├── macro_outcomes.py          # Daily 15:50–15:59 ET macro outcome features
├── tick_density.py            # 1m / 5s macro tick-density grids from ticks
├── volume_delta.py            # Macro + Globex signed volume/tick delta
├── input-data/                # Raw source datasets (CSV + large tick parquet)
├── outputs/                   # All generated parquet, CSV, figures, logs
│   ├── figs/                  # Per-study figure subdirs (fvg, ma, tick_density, ...)
│   ├── logs/                  # Run logs
│   └── trend_modeling/        # Modeling tables, experiment + ablation outputs
├── features/                  # Feature / research modules
│   ├── macro_*.py             # Flat macro microstructure studies (15+ modules)
│   ├── pm_*.py                # PM / 3 PM context + interaction studies
│   ├── lrlr/                  # Low Resistance Liquidity Run detection
│   └── trend/                 # Trend/regime indicators + modeling stack
│       ├── <Indicator dirs>/  # ADX, ATR Range, DRA, IRR, ... each with test/
│       └── modeling/          # registry, table, walkforward, cli, labels, target
├── viz/                       # Visualization scripts (parquet → figs/CSV)
├── utils/                     # Shared data helpers (minute_bars, tick_data, helper)
├── test/                      # Top-level pytest suite (mirrors root + features studies)
├── docs/                      # Plans, reports, experiment logs, research log
├── roadmap.md                 # Phase-based research roadmap
├── README.md                  # Project objectives + research questions
├── AGENTS.md                  # Contributor + agent conventions (authoritative)
└── requirements.txt           # Python dependencies
```

## Directory Purposes

**`input-data/`:**
- Purpose: Immutable raw source data; never written by studies.
- Contains: `nq_1m.csv`, `es_1m.csv` (minute bars), `merged_nq_ticks.parquet` (~2.9 GB ticks), `economic_events.parquet` (news), plus tick schema/example files.
- Key files: `merged_ticks_schema.txt` documents the tick column contract.

**`outputs/`:**
- Purpose: All generated artifacts; the inter-stage communication medium.
- Contains: `nq_macro_*.parquet` + `_summary.parquet` pairs, base minute parquet, symlinks (`macro_outcomes.parquet`, `nq_minute_base.parquet`).
- Key files: `nq_1m.parquet`, `nq_macro_outcomes.parquet`, `nq_macro_extreme_timing.parquet`.

**`outputs/figs/`:**
- Purpose: One subdirectory per study (`fvg`, `ma`, `tick_density`, `macro_1550_barrier`, `macro_extreme_timing`, `macro_regime_direction`, `macro_vwap_barrier_context`).
- Generated: Yes. Committed: Yes (figures live in the repo).

**`outputs/trend_modeling/`:**
- Purpose: Modeling table cache and walk-forward experiment/ablation outputs.
- Contains: `cache/`, `containment_experiments/`, `containment_v2_*`, `regime_3scalar*` subdirs keyed by session window and experiment name.

**`features/`:**
- Purpose: Derived research compute modules; the largest subsystem.
- Contains: flat `macro_*.py` and `pm_*.py` studies, plus `lrlr/` and `trend/` subpackages.
- Key files: `macro_extreme_timing.py`, `macro_fvg_study.py` (largest, ~60 KB), `macro_vwap_features.py`, `pm_3pm.py`.

**`features/trend/`:**
- Purpose: Trend/regime indicator families and the modeling stack.
- Contains: one directory per indicator (`ADX/`, `ATR Range/`, `DRA/`, `IRR/`, `Lag autocorr/`, `MSS tan/`, `Swing Point Density/`, `efficiency_ratio/`, `variance_ratio/`), each with its own `test/`; plus `modeling/`, `state_detector.py`, `historical_regimes.py`.
- Key files: `modeling/cli.py` (argparse entry), `modeling/registry.py` (feature-set definitions), `modeling/table.py`, `modeling/walkforward.py`.

**`features/lrlr/`:**
- Purpose: Low Resistance Liquidity Run pattern detection (contains a nested `.git`).
- Key files: `lrlr.py`, `test/test_lrlr.py` (runnable with `--no-viz`).

**`viz/`:**
- Purpose: Render derived parquet to figures + CSV summaries.
- Key files: `macro_analysis.py`, `tick_density_viz.py`, `macro_extreme_timing_viz.py`, `macro_1550_barrier_viz.py`, `macro_vwap_barrier_context_viz.py`. Older scripts (`macro_high.py`, `viz_outcome.py`, `pm_macro_viz.py`) may assume viz-relative paths or call `plt.show()`.

**`utils/`:**
- Purpose: Shared, dependency-free-of-studies helpers.
- Key files: `minute_bars.py` (canonical schema, session windows), `tick_data.py` (lazy tick access), `helper.py` (news classification + joins).

**`test/`:**
- Purpose: Top-level pytest suite; one `test_<study>.py` per root/feature study.
- Key files: `conftest.py` (matplotlib toolkit setup), `cpi.py` (CPI analysis helper).

**`docs/`:**
- Purpose: Human-facing research artifacts.
- Contains: `plans/` (dated design docs), `reports/` (dated findings), `experiments/` (numbered logs `NNNN-short-name.md`), `research_log.md`, `superpowers/`.

## Key File Locations

**Entry Points:**
- `session_tagger.py`, `macro_outcomes.py`, `tick_density.py`, `volume_delta.py`: root study scripts (`python <file>.py`).
- `features/trend/modeling/cli.py`: argparse multi-command modeling entry (`python -m features.trend.modeling.cli`).
- `features/macro_*.py`: module entry points (`python -m features.<module>`).

**Configuration:**
- `requirements.txt`: dependencies.
- `AGENTS.md`: authoritative conventions, data-flow, and command reference.
- `.venv/`: project virtualenv (always invoke as `.venv/bin/python`).

**Core Logic:**
- `utils/minute_bars.py`: schema + session/window definitions.
- `utils/tick_data.py`: memory-safe tick access.
- `utils/helper.py`: news event join logic.

**Testing:**
- `test/`: top-level suite.
- `features/trend/<Indicator>/test/`, `features/trend/modeling/test/`, `features/lrlr/test/`: co-located package tests.

## Naming Conventions

**Files:**
- Modules: `snake_case.py`. Studies named by subject: `macro_<topic>.py`, `pm_<topic>.py`.
- Tests: `test_<module>.py`, mirroring the module under test.
- Indicator directories under `features/trend/` use human-readable names, sometimes with spaces and capitals (`ATR Range`, `Lag autocorr`, `MSS tan`, `Swing Point Density`) — these are NOT dotted-importable and are loaded by file path in `state_detector.py`.

**Output parquet (per AGENTS.md):**
- `nq_macro_<feature>.parquet` — main per-study dataset.
- `nq_macro_<feature>_summary.parquet` — aggregated summary.
- `nq_macro_<feature>_5s.parquet` — 5-second-bucket variant.
- `nq_globex_<feature>_1m.parquet` — Globex-window 1-minute variant.

**Column naming (per AGENTS.md):**
- `datetime_utc` for UTC timestamps; ET derived via `MARKET_TZ`.
- Timing fields: `*_high_time`, `*_low_time`, `*_high_first`.
- `macro_minute_index` for ET minute markers; `bucket_index` / `macro_bucket_index` for 5-second buckets.
- Config constants: `UPPER_CASE` (e.g. `INPUT_PATH`, `OUTPUT_PATH`, `MACRO_WINDOW_NAME`).

## Where to Add New Code

**New macro feature study:**
- Implementation: `features/macro_<topic>.py` with module-level `INPUT_PATH` / `OUTPUT_PATH` / `SUMMARY_OUTPUT_PATH` constants, pure compute function(s), and a `main()` guarded by `if __name__ == "__main__"`.
- Inputs: read existing `outputs/*.parquet` where possible; use `utils.tick_data` for raw ticks.
- Tests: `test/test_macro_<topic>.py` covering a happy path and a schema-failure path.
- Outputs: follow `nq_macro_<feature>.parquet` + `_summary.parquet` naming into `outputs/`.

**New trend indicator:**
- Implementation: `features/trend/<IndicatorName>/<short>.py` with a co-located `test/` directory.
- Register feature columns in `features/trend/modeling/registry.py` if used in modeling experiments.

**New visualization:**
- Implementation: `viz/<study>_viz.py`, headless Matplotlib `Agg`, reading the study's derived parquet and writing to `outputs/figs/<study>/`.
- Tests: `test/test_<study>_viz.py` with a non-visual assertion path so it runs headless.

**Shared utility:**
- Add to `utils/minute_bars.py` (timestamps/sessions), `utils/tick_data.py` (tick access), or `utils/helper.py` (news/joins). Keep Polars-based.

**New documentation:**
- Design plan: `docs/plans/YYYY-MM-DD-<topic>.md`.
- Findings report: `docs/reports/YYYY-MM-DD-<topic>-findings.md`.
- Experiment log: `docs/experiments/NNNN-short-name.md`, indexed in `docs/research_log.md`.

## Special Directories

**`.venv/`:**
- Purpose: Project virtualenv (Python 3.12, Polars, Matplotlib, etc.).
- Generated: Yes. Committed: No (gitignored).

**`outputs/`:**
- Purpose: All generated artifacts; some entries are symlinks (`macro_outcomes.parquet` → `nq_macro_outcomes.parquet`, `nq_minute_base.parquet` → `nq_1m.parquet`).
- Generated: Yes. Committed: Partially (figures committed; large parquet typically not).

**`features/lrlr/.git`:**
- Purpose: Nested git repository inside the LRLR subpackage.
- Generated: Yes. Note: this is a vendored/embedded repo, not part of the top-level working tree.

**`__pycache__/`, `.pytest_cache/`, `.worktrees/`:**
- Purpose: Python/pytest/git tooling caches. Generated: Yes. Committed: No.

---

*Structure analysis: 2026-05-24*
