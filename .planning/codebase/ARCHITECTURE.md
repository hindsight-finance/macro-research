<!-- refreshed: 2026-05-24 -->
# Architecture

**Analysis Date:** 2026-05-24

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                      INPUT DATA (raw)                        │
│   `input-data/nq_1m.csv`  `input-data/es_1m.csv`             │
│   `input-data/merged_nq_ticks.parquet` (~2.9 GB ticks)       │
│   `input-data/economic_events.parquet` (news)               │
└────────┬──────────────────────────────────────┬─────────────┘
         │ minute bars                           │ ticks
         ▼                                       ▼
┌──────────────────────────────┐   ┌──────────────────────────┐
│  NORMALIZE / BASE LAYER      │   │  TICK STUDIES (top-level) │
│  `session_tagger.py`         │   │  `tick_density.py`        │
│  `macro_outcomes.py`         │   │  `volume_delta.py`        │
│  → `outputs/nq_1m.parquet`   │   │  → 1m/5s delta + density  │
│  → `nq_macro_outcomes.parquet│   │     parquet outputs       │
└────────┬─────────────────────┘   └────────┬──────────────────┘
         │ canonical minute bars + macro     │ derived tick datasets
         ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE / RESEARCH LAYER  `features/`           │
│  macro_* studies (barrier, vwap, fvg, delta, regime, ...)   │
│  `features/trend/` indicator families + modeling stack      │
│  `features/lrlr/` liquidity pattern detection               │
│  → `outputs/nq_macro_<feature>.parquet` + `_summary`        │
└────────┬────────────────────────────────────────────────────┘
         │ derived feature parquet
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   VISUALIZATION LAYER  `viz/`               │
│  reads derived parquet → CSV summaries + figures            │
│  → `outputs/figs/<study>/` and `outputs/*.csv`              │
└─────────────────────────────────────────────────────────────┘

         shared helpers used by all layers:
         `utils/minute_bars.py`  `utils/tick_data.py`  `utils/helper.py`
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Minute-base normalizer | Convert raw CSV/legacy timestamps to canonical UTC minute parquet | `session_tagger.py`, `utils/minute_bars.py` |
| Macro outcomes | Daily 15:50–15:59 ET macro range/direction/skew/timing features | `macro_outcomes.py` |
| Tick density study | 1-minute and 5-second macro tick-count density grids | `tick_density.py` |
| Volume delta study | Signed volume/tick delta for macro and Globex windows | `volume_delta.py` |
| Macro feature studies | Derived microstructure research datasets (barrier, vwap, fvg, delta reversal, etc.) | `features/macro_*.py` |
| Trend/regime modeling | Indicator families + walk-forward session modeling tables | `features/trend/` |
| Liquidity detection | Low Resistance Liquidity Run pattern detection | `features/lrlr/lrlr.py` |
| Visualization | Convert derived parquet to figures/CSV summaries | `viz/*.py` |
| Shared data utilities | Timestamp normalization, session windows, tick scanning, news joins | `utils/minute_bars.py`, `utils/tick_data.py`, `utils/helper.py` |

## Pattern Overview

**Overall:** Script-first research feature pipeline (DAG of parquet artifacts).

**Key Characteristics:**
- Each study is a standalone module with module-level `INPUT_PATH` / `OUTPUT_PATH` / `SUMMARY_OUTPUT_PATH` constants and a `main()` entry point.
- Stages communicate exclusively through parquet files in `outputs/` — there is no in-process orchestrator or shared service. Dependencies are implicit via input/output path constants.
- Polars is the default processing engine; Pandas appears only at modeling (`features/trend/modeling/`) and matplotlib boundaries.
- Tick processing is lazy/streaming-first (`pl.scan_parquet`, bounded filters, `collect(engine="streaming")`) because tick inputs are hundreds of millions of rows.
- Canonical schema contract: `datetime_utc, Open, High, Low, Close, Volume[, instrument]`, UTC stored internally, ET derived via `utils.minute_bars.MARKET_TZ`.

## Layers

**Normalize / base layer:**
- Purpose: Turn raw vendor CSV (and large tick parquet) into canonical, deduplicated, UTC-indexed minute bars and the core macro outcome table.
- Location: root scripts `session_tagger.py`, `macro_outcomes.py`; backed by `utils/minute_bars.py`.
- Contains: I/O, timezone normalization, session/window tagging, daily macro aggregation.
- Depends on: `input-data/`, `utils/minute_bars.py`.
- Used by: every feature study (reads `outputs/nq_1m.parquet`, `outputs/nq_minute_base.parquet`, `outputs/nq_macro_outcomes.parquet`).

**Tick-study layer (top-level):**
- Purpose: Produce tick-derived datasets (density, volume delta) consumed by macro feature studies.
- Location: `tick_density.py`, `volume_delta.py`; backed by `utils/tick_data.py`.
- Depends on: `input-data/merged_nq_ticks.parquet`.
- Used by: `features/macro_1550_delta_impulse.py`, `features/macro_bucket_path.py`, `features/macro_delta_reversal.py`.

**Feature / research layer:**
- Purpose: Compute derived research datasets and per-study summary tables.
- Location: `features/macro_*.py` (flat macro studies), `features/trend/` (indicators + modeling), `features/lrlr/` (liquidity).
- Depends on: base + tick outputs in `outputs/`, occasionally raw ticks directly.
- Used by: `viz/`, downstream feature studies, modeling.

**Visualization layer:**
- Purpose: Render figures and CSV summaries from derived parquet.
- Location: `viz/*.py`.
- Depends on: derived parquet in `outputs/`.
- Used by: human reports under `docs/`.

## Data Flow

### Primary Macro Pipeline

1. Raw minute CSV loaded and normalized to canonical UTC schema (`session_tagger.py`, via `utils.minute_bars.load_minute_bars`) → `outputs/nq_1m.parquet` (`session_tagger.py:1`).
2. Canonical bars read, session/window tagged, daily macro window (15:50–15:59 ET) aggregated into range/direction/skew/timing fields (`macro_outcomes.py:87` `compute_macro_outcomes`) → `outputs/nq_macro_outcomes.parquet` (`macro_outcomes.py:162`).
3. Optional regime context joined when `outputs/nq_regimes.parquet` exists (`macro_outcomes.py:49` `join_regime_context`).
4. Downstream feature studies read the macro outcomes / minute base and write `outputs/nq_macro_<feature>.parquet` + `_summary.parquet`.
5. Viz scripts read those parquet files and emit figures under `outputs/figs/<study>/`.

### Tick Microstructure Flow

1. Tick parquet lazily scanned, bounded to a UTC window, only required columns selected (`utils/tick_data.py:18` `scan_tick_data`, `utils/tick_data.py:28` `_bounded_filter`).
2. Streaming collect/aggregate to minute or 5-second buckets (`tick_density.py:153`, `volume_delta.py:230`).
3. Outputs feed delta/bucket macro studies in `features/`.

### Chained Feature Flow (example)

`features/macro_vwap_barrier_context.py` reads three upstream parquet inputs — raw ticks, `outputs/nq_macro_1550_barrier.parquet`, and `outputs/nq_macro_vwap_intramacro.parquet` (`features/macro_vwap_barrier_context.py:13`) — demonstrating that feature studies form a multi-hop DAG, not a single linear pass.

**State Management:**
- No runtime shared state; all state is materialized parquet in `outputs/`. Re-running a stage overwrites its outputs. `outputs/macro_outcomes.parquet` and `outputs/nq_minute_base.parquet` are symlinks to the NQ artifacts.

## Key Abstractions

**Canonical minute bar:**
- Purpose: Single agreed minute schema (`datetime_utc, Open, High, Low, Close, Volume[, instrument]`) all studies consume.
- Examples: `utils/minute_bars.py` (`normalize_minute_bars`, `load_minute_bars`).
- Pattern: UTC-internal, ET derived on demand to handle DST.

**Session / window tagging:**
- Purpose: Label each minute with `session` (ASIA/LONDON/NYAM/LUNCH/PM/OTHER) and `window` (H3PM/MACRO/POST/NONE).
- Examples: `utils/minute_bars.py:95` `derive_session_window`, `utils/minute_bars.py:82` `build_market_time_columns`.

**Tick window collector:**
- Purpose: Memory-safe lazy access to the large tick parquet.
- Examples: `utils/tick_data.py` (`scan_tick_data`, `collect_tick_window`, `ticks_to_minute_bars`).
- Pattern: Tick price stored as integer ticks; divide by `TICK_PRICE_DENOMINATOR` (4.0); `side` 2=buy/1=sell/0=none.

**Feature-set registry (modeling):**
- Purpose: Named immutable feature-column tuples (CORE5, ADX_PARTS, CONTAINMENT_V2, ablations) reused across experiments.
- Examples: `features/trend/modeling/registry.py`, configured into walk-forward runs via `features/trend/modeling/cli.py`.

**News/event join helpers:**
- Purpose: Classify economic events and join them to daily macro frames.
- Examples: `utils/helper.py` (`classify_event`, `merge_news_daily`, `build_macro_event_links`).

## Entry Points

**Top-level study scripts:**
- Location: `session_tagger.py`, `macro_outcomes.py`, `tick_density.py`, `volume_delta.py`.
- Triggers: `.venv/bin/python <script>.py` from repo root.
- Responsibilities: Each owns a `main()` guarded by `if __name__ == "__main__"`, with module-level path constants.

**Feature modules:**
- Location: `features/macro_*.py`.
- Triggers: `.venv/bin/python -m features.<module>` (run as module; relative imports of `utils.*`).
- Responsibilities: Most expose `main()` plus pure compute functions importable by tests.

**Modeling CLI (only argparse-based multi-command entry):**
- Location: `features/trend/modeling/cli.py`.
- Triggers: `.venv/bin/python -m features.trend.modeling.cli <build-table|run-experiments|...>`.
- Responsibilities: Build canonical session modeling tables and run walk-forward experiments into `outputs/trend_modeling/`.

**Viz scripts:**
- Location: `viz/*.py`.
- Triggers: `.venv/bin/python viz/<script>.py` from repo root.
- Responsibilities: Read derived parquet, write figures/CSV (headless `Agg` backend in current scripts).

## Architectural Constraints

- **Threading:** Single-threaded scripts; parallelism comes only from the Polars engine. Tick reads MUST use streaming/lazy collect, never eager full reads of `input-data/merged_nq_ticks.parquet`.
- **Global state:** Module-level path constants (`INPUT_PATH`, `OUTPUT_PATH`, etc.) act as configuration singletons in every study. `features/trend/state_detector.py` maintains a module-level `_IMPORT_CACHE` and mutates `sys.path` to load indicator modules from directories with spaces.
- **Timezone:** UTC must stay internal; ET is derived via `utils.minute_bars.MARKET_TZ`. Legacy ambiguous ET timestamps during DST fallback raise rather than silently mis-convert (`utils/minute_bars.py:46`).
- **Naming drift:** `session_tagger.py` writes `outputs/nq_1m.parquet`, while `macro_outcomes.py` defaults its input to `outputs/nq_minute_base.parquet` (a symlink). Callers/tests may override paths. Treat the symlinks as the contract.
- **Directories with spaces:** `features/trend/ATR Range/`, `features/trend/Lag autocorr/`, etc. cannot be imported via normal dotted imports; `state_detector.py` loads them by file path.

## Anti-Patterns

### Hard-coded path constants instead of arguments

**What happens:** Studies pin inputs/outputs as module-level `Path(...)` constants (e.g. `features/macro_1550_barrier.py:11`), and dependencies between stages are implicit.
**Why it's wrong:** Pipeline order is undocumented and must be inferred from path matching; re-pointing inputs requires editing source or test overrides.
**Do this instead:** Keep the constant defaults but expose compute functions that accept paths/frames as parameters (as `tick_density.py:153` and `volume_delta.py:230` do), so tests and chained callers can pass paths explicitly.

### Eager-reading the tick parquet

**What happens:** A study could `pl.read_parquet("input-data/merged_nq_ticks.parquet")`.
**Why it's wrong:** The file is ~2.9 GB / hundreds of millions of rows and will exhaust memory.
**Do this instead:** Use `utils.tick_data.scan_tick_data` + bounded filter + `collect(engine="streaming")` (`utils/tick_data.py:34`).

### Pandas in new processing paths

**What happens:** Reaching for Pandas for convenience in feature compute.
**Why it's wrong:** The codebase standardizes on Polars; mixing engines fragments the schema contract and timezone handling.
**Do this instead:** Use Polars for processing; restrict Pandas to the modeling stack (`features/trend/modeling/`) and the matplotlib boundary in `viz/`.

## Error Handling

**Strategy:** Fail fast with explicit validation at study boundaries.

**Patterns:**
- Required-column checks raise `ValueError` early (`macro_outcomes.py:88`, `utils/minute_bars.py:65`).
- Missing input files print `[ERROR] ... not found` to stderr and `sys.exit(1)` in `main()` (`macro_outcomes.py:166`, `tick_density.py:179`).
- Timestamp parse failures and duplicate/ambiguous timestamps raise rather than silently coerce (`utils/minute_bars.py:62`, `:72`).

## Cross-Cutting Concerns

**Logging:** `print()` with `[OK]` / `[WARN]` / `[ERROR]` prefixes; no logging framework. `outputs/logs/` holds run artifacts.
**Validation:** Required-column / schema validation at the start of each compute function (constants like `TICK_REQUIRED_COLUMNS`, `BARRIER_REQUIRED_COLUMNS` in `features/macro_vwap_barrier_context.py:20`).
**Authentication:** Not applicable (local research workspace, no network services).

---

*Architecture analysis: 2026-05-24*
