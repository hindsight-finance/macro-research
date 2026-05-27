# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A script-first Python research workspace studying NQ/ES intraday behavior around the
**market-close macro window (15:50–16:00 ET)** — its range, direction, timing, tick
microstructure, volume delta, FVG/event behavior, liquidity, and trend/regime context.
It is a research codebase, not an application: entry points are standalone scripts that read
datasets and write derived parquet/CSV/figures. There is no service or build step.

`AGENTS.md` is the authoritative, detailed contributor guide (data flow, naming conventions,
experiment-log protocol, per-feature notes). `.planning/codebase/` holds generated maps
(`ARCHITECTURE.md`, `STRUCTURE.md`, `CONVENTIONS.md`, `TESTING.md`, etc.). Consult those for
depth; this file is the orientation layer.

## Environment

Always use the project virtualenv — **never system `python`**. Run everything from the repo root:

```bash
.venv/bin/python <script>.py
.venv/bin/python -m features.<module>          # feature/package modules run with -m
.venv/bin/python -m pytest test -q             # main suite
.venv/bin/python -m pytest test/test_x.py -q   # single test file
```

Root scripts: `session_tagger.py`, `macro_outcomes.py`, `tick_density.py`, `volume_delta.py`.
Feature modules under `features/` and the trend modeling CLI
(`.venv/bin/python -m features.trend.modeling.cli`) use argparse subcommands.

## Data pipeline (the big picture)

The flow is a chain of parquet files, each script consuming the previous stage's output:

1. `session_tagger.py` normalizes `input-data/{nq,es}_1m.csv` → canonical minute parquet.
2. `macro_outcomes.py` computes the daily macro-window outcome table from canonical bars.
3. Tick studies (`tick_density.py`, `volume_delta.py`, most `features/macro_*`) read the large
   tick parquet `input-data/merged_nq_ticks.parquet`.
4. `features/` scripts produce derived research tables (`outputs/nq_macro_<feature>.parquet`).
5. `viz/` scripts turn derived parquet into CSV summaries + figures under `outputs/figs/`.

**Naming-drift gotcha:** `session_tagger.py` writes `outputs/nq_1m.parquet`, but
`macro_outcomes.py` defaults its input to `outputs/nq_minute_base.parquet`. Check/override the
path when wiring the full pipeline; tests pass explicit paths.

`input-data/` and `outputs/` are gitignored — never overwrite source files in `input-data/`,
and don't commit large generated artifacts unless they are intentional deliverables.

## Time handling (critical, easy to get wrong)

All logic keeps **UTC internally** (`datetime_utc`) and derives ET on demand via
`utils.minute_bars` (`MARKET_TZ = "America/New_York"`), so DST is handled correctly. Use
`build_market_time_columns` and `derive_session_window` rather than hand-rolling timezone math.
Window labels: `H3PM` = 15:00–15:49, `MACRO` = 15:50–15:59, `POST` = 16:00–16:10 ET. Sessions:
`ASIA`, `LONDON`, `NYAM`, `LUNCH`, `PM`, `OTHER`. Legacy ET inputs with ambiguous DST-fallback
timestamps are rejected by `normalize_minute_bars` — prefer `datetime_utc` inputs.

## Tick data & memory safety (non-negotiable)

Tick parquet files have hundreds of millions of rows. **Never eager-read
`merged_nq_ticks.parquet`.** Use `utils.tick_data` helpers: `get_tick_schema()` to validate from
metadata, `scan_tick_data()` for a lazy frame projecting only `TICK_COLUMNS`
(`ts_event, intra_ts_rank, side, price_ticks, size`), and `collect_tick_window()` which *requires*
bounded start/end UTC and uses `collect(engine="streaming")`. Filter to a bounded ET/UTC window
early; use `sink_parquet` for full-output pipelines. `side`: `2`=buy, `1`=sell, `0`=none;
prices are `price_ticks / 4.0`. Preserve empty 5-second buckets that are part of the analysis grid.

## Conventions

- **Polars by default** for data processing; keep Pandas out of new processing paths (acceptable
  only at the matplotlib boundary on small summarized frames). Validate required columns early.
- Timing/output column naming follows a fixed vocabulary — see `AGENTS.md` ("Coding Style" and
  per-feature sections) before inventing names (`*_high_time`, `*_high_first`,
  `macro_minute_index`, `bucket_index`, output stems like `nq_macro_<feature>_summary.parquet`).
- Visualization scripts use headless Matplotlib (`Agg`); a few legacy `viz/` scripts call
  `plt.show()` or assume `viz/`-relative paths.
- Tests live in `test/` (root pipeline) and beside feature packages (`features/lrlr/test`,
  `features/trend/test`, `features/trend/modeling/test`). Fixtures sit next to the tests using
  them. Cover happy-path *and* schema-failure paths; for tick studies cover DST, bounded-window
  filtering, and first-touch/tie behavior. If a change emits plots, add a non-visual assertion
  path so it runs headless.

## Experiment logs

When the user asks for an experiment/research/findings log, follow the schema and rules in
`AGENTS.md` ("Experiment Log Protocol"): index in `docs/research_log.md`, individual logs in
`docs/experiments/NNNN-short-name.md`. Always report sample sizes with rates and mark results
Exploratory vs. Validated. Don't hand-edit generated artifacts to fit a log — regenerate from
scripts or cite existing generated files.

## Concept graph

A browsable Obsidian vault under `docs/graph/` maps every concept and study to a node graph
(open `docs/` as the vault; hub `docs/graph/Concept Map.md`, protocol `docs/graph/README.md`).
**After completing or logging any new experiment, or introducing a new reusable concept, update
the graph before considering the work done:** add/update the `experiments/<slug>.md` or
`concepts/<slug>.md` node, set nested `tags:` (category → variant, e.g. `feature/vwap/anchor-1550`)
and **flat** frontmatter param keys (Obsidian Properties does not support nested maps — never a
`params:` object), link the concepts and source artifacts, and add it to the Concept Map hub.
Keep it additive; never rewrite a canonical artifact to fit a node. Params are frontmatter, never
nodes; variants are nested tags, promoted to nodes only when shared by ≥2 experiments.
