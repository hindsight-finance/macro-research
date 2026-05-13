# PM 1H FVG Macro Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a descriptive no-ML script that joins 12:00-15:00 ET 1H FVG/imbalance context to 15:50-15:59 ET macro direction outcomes.

**Architecture:** Add one focused Polars feature module plus one focused pytest file. The module loads canonical UTC minute bars, derives ET hour buckets, computes no-leak 1H FVG and 13:00-15:00 imbalance fields, joins macro outcomes, writes a context parquet and summary CSV.

**Tech Stack:** Python, Polars, pytest, project `.venv`.

---

## Files

- Create: `features/pm_1h_macro_context.py`
  - public helpers: `build_hourly_context`, `join_macro_outcomes`, `build_summary`, `run_pm_1h_macro_context`
  - CLI `main()` with default repo output paths.
- Create: `test/test_pm_1h_macro_context.py`
  - synthetic UTC fixtures, FVG detection, imbalance buckets, macro join, summary rates.

## Task 1: Failing tests for context builder

- [ ] Create `test/test_pm_1h_macro_context.py` with tests importing wished-for functions.
- [ ] Test bullish FVG: h12 high < h14 low → `fvg_direction == bullish`.
- [ ] Test bearish FVG: h12 low > h14 high → `fvg_direction == bearish`.
- [ ] Test imbalance buckets: bullish, bearish, neutral from 13:00-15:00 open/close/close_pos.
- [ ] Run: `/mnt/e/backup/code/Finance/research/macro/.venv/bin/python -m pytest test/test_pm_1h_macro_context.py -q`
- [ ] Expected: fail with missing module/function import.

## Task 2: Minimal context implementation

- [ ] Create `features/pm_1h_macro_context.py`.
- [ ] Implement minute validation + hourly aggregation from canonical UTC bars.
- [ ] Implement `build_hourly_context(minute_bars: pl.DataFrame) -> pl.DataFrame`.
- [ ] Re-run test file; expected: Task 1 tests pass.

## Task 3: Macro join and summary tests

- [ ] Add tests for `join_macro_outcomes()`:
  - joins on `date`
  - adds `macro_dir_sign`
  - adds `macro_direction`
- [ ] Add tests for `build_summary()`:
  - includes `fvg_direction`, `imbalance_direction`, `fvg_x_imbalance`, `fvg_size_bucket`
  - computes `n`, bull/bear counts/rates, macro dir/range means/medians.
- [ ] Run test file; expected: fail from missing join/summary behavior.

## Task 4: Implement join, summary, CLI

- [ ] Implement `join_macro_outcomes(context, macro, include_flat_macro=False)`.
- [ ] Implement `build_summary(joined, include_flat_macro=False)`.
- [ ] Implement `run_pm_1h_macro_context(...)` that reads inputs, writes parquet/csv, returns both frames.
- [ ] Implement `main()` CLI args:
  - `--minute-input`
  - `--macro-input`
  - `--context-output`
  - `--summary-output`
  - `--include-flat-macro`
- [ ] Run test file; expected: all pass.

## Task 5: Integration verification

- [ ] Run: `/mnt/e/backup/code/Finance/research/macro/.venv/bin/python -m pytest test/test_pm_1h_macro_context.py test/test_pm_3pm.py test/test_macro_outcomes.py -q`
- [ ] Run script on available outputs: `/mnt/e/backup/code/Finance/research/macro/.venv/bin/python -m features.pm_1h_macro_context`
- [ ] Confirm files written:
  - `outputs/nq_pm_1h_macro_context.parquet`
  - `outputs/nq_pm_1h_macro_summary.csv`
- [ ] Inspect summary head with Polars.

## Task 6: Commit

- [ ] `git add features/pm_1h_macro_context.py test/test_pm_1h_macro_context.py docs/superpowers/plans/2026-05-13-pm-1h-fvg-macro-context.md`
- [ ] `git commit -m "feat: add pm 1h fvg macro context study"`
