# FVG Delta Time Basis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add macro FVG delta-dominance summaries split by FVG creation minute and existing coarse minute block.

**Architecture:** Reuse existing enriched FVG event columns and `_group_success_context_stats`. Extend summary schemas with four new scopes: aligned/absolute delta quantile by creation minute, and aligned/absolute delta quantile by minute block. Keep the feature in `features/macro_fvg_study.py` so generated `nq_macro_fvg_summary.parquet` remains the single summary output.

**Tech Stack:** Python, Polars, pytest, existing script-first repo patterns. Use `.venv/bin/python`.

---

## File Structure

- Modify: `features/macro_fvg_study.py`
  - Add four summary builder functions.
  - Append them to `build_summary_tables()`.
- Modify: `test/test_macro_fvg_study.py`
  - Add tests using existing dominance fixtures.

### Task 1: Add failing time-basis summary tests

**Files:**
- Modify: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Add a test for creation-minute delta summaries**

Append before `test_run_macro_fvg_study_writes_parquet_and_figures`:

```python
def test_build_summary_tables_includes_delta_dominance_by_creation_minute():
    enriched = enrich_fvg_events_with_delta_dominance(
        make_delta_events_for_dominance(),
        make_delta_5s_for_dominance(),
    ).with_columns(
        assigned_minute_index=pl.Series([0, 0, 1, 1]),
        assigned_minute_hhmm=pl.Series(["15:50", "15:50", "15:51", "15:51"]),
        minute_block=pl.Series(["15:50-15:52", "15:50-15:52", "15:50-15:52", "15:50-15:52"]),
    )

    summary = macro_fvg_study.build_summary_tables(enriched)
    scopes = set(summary["summary_scope"].to_list())

    assert "success_context_creation_minute_aligned_delta_imbalance_quantile" in scopes
    assert "success_context_creation_minute_abs_delta_imbalance_quantile" in scopes

    row = filter_one(
        summary,
        (pl.col("summary_scope") == "success_context_creation_minute_aligned_delta_imbalance_quantile")
        & (pl.col("assigned_minute_hhmm") == "15:51")
        & (pl.col("aligned_delta_imbalance_quantile") == "q4_highest"),
    )
    assert row["n_confirmable"] == 1
    assert row["n_retraced"] == 1
    assert row["n_successful"] == 1
    assert row["successful_share_of_confirmable"] == 1.0
```

- [ ] **Step 2: Add a test for coarse block delta summaries**

Append before `test_run_macro_fvg_study_writes_parquet_and_figures`:

```python
def test_build_summary_tables_includes_delta_dominance_by_minute_block():
    enriched = enrich_fvg_events_with_delta_dominance(
        make_delta_events_for_dominance(),
        make_delta_5s_for_dominance(),
    ).with_columns(
        assigned_minute_index=pl.Series([0, 1, 5, 5]),
        assigned_minute_hhmm=pl.Series(["15:50", "15:51", "15:55", "15:55"]),
        minute_block=pl.Series(["15:50-15:52", "15:50-15:52", "15:53-15:57", "15:53-15:57"]),
    )

    summary = macro_fvg_study.build_summary_tables(enriched)
    scopes = set(summary["summary_scope"].to_list())

    assert "success_context_minute_block_aligned_delta_imbalance_quantile" in scopes
    assert "success_context_minute_block_abs_delta_imbalance_quantile" in scopes

    row = filter_one(
        summary,
        (pl.col("summary_scope") == "success_context_minute_block_abs_delta_imbalance_quantile")
        & (pl.col("minute_block") == "15:53-15:57")
        & (pl.col("abs_delta_imbalance_quantile") == "q4_highest"),
    )
    assert row["n_confirmable"] == 1
    assert row["n_retraced"] == 1
    assert row["n_successful"] == 1
    assert row["successful_share_of_confirmable"] == 1.0
```

- [ ] **Step 3: Run the new tests and verify they fail**

```bash
.venv/bin/python -m pytest \
  test/test_macro_fvg_study.py::test_build_summary_tables_includes_delta_dominance_by_creation_minute \
  test/test_macro_fvg_study.py::test_build_summary_tables_includes_delta_dominance_by_minute_block \
  -q
```

Expected: FAIL because the new summary scopes are not present.

### Task 2: Implement time-basis summary scopes

**Files:**
- Modify: `features/macro_fvg_study.py`

- [ ] **Step 1: Add builder functions**

Insert after `build_success_context_side_abs_delta_imbalance_quantile_summary()`:

```python
def build_success_context_creation_minute_aligned_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "aligned_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["assigned_minute_index", "assigned_minute_hhmm", column],
        "success_context_creation_minute_aligned_delta_imbalance_quantile",
    )


def build_success_context_creation_minute_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["assigned_minute_index", "assigned_minute_hhmm", column],
        "success_context_creation_minute_abs_delta_imbalance_quantile",
    )


def build_success_context_minute_block_aligned_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "aligned_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["minute_block", column],
        "success_context_minute_block_aligned_delta_imbalance_quantile",
    )


def build_success_context_minute_block_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["minute_block", column],
        "success_context_minute_block_abs_delta_imbalance_quantile",
    )
```

- [ ] **Step 2: Append builders to `build_summary_tables()`**

After `build_success_context_side_abs_delta_imbalance_quantile_summary(events),` add:

```python
        build_success_context_creation_minute_aligned_delta_imbalance_quantile_summary(events),
        build_success_context_creation_minute_abs_delta_imbalance_quantile_summary(events),
        build_success_context_minute_block_aligned_delta_imbalance_quantile_summary(events),
        build_success_context_minute_block_abs_delta_imbalance_quantile_summary(events),
```

- [ ] **Step 3: Run focused tests and verify pass**

```bash
.venv/bin/python -m pytest \
  test/test_macro_fvg_study.py::test_build_summary_tables_includes_delta_dominance_by_creation_minute \
  test/test_macro_fvg_study.py::test_build_summary_tables_includes_delta_dominance_by_minute_block \
  -q
```

Expected: PASS.

### Task 3: Full verification and output regeneration

**Files:**
- Runtime outputs under `outputs/`

- [ ] **Step 1: Run focused tests**

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Regenerate macro FVG outputs in the main workspace if input/output parquet files are available**

```bash
.venv/bin/python -m features.macro_fvg_study
```

Expected: writes `outputs/nq_macro_fvg_summary.parquet` with the four new time-basis scopes.

- [ ] **Step 3: Inspect new scopes**

```bash
.venv/bin/python - <<'PY'
import polars as pl
summary = pl.read_parquet('outputs/nq_macro_fvg_summary.parquet')
scopes = [
    'success_context_creation_minute_aligned_delta_imbalance_quantile',
    'success_context_creation_minute_abs_delta_imbalance_quantile',
    'success_context_minute_block_aligned_delta_imbalance_quantile',
    'success_context_minute_block_abs_delta_imbalance_quantile',
]
print(summary.filter(pl.col('summary_scope').is_in(scopes)).select([
    'summary_scope', 'minute_block', 'assigned_minute_hhmm',
    'aligned_delta_imbalance_quantile', 'abs_delta_imbalance_quantile',
    'n_confirmable', 'n_successful', 'successful_share_of_confirmable',
]).sort(['summary_scope', 'minute_block', 'assigned_minute_hhmm', 'aligned_delta_imbalance_quantile', 'abs_delta_imbalance_quantile']))
PY
```

Expected: printed rows for all four scopes.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-05-13-fvg-delta-time-basis.md features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add fvg delta time-basis summaries"
```
