# Macro FVG Success Context MFE Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the successful-FVG success-context summaries with successful-only MFE statistics and add two new matplotlib MFE figures.

**Architecture:** Keep the current event parquet and success-context summary scopes as the source of truth. Reuse the existing successful-FVG filter inside the success-context summary builders to populate MFE aggregates alongside MAE, then add two plotting helpers driven from the same summary parquet slices.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, pytest

---

### Task 1: Add successful-only MFE stats to success-context summaries

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write the failing tests**

Extend the success-context summary tests so they also assert:
- `mfe_pct_mean`
- `mfe_pct_median`
- `mfe_pct_p75`

using successful rows only.

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_builds_success_context_alignment_bucket_summary \
  test/test_macro_fvg_study.py::test_builds_success_context_stacked_flag_summary \
  test/test_macro_fvg_study.py::test_builds_success_context_alignment_bucket_stacked_flag_summary \
  -q
```

Expected:
- fail because success-context summaries do not yet populate MFE fields

**Step 3: Implement the minimal summary change**

Update `_group_success_context_stats()` so successful rows also populate:

```python
"mfe_pct_mean"
"mfe_pct_median"
"mfe_pct_p75"
```

from successful rows only, using the existing `mfe_pct_to_1559` event field.

**Step 4: Run the focused tests again**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_builds_success_context_alignment_bucket_summary \
  test/test_macro_fvg_study.py::test_builds_success_context_stacked_flag_summary \
  test/test_macro_fvg_study.py::test_builds_success_context_alignment_bucket_stacked_flag_summary \
  -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg success context mfe stats"
```

### Task 2: Add the two successful-FVG MFE figures

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write the failing tests**

Update the run smoke test to require:

```python
assert (figures_dir / "successful_fvg_mfe_by_alignment_bucket.png").exists()
assert (figures_dir / "successful_fvg_mfe_by_stacked_flag.png").exists()
```

**Step 2: Run the focused smoke test to verify it fails**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures \
  -q
```

Expected:
- fail because the new figure files are not generated yet

**Step 3: Implement the minimal plotting helpers**

Add:

```python
def plot_successful_fvg_mfe_by_alignment_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_successful_fvg_mfe_by_stacked_flag(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...
```

Behavior:
- drive from `success_context_alignment_bucket` and `success_context_stacked_flag`
- plot `mfe_pct_mean`, `mfe_pct_median`, `mfe_pct_p75` multiplied by 100
- create placeholders when the required summary slice is empty
- wire both figures into `plot_fvg_summary_figures()`

**Step 4: Run the focused smoke test again**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures \
  -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg success context mfe figures"
```

### Task 3: Verify end to end

**Files:**
- Modify: `features/macro_fvg_study.py` if cleanup is needed

**Step 1: Run the full focused test file**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- all tests pass

**Step 2: Regenerate the real study outputs**

Run:

```bash
python3 features/macro_fvg_study.py
```

Expected:
- rewrites the derived figure outputs
- includes:
  - `successful_fvg_mfe_by_alignment_bucket.png`
  - `successful_fvg_mfe_by_stacked_flag.png`

**Step 3: Verify the files exist**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

figures_dir = Path("outputs/figs/fvg")
for name in [
    "successful_fvg_mfe_by_alignment_bucket.png",
    "successful_fvg_mfe_by_stacked_flag.png",
]:
    path = figures_dir / name
    print(name, path.exists(), path.stat().st_size if path.exists() else 0)
PY
```

Expected:
- both files exist with non-zero size
