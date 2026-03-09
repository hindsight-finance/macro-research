# Macro FVG Success Context Figures Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two matplotlib sanity-check charts for the new macro FVG success-context summaries.

**Architecture:** Extend the existing plotting flow in `features/macro_fvg_study.py` so the new charts are driven from the `success_context_*` summary scopes already written into the summary parquet. Keep the existing runner and figures directory behavior, and add smoke-test coverage for the new figure files.

**Tech Stack:** Python 3, pandas, matplotlib, pytest

---

### Task 1: Add smoke coverage for the new success-context figure files

**Files:**
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write the failing test**

Extend `test_run_macro_fvg_study_writes_parquet_and_figures()` with:

```python
assert (figures_dir / "successful_fvg_mae_by_alignment_bucket.png").exists()
assert (figures_dir / "successful_fvg_mae_by_stacked_flag.png").exists()
```

**Step 2: Run the focused test to verify it fails**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures \
  -q
```

Expected:
- fail because the new figure files are not generated yet

**Step 3: Commit**

```bash
git add test/test_macro_fvg_study.py
git commit -m "test: cover macro fvg success context figures"
```

### Task 2: Add the two success-context matplotlib figures

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Implement the minimal plotting helpers**

Add:

```python
def plot_successful_fvg_mae_by_alignment_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_successful_fvg_mae_by_stacked_flag(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...
```

Behavior:
- read from `success_context_alignment_bucket` and `success_context_stacked_flag`
- plot `mae_pct_mean`, `mae_pct_median`, `mae_pct_p75` multiplied by 100
- save:
  - `successful_fvg_mae_by_alignment_bucket.png`
  - `successful_fvg_mae_by_stacked_flag.png`
- create placeholder figures when the needed summary slice is empty

**Step 2: Wire the figures into the existing plotting pipeline**

Update `plot_fvg_summary_figures()` so:
- the placeholder branch includes both new filenames
- the populated branch calls both new plotting helpers

**Step 3: Run the focused test to verify it passes**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures \
  -q
```

Expected:
- pass

**Step 4: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg success context figures"
```

### Task 3: Verify on the focused file and regenerate real figures

**Files:**
- Modify: `features/macro_fvg_study.py` if cleanup is needed

**Step 1: Run the full focused test file**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- all tests pass

**Step 2: Regenerate the real outputs**

Run:

```bash
python3 features/macro_fvg_study.py
```

Expected:
- updates the derived figures in `outputs/figs/fvg/`
- includes:
  - `successful_fvg_mae_by_alignment_bucket.png`
  - `successful_fvg_mae_by_stacked_flag.png`

**Step 3: Verify the files exist**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

figures_dir = Path("outputs/figs/fvg")
for name in [
    "successful_fvg_mae_by_alignment_bucket.png",
    "successful_fvg_mae_by_stacked_flag.png",
]:
    path = figures_dir / name
    print(name, path.exists(), path.stat().st_size if path.exists() else 0)
PY
```

Expected:
- both files exist with non-zero size

**Step 4: Commit if source changes were made during verification**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "chore: finalize macro fvg success context figures"
```
