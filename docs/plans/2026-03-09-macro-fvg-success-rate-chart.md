# Macro FVG Success Rate Chart Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the success-context alignment-bucket matplotlib chart so it also shows the percent of confirmable FVGs classified as successful.

**Architecture:** Keep the existing `successful_fvg_mae_by_alignment_bucket.png` file and add a secondary y-axis line driven by `successful_share_of_confirmable` from the existing `success_context_alignment_bucket` summary scope. Add a focused plotting test that verifies the overlay axis and success-rate series are present.

**Tech Stack:** Python 3, pandas, matplotlib, pytest

---

### Task 1: Add a failing plotting test for the success-rate overlay

**Files:**
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write the failing test**

Add a targeted test for `plot_successful_fvg_mae_by_alignment_bucket()` that:
- builds a small `success_context_alignment_bucket` summary frame
- monkeypatches `plt.close` so the figure can be inspected
- asserts the figure has two axes
- asserts the secondary axis label is `Success Rate`
- asserts the secondary axis line uses `successful_share_of_confirmable * 100`

**Step 2: Run the focused test to verify it fails**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_plot_successful_fvg_mae_by_alignment_bucket_overlays_success_rate \
  -q
```

Expected:
- fail because the current chart only has the MAE bars

### Task 2: Implement the success-rate overlay

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Add the minimal plotting change**

Update `plot_successful_fvg_mae_by_alignment_bucket()` so it:
- reads `successful_share_of_confirmable`
- creates a right-side y-axis with `ax.twinx()`
- plots success rate in percent as a line with markers
- keeps the MAE bars and existing file output

**Step 2: Run the focused test to verify it passes**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_plot_successful_fvg_mae_by_alignment_bucket_overlays_success_rate \
  -q
```

Expected:
- pass

**Step 3: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: show macro fvg success rate on alignment chart"
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
- rewrites `outputs/figs/fvg/successful_fvg_mae_by_alignment_bucket.png`

**Step 3: Verify the figure exists**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
path = Path("outputs/figs/fvg/successful_fvg_mae_by_alignment_bucket.png")
print(path.exists(), path.stat().st_size if path.exists() else 0)
PY
```

Expected:
- file exists with non-zero size
