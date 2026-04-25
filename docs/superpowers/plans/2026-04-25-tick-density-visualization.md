# Tick Density Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a matplotlib script that reads the requested macro tick-density parquet datasets, writes separate histogram and mean/p25/p75 band PNGs per dataset, and reports whether cross-day tick-density distributions are normal at each minute/bucket.

**Architecture:** Add a focused visualization module under `viz/` with small helpers for dataset loading, cross-day aggregation, normality testing, plotting, and artifact writing. Cover aggregation and normality behavior with pytest first, then run the script on the real parquet files to generate PNGs and CSV summaries in `outputs/figs/tick_density/`.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, scipy, pytest

---

### Task 1: Add failing tests for aggregation + normality

**Files:**
- Create: `test/test_tick_density_viz.py`
- Test: `test/test_tick_density_viz.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pandas as pd

from viz.tick_density_viz import summarize_tick_density_dataset


def test_summarize_tick_density_dataset_builds_cross_day_band_stats(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                [
                    "2025-01-02T20:50:00Z",
                    "2025-01-02T20:51:00Z",
                    "2025-01-03T20:50:00Z",
                    "2025-01-03T20:51:00Z",
                ],
                utc=True,
            ),
            "date_utc": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "macro_minute_index": [0, 1, 0, 1],
            "tick_count": [10, 20, 30, 40],
            "total_size": [1, 1, 1, 1],
            "buy_ticks": [5, 10, 15, 20],
            "sell_ticks": [5, 10, 15, 20],
            "none_ticks": [0, 0, 0, 0],
        }
    ).to_parquet(path, index=False)

    summary = summarize_tick_density_dataset(path)

    assert summary.index_column == "macro_minute_index"
    assert summary.band_stats["mean_tick_count"].round(4).tolist() == [20.0, 30.0]
    assert summary.band_stats["p25_tick_count"].round(4).tolist() == [15.0, 25.0]
    assert summary.band_stats["p75_tick_count"].round(4).tolist() == [25.0, 35.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_tick_density_viz.py::test_summarize_tick_density_dataset_builds_cross_day_band_stats -q`
Expected: FAIL with `ModuleNotFoundError` or missing symbol from `viz.tick_density_viz`

- [ ] **Step 3: Extend failing test for normality classification**

```python
import numpy as np


def test_summarize_tick_density_dataset_flags_mixed_bucket_normality(tmp_path: Path):
    rng = np.random.default_rng(7)
    n = 80
    normal_bucket = rng.normal(loc=100.0, scale=8.0, size=n)
    skewed_bucket = rng.exponential(scale=18.0, size=n) + 80.0
    path = tmp_path / "sample_5s.parquet"
    df = pd.DataFrame(
        {
            "datetime_utc": pd.date_range("2025-01-01", periods=n * 2, freq="5s", tz="UTC"),
            "date_utc": [f"2025-01-{(i % n) + 1:02d}" for i in range(n)] * 2,
            "bucket_index": [0] * n + [1] * n,
            "is_empty": [False] * (n * 2),
            "tick_count": np.concatenate([normal_bucket, skewed_bucket]),
            "total_size": [1] * (n * 2),
            "buy_ticks": [1] * (n * 2),
            "sell_ticks": [0] * (n * 2),
            "none_ticks": [0] * (n * 2),
        }
    )
    df.to_parquet(path, index=False)

    summary = summarize_tick_density_dataset(path)

    assert summary.normality["is_normal"].tolist() == [True, False]
    assert summary.overall_normality == "mixed"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python3 -m pytest test/test_tick_density_viz.py::test_summarize_tick_density_dataset_flags_mixed_bucket_normality -q`
Expected: FAIL because summary function is not implemented yet

### Task 2: Implement visualization module

**Files:**
- Create: `viz/tick_density_viz.py`
- Modify: `viz/tick_density_viz.py`
- Test: `test/test_tick_density_viz.py`

- [ ] **Step 1: Write minimal implementation**

```python
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


@dataclass
class TickDensitySummary:
    path: Path
    dataset_name: str
    index_column: str
    band_stats: pd.DataFrame
    normality: pd.DataFrame
    overall_normality: str
```

Add helpers to:
- detect `macro_minute_index` vs `bucket_index`
- aggregate `tick_count` by index across `date_utc` with mean/p25/p75
- run Shapiro per index on cross-day `tick_count`
- save histogram + band plot PNGs
- save per-dataset CSV summaries
- expose `summarize_tick_density_dataset()` and CLI `main()`

- [ ] **Step 2: Run tests to verify they pass**

Run: `python3 -m pytest test/test_tick_density_viz.py -q`
Expected: PASS

### Task 3: Run script on real datasets

**Files:**
- Modify: `outputs/figs/tick_density/*`

- [ ] **Step 1: Execute visualization script**

Run: `python3 viz/tick_density_viz.py`
Expected: PNG + CSV artifacts written under `outputs/figs/tick_density/`

- [ ] **Step 2: Verify output files exist**

Run: `find outputs/figs/tick_density -maxdepth 1 -type f | sort`
Expected: histogram and band PNGs for each requested parquet, plus CSV summaries
