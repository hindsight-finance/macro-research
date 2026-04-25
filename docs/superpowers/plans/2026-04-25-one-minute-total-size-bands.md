# One-Minute Total Size Bands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second 1-minute macro band chart for `total_size` in `viz/tick_density_viz.py` with mean / p25 / p75 across trading days by `macro_minute_index`.

**Architecture:** Keep the existing tick-count path intact. Add a small band-stat helper usable for the new `total_size` chart and CSV, then keep output writing explicit for the 1-minute dataset. Verify with pytest, then regenerate the visualization artifacts.

**Tech Stack:** Python 3, pandas, matplotlib, scipy, pytest

---

### Task 1: Add failing tests for total-size minute bands

**Files:**
- Modify: `test/test_tick_density_viz.py`
- Test: `test/test_tick_density_viz.py`

- [ ] **Step 1: Write the failing test**

```python
def test_summarize_metric_by_index_supports_total_size(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame(...).to_parquet(path, index=False)
    summary = summarize_tick_density_dataset(path)
    stats_df = summarize_metric_by_index(summary.raw_frame, summary.index_column, "total_size")
    assert stats_df["mean_total_size"].tolist()[:2] == [20.0, 30.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_tick_density_viz.py::test_summarize_metric_by_index_supports_total_size -q`
Expected: FAIL with missing helper or missing total-size output.

- [ ] **Step 3: Extend failing artifact test**

```python
assert (out_dir / "sample_total_size_bands.png").exists()
assert (out_dir / "sample_total_size_band_stats.csv").exists()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python3 -m pytest test/test_tick_density_viz.py::test_process_dataset_writes_band_outputs_only -q`
Expected: FAIL because total-size artifacts do not exist yet.

### Task 2: Implement total-size band flow

**Files:**
- Modify: `viz/tick_density_viz.py`
- Test: `test/test_tick_density_viz.py`

- [ ] **Step 1: Write minimal implementation**

```python
def summarize_metric_by_index(...):
    ...

if summary.dataset_name == "nq_macro_tick_density":
    total_size_stats = summarize_metric_by_index(raw_df, summary.index_column, "total_size")
    _plot_metric_bands(...)
    total_size_stats.to_csv(...)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python3 -m pytest test/test_tick_density_viz.py -q`
Expected: PASS

### Task 3: Regenerate outputs

**Files:**
- Modify: `outputs/figs/tick_density/*`

- [ ] **Step 1: Execute visualization script**

Run: `python3 viz/tick_density_viz.py`
Expected: `nq_macro_tick_density_total_size_bands.png` and CSV written.

- [ ] **Step 2: Verify output files exist**

Run: `find outputs/figs/tick_density -maxdepth 1 -type f | sort`
Expected: includes total-size PNG and CSV alongside existing band outputs.
