# Three-Scalar Regime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit `trend_score`, `containment_score`, and new `chop_score` outputs to the modeling table and enable three-scalar regime research without redesigning the frozen trend target or current containment target.

**Architecture:** Preserve existing scalar logic by aliasing `descriptive_target` to `trend_score` and `containment_target` to `containment_score`, then add a new `build_chop_target(...)` realized-window target in `target.py`. Wire the new scalar aliases and chop components into `table.py`, then extend the research runner so three-way labels can be derived from the three independent scalars for multiclass experiments.

**Tech Stack:** Python, NumPy, Pandas, pytest, existing trend modeling table and research runner

---

### Task 1: Add failing chop-target unit tests

**Files:**
- Modify: `features/trend/modeling/test/test_target.py`
- Test: `features/trend/modeling/test/test_target.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_chop_target_returns_bounded_components():
    result = build_chop_target(...)
    assert {
        "chop_flip_rate",
        "chop_path_waste",
        "chop_outside_share",
        "chop_instability",
        "chop_score",
        "chop_status",
    } <= set(result)


def test_chop_target_scores_ugly_chop_above_clean_rotation_and_trend():
    trend = build_chop_target(...)
    rotating = build_chop_target(...)
    chop = build_chop_target(...)

    assert chop["chop_score"] > rotating["chop_score"]
    assert chop["chop_score"] > trend["chop_score"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`
Expected: FAIL because `build_chop_target` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_chop_target(open_, high, low, close) -> dict:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/test/test_target.py features/trend/modeling/target.py
git commit -m "feat: add chop scalar target"
```

### Task 2: Add failing table tests for scalar aliases and chop columns

**Files:**
- Modify: `features/trend/modeling/test/test_table.py`
- Test: `features/trend/modeling/test/test_table.py`

- [ ] **Step 1: Write the failing assertions**

```python
assert {
    "trend_score",
    "containment_score",
    "chop_flip_rate",
    "chop_path_waste",
    "chop_outside_share",
    "chop_instability",
    "chop_score",
    "chop_status",
} <= set(table.columns)
```

```python
assert set(table["chop_status"]) == {"ok"}
assert table[["trend_score", "containment_score", "chop_score"]].notna().all().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py -q`
Expected: FAIL because scalar aliases and chop columns are missing.

- [ ] **Step 3: Write minimal implementation**

```python
row["trend_score"] = row["descriptive_target"]
row["containment_score"] = row["containment_target"]
row.update(build_chop_target(...))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/test/test_table.py features/trend/modeling/table.py
git commit -m "feat: add three-scalar regime outputs"
```

### Task 3: Add failing research-runner tests for three-scalar label derivation

**Files:**
- Modify: `features/trend/modeling/test/test_containment_research.py`
- Modify: `features/trend/modeling/containment_research.py`
- Test: `features/trend/modeling/test/test_containment_research.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_assign_three_scalar_labels_returns_three_classes():
    labeled = assign_three_scalar_labels(...)
    assert {"trend", "containment", "chop"} <= set(labeled["label"])
```

```python
def test_run_three_way_probe_returns_macro_metrics():
    results = run_three_way_probe(...)
    assert {"model", "macro_f1", "balanced_accuracy", "confusion_matrix"} <= set(results.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_containment_research.py -q`
Expected: FAIL because the three-scalar helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def assign_three_scalar_labels(...):
    ...


def run_three_way_probe(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_containment_research.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/test/test_containment_research.py features/trend/modeling/containment_research.py
git commit -m "feat: add three-scalar regime research helpers"
```

### Task 4: Run full verification

**Files:**
- Modify: none

- [ ] **Step 1: Run modeling tests**

Run: `python3 -m pytest features/trend/modeling/test -q`
Expected: PASS

- [ ] **Step 2: Run broader trend suite**

Run: `python3 -m pytest features/trend -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add features/trend/modeling docs/superpowers/specs/2026-04-17-three-scalar-regime-design.md docs/superpowers/plans/2026-04-17-three-scalar-regime.md
git commit -m "feat: add three-scalar regime research support"
```

### Task 5: Rebuild table and run three-scalar research outputs

**Files:**
- Modify: `outputs/trend_modeling/cache/nq_trend_modeling_table_regime_3scalar.parquet`
- Modify: `outputs/trend_modeling/regime_3scalar/...`

- [ ] **Step 1: Rebuild table**

Run:

```bash
python3 -m features.trend.modeling.cli build-table \
  --input-path /mnt/e/backup/code/finance/research/macro/outputs/nq_1m.parquet \
  --instrument NQ \
  --output-path /mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/cache/nq_trend_modeling_table_regime_3scalar.parquet
```

Expected: `Wrote ... rows to .../nq_trend_modeling_table_regime_3scalar.parquet`

- [ ] **Step 2: Run existing containment research runner against rebuilt table**

Run:

```bash
python3 -m features.trend.modeling.cli containment-research \
  --table-path /mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/cache/nq_trend_modeling_table_regime_3scalar.parquet \
  --output-dir /mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/regime_3scalar
```

Expected: updated scalar research summaries written under `regime_3scalar/`

- [ ] **Step 3: Run direct three-way probe**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
from features.trend.modeling.containment_research import load_post_covid_table, run_three_way_probe

table = load_post_covid_table('/mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/cache/nq_trend_modeling_table_regime_3scalar.parquet')
results = run_three_way_probe(table=table)
out = Path('/mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/regime_3scalar/three_way_probe.csv')
out.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(out, index=False)
print(results.to_string(index=False))
PY
```

Expected: macro-F1 / balanced accuracy / confusion results for `trend`, `containment`, `chop`

- [ ] **Step 4: Commit**

```bash
git add outputs/trend_modeling/cache/nq_trend_modeling_table_regime_3scalar.parquet outputs/trend_modeling/regime_3scalar
git commit -m "feat: generate three-scalar regime research outputs"
```
