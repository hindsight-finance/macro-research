# Containment Model-First Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small set of new containment features, formalize reproducible model-family bakeoff support for containment research, and rerun the strongest models on the expanded containment table for `1pm-3pm` and `3pm-3:50pm`.

**Architecture:** Keep the existing walk-forward regression harness intact for the core modeling path. Add new OHLCV-native containment feature builders in `target.py`, wire them into the modeling table, and create a separate `containment_research.py` module for regression/classification bakeoff workflows so research experiments do not destabilize the main CLI and registry path. Use TDD throughout: feature/unit tests first, then table integration, then research-module tests, then full verification and experiments.

**Tech Stack:** Python, NumPy, Pandas, pytest, scikit-learn, existing trend modeling table/walkforward helpers

---

### Task 1: Add failing tests for new containment expansion features

**Files:**
- Modify: `features/trend/modeling/test/test_target.py`
- Test: `features/trend/modeling/test/test_target.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_containment_expansion_features_returns_expected_columns():
    result = build_containment_expansion_features(
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        session_name="1pm-3pm",
    )

    assert {
        "containment_ib_extension_ratio",
        "containment_ib_asymmetry",
        "containment_bandwidth_squeeze",
        "containment_vwap_acceptance",
        "containment_excess_rejection",
    } <= set(result)


def test_containment_expansion_features_score_clean_rotation_above_trend():
    trend = build_containment_expansion_features(...)
    rotating = build_containment_expansion_features(...)

    assert rotating["containment_ib_extension_ratio"] < trend["containment_ib_extension_ratio"]
    assert rotating["containment_vwap_acceptance"] > trend["containment_vwap_acceptance"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`
Expected: FAIL because `build_containment_expansion_features` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_containment_expansion_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    session_name: str,
) -> dict:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/test/test_target.py features/trend/modeling/target.py
git commit -m "feat: add containment model-first feature builders"
```

### Task 2: Wire new feature columns into modeling table

**Files:**
- Modify: `features/trend/modeling/table.py`
- Modify: `features/trend/modeling/test/test_table.py`
- Test: `features/trend/modeling/test/test_table.py`

- [ ] **Step 1: Write the failing table assertions**

```python
assert {
    "containment_ib_extension_ratio",
    "containment_ib_asymmetry",
    "containment_bandwidth_squeeze",
    "containment_vwap_acceptance",
    "containment_excess_rejection",
} <= set(table.columns)

assert table[
    [
        "containment_ib_extension_ratio",
        "containment_ib_asymmetry",
        "containment_bandwidth_squeeze",
        "containment_vwap_acceptance",
        "containment_excess_rejection",
    ]
].notna().all().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py -q`
Expected: FAIL because new containment expansion columns are missing.

- [ ] **Step 3: Write minimal implementation**

```python
row.update(
    build_containment_expansion_features(
        open_=window_bars["open"].to_numpy(),
        high=window_bars["high"].to_numpy(),
        low=window_bars["low"].to_numpy(),
        close=window_bars["close"].to_numpy(),
        volume=window_bars["volume"].to_numpy() if "volume" in window_bars.columns else window_bars["Volume"].to_numpy(),
        session_name=session_name,
    )
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/table.py features/trend/modeling/test/test_table.py
git commit -m "feat: wire containment expansion features into modeling table"
```

### Task 3: Add reproducible containment research runner with failing tests first

**Files:**
- Create: `features/trend/modeling/containment_research.py`
- Create: `features/trend/modeling/test/test_containment_research.py`
- Test: `features/trend/modeling/test/test_containment_research.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_run_regression_bakeoff_returns_ranked_rows(tmp_path: Path):
    results = run_regression_bakeoff(
        table=table,
        session_name="1pm-3pm",
        feature_columns=("mss", "er", "containment_range_stability"),
    )

    assert {"model", "holdout_r2", "oos_r2"} <= set(results.columns)
    assert {"ridge", "hist_gbm", "random_forest", "extra_trees"} <= set(results["model"])


def test_run_classification_bakeoff_returns_pr_metrics(tmp_path: Path):
    results = run_classification_bakeoff(
        table=table,
        session_name="1pm-3pm",
        feature_columns=("mss", "er", "containment_range_stability"),
        target_column="containment_target",
    )

    assert {"model", "pr_auc", "precision_at_10pct", "lift_at_10pct"} <= set(results.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_containment_research.py -q`
Expected: FAIL because the module/functions do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def run_regression_bakeoff(...):
    ...
    return pd.DataFrame(rows)


def run_classification_bakeoff(...):
    ...
    return pd.DataFrame(rows)
```

```python
def summarize_top_models(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_containment_research.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/containment_research.py features/trend/modeling/test/test_containment_research.py
git commit -m "feat: add containment research bakeoff runner"
```

### Task 4: Add CLI entry for containment research runner

**Files:**
- Modify: `features/trend/modeling/cli.py`
- Modify: `features/trend/modeling/test/test_cli.py`
- Test: `features/trend/modeling/test/test_cli.py`

- [ ] **Step 1: Write the failing parser assertions**

```python
args = parser.parse_args(
    [
        "containment-research",
        "--table-path",
        "table.parquet",
        "--output-dir",
        "outputs/trend_modeling/research",
    ]
)

assert args.command == "containment-research"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_cli.py -q`
Expected: FAIL because the subcommand does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
research_parser = subparsers.add_parser("containment-research", help="Run containment regression/classification bakeoffs")
research_parser.add_argument("--table-path", required=True)
research_parser.add_argument("--output-dir", required=True)
```

```python
if args.command == "containment-research":
    return _containment_research_command(args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/cli.py features/trend/modeling/test/test_cli.py
git commit -m "feat: add containment research CLI command"
```

### Task 5: Run full verification, rebuild table, run research experiments

**Files:**
- Modify: `outputs/trend_modeling/cache/nq_trend_modeling_table_containment_v3.parquet`
- Modify: `outputs/trend_modeling/containment_model_first/...`

- [ ] **Step 1: Run modeling tests**

Run: `python3 -m pytest features/trend/modeling/test -q`
Expected: PASS

- [ ] **Step 2: Run broader trend suite**

Run: `python3 -m pytest features/trend -q`
Expected: PASS

- [ ] **Step 3: Rebuild containment table with new features**

Run:

```bash
python3 -m features.trend.modeling.cli build-table \
  --input-path /mnt/e/backup/code/finance/research/macro/outputs/nq_1m.parquet \
  --instrument NQ \
  --output-path /mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/cache/nq_trend_modeling_table_containment_v3.parquet
```

Expected: `Wrote ... rows to .../nq_trend_modeling_table_containment_v3.parquet`

- [ ] **Step 4: Run containment research bakeoff**

Run:

```bash
python3 -m features.trend.modeling.cli containment-research \
  --table-path /mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/cache/nq_trend_modeling_table_containment_v3.parquet \
  --output-dir /mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/containment_model_first
```

Expected: writes regression/classification summaries for `1pm-3pm` and `3pm-3:50pm`

- [ ] **Step 5: Compare against prior baselines**

Run:

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path
root = Path('/mnt/e/backup/code/finance/research/macro/outputs/trend_modeling/containment_model_first')
for name in ['regression_summary.csv', 'classification_summary.csv']:
    path = root / name
    print('\\n', name)
    print(pd.read_csv(path).to_string(index=False))
PY
```

Expected: at least one kept session beats current nonlinear baseline or clearly shows no incremental feature value.

- [ ] **Step 6: Commit**

```bash
git add features/trend/modeling docs/superpowers/specs/2026-04-16-containment-model-first-expansion-design.md docs/superpowers/plans/2026-04-16-containment-model-first-expansion.md
git commit -m "feat: add containment model-first research workflow"
```
