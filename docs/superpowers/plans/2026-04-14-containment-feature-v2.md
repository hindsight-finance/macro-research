# Containment Feature V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four descriptive containment-focused features to the modeling table, wire minimal containment-v2 experiment variants into the registry, and rerun quick containment experiments for `1pm-3pm` and `3pm-3:50pm`.

**Architecture:** Extend the existing modeling table builder with small OHLCV-derived descriptive feature helpers, then expose narrow registry variants that append those new columns onto the current winning containment representations. Keep the containment target, trendability target, and walk-forward harness unchanged; validate through focused unit tests first, then full modeling/trend suites, then a quick containment rerun on the two relevant sessions.

**Tech Stack:** Python, NumPy, Pandas, pytest, existing trend modeling registry/CLI/walkforward harness

---

### Task 1: Add Failing Tests For New Containment Feature Columns

**Files:**
- Modify: `features/trend/modeling/test/test_table.py`
- Test: `features/trend/modeling/test/test_table.py`

- [ ] **Step 1: Write the failing test**

```python
assert {
    "containment_overshoot_ratio",
    "containment_range_stability",
    "containment_mid_cross_count",
    "containment_swing_symmetry",
} <= set(table.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py::test_build_modeling_table_emits_one_row_per_date_and_session -q`
Expected: FAIL because the new containment-v2 columns are missing.

- [ ] **Step 3: Write minimal implementation**

```python
row.update(
    {
        "containment_overshoot_ratio": ...,
        "containment_range_stability": ...,
        "containment_mid_cross_count": ...,
        "containment_swing_symmetry": ...,
    }
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py::test_build_modeling_table_emits_one_row_per_date_and_session -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/test/test_table.py features/trend/modeling/table.py
git commit -m "feat: add containment v2 table columns"
```

### Task 2: Add Failing Unit Tests For Containment-V2 Feature Behavior

**Files:**
- Modify: `features/trend/modeling/test/test_target.py`
- Test: `features/trend/modeling/test/test_target.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_containment_features_returns_expected_columns():
    result = build_containment_features(...)
    assert {
        "containment_overshoot_ratio",
        "containment_range_stability",
        "containment_mid_cross_count",
        "containment_swing_symmetry",
    } <= set(result)


def test_containment_features_score_clean_rotation_above_trend_and_noisy_chop():
    trend = build_containment_features(...)
    rotating = build_containment_features(...)
    chop = build_containment_features(...)

    assert rotating["containment_overshoot_ratio"] < chop["containment_overshoot_ratio"]
    assert rotating["containment_range_stability"] > chop["containment_range_stability"]
    assert rotating["containment_swing_symmetry"] > chop["containment_swing_symmetry"]
    assert rotating["containment_mid_cross_count"] > trend["containment_mid_cross_count"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`
Expected: FAIL because `build_containment_features` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_containment_features(open_, high, low, close) -> dict:
    ...
    return {
        "containment_overshoot_ratio": ...,
        "containment_range_stability": ...,
        "containment_mid_cross_count": ...,
        "containment_swing_symmetry": ...,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/test/test_target.py features/trend/modeling/target.py
git commit -m "feat: add containment v2 feature calculations"
```

### Task 3: Wire Containment-V2 Features Into The Modeling Table

**Files:**
- Modify: `features/trend/modeling/table.py`
- Test: `features/trend/modeling/test/test_table.py`

- [ ] **Step 1: Write the failing integration assertion**

```python
assert set(table["feature_status"]) == {"ok"}
assert table[
    [
        "containment_overshoot_ratio",
        "containment_range_stability",
        "containment_mid_cross_count",
        "containment_swing_symmetry",
    ]
].notna().all().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py -q`
Expected: FAIL because the new columns are missing or null.

- [ ] **Step 3: Write minimal implementation**

```python
row.update(
    build_containment_features(
        open_=window_bars["open"].to_numpy(),
        high=window_bars["high"].to_numpy(),
        low=window_bars["low"].to_numpy(),
        close=window_bars["close"].to_numpy(),
    )
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/table.py features/trend/modeling/test/test_table.py
git commit -m "feat: wire containment v2 features into modeling table"
```

### Task 4: Add Narrow Registry Variants For Quick Containment V2 Experiments

**Files:**
- Modify: `features/trend/modeling/registry.py`
- Modify: `features/trend/modeling/test/test_registry.py`
- Modify: `features/trend/modeling/cli.py`
- Test: `features/trend/modeling/test/test_registry.py`

- [ ] **Step 1: Write the failing registry test**

```python
def test_build_containment_v2_registry_contains_post_winner_variants():
    registry = build_containment_v2_registry(session_name="1pm-3pm")
    ids = {experiment.experiment_id for experiment in registry}

    assert "EXP40_post_core5_dra_containment_v2" in ids
    assert "EXP41_post_adx_parts_containment_v2" in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_registry.py -q`
Expected: FAIL because the registry helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
CONTAINMENT_V2_FEATURE_COLUMNS = (
    "containment_overshoot_ratio",
    "containment_range_stability",
    "containment_mid_cross_count",
    "containment_swing_symmetry",
)

CORE5_DRA_CONTAINMENT_V2_FEATURE_COLUMNS = CORE5_DRA_FEATURE_COLUMNS + CONTAINMENT_V2_FEATURE_COLUMNS
ADX_PARTS_CONTAINMENT_V2_FEATURE_COLUMNS = ADX_PARTS_FEATURE_COLUMNS + CONTAINMENT_V2_FEATURE_COLUMNS
```

```python
def build_containment_v2_registry(session_name: str, ridge_alpha: float = 1.0) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(..., "core5_dra_containment_v2", CORE5_DRA_CONTAINMENT_V2_FEATURE_COLUMNS, ...),
        ExperimentSpec(..., "adx_parts_containment_v2", ADX_PARTS_CONTAINMENT_V2_FEATURE_COLUMNS, ...),
    ]
```

```python
if args.experiment_group == "containment_v2":
    return build_containment_v2_registry(session_name=args.session_name, ridge_alpha=args.ridge_alpha)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_registry.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/registry.py features/trend/modeling/test/test_registry.py features/trend/modeling/cli.py
git commit -m "feat: add containment v2 experiment registry"
```

### Task 5: Run Full Verification And Quick Containment Rerun

**Files:**
- Modify: `outputs/trend_modeling/cache/nq_trend_modeling_table_containment_v2.parquet`
- Modify: `outputs/trend_modeling/containment_v2_experiments/...`

- [ ] **Step 1: Run modeling tests**

Run: `python3 -m pytest features/trend/modeling/test -q`
Expected: PASS

- [ ] **Step 2: Run broader trend suite**

Run: `python3 -m pytest features/trend -q`
Expected: PASS

- [ ] **Step 3: Rebuild the containment table cache**

Run:

```bash
python3 -m features.trend.modeling.cli build-table \
  --input-path /mnt/e/backup/code/Finance/research/macro/outputs/nq_1m.parquet \
  --instrument NQ \
  --output-path /mnt/e/backup/code/Finance/research/macro/outputs/trend_modeling/cache/nq_trend_modeling_table_containment_v2.parquet
```

Expected: `Wrote ... rows to .../nq_trend_modeling_table_containment_v2.parquet`

- [ ] **Step 4: Run quick containment-v2 experiments for `1pm-3pm`**

Run:

```bash
python3 -m features.trend.modeling.cli run-experiments \
  --table-path /mnt/e/backup/code/Finance/research/macro/outputs/trend_modeling/cache/nq_trend_modeling_table_containment_v2.parquet \
  --session-name 1pm-3pm \
  --output-dir /mnt/e/backup/code/Finance/research/macro/outputs/trend_modeling/containment_v2_experiments \
  --experiment-group containment_v2 \
  --target-column containment_target
```

Expected: summary table with `EXP40_post_core5_dra_containment_v2` and `EXP41_post_adx_parts_containment_v2`

- [ ] **Step 5: Run quick containment-v2 experiments for `3pm-3:50pm`**

Run:

```bash
python3 -m features.trend.modeling.cli run-experiments \
  --table-path /mnt/e/backup/code/Finance/research/macro/outputs/trend_modeling/cache/nq_trend_modeling_table_containment_v2.parquet \
  --session-name 3pm-3:50pm \
  --output-dir /mnt/e/backup/code/Finance/research/macro/outputs/trend_modeling/containment_v2_experiments \
  --experiment-group containment_v2 \
  --target-column containment_target
```

Expected: summary table with `EXP40_post_core5_dra_containment_v2` and `EXP41_post_adx_parts_containment_v2`

- [ ] **Step 6: Commit**

```bash
git add features/trend/modeling outputs/trend_modeling/cache/nq_trend_modeling_table_containment_v2.parquet docs/superpowers/specs/2026-04-14-containment-feature-v2-design.md docs/superpowers/plans/2026-04-14-containment-feature-v2.md
git commit -m "feat: add containment v2 feature set"
```
