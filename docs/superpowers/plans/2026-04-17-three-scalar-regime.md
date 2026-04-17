# Three-Scalar Regime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a realized `chop_score`, expose explicit scalar aliases (`trend_score`, `containment_score`), and wire the new chop diagnostics into the trend modeling table without changing the existing frozen trend or containment targets.

**Architecture:** Keep the current target builders intact, add a new `build_chop_target(...)` helper in `features/trend/modeling/target.py`, and alias the existing target columns inside `features/trend/modeling/table.py`. The walk-forward harness already accepts arbitrary `target_column` values, so this round only needs target/table/test wiring plus a small package-export update. Use TDD: failing target tests first, then implementation, then failing table/export tests, then integration and verification.

**Tech Stack:** Python, NumPy, Pandas, pytest, existing trend modeling target/table/walkforward helpers

---

### Task 1: Add failing chop-target tests

**Files:**
- Modify: `features/trend/modeling/test/test_target.py`
- Test: `features/trend/modeling/test/test_target.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_chop_target_returns_bounded_components():
    open_, high, low, close = _make_ohlc([100.0, 101.7, 99.0, 101.5, 98.8, 101.4, 99.9])

    result = build_chop_target(open_=open_, high=high, low=low, close=close)

    assert {
        "chop_flip_rate",
        "chop_path_waste",
        "chop_outside_share",
        "chop_instability",
        "chop_score",
        "chop_status",
    } <= set(result)
    assert 0.0 <= result["chop_score"] <= 1.0


def test_chop_target_scores_ugly_chop_above_rotation_and_trend():
    trend_open, trend_high, trend_low, trend_close = _make_ohlc([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    rotating_open, rotating_high, rotating_low, rotating_close = _make_ohlc([100.0, 101.6, 100.9, 99.3, 100.0, 101.1, 100.2])
    chop_open, chop_high, chop_low, chop_close = _make_ohlc([100.0, 101.7, 99.0, 101.5, 98.8, 101.4, 99.9])

    trend = build_chop_target(
        open_=trend_open,
        high=trend_high,
        low=trend_low,
        close=trend_close,
    )
    rotating = build_chop_target(
        open_=rotating_open,
        high=rotating_high,
        low=rotating_low,
        close=rotating_close,
    )
    chop = build_chop_target(
        open_=chop_open,
        high=chop_high,
        low=chop_low,
        close=chop_close,
    )

    assert chop["chop_score"] > rotating["chop_score"]
    assert chop["chop_score"] > trend["chop_score"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`
Expected: FAIL because `build_chop_target` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_chop_target(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict:
    _, high_arr, low_arr, close_arr = _coerce_ohlc_arrays(
        open_=open_,
        high=high,
        low=low,
        close=close,
    )
    realized_range, close_pos = _compute_realized_range_and_close_pos(
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
    )
    returns = np.diff(np.log(close_arr))
    nonzero_returns = returns[returns != 0]
    sign_changes = (
        np.count_nonzero(np.sign(nonzero_returns[1:]) != np.sign(nonzero_returns[:-1]))
        if nonzero_returns.size > 1
        else 0
    )
    flip_rate = float(sign_changes / max(nonzero_returns.size - 1, 1))
    path_length = float(np.sum(np.abs(np.diff(close_arr))))
    path_waste = float(np.clip(path_length / (realized_range + 1e-12), 0.0, 4.0) / 4.0)
    outside_share = float(np.mean((close_pos < 0.05) | (close_pos > 0.95)))
    block_indexes = [
        block
        for block in np.array_split(np.arange(close_arr.size), min(4, close_arr.size))
        if block.size
    ]
    block_ranges = [
        float(high_arr[idx].max() - low_arr[idx].min()) / (realized_range + 1e-12)
        for idx in block_indexes
    ]
    instability = float(np.clip(np.std(block_ranges, ddof=0), 0.0, 1.0))
    chop_score = float(
        np.clip(
            0.35 * path_waste
            + 0.30 * flip_rate
            + 0.20 * outside_share
            + 0.15 * instability,
            0.0,
            1.0,
        )
    )
    return {
        "chop_flip_rate": flip_rate,
        "chop_path_waste": path_waste,
        "chop_outside_share": outside_share,
        "chop_instability": instability,
        "chop_score": chop_score,
        "chop_status": "ok",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/test/test_target.py features/trend/modeling/target.py
git commit -m "feat: add realized chop target"
```

### Task 2: Add failing scalar-alias and chop-column table tests

**Files:**
- Modify: `features/trend/modeling/test/test_table.py`
- Modify: `features/trend/modeling/test/test_init.py`
- Test: `features/trend/modeling/test/test_table.py`
- Test: `features/trend/modeling/test/test_init.py`

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

assert table["trend_score"].equals(table["descriptive_target"])
assert table["containment_score"].equals(table["containment_target"])
assert table[
    ["chop_flip_rate", "chop_path_waste", "chop_outside_share", "chop_instability", "chop_score"]
].notna().all().all()
assert set(table["chop_status"]) == {"ok"}
```

```python
assert callable(modeling.build_chop_target)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py features/trend/modeling/test/test_init.py -q`
Expected: FAIL because the alias columns, chop columns, or export are missing.

- [ ] **Step 3: Write minimal implementation**

```python
row["trend_score"] = row["descriptive_target"]
row["containment_score"] = row["containment_target"]
row.update(
    build_chop_target(
        open_=window_bars["open"].to_numpy(),
        high=window_bars["high"].to_numpy(),
        low=window_bars["low"].to_numpy(),
        close=window_bars["close"].to_numpy(),
    )
)
```

```python
from .target import build_chop_target, build_descriptive_target
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py features/trend/modeling/test/test_init.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/test/test_table.py features/trend/modeling/test/test_init.py features/trend/modeling/table.py features/trend/modeling/__init__.py
git commit -m "feat: add three-scalar regime columns to modeling table"
```

### Task 3: Run focused verification for trend modeling

**Files:**
- Modify: `features/trend/modeling/target.py`
- Modify: `features/trend/modeling/table.py`
- Modify: `features/trend/modeling/__init__.py`
- Test: `features/trend/modeling/test/test_target.py`
- Test: `features/trend/modeling/test/test_table.py`
- Test: `features/trend/modeling/test/test_init.py`
- Test: `features/trend/modeling/test/test_walkforward.py`

- [ ] **Step 1: Run focused modeling verification**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py features/trend/modeling/test/test_table.py features/trend/modeling/test/test_init.py features/trend/modeling/test/test_walkforward.py -q`
Expected: PASS with the new chop target, aliases, and existing generic walk-forward path still green.

- [ ] **Step 2: Run broader trend verification**

Run: `python3 -m pytest features/trend -q`
Expected: PASS, proving the new target/table wiring does not regress the wider trend package.

- [ ] **Step 3: Inspect diff**

Run: `git status --short && git diff -- features/trend/modeling/target.py features/trend/modeling/table.py features/trend/modeling/__init__.py features/trend/modeling/test/test_target.py features/trend/modeling/test/test_table.py features/trend/modeling/test/test_init.py`
Expected: Only the planned three-scalar regime changes appear.

- [ ] **Step 4: Commit**

```bash
git add features/trend/modeling/target.py features/trend/modeling/table.py features/trend/modeling/__init__.py features/trend/modeling/test/test_target.py features/trend/modeling/test/test_table.py features/trend/modeling/test/test_init.py
git commit -m "feat: add three-scalar regime modeling targets"
```
