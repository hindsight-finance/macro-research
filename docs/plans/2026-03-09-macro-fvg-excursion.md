# Macro FVG Conditional Entry Excursion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing macro FVG study so each event tracks conditional `bar3` breakout entry triggers and post-trigger MAE/MFE through the `15:59` close, then summarize those excursion distributions by the existing research controls.

**Architecture:** Modify `features/macro_fvg_study.py` in place so the existing event parquet remains the source of truth. Add entry-trigger and excursion fields during outcome scanning, append excursion summary scopes to the current summary parquet, and add a small set of excursion figures under `outputs/figs/fvg/` without removing current outputs.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, pytest, pyarrow/parquet

---

### Task 1: Add event-level entry trigger and excursion fields

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`
- Reference: `docs/plans/2026-03-09-macro-fvg-excursion-design.md`

**Step 1: Write the failing tests for entry price, trigger detection, and percent excursions**

Add focused tests like:

```python
def test_bullish_entry_trigger_and_excursions_use_bar3_high():
    bars = make_bars([
        {
            "DateTime_ET": "2025-01-02 15:50:00",
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:51:00",
            "Open": 100.5, "High": 101.5, "Low": 100.0, "Close": 101.0,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:52:00",
            "Open": 102.0, "High": 103.0, "Low": 102.0, "Close": 102.5,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:54:00",
            "Open": 102.5, "High": 104.0, "Low": 101.0, "Close": 103.0,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:59:00",
            "Open": 103.0, "High": 105.0, "Low": 100.5, "Close": 104.0,
            "window": "MACRO",
        },
    ])

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["entry_price"] == 103.0
    assert event["entry_triggered_by_1559"]
    assert event["first_entry_trigger_at"] == pd.Timestamp("2025-01-02 15:54:00")
    assert event["entry_trigger_minute_hhmm"] == "15:54"
    assert event["entry_trigger_minute_index"] == 4
    assert event["mfe_pct_to_1559"] == pytest.approx((105.0 - 103.0) / 103.0)
    assert event["mae_pct_to_1559"] == pytest.approx((103.0 - 100.5) / 103.0)
```

Add additional focused tests for:
- bearish trigger and percent excursion math using `bar3_low`
- no-trigger case keeps `first_entry_trigger_at`, `mfe_pct_to_1559`, and `mae_pct_to_1559` null
- trigger on `15:59` marks `entry_triggered_by_1559=True` but leaves MAE/MFE null

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_bullish_entry_trigger_and_excursions_use_bar3_high \
  test/test_macro_fvg_study.py::test_bearish_entry_trigger_and_excursions_use_bar3_low \
  test/test_macro_fvg_study.py::test_entry_excursion_fields_stay_null_when_entry_never_triggers \
  test/test_macro_fvg_study.py::test_entry_trigger_on_1559_sets_trigger_without_excursions \
  -q
```

Expected:
- fail because the new entry/excursion fields do not exist yet

**Step 3: Implement the minimal trigger and excursion scan**

Add helpers in `features/macro_fvg_study.py` along the lines of:

```python
def assign_entry_price(event: pd.Series) -> float:
    if event["fvg_side"] == "bullish":
        return float(event["bar3_high"])
    return float(event["bar3_low"])


def _bar_triggers_entry(bar: pd.Series, fvg_side: str, entry_price: float) -> bool:
    if fvg_side == "bullish":
        return float(bar["High"]) >= entry_price
    return float(bar["Low"]) <= entry_price


def _calculate_excursions_from_entry(
    scan_df: pd.DataFrame,
    fvg_side: str,
    entry_price: float,
) -> tuple[float | None, float | None]:
    if scan_df.empty:
        return np.nan, np.nan

    max_high = float(scan_df["High"].max())
    min_low = float(scan_df["Low"].min())

    if fvg_side == "bullish":
        mfe_pct = (max_high - entry_price) / entry_price
        mae_pct = (entry_price - min_low) / entry_price
    else:
        mfe_pct = (entry_price - min_low) / entry_price
        mae_pct = (max_high - entry_price) / entry_price

    return float(mfe_pct), float(mae_pct)
```

Update `detect_macro_fvgs()` to retain:

```python
"bar3_high",
"bar3_low",
```

Update `scan_fvg_outcomes_until_1559_close()` so each event row stores:

```python
entry_price = assign_entry_price(event)
first_entry_trigger_at = first bar in scan_df where _bar_triggers_entry(...)
entry_triggered_by_1559 = pd.notna(first_entry_trigger_at)

if pd.isna(first_entry_trigger_at):
    mfe_pct_to_1559 = np.nan
    mae_pct_to_1559 = np.nan
else:
    post_trigger_df = day_bars[
        (day_bars["DateTime_ET"] > first_entry_trigger_at)
        & (day_bars["DateTime_ET"] <= scan_end)
    ]
    mfe_pct_to_1559, mae_pct_to_1559 = _calculate_excursions_from_entry(
        post_trigger_df,
        event_dict["fvg_side"],
        entry_price,
    )
```

Also store:

```python
event_dict["entry_trigger_minute_hhmm"] = first_entry_trigger_at.strftime("%H:%M") if pd.notna(...) else pd.NA
event_dict["entry_trigger_minute_index"] = (
    first_entry_trigger_at.hour * 60 + first_entry_trigger_at.minute - (15 * 60 + 50)
) if pd.notna(...) else pd.NA
```

For non-confirmable events, keep the new fields null / false.

**Step 4: Run the focused tests again**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_bullish_entry_trigger_and_excursions_use_bar3_high \
  test/test_macro_fvg_study.py::test_bearish_entry_trigger_and_excursions_use_bar3_low \
  test/test_macro_fvg_study.py::test_entry_excursion_fields_stay_null_when_entry_never_triggers \
  test/test_macro_fvg_study.py::test_entry_trigger_on_1559_sets_trigger_without_excursions \
  -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg entry excursions"
```

### Task 2: Add excursion summary builders and summary schema

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write the failing tests for excursion summary scopes**

Add tests like:

```python
def test_builds_entry_excursion_alignment_bucket_summary():
    events = pd.DataFrame([
        {
            "alignment_bucket": "3_aligned",
            "minute_block": "15:50-15:52",
            "gap_size_bucket_225": ">=2.25",
            "is_confirmable_by_1559": True,
            "entry_triggered_by_1559": True,
            "mfe_pct_to_1559": 0.012,
            "mae_pct_to_1559": 0.004,
        },
        {
            "alignment_bucket": "3_aligned",
            "minute_block": "15:50-15:52",
            "gap_size_bucket_225": ">=2.25",
            "is_confirmable_by_1559": True,
            "entry_triggered_by_1559": False,
            "mfe_pct_to_1559": np.nan,
            "mae_pct_to_1559": np.nan,
        },
    ])

    summary = build_entry_excursion_alignment_bucket_summary(events)

    row = summary.iloc[0]
    assert row["alignment_bucket"] == "3_aligned"
    assert row["n_confirmable"] == 2
    assert row["n_triggered"] == 1
    assert row["entry_trigger_rate"] == 0.5
    assert row["mfe_pct_mean"] == 0.012
    assert row["mae_pct_mean"] == 0.004
```

Add a second test for:
- `build_entry_excursion_alignment_bucket_minute_block_summary()`

Add a third test for:
- `build_entry_excursion_gap_bucket_summary()`

**Step 2: Run the focused summary tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_builds_entry_excursion_alignment_bucket_summary \
  test/test_macro_fvg_study.py::test_builds_entry_excursion_alignment_bucket_minute_block_summary \
  test/test_macro_fvg_study.py::test_builds_entry_excursion_gap_bucket_summary \
  -q
```

Expected:
- fail because the excursion summary builders do not exist yet

**Step 3: Implement excursion summary helpers**

Extend the summary schema in `features/macro_fvg_study.py` with nullable columns:

```python
"n_triggered",
"entry_trigger_rate",
"mfe_pct_mean",
"mfe_pct_median",
"mfe_pct_p75",
"mfe_pct_p90",
"mae_pct_mean",
"mae_pct_median",
"mae_pct_p75",
"mae_pct_p90",
```

Add a helper like:

```python
def _group_entry_excursion_stats(
    events: pd.DataFrame,
    group_cols: list[str],
    scope_name: str,
) -> pd.DataFrame:
    ...
```

Recommended logic:
- `n_confirmable = sum(is_confirmable_by_1559)`
- `n_triggered = sum(entry_triggered_by_1559)`
- `entry_trigger_rate = n_triggered / n_confirmable`
- compute MAE/MFE aggregates from triggered rows only
- leave MAE/MFE aggregates null when `n_triggered == 0`

Add builders:

```python
def build_entry_excursion_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_entry_excursion_stats(events, [], "entry_excursion_overall")


def build_entry_excursion_alignment_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_entry_excursion_stats(
        events,
        ["alignment_bucket"],
        "entry_excursion_alignment_bucket",
    )


def build_entry_excursion_minute_block_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_entry_excursion_stats(
        events,
        ["minute_block"],
        "entry_excursion_minute_block",
    )


def build_entry_excursion_gap_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_entry_excursion_stats(
        events,
        ["gap_size_bucket_225"],
        "entry_excursion_gap_bucket",
    )


def build_entry_excursion_alignment_bucket_minute_block_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    return _group_entry_excursion_stats(
        events,
        ["alignment_bucket", "minute_block"],
        "entry_excursion_alignment_bucket_minute_block",
    )
```

Append these frames to the existing combined summary output.

**Step 4: Run the focused summary tests again**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_builds_entry_excursion_alignment_bucket_summary \
  test/test_macro_fvg_study.py::test_builds_entry_excursion_alignment_bucket_minute_block_summary \
  test/test_macro_fvg_study.py::test_builds_entry_excursion_gap_bucket_summary \
  -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg excursion summaries"
```

### Task 3: Add first-pass excursion figures and extend the smoke test

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write failing smoke assertions for the new excursion figures**

Extend the end-to-end smoke test so it also checks for:

```python
assert (figures_dir / "entry_trigger_rate_by_alignment_bucket.png").exists()
assert (figures_dir / "mfe_mae_pct_by_alignment_bucket.png").exists()
assert (figures_dir / "mfe_pct_by_minute_block.png").exists()
assert (figures_dir / "mfe_pct_by_gap_bucket.png").exists()
```

**Step 2: Run the smoke test to verify it fails**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures -q
```

Expected:
- fail because the new excursion figure files are not created yet

**Step 3: Implement the new plotting branches**

Add or extend plotting in `features/macro_fvg_study.py`:

```python
def plot_entry_trigger_rate_by_alignment_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_mfe_mae_pct_by_alignment_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_mfe_pct_by_minute_block(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_mfe_pct_by_gap_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...
```

Recommended content:
- `entry_trigger_rate_by_alignment_bucket.png`
  - bar chart of `entry_trigger_rate` by `alignment_bucket`
- `mfe_mae_pct_by_alignment_bucket.png`
  - grouped bars of `mfe_pct_mean` and `mae_pct_mean` by `alignment_bucket`
- `mfe_pct_by_minute_block.png`
  - grouped bars of `mfe_pct_mean` by `minute_block`
- `mfe_pct_by_gap_bucket.png`
  - grouped bars of `mfe_pct_mean` by `gap_size_bucket_225`

Update `plot_fvg_summary_figures()` to call the four new plotters and generate placeholder figures if the event table is empty.

**Step 4: Run the smoke test again**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg excursion figures"
```

### Task 4: Run focused verification and inspect real outputs

**Files:**
- Modify: `features/macro_fvg_study.py` if cleanup is needed after the real-data run

**Step 1: Run the full focused test file**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- all tests pass

**Step 2: Run the study on the real tagged data**

Run:

```bash
python3 features/macro_fvg_study.py
```

Expected:
- updates `outputs/nq_macro_fvg_events.parquet`
- updates `outputs/nq_macro_fvg_summary.parquet`
- writes the new excursion figures under `outputs/figs/fvg/`

**Step 3: Inspect the new columns and outputs**

Run:

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

events = pd.read_parquet("outputs/nq_macro_fvg_events.parquet")
summary = pd.read_parquet("outputs/nq_macro_fvg_summary.parquet")
figures_dir = Path("outputs/figs/fvg")

print(events[
    [
        "entry_price",
        "entry_triggered_by_1559",
        "first_entry_trigger_at",
        "entry_trigger_minute_hhmm",
        "entry_trigger_minute_index",
        "mfe_pct_to_1559",
        "mae_pct_to_1559",
    ]
].head())
print(summary[summary["summary_scope"].str.contains("entry_excursion", na=False)].head())
print("triggered_confirmable_rows", int(events["entry_triggered_by_1559"].fillna(False).sum()))
for filename in [
    "entry_trigger_rate_by_alignment_bucket.png",
    "mfe_mae_pct_by_alignment_bucket.png",
    "mfe_pct_by_minute_block.png",
    "mfe_pct_by_gap_bucket.png",
]:
    path = figures_dir / filename
    print(filename, path.exists(), path.stat().st_size if path.exists() else 0)
PY
```

Expected:
- event parquet contains the new entry-excursion fields
- summary parquet contains the new entry-excursion scopes
- the new figure files are present under `outputs/figs/fvg/`

**Step 4: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: extend macro fvg study with entry excursions"
```
