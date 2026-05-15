# Macro VWAP Barrier Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a VWAP/barrier context study that compares 15:50 first-10-second barrier behavior with macro-open VWAP context, post-barrier wrong-side VWAP behavior, and 15:55 decision context, plus distribution summaries and visualization outputs.

**Architecture:** Add one feature module for raw/event + summary parquet generation and one viz module for distribution CSVs/figures. The feature module joins existing barrier/VWAP daily outputs with bounded 15:50-16:00 tick-derived metrics. Tick processing must use bounded lazy scans or PyArrow batch streaming and never eager-read the full tick file.

**Tech Stack:** Python, Polars, PyArrow parquet metadata/batch reading, Matplotlib Agg, pytest, project `.venv`.

---

## File Structure

- Create: `features/macro_vwap_barrier_context.py`
  - Schema constants.
  - Input validation.
  - Direction-aware constructive/wrong-side helpers.
  - Bounded tick metric builder for 15:50-16:00 ET.
  - Daily raw context builder.
  - Summary builder.
  - Writer and CLI entrypoint.
- Create: `viz/macro_vwap_barrier_context_viz.py`
  - Read raw + summary parquet.
  - Write CSV distribution tables.
  - Write Matplotlib Agg figures.
- Create: `test/test_macro_vwap_barrier_context.py`
  - Synthetic barrier/VWAP/tick fixtures.
  - Unit tests for calculations and writer.
- Create: `test/test_macro_vwap_barrier_context_viz.py`
  - Tiny fixture raw/summary frames.
  - Non-visual output existence/schema checks.

---

### Task 1: Feature module shell and failing tests

**Files:**
- Create: `features/macro_vwap_barrier_context.py`
- Create: `test/test_macro_vwap_barrier_context.py`

- [ ] **Step 1: Create failing tests**

Create `test/test_macro_vwap_barrier_context.py`:

```python
from pathlib import Path

import polars as pl
import pytest

from features.macro_vwap_barrier_context import (
    MACRO_VWAP_BARRIER_CONTEXT_COLUMNS,
    MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS,
    build_macro_vwap_barrier_context,
    classify_constructive_side,
    summarize_macro_vwap_barrier_context,
    write_macro_vwap_barrier_context,
)


def _tick(ts: str, price: float, size: int = 1, rank: int = 0) -> dict:
    return {"ts_event": ts, "intra_ts_rank": rank, "price_ticks": int(price * 4), "size": size}


def _write_ticks(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", time_zone="UTC")),
        pl.col("intra_ts_rank").cast(pl.Int64),
        pl.col("price_ticks").cast(pl.Int64),
        pl.col("size").cast(pl.Int64),
    ).write_parquet(path)


def _barrier_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "macro_trend_state": ["bullish", "bearish"],
            "barrier_extreme": ["low", "high"],
            "barrier_price": [99.0, 201.0],
            "barrier_time": [5, 8],
            "barrier_first10": [True, True],
            "barrier_is_macro_extreme": [True, False],
            "barrier_holds": [True, False],
            "edge_case": [False, True],
        }
    ).with_columns(pl.col("date").str.to_date())


def _vwap_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "macro_1550_at_1550_10s_vwap_side": ["above", "below"],
            "macro_1550_at_1550_10s_vwap_dist_points": [1.0, -1.25],
            "macro_1550_at_1550_10s_vwap_dist_bps": [10.0, -12.0],
            "macro_1550_at_1555_vwap_side": ["above", "above"],
            "macro_1550_at_1555_vwap_dist_points": [2.0, 3.0],
            "macro_1550_at_1555_vwap_dist_bps": [20.0, 30.0],
            "target_1550_1554_points": [4.0, -3.0],
            "target_1550_1554_sign": [1, -1],
            "target_1550_1554_state": ["bullish", "bearish"],
            "target_1555_1559_points": [3.0, -4.0],
            "target_1555_1559_sign": [1, -1],
            "target_1555_1559_state": ["bullish", "bearish"],
            "target_1550_1559_points": [7.0, -7.0],
            "target_1550_1559_sign": [1, -1],
            "target_1550_1559_state": ["bullish", "bearish"],
        }
    ).with_columns(pl.col("date").str.to_date())


def test_classify_constructive_side_is_direction_aware():
    assert classify_constructive_side("bullish", "above") == "constructive"
    assert classify_constructive_side("bullish", "touch") == "touch"
    assert classify_constructive_side("bullish", "below") == "wrong"
    assert classify_constructive_side("bearish", "below") == "constructive"
    assert classify_constructive_side("bearish", "above") == "wrong"
    assert classify_constructive_side("neutral", "above") == "unknown"
    assert classify_constructive_side("bullish", None) == "unknown"


def test_build_context_adds_vwap_barrier_and_wrong_side_metrics(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:05Z", 99.0, 1),
            _tick("2025-01-02T20:50:10Z", 101.0, 1),
            _tick("2025-01-02T20:50:30Z", 98.0, 1),
            _tick("2025-01-02T20:50:59Z", 100.0, 1),
            _tick("2025-01-02T20:54:59Z", 104.0, 1),
            _tick("2025-01-02T20:55:00Z", 105.0, 1),
            _tick("2025-01-02T20:59:59Z", 108.0, 1),
            _tick("2025-01-03T20:50:00Z", 200.0, 1),
            _tick("2025-01-03T20:50:08Z", 201.0, 1),
            _tick("2025-01-03T20:50:20Z", 203.0, 1),
            _tick("2025-01-03T20:50:59Z", 202.0, 1),
            _tick("2025-01-03T20:54:59Z", 198.0, 1),
            _tick("2025-01-03T20:55:00Z", 197.0, 1),
            _tick("2025-01-03T20:59:59Z", 193.0, 1),
        ],
    )

    out = build_macro_vwap_barrier_context(_barrier_frame(), _vwap_frame(), tick_path)

    assert out.columns == MACRO_VWAP_BARRIER_CONTEXT_COLUMNS
    bull = out.filter(pl.col("date") == pl.date(2025, 1, 2)).row(0, named=True)
    assert bull["vwap_10s_constructive"] == "constructive"
    assert bull["barrier_first10_and_vwap_constructive"] is True
    assert bull["closed_wrong_side_1550"] is True
    assert bull["closed_wrong_side_more_than_1tick"] is True
    assert bull["worst_wrong_side_dist_points"] > 0
    assert bull["wrong_side_share_1550"] > 0
    assert bull["target_1550_10s_1554_points"] == 3.0
    assert bull["target_1550_10s_1559_points"] == 7.0
    assert bull["target_1551_1559_points"] == 7.0

    bear = out.filter(pl.col("date") == pl.date(2025, 1, 3)).row(0, named=True)
    assert bear["vwap_10s_constructive"] == "constructive"
    assert bear["vwap_1555_constructive"] == "wrong"
    assert bear["vwap_context_10s_to_1555"] == "constructive_to_wrong"


def test_summarize_context_reports_distribution_scopes(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    rows = []
    for day in range(1, 13):
        date = f"2025-01-{day:02d}"
        rows.extend([
            _tick(f"{date}T20:50:00Z", 100.0 + day, 1),
            _tick(f"{date}T20:50:05Z", 99.0 + day, 1),
            _tick(f"{date}T20:50:10Z", 101.0 + day, 1),
            _tick(f"{date}T20:50:59Z", 100.0 + day, 1),
            _tick(f"{date}T20:54:59Z", 102.0 + day, 1),
            _tick(f"{date}T20:55:00Z", 103.0 + day, 1),
            _tick(f"{date}T20:59:59Z", 104.0 + day, 1),
        ])
    _write_ticks(tick_path, rows)
    barrier = pl.DataFrame(
        {
            "date": [f"2025-01-{day:02d}" for day in range(1, 13)],
            "macro_trend_state": ["bullish"] * 12,
            "barrier_extreme": ["low"] * 12,
            "barrier_price": [99.0] * 12,
            "barrier_time": [5] * 12,
            "barrier_first10": [True, False] * 6,
            "barrier_is_macro_extreme": [True] * 12,
            "barrier_holds": [True, True, False] * 4,
            "edge_case": [False] * 12,
        }
    ).with_columns(pl.col("date").str.to_date())
    vwap = pl.DataFrame(
        {
            "date": [f"2025-01-{day:02d}" for day in range(1, 13)],
            "macro_1550_at_1550_10s_vwap_side": ["above", "below", "touch"] * 4,
            "macro_1550_at_1550_10s_vwap_dist_points": [1.0, -1.0, 0.0] * 4,
            "macro_1550_at_1550_10s_vwap_dist_bps": [10.0, -10.0, 0.0] * 4,
            "macro_1550_at_1555_vwap_side": ["above", "below"] * 6,
            "macro_1550_at_1555_vwap_dist_points": [2.0, -2.0] * 6,
            "macro_1550_at_1555_vwap_dist_bps": [20.0, -20.0] * 6,
            "target_1550_1554_points": list(range(12)),
            "target_1550_1554_sign": [1] * 12,
            "target_1550_1554_state": ["bullish"] * 12,
            "target_1555_1559_points": list(range(12)),
            "target_1555_1559_sign": [1] * 12,
            "target_1555_1559_state": ["bullish"] * 12,
            "target_1550_1559_points": list(range(12)),
            "target_1550_1559_sign": [1] * 12,
            "target_1550_1559_state": ["bullish"] * 12,
        }
    ).with_columns(pl.col("date").str.to_date())

    context = build_macro_vwap_barrier_context(barrier, vwap, tick_path)
    summary = summarize_macro_vwap_barrier_context(context)

    assert summary.columns == MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS
    assert summary.filter(pl.col("scope") == "barrier_only").height > 0
    assert summary.filter(pl.col("scope") == "vwap_10s_only").height > 0
    assert summary.filter(pl.col("scope") == "wrong_side_close_bucket").height > 0
    assert summary.filter(pl.col("scope") == "wrong_side_share_decile").height == 40
    assert summary.filter(pl.col("scope") == "vwap_1555_decision").height > 0


def test_write_macro_vwap_barrier_context_persists_outputs(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    barrier_path = tmp_path / "barrier.parquet"
    vwap_path = tmp_path / "vwap.parquet"
    output_path = tmp_path / "context.parquet"
    summary_path = tmp_path / "summary.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:05Z", 99.0, 1),
            _tick("2025-01-02T20:50:10Z", 101.0, 1),
            _tick("2025-01-02T20:54:59Z", 104.0, 1),
            _tick("2025-01-02T20:55:00Z", 105.0, 1),
            _tick("2025-01-02T20:59:59Z", 108.0, 1),
        ],
    )
    _barrier_frame().filter(pl.col("date") == pl.date(2025, 1, 2)).write_parquet(barrier_path)
    _vwap_frame().filter(pl.col("date") == pl.date(2025, 1, 2)).write_parquet(vwap_path)

    wrote = write_macro_vwap_barrier_context(tick_path, barrier_path, vwap_path, output_path, summary_path)

    assert wrote == (output_path, summary_path)
    assert pl.read_parquet(output_path).columns == MACRO_VWAP_BARRIER_CONTEXT_COLUMNS
    assert pl.read_parquet(summary_path).columns == MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'features.macro_vwap_barrier_context'`.

- [ ] **Step 3: Add feature module shell**

Create `features/macro_vwap_barrier_context.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR, get_tick_schema

TICK_INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
BARRIER_INPUT_PATH = Path("outputs/nq_macro_1550_barrier.parquet")
VWAP_INPUT_PATH = Path("outputs/nq_macro_vwap_intramacro.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context_summary.parquet")
UTC_NS = pl.Datetime("ns", time_zone="UTC")

TICK_REQUIRED_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks", "size"}
BARRIER_REQUIRED_COLUMNS = {
    "date", "macro_trend_state", "barrier_extreme", "barrier_price", "barrier_time",
    "barrier_first10", "barrier_is_macro_extreme", "barrier_holds", "edge_case",
}
VWAP_REQUIRED_COLUMNS = {
    "date",
    "macro_1550_at_1550_10s_vwap_side",
    "macro_1550_at_1550_10s_vwap_dist_points",
    "macro_1550_at_1550_10s_vwap_dist_bps",
    "macro_1550_at_1555_vwap_side",
    "macro_1550_at_1555_vwap_dist_points",
    "macro_1550_at_1555_vwap_dist_bps",
    "target_1550_1554_points", "target_1550_1554_sign", "target_1550_1554_state",
    "target_1555_1559_points", "target_1555_1559_sign", "target_1555_1559_state",
    "target_1550_1559_points", "target_1550_1559_sign", "target_1550_1559_state",
}

TARGET_PREFIXES = (
    "target_1550_10s_1554",
    "target_1550_10s_1559",
    "target_1551_1559",
    "target_1555_1559",
)

MACRO_VWAP_BARRIER_CONTEXT_COLUMNS = [
    "date",
    "macro_trend_state",
    "barrier_extreme",
    "barrier_price",
    "barrier_time",
    "barrier_first10",
    "barrier_is_macro_extreme",
    "barrier_holds",
    "edge_case",
    "vwap_10s_side",
    "vwap_10s_dist_points",
    "vwap_10s_dist_bps",
    "vwap_10s_constructive",
    "barrier_first10_and_vwap_constructive",
    "barrier_ts_utc",
    "post_barrier_tick_count_1550",
    "vwap_side_at_barrier",
    "vwap_dist_at_barrier_points",
    "vwap_side_at_1550_close",
    "vwap_dist_at_1550_close_points",
    "closed_wrong_side_1550",
    "closed_wrong_side_more_than_1tick",
    "closed_wrong_side_more_than_2pts",
    "closed_wrong_side_more_than_5pts",
    "worst_wrong_side_dist_points",
    "worst_wrong_side_dist_bps",
    "seconds_wrong_side_vwap",
    "wrong_side_share_1550",
    "vwap_1555_side",
    "vwap_1555_dist_points",
    "vwap_1555_dist_bps",
    "vwap_1555_constructive",
    "vwap_context_10s_to_1555",
    "barrier_holds_and_1555_constructive",
    "barrier_first10_and_1555_constructive",
    "target_1550_1554_points", "target_1550_1554_sign", "target_1550_1554_state",
    "target_1555_1559_points", "target_1555_1559_sign", "target_1555_1559_state",
    "target_1550_1559_points", "target_1550_1559_sign", "target_1550_1559_state",
    "target_1550_10s_1554_points", "target_1550_10s_1554_sign", "target_1550_10s_1554_state",
    "target_1550_10s_1559_points", "target_1550_10s_1559_sign", "target_1550_10s_1559_state",
    "target_1551_1559_points", "target_1551_1559_sign", "target_1551_1559_state",
]

MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS = [
    "scope", "bucket", "target_name", "sample_size", "bullish_count", "bearish_count", "neutral_count",
    "bullish_pct", "bearish_pct", "neutral_pct", "avg_target_points", "median_target_points",
    "p10_target_points", "p25_target_points", "p75_target_points", "p90_target_points",
]


def classify_constructive_side(macro_trend_state: str | None, vwap_side: str | None) -> str:
    raise NotImplementedError


def build_macro_vwap_barrier_context(barrier: pl.DataFrame, vwap: pl.DataFrame, tick_path: str | Path) -> pl.DataFrame:
    raise NotImplementedError


def summarize_macro_vwap_barrier_context(df: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError


def write_macro_vwap_barrier_context(
    tick_path: str | Path = TICK_INPUT_PATH,
    barrier_path: str | Path = BARRIER_INPUT_PATH,
    vwap_path: str | Path = VWAP_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify NotImplemented failures**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py -q
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 5: Commit shell/tests**

```bash
git add features/macro_vwap_barrier_context.py test/test_macro_vwap_barrier_context.py
git commit -m "test: add macro vwap barrier context coverage"
```

---

### Task 2: Implement classification, validation, and target helpers

**Files:**
- Modify: `features/macro_vwap_barrier_context.py`

- [ ] **Step 1: Add helpers after constants**

```python
def _et_second(hour: int, minute: int, second: int = 0) -> int:
    return hour * 3600 + minute * 60 + second


def _missing_columns(df: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(df.columns))


def _validate_frame(df: pl.DataFrame, required: set[str], name: str) -> None:
    missing = _missing_columns(df, required)
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = sorted(TICK_REQUIRED_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def classify_constructive_side(macro_trend_state: str | None, vwap_side: str | None) -> str:
    if macro_trend_state not in {"bullish", "bearish"} or vwap_side is None:
        return "unknown"
    if vwap_side == "touch":
        return "touch"
    if macro_trend_state == "bullish":
        return "constructive" if vwap_side == "above" else "wrong"
    return "constructive" if vwap_side == "below" else "wrong"


def _sign_state(points: float | None) -> tuple[int | None, str | None]:
    if points is None:
        return None, None
    if points > 0:
        return 1, "bullish"
    if points < 0:
        return -1, "bearish"
    return 0, "neutral"


def _side_from_signed_dist(value: float | None) -> str | None:
    if value is None:
        return None
    if abs(value) <= 0.25:
        return "touch"
    return "above" if value > 0 else "below"


def _wrong_side_close_bucket(value: float | None) -> str:
    if value is None or value <= 0:
        return "no_wrong_side_close"
    if value <= 0.25:
        return "wrong_le_1tick"
    if value <= 2.0:
        return "wrong_1tick_to_2pts"
    if value <= 5.0:
        return "wrong_2_to_5pts"
    return "wrong_gt_5pts"
```

- [ ] **Step 2: Run classification test**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py::test_classify_constructive_side_is_direction_aware -q
```

Expected: PASS.

- [ ] **Step 3: Commit helper implementation**

```bash
git add features/macro_vwap_barrier_context.py
git commit -m "feat: add vwap barrier context helpers"
```

---

### Task 3: Implement bounded tick metrics and raw context builder

**Files:**
- Modify: `features/macro_vwap_barrier_context.py`

- [ ] **Step 1: Add tick metric builder before `build_macro_vwap_barrier_context`**

```python
def _scan_macro_ticks(path: str | Path) -> pl.DataFrame:
    _validate_tick_schema(path)
    ts_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ)
    return (
        pl.scan_parquet(path)
        .select(
            pl.col("ts_event").cast(UTC_NS).alias("ts_event"),
            pl.col("intra_ts_rank").cast(pl.Int64),
            pl.col("price_ticks").cast(pl.Int64),
            pl.col("size").cast(pl.Int64),
        )
        .with_columns(
            datetime_et=ts_et,
            date=ts_et.dt.date(),
            et_second=(ts_et.dt.hour().cast(pl.Int32) * 3600 + ts_et.dt.minute().cast(pl.Int32) * 60 + ts_et.dt.second().cast(pl.Int32)),
            second_in_1550=ts_et.dt.second().cast(pl.Int32),
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .filter((pl.col("et_second") >= _et_second(15, 50)) & (pl.col("et_second") < _et_second(16, 0)))
        .sort("date", "ts_event", "intra_ts_rank")
        .collect(engine="streaming")
    )


def _macro_open_vwap_ticks(ticks: pl.DataFrame) -> pl.DataFrame:
    if ticks.is_empty():
        return ticks.with_columns(vwap_1550=pl.lit(None, dtype=pl.Float64))
    return ticks.with_columns(
        pv=pl.col("price") * pl.col("size").cast(pl.Float64),
        is_1550=pl.col("et_second").is_between(_et_second(15, 50), _et_second(15, 51) - 1),
    ).with_columns(
        cum_pv_1550=pl.col("pv").filter(pl.col("is_1550")).cum_sum().over("date"),
        cum_size_1550=pl.col("size").filter(pl.col("is_1550")).cum_sum().over("date"),
    ).with_columns(
        vwap_1550=pl.when(pl.col("cum_size_1550") > 0).then(pl.col("cum_pv_1550") / pl.col("cum_size_1550")).otherwise(None)
    )


def _target_from_ticks(day_ticks: pl.DataFrame, start_second: int, end_second: int) -> float | None:
    window = day_ticks.filter((pl.col("et_second") >= start_second) & (pl.col("et_second") < end_second)).sort("ts_event", "intra_ts_rank")
    if window.is_empty():
        return None
    return float(window.item(window.height - 1, "price") - window.item(0, "price"))


def _tick_metrics_for_day(day: dict, ticks: pl.DataFrame) -> dict:
    date = day["date"]
    trend = day["macro_trend_state"]
    barrier_time = day["barrier_time"]
    day_ticks = ticks.filter(pl.col("date") == date).sort("ts_event", "intra_ts_rank")
    if day_ticks.is_empty() or barrier_time is None:
        return {"date": date}

    barrier_second = _et_second(15, 50, int(barrier_time))
    barrier_rows = day_ticks.filter((pl.col("et_second") >= barrier_second) & (pl.col("et_second") < _et_second(15, 51)))
    tick_1550 = day_ticks.filter((pl.col("et_second") >= _et_second(15, 50)) & (pl.col("et_second") < _et_second(15, 51)))
    post_barrier = barrier_rows

    def signed_dist(price: float | None, vwap: float | None) -> float | None:
        if price is None or vwap is None:
            return None
        if trend == "bullish":
            return price - vwap
        if trend == "bearish":
            return vwap - price
        return None

    if post_barrier.is_empty():
        at_barrier = None
        at_close = None
        signed_values = []
    else:
        at_barrier = post_barrier.row(0, named=True)
        at_close = post_barrier.row(post_barrier.height - 1, named=True)
        signed_values = [signed_dist(row["price"], row["vwap_1550"]) for row in post_barrier.iter_rows(named=True)]
        signed_values = [value for value in signed_values if value is not None]

    wrong_values = [max(-value, 0.0) for value in signed_values]
    worst_wrong = max(wrong_values) if wrong_values else None
    wrong_count = sum(value > 0 for value in wrong_values)
    post_count = post_barrier.height
    close_signed = signed_dist(at_close["price"], at_close["vwap_1550"]) if at_close else None
    close_wrong = max(-close_signed, 0.0) if close_signed is not None else None
    barrier_signed = signed_dist(at_barrier["price"], at_barrier["vwap_1550"]) if at_barrier else None
    target_10s_1554 = _target_from_ticks(day_ticks, _et_second(15, 50, 10), _et_second(15, 55))
    target_10s_1559 = _target_from_ticks(day_ticks, _et_second(15, 50, 10), _et_second(16, 0))
    target_1551_1559 = _target_from_ticks(day_ticks, _et_second(15, 51), _et_second(16, 0))

    row = {
        "date": date,
        "barrier_ts_utc": at_barrier["ts_event"] if at_barrier else None,
        "post_barrier_tick_count_1550": post_count,
        "vwap_side_at_barrier": _side_from_signed_dist(barrier_signed),
        "vwap_dist_at_barrier_points": barrier_signed,
        "vwap_side_at_1550_close": _side_from_signed_dist(close_signed),
        "vwap_dist_at_1550_close_points": close_signed,
        "closed_wrong_side_1550": close_wrong is not None and close_wrong > 0,
        "closed_wrong_side_more_than_1tick": close_wrong is not None and close_wrong > 0.25,
        "closed_wrong_side_more_than_2pts": close_wrong is not None and close_wrong > 2.0,
        "closed_wrong_side_more_than_5pts": close_wrong is not None and close_wrong > 5.0,
        "worst_wrong_side_dist_points": worst_wrong,
        "worst_wrong_side_dist_bps": (worst_wrong / at_close["vwap_1550"] * 10000.0) if worst_wrong is not None and at_close and at_close["vwap_1550"] not in (None, 0) else None,
        "seconds_wrong_side_vwap": wrong_count,
        "wrong_side_share_1550": (wrong_count / post_count) if post_count else None,
        "target_1550_10s_1554_points": target_10s_1554,
        "target_1550_10s_1559_points": target_10s_1559,
        "target_1551_1559_points": target_1551_1559,
    }
    for prefix in ["target_1550_10s_1554", "target_1550_10s_1559", "target_1551_1559"]:
        sign, state = _sign_state(row[f"{prefix}_points"])
        row[f"{prefix}_sign"] = sign
        row[f"{prefix}_state"] = state
    return row


def _tick_context_metrics(barrier: pl.DataFrame, tick_path: str | Path) -> pl.DataFrame:
    ticks = _macro_open_vwap_ticks(_scan_macro_ticks(tick_path))
    rows = [_tick_metrics_for_day(day, ticks) for day in barrier.iter_rows(named=True)]
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame({"date": []}, schema={"date": pl.Date})
```

- [ ] **Step 2: Implement raw builder**

Replace `build_macro_vwap_barrier_context`:

```python
def _transition_label(start: str, end: str) -> str:
    if start == "unknown" or end == "unknown":
        return "unknown"
    if "touch" in {start, end}:
        return "touch_mixed"
    return f"{start}_to_{end}"


def build_macro_vwap_barrier_context(barrier: pl.DataFrame, vwap: pl.DataFrame, tick_path: str | Path) -> pl.DataFrame:
    _validate_frame(barrier, BARRIER_REQUIRED_COLUMNS, "barrier")
    _validate_frame(vwap, VWAP_REQUIRED_COLUMNS, "VWAP")
    base = barrier.join(vwap, on="date", how="inner")
    tick_metrics = _tick_context_metrics(base, tick_path)
    out = base.join(tick_metrics, on="date", how="left")
    rows = []
    for row in out.iter_rows(named=True):
        vwap_10s = classify_constructive_side(row["macro_trend_state"], row["macro_1550_at_1550_10s_vwap_side"])
        vwap_1555 = classify_constructive_side(row["macro_trend_state"], row["macro_1550_at_1555_vwap_side"])
        result = {
            **row,
            "vwap_10s_side": row["macro_1550_at_1550_10s_vwap_side"],
            "vwap_10s_dist_points": row["macro_1550_at_1550_10s_vwap_dist_points"],
            "vwap_10s_dist_bps": row["macro_1550_at_1550_10s_vwap_dist_bps"],
            "vwap_10s_constructive": vwap_10s,
            "barrier_first10_and_vwap_constructive": bool(row["barrier_first10"] and vwap_10s in {"constructive", "touch"}),
            "vwap_1555_side": row["macro_1550_at_1555_vwap_side"],
            "vwap_1555_dist_points": row["macro_1550_at_1555_vwap_dist_points"],
            "vwap_1555_dist_bps": row["macro_1550_at_1555_vwap_dist_bps"],
            "vwap_1555_constructive": vwap_1555,
            "vwap_context_10s_to_1555": _transition_label(vwap_10s, vwap_1555),
            "barrier_holds_and_1555_constructive": bool(row["barrier_holds"] and vwap_1555 in {"constructive", "touch"}),
            "barrier_first10_and_1555_constructive": bool(row["barrier_first10"] and vwap_1555 in {"constructive", "touch"}),
        }
        result["wrong_side_close_bucket"] = _wrong_side_close_bucket(result.get("vwap_dist_at_1550_close_points") and max(-result["vwap_dist_at_1550_close_points"], 0.0))
        rows.append(result)
    return pl.DataFrame(rows, infer_schema_length=None).select(MACRO_VWAP_BARRIER_CONTEXT_COLUMNS).sort("date")
```

- [ ] **Step 3: Run raw builder test**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py::test_build_context_adds_vwap_barrier_and_wrong_side_metrics -q
```

Expected: PASS.

- [ ] **Step 4: Commit raw builder**

```bash
git add features/macro_vwap_barrier_context.py
git commit -m "feat: build macro vwap barrier context rows"
```

---

### Task 4: Implement summary builder and writer

**Files:**
- Modify: `features/macro_vwap_barrier_context.py`

- [ ] **Step 1: Add summary helpers before `summarize_macro_vwap_barrier_context`**

```python
def _pct(count: int, denom: int) -> float | None:
    return count / denom * 100.0 if denom else None


def _summary_row(df: pl.DataFrame, scope: str, bucket: str, target_name: str) -> dict:
    state_col = f"{target_name}_state"
    points_col = f"{target_name}_points"
    sample_size = df.height
    bullish_count = df.filter(pl.col(state_col) == "bullish").height if sample_size else 0
    bearish_count = df.filter(pl.col(state_col) == "bearish").height if sample_size else 0
    neutral_count = df.filter(pl.col(state_col) == "neutral").height if sample_size else 0
    return {
        "scope": scope,
        "bucket": bucket,
        "target_name": target_name,
        "sample_size": sample_size,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "bullish_pct": _pct(bullish_count, sample_size),
        "bearish_pct": _pct(bearish_count, sample_size),
        "neutral_pct": _pct(neutral_count, sample_size),
        "avg_target_points": df.select(pl.col(points_col).mean()).item() if sample_size else None,
        "median_target_points": df.select(pl.col(points_col).median()).item() if sample_size else None,
        "p10_target_points": df.select(pl.col(points_col).quantile(0.10)).item() if sample_size else None,
        "p25_target_points": df.select(pl.col(points_col).quantile(0.25)).item() if sample_size else None,
        "p75_target_points": df.select(pl.col(points_col).quantile(0.75)).item() if sample_size else None,
        "p90_target_points": df.select(pl.col(points_col).quantile(0.90)).item() if sample_size else None,
    }


def _deciled(df: pl.DataFrame, value_col: str) -> pl.DataFrame | None:
    non_null = df.filter(pl.col(value_col).is_not_null())
    if non_null.height < 10 or non_null.select(pl.col(value_col).n_unique()).item() < 10:
        return None
    return non_null.with_columns(
        (((pl.col(value_col).rank(method="ordinal") - 1) * 10 / non_null.height).floor().cast(pl.Int64).clip(0, 9) + 1)
        .cast(pl.String)
        .alias("_bucket")
    )
```

- [ ] **Step 2: Replace summary and writer functions**

```python
def summarize_macro_vwap_barrier_context(df: pl.DataFrame) -> pl.DataFrame:
    missing = _missing_columns(df, set(MACRO_VWAP_BARRIER_CONTEXT_COLUMNS))
    if missing:
        raise ValueError(f"Missing context columns: {missing}")
    rows = []
    for target in TARGET_PREFIXES:
        scopes = [
            ("barrier_only", "first10_true", df.filter(pl.col("barrier_first10"))),
            ("barrier_only", "first10_false", df.filter(~pl.col("barrier_first10"))),
            ("barrier_only", "holds_true", df.filter(pl.col("barrier_holds"))),
            ("barrier_only", "holds_false", df.filter(~pl.col("barrier_holds"))),
            ("vwap_10s_only", "constructive", df.filter(pl.col("vwap_10s_constructive") == "constructive")),
            ("vwap_10s_only", "wrong", df.filter(pl.col("vwap_10s_constructive") == "wrong")),
            ("vwap_10s_only", "touch", df.filter(pl.col("vwap_10s_constructive") == "touch")),
            ("vwap_10s_only", "unknown", df.filter(pl.col("vwap_10s_constructive") == "unknown")),
            ("barrier_vwap_10s", "first10_constructive", df.filter(pl.col("barrier_first10") & pl.col("vwap_10s_constructive").is_in(["constructive", "touch"]))),
            ("barrier_vwap_10s", "first10_wrong", df.filter(pl.col("barrier_first10") & (pl.col("vwap_10s_constructive") == "wrong"))),
            ("barrier_1555_context", "holds_1555_constructive", df.filter(pl.col("barrier_holds_and_1555_constructive"))),
            ("barrier_1555_context", "holds_1555_not_constructive", df.filter(pl.col("barrier_holds") & ~pl.col("barrier_holds_and_1555_constructive"))),
        ]
        for scope, bucket, subset in scopes:
            rows.append(_summary_row(subset, scope, bucket, target))
        for bucket in ["no_wrong_side_close", "wrong_le_1tick", "wrong_1tick_to_2pts", "wrong_2_to_5pts", "wrong_gt_5pts"]:
            close_wrong = pl.when(pl.col("vwap_dist_at_1550_close_points").is_not_null()).then((-pl.col("vwap_dist_at_1550_close_points")).clip(0.0)).otherwise(None)
            bucketed = df.with_columns(
                pl.when(close_wrong.is_null() | (close_wrong <= 0)).then(pl.lit("no_wrong_side_close"))
                .when(close_wrong <= 0.25).then(pl.lit("wrong_le_1tick"))
                .when(close_wrong <= 2.0).then(pl.lit("wrong_1tick_to_2pts"))
                .when(close_wrong <= 5.0).then(pl.lit("wrong_2_to_5pts"))
                .otherwise(pl.lit("wrong_gt_5pts")).alias("_bucket")
            )
            rows.append(_summary_row(bucketed.filter(pl.col("_bucket") == bucket), "wrong_side_close_bucket", bucket, target))
        for value_col, scope in [("wrong_side_share_1550", "wrong_side_share_decile"), ("worst_wrong_side_dist_points", "worst_wrong_side_dist_decile")]:
            deciled = _deciled(df, value_col)
            if deciled is not None:
                for decile in [str(i) for i in range(1, 11)]:
                    rows.append(_summary_row(deciled.filter(pl.col("_bucket") == decile), scope, decile, target))
        for bucket in sorted(df.select(pl.col("vwap_context_10s_to_1555").drop_nulls().unique()).to_series().to_list()):
            rows.append(_summary_row(df.filter(pl.col("vwap_context_10s_to_1555") == bucket), "vwap_1555_decision", bucket, target))
    return pl.DataFrame(rows, infer_schema_length=None).select(MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS)


def write_macro_vwap_barrier_context(
    tick_path: str | Path = TICK_INPUT_PATH,
    barrier_path: str | Path = BARRIER_INPUT_PATH,
    vwap_path: str | Path = VWAP_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    barrier = pl.read_parquet(barrier_path)
    vwap = pl.read_parquet(vwap_path)
    context = build_macro_vwap_barrier_context(barrier, vwap, tick_path)
    summary = summarize_macro_vwap_barrier_context(context)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    context.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    for path in [TICK_INPUT_PATH, BARRIER_INPUT_PATH, VWAP_INPUT_PATH]:
        if not Path(path).exists():
            print(f"[ERROR] Input not found: {path}", file=sys.stderr)
            sys.exit(1)
    output, summary = write_macro_vwap_barrier_context()
    print(f"[OK] Wrote macro VWAP barrier context -> {output}")
    print(f"[OK] Wrote macro VWAP barrier context summary -> {summary}")
```

- [ ] **Step 3: Run feature tests**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit summary/writer**

```bash
git add features/macro_vwap_barrier_context.py test/test_macro_vwap_barrier_context.py
git commit -m "feat: summarize macro vwap barrier context"
```

---

### Task 5: Add visualization script and tests

**Files:**
- Create: `viz/macro_vwap_barrier_context_viz.py`
- Create: `test/test_macro_vwap_barrier_context_viz.py`

- [ ] **Step 1: Create failing viz tests**

Create `test/test_macro_vwap_barrier_context_viz.py`:

```python
from pathlib import Path

import polars as pl

from viz.macro_vwap_barrier_context_viz import process_dataset


def test_process_dataset_writes_distribution_csvs_and_figures(tmp_path: Path):
    context_path = tmp_path / "context.parquet"
    summary_path = tmp_path / "summary.parquet"
    output_dir = tmp_path / "figs"
    pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03", "2025-01-04"],
            "vwap_10s_dist_points": [1.0, -1.0, 0.25],
            "vwap_1555_dist_points": [2.0, -2.0, 0.5],
            "worst_wrong_side_dist_points": [0.0, 3.0, 1.0],
            "wrong_side_share_1550": [0.0, 0.5, 0.25],
            "vwap_context_10s_to_1555": ["constructive_to_constructive", "wrong_to_wrong", "touch_mixed"],
            "target_1555_1559_points": [5.0, -4.0, 1.0],
            "target_1550_10s_1559_points": [6.0, -5.0, 2.0],
        }
    ).with_columns(pl.col("date").str.to_date()).write_parquet(context_path)
    pl.DataFrame(
        {
            "scope": ["vwap_10s_only"],
            "bucket": ["constructive"],
            "target_name": ["target_1555_1559"],
            "sample_size": [1],
            "bullish_count": [1],
            "bearish_count": [0],
            "neutral_count": [0],
            "bullish_pct": [100.0],
            "bearish_pct": [0.0],
            "neutral_pct": [0.0],
            "avg_target_points": [5.0],
            "median_target_points": [5.0],
            "p10_target_points": [5.0],
            "p25_target_points": [5.0],
            "p75_target_points": [5.0],
            "p90_target_points": [5.0],
        }
    ).write_parquet(summary_path)

    wrote = process_dataset(context_path, summary_path, output_dir)

    names = {path.name for path in wrote}
    assert "summary_by_scope.csv" in names
    assert "wrong_side_quantiles.csv" in names
    assert "target_quantiles_by_bucket.csv" in names
    assert "vwap_1555_decision_summary.csv" in names
    assert "vwap_10s_dist_hist.png" in names
    assert "wrong_side_dist_ecdf.png" in names
    assert all(path.exists() for path in wrote)
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context_viz.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing `process_dataset`.

- [ ] **Step 3: Create viz script**

Create `viz/macro_vwap_barrier_context_viz.py`:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

CONTEXT_INPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context.parquet")
SUMMARY_INPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context_summary.parquet")
OUTPUT_DIR = Path("outputs/figs/macro_vwap_barrier_context")


def _write_csv(df: pl.DataFrame, path: Path, wrote: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    wrote.append(path)


def _save_hist(values: list[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=30, color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("points")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_ecdf(values: list[float], path: Path, title: str) -> None:
    sorted_values = sorted(values)
    y = [(idx + 1) / len(sorted_values) for idx in range(len(sorted_values))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted_values, y, color="#f58518")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("ECDF")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_scatter(df: pl.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    small = df.select(x, y).drop_nulls()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(small[x].to_list(), small[y].to_list(), s=12, alpha=0.65)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def process_dataset(
    context_path: str | Path = CONTEXT_INPUT_PATH,
    summary_path: str | Path = SUMMARY_INPUT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> list[Path]:
    context = pl.read_parquet(context_path)
    summary = pl.read_parquet(summary_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wrote: list[Path] = []

    _write_csv(summary.sort("scope", "target_name", "bucket"), out / "summary_by_scope.csv", wrote)
    _write_csv(
        context.select(
            pl.col("worst_wrong_side_dist_points").quantile(0.10).alias("p10_worst_wrong_side_dist_points"),
            pl.col("worst_wrong_side_dist_points").quantile(0.25).alias("p25_worst_wrong_side_dist_points"),
            pl.col("worst_wrong_side_dist_points").median().alias("median_worst_wrong_side_dist_points"),
            pl.col("worst_wrong_side_dist_points").quantile(0.75).alias("p75_worst_wrong_side_dist_points"),
            pl.col("worst_wrong_side_dist_points").quantile(0.90).alias("p90_worst_wrong_side_dist_points"),
            pl.col("wrong_side_share_1550").quantile(0.10).alias("p10_wrong_side_share_1550"),
            pl.col("wrong_side_share_1550").median().alias("median_wrong_side_share_1550"),
            pl.col("wrong_side_share_1550").quantile(0.90).alias("p90_wrong_side_share_1550"),
        ),
        out / "wrong_side_quantiles.csv",
        wrote,
    )
    _write_csv(
        context.group_by("vwap_context_10s_to_1555").agg(
            pl.len().alias("sample_size"),
            pl.col("target_1555_1559_points").mean().alias("avg_target_1555_1559_points"),
            pl.col("target_1555_1559_points").median().alias("median_target_1555_1559_points"),
        ).sort("vwap_context_10s_to_1555"),
        out / "vwap_1555_decision_summary.csv",
        wrote,
    )
    _write_csv(
        summary.filter(pl.col("scope").is_in(["wrong_side_close_bucket", "vwap_1555_decision", "barrier_1555_context"])),
        out / "target_quantiles_by_bucket.csv",
        wrote,
    )

    for col, filename, title in [
        ("vwap_10s_dist_points", "vwap_10s_dist_hist.png", "VWAP distance at 15:50:10"),
        ("vwap_1555_dist_points", "vwap_1555_dist_hist.png", "VWAP distance at 15:55"),
    ]:
        values = context.select(col).drop_nulls().to_series().to_list()
        if values:
            path = out / filename
            _save_hist(values, path, title)
            wrote.append(path)
    for col, filename, title in [
        ("worst_wrong_side_dist_points", "wrong_side_dist_ecdf.png", "Worst wrong-side VWAP distance ECDF"),
        ("wrong_side_share_1550", "wrong_side_share_ecdf.png", "Wrong-side VWAP share ECDF"),
    ]:
        values = context.select(col).drop_nulls().to_series().to_list()
        if values:
            path = out / filename
            _save_ecdf(values, path, title)
            wrote.append(path)
    scatter_path = out / "vwap_1555_scatter_target.png"
    _save_scatter(context, "vwap_1555_dist_points", "target_1555_1559_points", scatter_path, "15:55 VWAP distance vs 15:55-15:59 target")
    wrote.append(scatter_path)
    return wrote


def main() -> None:
    wrote = process_dataset()
    for path in wrote:
        print(f"[OK] Wrote macro VWAP barrier context viz -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run viz test**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context_viz.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit viz**

```bash
git add viz/macro_vwap_barrier_context_viz.py test/test_macro_vwap_barrier_context_viz.py
git commit -m "feat: visualize macro vwap barrier context distributions"
```

---

### Task 6: Runtime verification and output generation

**Files:**
- Modify only if tests/runtime reveal defects.

- [ ] **Step 1: Run feature tests**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py test/test_macro_vwap_barrier_context_viz.py -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent VWAP/barrier tests**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py test/test_macro_vwap_barrier_context_viz.py test/test_macro_vwap_features.py test/test_macro_1550_barrier.py -q
```

Expected: PASS.

- [ ] **Step 3: Generate feature outputs**

Use PowerShell pass-through if running full tick parquet on Windows is safer:

```bash
powershell.exe -NoProfile -Command '$env:POLARS_MAX_THREADS="2"; Set-Location "E:\backup\code\Finance\research\macro\.worktrees\macro-vwap-features"; python -m features.macro_vwap_barrier_context'
```

Expected:

```text
[OK] Wrote macro VWAP barrier context -> outputs\nq_macro_vwap_barrier_context.parquet
[OK] Wrote macro VWAP barrier context summary -> outputs\nq_macro_vwap_barrier_context_summary.parquet
```

- [ ] **Step 4: Generate viz outputs**

```bash
.venv/bin/python viz/macro_vwap_barrier_context_viz.py
```

Expected: CSVs + PNGs under `outputs/figs/macro_vwap_barrier_context/`.

- [ ] **Step 5: Inspect generated outputs**

```bash
.venv/bin/python - <<'PY'
import polars as pl
for path in [
    "outputs/nq_macro_vwap_barrier_context.parquet",
    "outputs/nq_macro_vwap_barrier_context_summary.parquet",
]:
    df = pl.read_parquet(path)
    print(path, df.height, len(df.columns))
    print(df.head(3))
PY
ls -lh outputs/figs/macro_vwap_barrier_context
```

Expected:
- raw context has non-zero rows.
- summary has non-zero rows.
- figure directory contains CSV + PNG files.

- [ ] **Step 6: Commit runtime fixes if any**

If runtime required code fixes:

```bash
git add features/macro_vwap_barrier_context.py viz/macro_vwap_barrier_context_viz.py test/test_macro_vwap_barrier_context.py test/test_macro_vwap_barrier_context_viz.py
git commit -m "fix: harden macro vwap barrier context runtime"
```

---

### Task 7: Final verification

**Files:**
- None unless defects found.

- [ ] **Step 1: Run final test command**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_barrier_context.py test/test_macro_vwap_barrier_context_viz.py test/test_macro_vwap_features.py test/test_macro_1550_barrier.py -q
```

Expected: PASS.

- [ ] **Step 2: Check git status**

```bash
git status --short
```

Expected: no uncommitted code/test changes. Generated outputs are ignored; do not commit them unless requested.

- [ ] **Step 3: Report evidence**

Final response should include:
- commits made
- tests run
- runtime commands run
- output files generated
- notable findings/caveats
