# Macro VWAP Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build tick-derived pre-macro and intramacro VWAP context datasets plus diagnostic summaries for the NQ 15:50-15:59 ET macro window.

**Architecture:** Add one focused feature module, `features/macro_vwap_features.py`, with lazy tick scanning, reusable anchored-VWAP/target helpers, separate A/B builders, optional barrier context join, long-form summaries, and a script entrypoint. Add `test/test_macro_vwap_features.py` with synthetic tick fixtures covering math, strict checkpoints, targets, touch classification, null behavior, DST, barrier join, and writer outputs.

**Tech Stack:** Python, Polars lazy/eager APIs, PyArrow parquet schema metadata, pytest, project `.venv`.

---

## File Structure

- Create: `features/macro_vwap_features.py`
  - Constants for inputs/outputs, anchors, checkpoints, target windows, schemas.
  - Lazy tick scan with schema validation.
  - Anchored VWAP aggregation helper.
  - Strict checkpoint price helper.
  - Tick-derived target helper.
  - `build_macro_vwap_premacro(...)` returning `pl.LazyFrame`.
  - `build_macro_vwap_intramacro(...)` returning `pl.LazyFrame`.
  - `summarize_macro_vwap_features(...)` returning eager summary `pl.DataFrame`.
  - `write_macro_vwap_features(...)` writing four parquet outputs.
  - `main()` for `.venv/bin/python -m features.macro_vwap_features`.
- Create: `test/test_macro_vwap_features.py`
  - Synthetic parquet tick writer.
  - Unit tests for each core behavior.
- No changes to `input-data/`.
- No changes to existing feature modules.

---

### Task 1: Add failing tests and public API shell

**Files:**
- Create: `test/test_macro_vwap_features.py`
- Create: `features/macro_vwap_features.py`

- [ ] **Step 1: Create initial failing tests**

Create `test/test_macro_vwap_features.py` with this content:

```python
from pathlib import Path

import polars as pl
import pytest

from features.macro_vwap_features import (
    INTRAMACRO_COLUMNS,
    PREMACRO_COLUMNS,
    SUMMARY_COLUMNS,
    build_macro_vwap_intramacro,
    build_macro_vwap_premacro,
    summarize_macro_vwap_features,
    write_macro_vwap_features,
)


def _write_ticks(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", time_zone="UTC")),
        pl.col("intra_ts_rank").cast(pl.Int64),
        pl.col("price_ticks").cast(pl.Int64),
        pl.col("size").cast(pl.Int64),
    ).write_parquet(path)


def _tick(ts: str, price: float, size: int = 1, rank: int = 0) -> dict:
    return {
        "ts_event": ts,
        "intra_ts_rank": rank,
        "price_ticks": int(price * 4),
        "size": size,
    }


def test_build_macro_vwap_premacro_computes_tick_weighted_vwap_and_targets(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T14:30:00Z", 100.0, 1),
            _tick("2025-01-02T18:00:00Z", 102.0, 3),
            _tick("2025-01-02T20:00:00Z", 104.0, 2),
            _tick("2025-01-02T20:49:59Z", 106.0, 4),
            _tick("2025-01-02T20:50:00Z", 107.0, 1),
            _tick("2025-01-02T20:54:59Z", 110.0, 1),
            _tick("2025-01-02T20:55:00Z", 109.0, 1),
            _tick("2025-01-02T20:59:59Z", 105.0, 1),
            _tick("2025-01-02T21:00:00Z", 999.0, 1),
        ],
    )

    out = build_macro_vwap_premacro(path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert out.columns == PREMACRO_COLUMNS
    assert row["date"].isoformat() == "2025-01-02"
    assert row["rth_0930_vwap"] == pytest.approx((100 * 1 + 102 * 3 + 104 * 2 + 106 * 4) / 10)
    assert row["pm_1300_vwap"] == pytest.approx((102 * 3 + 104 * 2 + 106 * 4) / 9)
    assert row["h3pm_1500_vwap"] == pytest.approx((104 * 2 + 106 * 4) / 6)
    assert row["rth_0930_price"] == 106.0
    assert row["rth_0930_vwap_dist_points"] == pytest.approx(106.0 - row["rth_0930_vwap"])
    assert row["rth_0930_vwap_dist_bps"] == pytest.approx((106.0 / row["rth_0930_vwap"] - 1.0) * 10000.0)
    assert row["rth_0930_vwap_side"] == "above"
    assert row["target_1550_1554_points"] == 3.0
    assert row["target_1555_1559_points"] == -4.0
    assert row["target_1550_1559_points"] == -2.0
    assert row["target_1550_1554_state"] == "bullish"
    assert row["target_1555_1559_state"] == "bearish"
    assert row["target_1550_1559_state"] == "bearish"


def test_strict_checkpoint_boundaries_exclude_equal_timestamp_ticks(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:09Z", 101.0, 1),
            _tick("2025-01-02T20:50:10Z", 150.0, 1),
            _tick("2025-01-02T20:54:59Z", 102.0, 1),
            _tick("2025-01-02T20:55:00Z", 200.0, 1),
            _tick("2025-01-02T20:55:01Z", 103.0, 1),
            _tick("2025-01-02T20:59:59Z", 104.0, 1),
            _tick("2025-01-02T21:00:00Z", 300.0, 1),
        ],
    )

    out = build_macro_vwap_intramacro(path, barrier_path=None).collect(engine="streaming")
    row = out.row(0, named=True)

    assert row["macro_1550_at_1550_10s_price"] == 101.0
    assert row["macro_1550_at_1555_price"] == 102.0
    assert row["macro_1550_at_1600_price"] == 104.0
    assert row["eoii_1555_at_1600_vwap"] == pytest.approx((103.0 + 104.0) / 2)
    assert row["target_1550_1554_points"] == 2.0
    assert row["target_1555_1559_points"] == -96.0
    assert row["target_1550_1559_points"] == 4.0


def test_vwap_side_uses_one_tick_touch_threshold(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T14:30:00Z", 100.0, 1),
            _tick("2025-01-02T20:49:59Z", 100.25, 1),
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:54:59Z", 100.0, 1),
            _tick("2025-01-02T20:55:00Z", 100.0, 1),
            _tick("2025-01-02T20:59:59Z", 100.0, 1),
        ],
    )

    out = build_macro_vwap_premacro(path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert row["rth_0930_vwap_dist_points"] == pytest.approx(0.125)
    assert row["rth_0930_vwap_side"] == "touch"


def test_missing_and_zero_size_windows_produce_null_context(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T14:30:00Z", 100.0, 0),
            _tick("2025-01-02T20:50:00Z", 101.0, 1),
            _tick("2025-01-02T20:54:59Z", 102.0, 1),
            _tick("2025-01-02T20:55:00Z", 103.0, 1),
            _tick("2025-01-02T20:59:59Z", 104.0, 1),
        ],
    )

    out = build_macro_vwap_premacro(path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert row["rth_0930_vwap"] is None
    assert row["rth_0930_price"] == 100.0
    assert row["rth_0930_vwap_side"] is None
    assert row["pm_1300_vwap"] is None
    assert row["pm_1300_price"] is None


def test_build_macro_vwap_intramacro_joins_optional_barrier_context(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    barrier_path = tmp_path / "barrier.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:09Z", 101.0, 1),
            _tick("2025-01-02T20:54:59Z", 102.0, 1),
            _tick("2025-01-02T20:55:00Z", 103.0, 1),
            _tick("2025-01-02T20:59:59Z", 104.0, 1),
        ],
    )
    pl.DataFrame(
        {
            "date": ["2025-01-02"],
            "macro_trend_state": ["bullish"],
            "barrier_extreme": ["low"],
            "barrier_price": [99.0],
            "barrier_time": [5],
            "barrier_first10": [True],
            "barrier_is_macro_extreme": [True],
            "barrier_holds": [True],
            "edge_case": [False],
        }
    ).with_columns(pl.col("date").str.to_date()).write_parquet(barrier_path)

    out = build_macro_vwap_intramacro(tick_path, barrier_path=barrier_path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert out.columns == INTRAMACRO_COLUMNS
    assert row["barrier_macro_trend_state"] == "bullish"
    assert row["barrier_first10"] is True
    assert row["barrier_holds"] is True


def test_summer_dst_uses_new_york_market_time(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-07-02T13:30:00Z", 100.0, 1),
            _tick("2025-07-02T19:49:59Z", 101.0, 1),
            _tick("2025-07-02T19:50:00Z", 102.0, 1),
            _tick("2025-07-02T19:54:59Z", 103.0, 1),
            _tick("2025-07-02T19:55:00Z", 104.0, 1),
            _tick("2025-07-02T19:59:59Z", 105.0, 1),
        ],
    )

    out = build_macro_vwap_premacro(path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert row["date"].isoformat() == "2025-07-02"
    assert row["rth_0930_price"] == 101.0
    assert row["target_1550_1559_points"] == 3.0


def test_summarize_macro_vwap_features_reports_side_bands_deciles_and_confluence():
    df = pl.DataFrame(
        {
            "date": [f"2025-01-{day:02d}" for day in range(1, 13)],
            "rth_0930_vwap_dist_bps": [-30, -15, -7, -3, -1, 0, 1, 3, 7, 15, 30, 40],
            "rth_0930_vwap_side": ["below", "below", "below", "below", "touch", "touch", "touch", "above", "above", "above", "above", "above"],
            "premacro_net_side_score": [-1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1],
            "target_1550_1554_points": [-1, -2, -3, -4, 0, 0, 1, 2, 3, 4, 5, 6],
            "target_1550_1554_state": ["bearish", "bearish", "bearish", "bearish", "neutral", "neutral", "bullish", "bullish", "bullish", "bullish", "bullish", "bullish"],
            "target_1555_1559_points": [1] * 12,
            "target_1555_1559_state": ["bullish"] * 12,
            "target_1550_1559_points": [1] * 12,
            "target_1550_1559_state": ["bullish"] * 12,
        }
    ).with_columns(pl.col("date").str.to_date())

    summary = summarize_macro_vwap_features(df, feature_set="premacro")

    assert summary.columns == SUMMARY_COLUMNS
    side = summary.filter(
        (pl.col("feature_name") == "rth_0930")
        & (pl.col("target_name") == "target_1550_1554")
        & (pl.col("scope") == "side")
        & (pl.col("bucket") == "below")
    ).row(0, named=True)
    assert side["sample_size"] == 4
    assert side["bearish_pct"] == 100.0
    assert summary.filter(pl.col("scope") == "fixed_bps_band").height > 0
    assert summary.filter(pl.col("scope") == "decile").height == 10
    assert summary.filter(pl.col("scope") == "confluence").height > 0


def test_write_macro_vwap_features_writes_four_outputs(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    premacro_path = tmp_path / "premacro.parquet"
    premacro_summary_path = tmp_path / "premacro_summary.parquet"
    intramacro_path = tmp_path / "intramacro.parquet"
    intramacro_summary_path = tmp_path / "intramacro_summary.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T14:30:00Z", 100.0, 1),
            _tick("2025-01-02T18:00:00Z", 101.0, 1),
            _tick("2025-01-02T20:00:00Z", 102.0, 1),
            _tick("2025-01-02T20:49:59Z", 103.0, 1),
            _tick("2025-01-02T20:50:00Z", 104.0, 1),
            _tick("2025-01-02T20:50:09Z", 105.0, 1),
            _tick("2025-01-02T20:54:59Z", 106.0, 1),
            _tick("2025-01-02T20:55:00Z", 107.0, 1),
            _tick("2025-01-02T20:59:59Z", 108.0, 1),
        ],
    )

    wrote = write_macro_vwap_features(
        input_path=tick_path,
        premacro_output_path=premacro_path,
        premacro_summary_output_path=premacro_summary_path,
        intramacro_output_path=intramacro_path,
        intramacro_summary_output_path=intramacro_summary_path,
        barrier_path=None,
    )

    assert wrote == (premacro_path, premacro_summary_path, intramacro_path, intramacro_summary_path)
    assert pl.read_parquet(premacro_path).columns == PREMACRO_COLUMNS
    assert pl.read_parquet(intramacro_path).columns == INTRAMACRO_COLUMNS
    assert pl.read_parquet(premacro_summary_path).columns == SUMMARY_COLUMNS
    assert pl.read_parquet(intramacro_summary_path).columns == SUMMARY_COLUMNS
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'features.macro_vwap_features'`.

- [ ] **Step 3: Add public API shell**

Create `features/macro_vwap_features.py` with this initial shell:

```python
from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR, get_tick_schema

INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
DEFAULT_BARRIER_PATH = Path("outputs/nq_macro_1550_barrier.parquet")
PREMACRO_OUTPUT_PATH = Path("outputs/nq_macro_vwap_premacro.parquet")
PREMACRO_SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_vwap_premacro_summary.parquet")
INTRAMACRO_OUTPUT_PATH = Path("outputs/nq_macro_vwap_intramacro.parquet")
INTRAMACRO_SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_vwap_intramacro_summary.parquet")

UTC_NS = pl.Datetime("ns", time_zone="UTC")
TOUCH_THRESHOLD_POINTS = 0.25
_REQUIRED_TICK_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks", "size"}

TARGET_PREFIXES = ("target_1550_1554", "target_1555_1559", "target_1550_1559")
PREMACRO_FEATURE_PREFIXES = ("rth_0930", "pm_1300", "h3pm_1500")
INTRAMACRO_FEATURE_PREFIXES = (
    "macro_1550_at_1550_10s",
    "macro_1550_at_1555",
    "macro_1550_at_1600",
    "eoii_1555_at_1600",
)
BARRIER_COLUMNS = [
    "barrier_macro_trend_state",
    "barrier_extreme",
    "barrier_price",
    "barrier_time",
    "barrier_first10",
    "barrier_is_macro_extreme",
    "barrier_holds",
    "barrier_edge_case",
]

_TARGET_COLUMNS = [col for prefix in TARGET_PREFIXES for col in (f"{prefix}_points", f"{prefix}_sign", f"{prefix}_state")]
_PREMACRO_VWAP_COLUMNS = [col for prefix in PREMACRO_FEATURE_PREFIXES for col in (f"{prefix}_vwap", f"{prefix}_price", f"{prefix}_vwap_dist_points", f"{prefix}_vwap_dist_bps", f"{prefix}_vwap_side")]
_INTRAMACRO_VWAP_COLUMNS = [col for prefix in INTRAMACRO_FEATURE_PREFIXES for col in (f"{prefix}_vwap", f"{prefix}_price", f"{prefix}_vwap_dist_points", f"{prefix}_vwap_dist_bps", f"{prefix}_vwap_side")]

PREMACRO_COLUMNS = [
    "date",
    *_PREMACRO_VWAP_COLUMNS,
    "premacro_above_count",
    "premacro_below_count",
    "premacro_touch_count",
    "premacro_net_side_score",
    *_TARGET_COLUMNS,
]
INTRAMACRO_COLUMNS = [
    "date",
    *_INTRAMACRO_VWAP_COLUMNS,
    "intramacro_above_count",
    "intramacro_below_count",
    "intramacro_touch_count",
    "intramacro_net_side_score",
    *_TARGET_COLUMNS,
    *BARRIER_COLUMNS,
]
SUMMARY_COLUMNS = [
    "feature_set",
    "feature_name",
    "target_name",
    "scope",
    "bucket",
    "sample_size",
    "bullish_count",
    "bearish_count",
    "neutral_count",
    "bullish_pct",
    "bearish_pct",
    "neutral_pct",
    "avg_target_points",
    "median_target_points",
]


def build_macro_vwap_premacro(path: str | Path = INPUT_PATH) -> pl.LazyFrame:
    raise NotImplementedError


def build_macro_vwap_intramacro(
    path: str | Path = INPUT_PATH,
    barrier_path: str | Path | None = DEFAULT_BARRIER_PATH,
) -> pl.LazyFrame:
    raise NotImplementedError


def summarize_macro_vwap_features(df: pl.DataFrame, feature_set: str) -> pl.DataFrame:
    raise NotImplementedError


def write_macro_vwap_features(*args, **kwargs) -> tuple[Path, Path, Path, Path]:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify API shell fails on behavior**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py -q
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 5: Commit failing tests and API shell**

```bash
git add test/test_macro_vwap_features.py features/macro_vwap_features.py
git commit -m "test: add macro vwap feature coverage"
```

---

### Task 2: Implement lazy tick scan, VWAP helpers, and pre-macro features

**Files:**
- Modify: `features/macro_vwap_features.py`

- [ ] **Step 1: Add helper constants/functions below `SUMMARY_COLUMNS`**

```python
def _et_second(hour: int, minute: int, second: int = 0) -> int:
    return hour * 3600 + minute * 60 + second


PREMACRO_SPECS = {
    "rth_0930": (_et_second(9, 30), _et_second(15, 50)),
    "pm_1300": (_et_second(13, 0), _et_second(15, 50)),
    "h3pm_1500": (_et_second(15, 0), _et_second(15, 50)),
}
INTRAMACRO_SPECS = {
    "macro_1550_at_1550_10s": (_et_second(15, 50), _et_second(15, 50, 10)),
    "macro_1550_at_1555": (_et_second(15, 50), _et_second(15, 55)),
    "macro_1550_at_1600": (_et_second(15, 50), _et_second(16, 0)),
    "eoii_1555_at_1600": (_et_second(15, 55), _et_second(16, 0)),
}
TARGET_SPECS = {
    "target_1550_1554": (_et_second(15, 50), _et_second(15, 55)),
    "target_1555_1559": (_et_second(15, 55), _et_second(16, 0)),
    "target_1550_1559": (_et_second(15, 50), _et_second(16, 0)),
}


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = sorted(_REQUIRED_TICK_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def _scan_ticks(path: str | Path) -> pl.LazyFrame:
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
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .with_columns(pv=pl.col("price") * pl.col("size").cast(pl.Float64))
        .filter((pl.col("et_second") >= _et_second(9, 30)) & (pl.col("et_second") < _et_second(16, 0)))
    )


def _safe_vwap_expr(prefix: str) -> pl.Expr:
    return pl.when(pl.col(f"{prefix}_total_size") > 0).then(pl.col(f"{prefix}_pv") / pl.col(f"{prefix}_total_size")).otherwise(None)


def _side_expr(prefix: str) -> pl.Expr:
    dist = pl.col(f"{prefix}_vwap_dist_points")
    return (
        pl.when(dist.is_null())
        .then(None)
        .when(dist.abs() <= TOUCH_THRESHOLD_POINTS)
        .then(pl.lit("touch"))
        .when(dist > TOUCH_THRESHOLD_POINTS)
        .then(pl.lit("above"))
        .otherwise(pl.lit("below"))
    )


def _add_distance_columns(lf: pl.LazyFrame, prefix: str) -> pl.LazyFrame:
    return (
        lf.with_columns(
            (pl.col(f"{prefix}_price") - pl.col(f"{prefix}_vwap")).alias(f"{prefix}_vwap_dist_points"),
            pl.when(pl.col(f"{prefix}_vwap") != 0)
            .then((pl.col(f"{prefix}_price") / pl.col(f"{prefix}_vwap") - 1.0) * 10000.0)
            .otherwise(None)
            .alias(f"{prefix}_vwap_dist_bps"),
        )
        .with_columns(_side_expr(prefix).alias(f"{prefix}_vwap_side"))
    )


def _anchored_vwap(base: pl.LazyFrame, prefix: str, anchor_second: int, checkpoint_second: int) -> pl.LazyFrame:
    window = (
        base.filter((pl.col("et_second") >= anchor_second) & (pl.col("et_second") < checkpoint_second))
        .sort("date", "ts_event", "intra_ts_rank")
        .group_by("date")
        .agg(
            pl.col("pv").sum().alias(f"{prefix}_pv"),
            pl.col("size").sum().alias(f"{prefix}_total_size"),
            pl.col("price").last().alias(f"{prefix}_price"),
        )
        .with_columns(_safe_vwap_expr(prefix).alias(f"{prefix}_vwap"))
        .select("date", f"{prefix}_vwap", f"{prefix}_price")
    )
    return _add_distance_columns(window, prefix)


def _count_side_expr(prefixes: tuple[str, ...], side: str) -> pl.Expr:
    expr = pl.lit(0, dtype=pl.Int16)
    for prefix in prefixes:
        expr = expr + (pl.col(f"{prefix}_vwap_side") == side).cast(pl.Int16).fill_null(0)
    return expr


def _add_confluence(lf: pl.LazyFrame, prefixes: tuple[str, ...], output_prefix: str) -> pl.LazyFrame:
    return lf.with_columns(
        _count_side_expr(prefixes, "above").alias(f"{output_prefix}_above_count"),
        _count_side_expr(prefixes, "below").alias(f"{output_prefix}_below_count"),
        _count_side_expr(prefixes, "touch").alias(f"{output_prefix}_touch_count"),
    ).with_columns(
        (pl.col(f"{output_prefix}_above_count") - pl.col(f"{output_prefix}_below_count")).alias(f"{output_prefix}_net_side_score")
    )
```

- [ ] **Step 2: Implement `build_macro_vwap_premacro` without targets yet**

Replace `build_macro_vwap_premacro` with:

```python
def build_macro_vwap_premacro(path: str | Path = INPUT_PATH) -> pl.LazyFrame:
    base = _scan_ticks(path)
    dates = base.select("date").unique()
    out = dates
    for prefix, (anchor_second, checkpoint_second) in PREMACRO_SPECS.items():
        out = out.join(_anchored_vwap(base, prefix, anchor_second, checkpoint_second), on="date", how="left")
    out = _add_confluence(out, PREMACRO_FEATURE_PREFIXES, "premacro")
    for col in _TARGET_COLUMNS:
        out = out.with_columns(pl.lit(None).alias(col))
    return out.select(PREMACRO_COLUMNS).sort("date")
```

- [ ] **Step 3: Run focused VWAP tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py::test_vwap_side_uses_one_tick_touch_threshold test/test_macro_vwap_features.py::test_missing_and_zero_size_windows_produce_null_context -q
```

Expected: PASS for touch/null tests, target assertions still fail in broader suite.

- [ ] **Step 4: Commit pre-macro VWAP helper implementation**

```bash
git add features/macro_vwap_features.py
git commit -m "feat: add tick anchored premacro vwap features"
```

---

### Task 3: Implement tick-derived macro targets

**Files:**
- Modify: `features/macro_vwap_features.py`

- [ ] **Step 1: Add target helper functions after `_anchored_vwap`**

```python
def _state_expr(points_col: str) -> pl.Expr:
    points = pl.col(points_col)
    return (
        pl.when(points.is_null())
        .then(None)
        .when(points > 0)
        .then(pl.lit("bullish"))
        .when(points < 0)
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("neutral"))
    )


def _sign_expr(points_col: str) -> pl.Expr:
    points = pl.col(points_col)
    return pl.when(points.is_null()).then(None).when(points > 0).then(1).when(points < 0).then(-1).otherwise(0)


def _target_window(base: pl.LazyFrame, prefix: str, start_second: int, end_second: int) -> pl.LazyFrame:
    points_col = f"{prefix}_points"
    return (
        base.filter((pl.col("et_second") >= start_second) & (pl.col("et_second") < end_second))
        .sort("date", "ts_event", "intra_ts_rank")
        .group_by("date")
        .agg(
            pl.col("price").first().alias(f"{prefix}_open"),
            pl.col("price").last().alias(f"{prefix}_close"),
        )
        .with_columns((pl.col(f"{prefix}_close") - pl.col(f"{prefix}_open")).alias(points_col))
        .with_columns(
            _sign_expr(points_col).cast(pl.Int8).alias(f"{prefix}_sign"),
            _state_expr(points_col).alias(f"{prefix}_state"),
        )
        .select("date", points_col, f"{prefix}_sign", f"{prefix}_state")
    )


def _target_frame(base: pl.LazyFrame) -> pl.LazyFrame:
    targets = None
    for prefix, (start_second, end_second) in TARGET_SPECS.items():
        target = _target_window(base, prefix, start_second, end_second)
        targets = target if targets is None else targets.join(target, on="date", how="full", coalesce=True)
    if targets is None:
        return pl.LazyFrame(schema={"date": pl.Date})
    return targets
```

- [ ] **Step 2: Join targets in `build_macro_vwap_premacro`**

Replace the target-null block in `build_macro_vwap_premacro`:

```python
    out = _add_confluence(out, PREMACRO_FEATURE_PREFIXES, "premacro")
    for col in _TARGET_COLUMNS:
        out = out.with_columns(pl.lit(None).alias(col))
    return out.select(PREMACRO_COLUMNS).sort("date")
```

with:

```python
    out = _add_confluence(out, PREMACRO_FEATURE_PREFIXES, "premacro")
    out = out.join(_target_frame(base), on="date", how="left")
    return out.select(PREMACRO_COLUMNS).sort("date")
```

- [ ] **Step 3: Run pre-macro target tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py::test_build_macro_vwap_premacro_computes_tick_weighted_vwap_and_targets test/test_macro_vwap_features.py::test_summer_dst_uses_new_york_market_time -q
```

Expected: PASS.

- [ ] **Step 4: Commit target implementation**

```bash
git add features/macro_vwap_features.py
git commit -m "feat: add tick-derived macro vwap targets"
```

---

### Task 4: Implement intramacro features and optional barrier join

**Files:**
- Modify: `features/macro_vwap_features.py`

- [ ] **Step 1: Add barrier helper after `_target_frame`**

```python
def _empty_barrier_context() -> pl.LazyFrame:
    return pl.LazyFrame(
        schema={
            "date": pl.Date,
            "barrier_macro_trend_state": pl.String,
            "barrier_extreme": pl.String,
            "barrier_price": pl.Float64,
            "barrier_time": pl.Int64,
            "barrier_first10": pl.Boolean,
            "barrier_is_macro_extreme": pl.Boolean,
            "barrier_holds": pl.Boolean,
            "barrier_edge_case": pl.Boolean,
        }
    )


def _barrier_context(barrier_path: str | Path | None) -> pl.LazyFrame:
    if barrier_path is None:
        return _empty_barrier_context()
    path = Path(barrier_path)
    if not path.exists():
        return _empty_barrier_context()
    return pl.scan_parquet(path).select(
        pl.col("date").cast(pl.Date),
        pl.col("macro_trend_state").alias("barrier_macro_trend_state"),
        "barrier_extreme",
        "barrier_price",
        "barrier_time",
        "barrier_first10",
        "barrier_is_macro_extreme",
        "barrier_holds",
        pl.col("edge_case").alias("barrier_edge_case"),
    )
```

- [ ] **Step 2: Replace `build_macro_vwap_intramacro`**

```python
def build_macro_vwap_intramacro(
    path: str | Path = INPUT_PATH,
    barrier_path: str | Path | None = DEFAULT_BARRIER_PATH,
) -> pl.LazyFrame:
    base = _scan_ticks(path)
    dates = base.select("date").unique()
    out = dates
    for prefix, (anchor_second, checkpoint_second) in INTRAMACRO_SPECS.items():
        out = out.join(_anchored_vwap(base, prefix, anchor_second, checkpoint_second), on="date", how="left")
    out = _add_confluence(out, INTRAMACRO_FEATURE_PREFIXES, "intramacro")
    out = out.join(_target_frame(base), on="date", how="left")
    out = out.join(_barrier_context(barrier_path), on="date", how="left")
    return out.select(INTRAMACRO_COLUMNS).sort("date")
```

- [ ] **Step 3: Run intramacro tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py::test_strict_checkpoint_boundaries_exclude_equal_timestamp_ticks test/test_macro_vwap_features.py::test_build_macro_vwap_intramacro_joins_optional_barrier_context -q
```

Expected: PASS.

- [ ] **Step 4: Run all builder tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py -q
```

Expected: summary/writer tests still fail with `NotImplementedError`; builder tests pass.

- [ ] **Step 5: Commit intramacro implementation**

```bash
git add features/macro_vwap_features.py
git commit -m "feat: add intramacro vwap context features"
```

---

### Task 5: Implement diagnostic summaries

**Files:**
- Modify: `features/macro_vwap_features.py`

- [ ] **Step 1: Add summary helpers before `summarize_macro_vwap_features`**

```python
def _feature_prefixes(feature_set: str) -> tuple[str, ...]:
    if feature_set == "premacro":
        return PREMACRO_FEATURE_PREFIXES
    if feature_set == "intramacro":
        return INTRAMACRO_FEATURE_PREFIXES
    raise ValueError(f"feature_set must be 'premacro' or 'intramacro', got {feature_set!r}")


def _confluence_column(feature_set: str) -> str:
    return "premacro_net_side_score" if feature_set == "premacro" else "intramacro_net_side_score"


def _pct(count: int, denom: int) -> float | None:
    return (count / denom) * 100.0 if denom else None


def _summary_row(subset: pl.DataFrame, feature_set: str, feature_name: str, target_name: str, scope: str, bucket: str) -> dict:
    state_col = f"{target_name}_state"
    points_col = f"{target_name}_points"
    sample_size = subset.height
    bullish_count = subset.filter(pl.col(state_col) == "bullish").height if sample_size else 0
    bearish_count = subset.filter(pl.col(state_col) == "bearish").height if sample_size else 0
    neutral_count = subset.filter(pl.col(state_col) == "neutral").height if sample_size else 0
    return {
        "feature_set": feature_set,
        "feature_name": feature_name,
        "target_name": target_name,
        "scope": scope,
        "bucket": str(bucket),
        "sample_size": sample_size,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "bullish_pct": _pct(bullish_count, sample_size),
        "bearish_pct": _pct(bearish_count, sample_size),
        "neutral_pct": _pct(neutral_count, sample_size),
        "avg_target_points": subset.select(pl.col(points_col).mean()).item() if sample_size else None,
        "median_target_points": subset.select(pl.col(points_col).median()).item() if sample_size else None,
    }


def _bps_band_expr(prefix: str) -> pl.Expr:
    bps = pl.col(f"{prefix}_vwap_dist_bps")
    side = pl.col(f"{prefix}_vwap_side")
    abs_bps = bps.abs()
    return (
        pl.when(bps.is_null())
        .then(None)
        .when(side == "touch")
        .then(pl.lit("touch"))
        .when((bps < 0) & (abs_bps > 20))
        .then(pl.lit("below_gt_20"))
        .when((bps < 0) & (abs_bps > 10))
        .then(pl.lit("below_10_20"))
        .when((bps < 0) & (abs_bps > 5))
        .then(pl.lit("below_5_10"))
        .when((bps < 0) & (abs_bps > 2))
        .then(pl.lit("below_2_5"))
        .when(bps < 0)
        .then(pl.lit("below_0_2"))
        .when((bps > 0) & (abs_bps <= 2))
        .then(pl.lit("above_0_2"))
        .when((bps > 0) & (abs_bps <= 5))
        .then(pl.lit("above_2_5"))
        .when((bps > 0) & (abs_bps <= 10))
        .then(pl.lit("above_5_10"))
        .when((bps > 0) & (abs_bps <= 20))
        .then(pl.lit("above_10_20"))
        .otherwise(pl.lit("above_gt_20"))
    )


def _deciled_frame(df: pl.DataFrame, prefix: str) -> pl.DataFrame | None:
    value_col = f"{prefix}_vwap_dist_bps"
    non_null = df.filter(pl.col(value_col).is_not_null())
    if non_null.height < 10:
        return None
    if non_null.select(pl.col(value_col).n_unique()).item() < 10:
        return None
    return non_null.with_columns(
        ((pl.col(value_col).rank(method="ordinal") - 1) * 10 / non_null.height)
        .floor()
        .cast(pl.Int64)
        .clip(0, 9)
        .add(1)
        .cast(pl.String)
        .alias("_bucket")
    )


def _target_available(df: pl.DataFrame, target_name: str) -> bool:
    return f"{target_name}_points" in df.columns and f"{target_name}_state" in df.columns
```

- [ ] **Step 2: Replace `summarize_macro_vwap_features`**

```python
def summarize_macro_vwap_features(df: pl.DataFrame, feature_set: str) -> pl.DataFrame:
    prefixes = _feature_prefixes(feature_set)
    rows: list[dict] = []
    targets = [target for target in TARGET_PREFIXES if _target_available(df, target)]
    for prefix in prefixes:
        side_col = f"{prefix}_vwap_side"
        bps_col = f"{prefix}_vwap_dist_bps"
        if side_col not in df.columns or bps_col not in df.columns:
            continue
        banded = df.with_columns(_bps_band_expr(prefix).alias("_bucket"))
        deciled = _deciled_frame(df, prefix)
        for target in targets:
            for side in ["above", "below", "touch"]:
                subset = df.filter(pl.col(side_col) == side)
                rows.append(_summary_row(subset, feature_set, prefix, target, "side", side))
            for band in [
                "below_gt_20", "below_10_20", "below_5_10", "below_2_5", "below_0_2",
                "touch", "above_0_2", "above_2_5", "above_5_10", "above_10_20", "above_gt_20",
            ]:
                subset = banded.filter(pl.col("_bucket") == band)
                rows.append(_summary_row(subset, feature_set, prefix, target, "fixed_bps_band", band))
            if deciled is not None:
                for decile in [str(i) for i in range(1, 11)]:
                    rows.append(_summary_row(deciled.filter(pl.col("_bucket") == decile), feature_set, prefix, target, "decile", decile))

    confluence_col = _confluence_column(feature_set)
    if confluence_col in df.columns:
        for target in targets:
            for bucket in sorted(df.select(pl.col(confluence_col).drop_nulls().unique()).to_series().to_list()):
                subset = df.filter(pl.col(confluence_col) == bucket)
                rows.append(_summary_row(subset, feature_set, confluence_col, target, "confluence", str(bucket)))

    if feature_set == "intramacro" and "barrier_first10" in df.columns:
        for prefix in prefixes:
            side_col = f"{prefix}_vwap_side"
            for target in targets:
                for flag_col, scope in [("barrier_first10", "barrier_first10_by_side"), ("barrier_holds", "barrier_holds_by_side")]:
                    if flag_col not in df.columns:
                        continue
                    for side in ["above", "below", "touch"]:
                        for flag_value in [True, False]:
                            subset = df.filter((pl.col(side_col) == side) & (pl.col(flag_col) == flag_value))
                            rows.append(_summary_row(subset, feature_set, prefix, target, scope, f"{side}_{str(flag_value).lower()}"))

    if not rows:
        return pl.DataFrame(schema={col: pl.Null for col in SUMMARY_COLUMNS})
    return pl.DataFrame(rows, infer_schema_length=None).select(SUMMARY_COLUMNS)
```

- [ ] **Step 3: Run summary test**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py::test_summarize_macro_vwap_features_reports_side_bands_deciles_and_confluence -q
```

Expected: PASS.

- [ ] **Step 4: Commit summaries**

```bash
git add features/macro_vwap_features.py
git commit -m "feat: summarize macro vwap diagnostics"
```

---

### Task 6: Implement writers and script entrypoint

**Files:**
- Modify: `features/macro_vwap_features.py`

- [ ] **Step 1: Replace `write_macro_vwap_features` and `main`**

```python
def _write_df(path: str | Path, df: pl.DataFrame) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)
    return output


def write_macro_vwap_features(
    input_path: str | Path = INPUT_PATH,
    premacro_output_path: str | Path = PREMACRO_OUTPUT_PATH,
    premacro_summary_output_path: str | Path = PREMACRO_SUMMARY_OUTPUT_PATH,
    intramacro_output_path: str | Path = INTRAMACRO_OUTPUT_PATH,
    intramacro_summary_output_path: str | Path = INTRAMACRO_SUMMARY_OUTPUT_PATH,
    barrier_path: str | Path | None = DEFAULT_BARRIER_PATH,
) -> tuple[Path, Path, Path, Path]:
    premacro = build_macro_vwap_premacro(input_path).collect(engine="streaming")
    intramacro = build_macro_vwap_intramacro(input_path, barrier_path=barrier_path).collect(engine="streaming")
    premacro_summary = summarize_macro_vwap_features(premacro, feature_set="premacro")
    intramacro_summary = summarize_macro_vwap_features(intramacro, feature_set="intramacro")

    premacro_output = _write_df(premacro_output_path, premacro)
    premacro_summary_output = _write_df(premacro_summary_output_path, premacro_summary)
    intramacro_output = _write_df(intramacro_output_path, intramacro)
    intramacro_summary_output = _write_df(intramacro_summary_output_path, intramacro_summary)
    return premacro_output, premacro_summary_output, intramacro_output, intramacro_summary_output


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    outputs = write_macro_vwap_features()
    print(f"[OK] Wrote macro VWAP premacro features -> {outputs[0]}")
    print(f"[OK] Wrote macro VWAP premacro summary -> {outputs[1]}")
    print(f"[OK] Wrote macro VWAP intramacro features -> {outputs[2]}")
    print(f"[OK] Wrote macro VWAP intramacro summary -> {outputs[3]}")
```

- [ ] **Step 2: Run writer test**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py::test_write_macro_vwap_features_writes_four_outputs -q
```

Expected: PASS.

- [ ] **Step 3: Run full VWAP test file**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py -q
```

Expected: all tests PASS.

- [ ] **Step 4: Commit writer/entrypoint**

```bash
git add features/macro_vwap_features.py test/test_macro_vwap_features.py
git commit -m "feat: write macro vwap feature outputs"
```

---

### Task 7: Regression tests and runtime generation

**Files:**
- Modify only if verification reveals a concrete defect:
  - `features/macro_vwap_features.py`
  - `test/test_macro_vwap_features.py`

- [ ] **Step 1: Run VWAP tests**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent tick/macro tests**

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py test/test_macro_1550_barrier.py test/test_macro_extreme_timing.py test/test_tick_density.py test/test_volume_delta.py -q
```

Expected: PASS.

- [ ] **Step 3: Generate real outputs**

```bash
.venv/bin/python -m features.macro_vwap_features
```

Expected output lines:

```text
[OK] Wrote macro VWAP premacro features -> outputs/nq_macro_vwap_premacro.parquet
[OK] Wrote macro VWAP premacro summary -> outputs/nq_macro_vwap_premacro_summary.parquet
[OK] Wrote macro VWAP intramacro features -> outputs/nq_macro_vwap_intramacro.parquet
[OK] Wrote macro VWAP intramacro summary -> outputs/nq_macro_vwap_intramacro_summary.parquet
```

- [ ] **Step 4: Inspect output schemas and row counts**

Run:

```bash
.venv/bin/python - <<'PY'
import polars as pl
for path in [
    "outputs/nq_macro_vwap_premacro.parquet",
    "outputs/nq_macro_vwap_premacro_summary.parquet",
    "outputs/nq_macro_vwap_intramacro.parquet",
    "outputs/nq_macro_vwap_intramacro_summary.parquet",
]:
    df = pl.read_parquet(path)
    print(path, df.height, len(df.columns))
    print(df.head(3))
PY
```

Expected:
- Feature outputs have one row per trade date with available ticks.
- Summary outputs have non-zero rows.
- No source files under `input-data/` changed.

- [ ] **Step 5: Commit final verified implementation**

If runtime generation required no code changes, commit only code/tests already staged by prior tasks is unnecessary. If fixes were made:

```bash
git add features/macro_vwap_features.py test/test_macro_vwap_features.py
git commit -m "fix: harden macro vwap runtime generation"
```

---

### Task 8: Final verification checklist

**Files:**
- No file changes unless previous verification found defects.

- [ ] **Step 1: Check git status**

```bash
git status --short
```

Expected: no uncommitted code/test changes. Generated output parquet files may appear ignored/untracked depending on `.gitignore`; do not commit large outputs unless explicitly requested.

- [ ] **Step 2: Record verification evidence in final response**

Include:
- Test commands run.
- Runtime command run.
- Output files generated.
- Any caveats, especially if real tick runtime was skipped.

