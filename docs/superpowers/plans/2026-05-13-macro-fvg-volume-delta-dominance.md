# Macro FVG Volume Delta Dominance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add integrated macro FVG summaries that compare existing FVG win rate across aligned and absolute 5-second volume-delta dominance quantiles.

**Architecture:** Create a focused helper module, `features/macro_fvg_delta_dominance.py`, responsible for validating 5-second volume-delta inputs, joining FVG events to the confirmation-time delta bucket, computing aligned/absolute dominance columns, and assigning quartile labels. Modify `features/macro_fvg_study.py` only to call the helper, extend event/summary schemas, and append four success-context summary scopes.

**Tech Stack:** Python, Polars, pytest, existing script-first repo patterns. Use `.venv/bin/python` for all commands.

---

## File Structure

- Create: `features/macro_fvg_delta_dominance.py`
  - Owns all delta-dominance enrichment and quantile assignment.
  - Public functions:
    - `load_macro_volume_delta_5s(path: str | Path) -> pl.DataFrame`
    - `enrich_fvg_events_with_delta_dominance(events: pl.DataFrame, volume_delta_5s: pl.DataFrame, quantile_count: int = 4) -> pl.DataFrame`
    - `try_enrich_fvg_events_with_delta_dominance(events: pl.DataFrame, volume_delta_path: str | Path, quantile_count: int = 4) -> pl.DataFrame`
- Modify: `features/macro_fvg_study.py`
  - Import helper and `OUTPUT_MACRO_5S_PATH`.
  - Extend `EVENT_COLUMNS` and `SUMMARY_COLUMNS`.
  - Add dominance summary builders that reuse `_group_success_context_stats`.
  - Call enrichment before summary building in `run_macro_fvg_study()`.
- Modify: `test/test_macro_fvg_study.py`
  - Add helper imports and synthetic delta fixtures.
  - Add focused unit tests for enrichment, quantiles, summaries, missing path, and schema errors.

---

### Task 1: Create failing enrichment tests

**Files:**
- Modify: `test/test_macro_fvg_study.py`
- Create later: `features/macro_fvg_delta_dominance.py`

- [ ] **Step 1: Add imports for the future helper functions**

Edit the import block near the top of `test/test_macro_fvg_study.py`. After the existing `from features.macro_fvg_study import (...)` block, add:

```python
from features.macro_fvg_delta_dominance import (
    enrich_fvg_events_with_delta_dominance,
    load_macro_volume_delta_5s,
    try_enrich_fvg_events_with_delta_dominance,
)
```

- [ ] **Step 2: Add minimal synthetic event and delta fixture helpers**

Append this helper code after `make_utc_bars()` in `test/test_macro_fvg_study.py`:

```python
def make_delta_events_for_dominance():
    return pl.DataFrame(
        [
            {
                "date": datetime(2025, 1, 2).date(),
                "fvg_side": "bullish",
                "confirmed_at": datetime(2025, 1, 2, 15, 52, 10),
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": True,
                "mfe_pct_to_1559": 0.010,
                "mae_pct_to_1559": 0.002,
            },
            {
                "date": datetime(2025, 1, 2).date(),
                "fvg_side": "bearish",
                "confirmed_at": datetime(2025, 1, 2, 15, 52, 15),
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
            {
                "date": datetime(2025, 1, 2).date(),
                "fvg_side": "bullish",
                "confirmed_at": datetime(2025, 1, 2, 15, 52, 20),
                "is_confirmable_by_1559": True,
                "retraced_by_1559": False,
                "successful_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
            {
                "date": datetime(2025, 1, 2).date(),
                "fvg_side": "bearish",
                "confirmed_at": datetime(2025, 1, 2, 15, 52, 25),
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": True,
                "mfe_pct_to_1559": 0.020,
                "mae_pct_to_1559": 0.003,
            },
        ]
    )


def make_delta_5s_for_dominance():
    return pl.DataFrame(
        [
            {
                "trade_date_et": datetime(2025, 1, 2).date(),
                "macro_bucket_index": 26,
                "volume_delta": 20,
                "delta_imbalance": 0.20,
                "tick_delta": 2,
                "classified_share": 0.90,
                "total_size": 100,
                "is_empty": False,
            },
            {
                "trade_date_et": datetime(2025, 1, 2).date(),
                "macro_bucket_index": 27,
                "volume_delta": 30,
                "delta_imbalance": 0.30,
                "tick_delta": 3,
                "classified_share": 0.85,
                "total_size": 100,
                "is_empty": False,
            },
            {
                "trade_date_et": datetime(2025, 1, 2).date(),
                "macro_bucket_index": 28,
                "volume_delta": -40,
                "delta_imbalance": -0.40,
                "tick_delta": -4,
                "classified_share": 0.95,
                "total_size": 100,
                "is_empty": False,
            },
            {
                "trade_date_et": datetime(2025, 1, 2).date(),
                "macro_bucket_index": 29,
                "volume_delta": -60,
                "delta_imbalance": -0.60,
                "tick_delta": -6,
                "classified_share": 0.80,
                "total_size": 100,
                "is_empty": False,
            },
        ]
    )
```

- [ ] **Step 3: Add tests for bucket join, aligned signs, absolute dominance, and quantiles**

Append these tests before `test_run_macro_fvg_study_writes_parquet_and_figures`:

```python
def test_enriches_fvg_events_with_confirmation_bucket_delta_dominance():
    enriched = enrich_fvg_events_with_delta_dominance(
        make_delta_events_for_dominance(),
        make_delta_5s_for_dominance(),
    )

    first = enriched.row(0, named=True)
    assert first["fvg_delta_bucket_index"] == 26
    assert first["fvg_delta_volume_delta"] == 20
    assert first["fvg_delta_imbalance"] == 0.20
    assert first["fvg_delta_tick_delta"] == 2
    assert first["fvg_delta_classified_share"] == 0.90
    assert first["fvg_delta_total_size"] == 100
    assert not first["fvg_delta_is_empty"]


def test_delta_dominance_alignment_and_absolute_values_respect_fvg_side():
    enriched = enrich_fvg_events_with_delta_dominance(
        make_delta_events_for_dominance(),
        make_delta_5s_for_dominance(),
    )

    rows = enriched.to_dicts()
    assert rows[0]["fvg_side"] == "bullish"
    assert rows[0]["aligned_delta_imbalance"] == pytest.approx(0.20)
    assert rows[0]["abs_delta_imbalance"] == pytest.approx(0.20)
    assert rows[0]["aligned_volume_delta"] == 20
    assert rows[0]["abs_volume_delta"] == 20
    assert rows[0]["aligned_tick_delta"] == 2
    assert rows[0]["abs_tick_delta"] == 2

    assert rows[1]["fvg_side"] == "bearish"
    assert rows[1]["fvg_delta_imbalance"] == pytest.approx(0.30)
    assert rows[1]["aligned_delta_imbalance"] == pytest.approx(-0.30)
    assert rows[1]["abs_delta_imbalance"] == pytest.approx(0.30)
    assert rows[1]["aligned_volume_delta"] == -30
    assert rows[1]["abs_volume_delta"] == 30
    assert rows[1]["aligned_tick_delta"] == -3
    assert rows[1]["abs_tick_delta"] == 3

    assert rows[3]["fvg_side"] == "bearish"
    assert rows[3]["fvg_delta_imbalance"] == pytest.approx(-0.60)
    assert rows[3]["aligned_delta_imbalance"] == pytest.approx(0.60)
    assert rows[3]["abs_delta_imbalance"] == pytest.approx(0.60)


def test_delta_dominance_quantiles_are_ranked_low_to_high():
    enriched = enrich_fvg_events_with_delta_dominance(
        make_delta_events_for_dominance(),
        make_delta_5s_for_dominance(),
    )

    aligned_by_value = (
        enriched.select("aligned_delta_imbalance", "aligned_delta_imbalance_quantile")
        .sort("aligned_delta_imbalance")
        .to_dicts()
    )
    assert [row["aligned_delta_imbalance_quantile"] for row in aligned_by_value] == [
        "q1_lowest",
        "q2",
        "q3",
        "q4_highest",
    ]

    abs_by_value = (
        enriched.select("abs_delta_imbalance", "abs_delta_imbalance_quantile")
        .sort("abs_delta_imbalance")
        .to_dicts()
    )
    assert [row["abs_delta_imbalance_quantile"] for row in abs_by_value] == [
        "q1_lowest",
        "q2",
        "q3",
        "q4_highest",
    ]
```

- [ ] **Step 4: Run focused tests and verify they fail for the missing module**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py::test_enriches_fvg_events_with_confirmation_bucket_delta_dominance -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'features.macro_fvg_delta_dominance'`.

---

### Task 2: Implement delta-dominance helper module

**Files:**
- Create: `features/macro_fvg_delta_dominance.py`
- Test: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Create helper module with validation, join, dominance, and quantiles**

Create `features/macro_fvg_delta_dominance.py` with this complete content:

```python
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import polars as pl

try:
    from volume_delta import OUTPUT_MACRO_5S_PATH
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from volume_delta import OUTPUT_MACRO_5S_PATH

DELTA_DOMINANCE_COLUMNS = [
    "fvg_delta_bucket_index",
    "fvg_delta_volume_delta",
    "fvg_delta_imbalance",
    "fvg_delta_tick_delta",
    "fvg_delta_classified_share",
    "fvg_delta_total_size",
    "fvg_delta_is_empty",
    "aligned_delta_imbalance",
    "abs_delta_imbalance",
    "aligned_volume_delta",
    "abs_volume_delta",
    "aligned_tick_delta",
    "abs_tick_delta",
    "aligned_delta_imbalance_quantile",
    "abs_delta_imbalance_quantile",
]

REQUIRED_EVENT_COLUMNS = {"date", "fvg_side", "confirmed_at"}
REQUIRED_DELTA_COLUMNS = {
    "trade_date_et",
    "macro_bucket_index",
    "volume_delta",
    "delta_imbalance",
    "tick_delta",
    "classified_share",
    "total_size",
    "is_empty",
}

QUANTILE_LOW_LABEL = "q1_lowest"
QUANTILE_HIGH_LABEL_TEMPLATE = "q{bucket}_highest"


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _null_expr_for_dtype(column: str) -> pl.Expr:
    if column in {
        "fvg_delta_bucket_index",
        "fvg_delta_volume_delta",
        "fvg_delta_tick_delta",
        "fvg_delta_total_size",
        "aligned_volume_delta",
        "abs_volume_delta",
        "aligned_tick_delta",
        "abs_tick_delta",
    }:
        return pl.lit(None, dtype=pl.Int64).alias(column)
    if column == "fvg_delta_is_empty":
        return pl.lit(None, dtype=pl.Boolean).alias(column)
    return pl.lit(None, dtype=pl.Utf8 if column.endswith("_quantile") else pl.Float64).alias(column)


def _with_empty_delta_columns(events: pl.DataFrame) -> pl.DataFrame:
    existing = set(events.columns)
    return events.with_columns(
        [_null_expr_for_dtype(column) for column in DELTA_DOMINANCE_COLUMNS if column not in existing]
    )


def load_macro_volume_delta_5s(path: str | Path = OUTPUT_MACRO_5S_PATH) -> pl.DataFrame:
    delta_path = Path(path)
    delta = pl.read_parquet(delta_path)
    missing = _missing_columns(delta, REQUIRED_DELTA_COLUMNS)
    if missing:
        raise ValueError(f"Missing volume-delta columns: {missing}")
    return delta


def _with_confirmation_bucket(events: pl.DataFrame) -> pl.DataFrame:
    confirmed = pl.col("confirmed_at")
    seconds_since_macro_start = (
        (confirmed.dt.hour() - 15).cast(pl.Int64) * 3600
        + (confirmed.dt.minute() - 50).cast(pl.Int64) * 60
        + confirmed.dt.second().cast(pl.Int64)
    )
    return events.with_columns(
        fvg_delta_bucket_index=pl.when((seconds_since_macro_start >= 0) & (seconds_since_macro_start < 600))
        .then((seconds_since_macro_start // 5).cast(pl.Int64))
        .otherwise(None)
    )


def _quantile_labels(bucket_count: int) -> list[str]:
    return [
        QUANTILE_LOW_LABEL if i == 1 else (QUANTILE_HIGH_LABEL_TEMPLATE.format(bucket=i) if i == bucket_count else f"q{i}")
        for i in range(1, bucket_count + 1)
    ]


def _with_rank_quantile(frame: pl.DataFrame, value_col: str, output_col: str, bucket_count: int) -> pl.DataFrame:
    if frame.is_empty() or value_col not in frame.columns:
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(output_col))

    non_null_count = frame.select(pl.col(value_col).drop_nulls().len()).item()
    if non_null_count == 0:
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(output_col))

    unique_count = frame.select(pl.col(value_col).drop_nulls().n_unique()).item()
    q = int(min(bucket_count, unique_count, non_null_count))
    if q <= 0:
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(output_col))

    labels = _quantile_labels(q)
    label_lookup = pl.DataFrame(
        {
            "_bucket": list(range(q)),
            output_col: labels,
        }
    )

    ranked = frame.with_columns(
        pl.when(pl.col(value_col).is_not_null())
        .then(pl.col(value_col).rank(method="ordinal") - 1)
        .otherwise(None)
        .alias("_rank")
    ).with_columns(
        pl.when(pl.col("_rank").is_not_null())
        .then(((pl.col("_rank") * q) / non_null_count).floor().cast(pl.Int64).clip(0, q - 1))
        .otherwise(None)
        .alias("_bucket")
    )

    return ranked.join(label_lookup, on="_bucket", how="left").drop(["_rank", "_bucket"])


def enrich_fvg_events_with_delta_dominance(
    events: pl.DataFrame,
    volume_delta_5s: pl.DataFrame,
    quantile_count: int = 4,
) -> pl.DataFrame:
    if quantile_count < 1:
        raise ValueError(f"quantile_count must be >= 1, got {quantile_count}")

    event_missing = _missing_columns(events, REQUIRED_EVENT_COLUMNS)
    if event_missing:
        raise ValueError(f"Missing FVG event columns: {event_missing}")

    delta_missing = _missing_columns(volume_delta_5s, REQUIRED_DELTA_COLUMNS)
    if delta_missing:
        raise ValueError(f"Missing volume-delta columns: {delta_missing}")

    if events.is_empty():
        return _with_empty_delta_columns(events)

    delta = volume_delta_5s.select(
        pl.col("trade_date_et").alias("date"),
        pl.col("macro_bucket_index").cast(pl.Int64).alias("fvg_delta_bucket_index"),
        pl.col("volume_delta").cast(pl.Int64).alias("fvg_delta_volume_delta"),
        pl.col("delta_imbalance").cast(pl.Float64).alias("fvg_delta_imbalance"),
        pl.col("tick_delta").cast(pl.Int64).alias("fvg_delta_tick_delta"),
        pl.col("classified_share").cast(pl.Float64).alias("fvg_delta_classified_share"),
        pl.col("total_size").cast(pl.Int64).alias("fvg_delta_total_size"),
        pl.col("is_empty").cast(pl.Boolean).alias("fvg_delta_is_empty"),
    )

    enriched = (
        _with_confirmation_bucket(events)
        .join(delta, on=["date", "fvg_delta_bucket_index"], how="left")
        .with_columns(
            aligned_delta_imbalance=pl.when(pl.col("fvg_side") == "bearish")
            .then(-pl.col("fvg_delta_imbalance"))
            .otherwise(pl.col("fvg_delta_imbalance")),
            abs_delta_imbalance=pl.col("fvg_delta_imbalance").abs(),
            aligned_volume_delta=pl.when(pl.col("fvg_side") == "bearish")
            .then(-pl.col("fvg_delta_volume_delta"))
            .otherwise(pl.col("fvg_delta_volume_delta")),
            abs_volume_delta=pl.col("fvg_delta_volume_delta").abs(),
            aligned_tick_delta=pl.when(pl.col("fvg_side") == "bearish")
            .then(-pl.col("fvg_delta_tick_delta"))
            .otherwise(pl.col("fvg_delta_tick_delta")),
            abs_tick_delta=pl.col("fvg_delta_tick_delta").abs(),
        )
    )
    enriched = _with_rank_quantile(
        enriched,
        "aligned_delta_imbalance",
        "aligned_delta_imbalance_quantile",
        quantile_count,
    )
    return _with_rank_quantile(
        enriched,
        "abs_delta_imbalance",
        "abs_delta_imbalance_quantile",
        quantile_count,
    )


def try_enrich_fvg_events_with_delta_dominance(
    events: pl.DataFrame,
    volume_delta_path: str | Path = OUTPUT_MACRO_5S_PATH,
    quantile_count: int = 4,
) -> pl.DataFrame:
    delta_path = Path(volume_delta_path)
    if not delta_path.exists():
        warnings.warn(
            f"Skipping macro FVG volume-delta dominance: missing {delta_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _with_empty_delta_columns(events)
    return enrich_fvg_events_with_delta_dominance(
        events,
        load_macro_volume_delta_5s(delta_path),
        quantile_count=quantile_count,
    )
```

- [ ] **Step 2: Run focused enrichment tests**

Run:

```bash
.venv/bin/python -m pytest \
  test/test_macro_fvg_study.py::test_enriches_fvg_events_with_confirmation_bucket_delta_dominance \
  test/test_macro_fvg_study.py::test_delta_dominance_alignment_and_absolute_values_respect_fvg_side \
  test/test_macro_fvg_study.py::test_delta_dominance_quantiles_are_ranked_low_to_high \
  -q
```

Expected: PASS.

- [ ] **Step 3: Commit helper and tests**

Run:

```bash
git add features/macro_fvg_delta_dominance.py test/test_macro_fvg_study.py
git commit -m "feat: enrich macro fvgs with delta dominance"
```

---

### Task 3: Add missing-path and schema-error tests

**Files:**
- Modify: `test/test_macro_fvg_study.py`
- Modify if needed: `features/macro_fvg_delta_dominance.py`

- [ ] **Step 1: Add tests for optional missing parquet and required delta columns**

Append these tests before `test_run_macro_fvg_study_writes_parquet_and_figures`:

```python
def test_try_enrich_delta_dominance_missing_path_preserves_events_with_null_delta_columns(tmp_path):
    missing_path = tmp_path / "missing_delta.parquet"

    with pytest.warns(RuntimeWarning, match="Skipping macro FVG volume-delta dominance"):
        enriched = try_enrich_fvg_events_with_delta_dominance(
            make_delta_events_for_dominance(),
            missing_path,
        )

    assert enriched.height == 4
    assert "aligned_delta_imbalance_quantile" in enriched.columns
    assert enriched["aligned_delta_imbalance_quantile"].null_count() == 4
    assert enriched["fvg_delta_imbalance"].null_count() == 4


def test_load_macro_volume_delta_5s_requires_expected_columns(tmp_path):
    bad_path = tmp_path / "bad_delta.parquet"
    pl.DataFrame(
        [
            {
                "trade_date_et": datetime(2025, 1, 2).date(),
                "macro_bucket_index": 1,
                "volume_delta": 10,
            }
        ]
    ).write_parquet(bad_path)

    with pytest.raises(ValueError, match="Missing volume-delta columns"):
        load_macro_volume_delta_5s(bad_path)
```

- [ ] **Step 2: Run the new tests**

Run:

```bash
.venv/bin/python -m pytest \
  test/test_macro_fvg_study.py::test_try_enrich_delta_dominance_missing_path_preserves_events_with_null_delta_columns \
  test/test_macro_fvg_study.py::test_load_macro_volume_delta_5s_requires_expected_columns \
  -q
```

Expected: PASS. If `test_load_macro_volume_delta_5s_requires_expected_columns` fails because `pl.read_parquet` itself errors, adjust only the fixture to include valid parquet data with the shown minimal columns; do not weaken the required-column validation.

- [ ] **Step 3: Commit error-handling tests**

Run:

```bash
git add test/test_macro_fvg_study.py features/macro_fvg_delta_dominance.py
git commit -m "test: cover macro fvg delta dominance inputs"
```

---

### Task 4: Integrate enrichment and summary scopes into macro FVG study

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Add tests for dominance summary scopes and run integration**

Append this test before `test_run_macro_fvg_study_writes_parquet_and_figures`:

```python
def test_build_summary_tables_includes_delta_dominance_success_context_scopes():
    enriched = enrich_fvg_events_with_delta_dominance(
        make_delta_events_for_dominance(),
        make_delta_5s_for_dominance(),
    )

    summary = macro_fvg_study.build_summary_tables(enriched)
    scopes = set(summary["summary_scope"].to_list())

    assert "success_context_aligned_delta_imbalance_quantile" in scopes
    assert "success_context_abs_delta_imbalance_quantile" in scopes
    assert "success_context_side_aligned_delta_imbalance_quantile" in scopes
    assert "success_context_side_abs_delta_imbalance_quantile" in scopes

    high_aligned = filter_one(
        summary,
        (pl.col("summary_scope") == "success_context_aligned_delta_imbalance_quantile")
        & (pl.col("aligned_delta_imbalance_quantile") == "q4_highest"),
    )
    assert high_aligned["n_confirmable"] == 1
    assert high_aligned["n_retraced"] == 1
    assert high_aligned["n_successful"] == 1
    assert high_aligned["successful_share_of_confirmable"] == 1.0

    side_abs = filter_one(
        summary,
        (pl.col("summary_scope") == "success_context_side_abs_delta_imbalance_quantile")
        & (pl.col("fvg_side") == "bearish")
        & (pl.col("abs_delta_imbalance_quantile") == "q4_highest"),
    )
    assert side_abs["n_confirmable"] == 1
    assert side_abs["n_successful"] == 1
    assert side_abs["successful_share_of_confirmable"] == 1.0
```

- [ ] **Step 2: Run the new summary test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py::test_build_summary_tables_includes_delta_dominance_success_context_scopes -q
```

Expected: FAIL because the new summary scopes are not present yet.

- [ ] **Step 3: Import helper and extend constants in `features/macro_fvg_study.py`**

Add this import block after the existing `utils.minute_bars` import fallback block:

```python
try:
    from features.macro_fvg_delta_dominance import (
        DELTA_DOMINANCE_COLUMNS,
        try_enrich_fvg_events_with_delta_dominance,
    )
    from volume_delta import OUTPUT_MACRO_5S_PATH
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from features.macro_fvg_delta_dominance import (
        DELTA_DOMINANCE_COLUMNS,
        try_enrich_fvg_events_with_delta_dominance,
    )
    from volume_delta import OUTPUT_MACRO_5S_PATH
```

In `SUMMARY_COLUMNS`, add these two columns immediately after `bar2_volume_bucket`:

```python
    "aligned_delta_imbalance_quantile",
    "abs_delta_imbalance_quantile",
```

Replace the end of `EVENT_COLUMNS` so it includes the delta dominance columns. Change:

```python
    "gap_size_bucket_225", "stacked_continuation_fvg", "stack_predecessor_assigned_at",
]
```

To:

```python
    "gap_size_bucket_225", "stacked_continuation_fvg", "stack_predecessor_assigned_at",
    *DELTA_DOMINANCE_COLUMNS,
]
```

- [ ] **Step 4: Add dominance summary builder functions**

In `features/macro_fvg_study.py`, insert these functions after `build_success_context_alignment_bucket_stacked_flag_summary()`:

```python
def _filter_non_null(events: pl.DataFrame, column: str) -> pl.DataFrame:
    return events.filter(pl.col(column).is_not_null()) if column in events.columns else events.clear()


def build_success_context_aligned_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    filtered = _filter_non_null(events, "aligned_delta_imbalance_quantile")
    return _group_success_context_stats(
        filtered,
        ["aligned_delta_imbalance_quantile"],
        "success_context_aligned_delta_imbalance_quantile",
    )


def build_success_context_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    filtered = _filter_non_null(events, "abs_delta_imbalance_quantile")
    return _group_success_context_stats(
        filtered,
        ["abs_delta_imbalance_quantile"],
        "success_context_abs_delta_imbalance_quantile",
    )


def build_success_context_side_aligned_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    filtered = _filter_non_null(events, "aligned_delta_imbalance_quantile")
    return _group_success_context_stats(
        filtered,
        ["fvg_side", "aligned_delta_imbalance_quantile"],
        "success_context_side_aligned_delta_imbalance_quantile",
    )


def build_success_context_side_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    filtered = _filter_non_null(events, "abs_delta_imbalance_quantile")
    return _group_success_context_stats(
        filtered,
        ["fvg_side", "abs_delta_imbalance_quantile"],
        "success_context_side_abs_delta_imbalance_quantile",
    )
```

Then update `build_summary_tables()` by appending these four builders to the `frames` list after `build_success_context_alignment_bucket_stacked_flag_summary(events),`:

```python
        build_success_context_aligned_delta_imbalance_quantile_summary(events),
        build_success_context_abs_delta_imbalance_quantile_summary(events),
        build_success_context_side_aligned_delta_imbalance_quantile_summary(events),
        build_success_context_side_abs_delta_imbalance_quantile_summary(events),
```

- [ ] **Step 5: Update `run_macro_fvg_study()` to enrich before summary creation**

Change the function signature from:

```python
def run_macro_fvg_study(
    input_path: Path = INPUT_PATH,
    events_output_path: Path = EVENTS_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[pl.DataFrame, pl.DataFrame]:
```

To:

```python
def run_macro_fvg_study(
    input_path: Path = INPUT_PATH,
    events_output_path: Path = EVENTS_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    figures_dir: Path = FIGURES_DIR,
    volume_delta_5s_path: Path = OUTPUT_MACRO_5S_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
```

Then insert enrichment after outcome scanning:

```python
    events = scan_fvg_outcomes_until_1559_close(events, bars)
    events = try_enrich_fvg_events_with_delta_dominance(events, volume_delta_5s_path)
    summary = build_summary_tables(events)
```

- [ ] **Step 6: Run the new summary test**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py::test_build_summary_tables_includes_delta_dominance_success_context_scopes -q
```

Expected: PASS.

- [ ] **Step 7: Commit integration**

Run:

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: summarize macro fvg delta dominance"
```

---

### Task 5: Add end-to-end run test with delta parquet

**Files:**
- Modify: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Update the existing run test to pass a delta parquet**

In `test_run_macro_fvg_study_writes_parquet_and_figures`, add this path next to the existing output paths:

```python
    delta_path = tmp_path / "nq_macro_volume_delta_5s.parquet"
```

Before calling `run_macro_fvg_study(...)`, write a small delta parquet that covers the expected event confirmation buckets:

```python
    pl.DataFrame(
        [
            {
                "trade_date_et": datetime(2025, 1, 2).date(),
                "macro_bucket_index": 24,
                "volume_delta": -50,
                "delta_imbalance": -0.50,
                "tick_delta": -5,
                "classified_share": 1.0,
                "total_size": 100,
                "is_empty": False,
            },
            {
                "trade_date_et": datetime(2025, 1, 2).date(),
                "macro_bucket_index": 84,
                "volume_delta": -20,
                "delta_imbalance": -0.20,
                "tick_delta": -2,
                "classified_share": 1.0,
                "total_size": 100,
                "is_empty": False,
            },
        ]
    ).write_parquet(delta_path)
```

Pass the new argument in the run call:

```python
    run_macro_fvg_study(
        input_path=input_path,
        events_output_path=events_path,
        summary_output_path=summary_path,
        figures_dir=figures_dir,
        volume_delta_5s_path=delta_path,
    )
```

Add these assertions after reading `events` and `summary`:

```python
    assert "aligned_delta_imbalance" in events.columns
    assert "abs_delta_imbalance_quantile" in events.columns
    assert "success_context_aligned_delta_imbalance_quantile" in set(summary["summary_scope"].to_list())
    assert "success_context_abs_delta_imbalance_quantile" in set(summary["summary_scope"].to_list())
```

- [ ] **Step 2: Run the end-to-end test**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures -q
```

Expected: PASS.

- [ ] **Step 3: Commit end-to-end coverage**

Run:

```bash
git add test/test_macro_fvg_study.py
git commit -m "test: cover fvg delta dominance pipeline"
```

---

### Task 6: Full verification and real output regeneration

**Files:**
- Runtime outputs under `outputs/`
- No source-code changes expected unless verification fails.

- [ ] **Step 1: Run focused macro FVG tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run related volume-delta tests**

Run:

```bash
.venv/bin/python -m pytest test/test_volume_delta.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run the main tick/macro focused set**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py test/test_volume_delta.py test/test_tick_density.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Ensure 5-second volume-delta output exists**

Run:

```bash
ls -lh outputs/nq_macro_volume_delta_5s.parquet
```

Expected: file exists. If missing, regenerate it:

```bash
.venv/bin/python volume_delta.py
```

Expected output includes:

```text
[OK] Wrote volume delta → outputs/nq_macro_volume_delta_5s.parquet
```

- [ ] **Step 5: Regenerate macro FVG outputs**

Run:

```bash
.venv/bin/python -m features.macro_fvg_study
```

Expected: command exits 0 and writes:

- `outputs/nq_macro_fvg_events.parquet`
- `outputs/nq_macro_fvg_summary.parquet`
- existing FVG figures under `outputs/figs/fvg/`

- [ ] **Step 6: Inspect new summary scopes**

Run:

```bash
.venv/bin/python - <<'PY'
import polars as pl
summary = pl.read_parquet('outputs/nq_macro_fvg_summary.parquet')
scopes = [
    'success_context_aligned_delta_imbalance_quantile',
    'success_context_abs_delta_imbalance_quantile',
    'success_context_side_aligned_delta_imbalance_quantile',
    'success_context_side_abs_delta_imbalance_quantile',
]
print(summary.filter(pl.col('summary_scope').is_in(scopes)).select([
    'summary_scope',
    'fvg_side',
    'aligned_delta_imbalance_quantile',
    'abs_delta_imbalance_quantile',
    'n_total',
    'n_confirmable',
    'n_retraced',
    'n_successful',
    'retrace_rate',
    'success_after_retrace_rate',
    'successful_share_of_confirmable',
]).sort(['summary_scope', 'fvg_side', 'aligned_delta_imbalance_quantile', 'abs_delta_imbalance_quantile']))
PY
```

Expected: printed table contains rows for the four new scopes, with non-null quantile labels in the appropriate quantile column.

- [ ] **Step 7: Check git status**

Run:

```bash
git status --short
```

Expected: source files changed only if not yet committed; generated outputs may appear modified/untracked depending repo tracking. Do not commit large generated outputs unless they are intentionally tracked already.

- [ ] **Step 8: Final commit if any source changes remain**

Run:

```bash
git add features/macro_fvg_delta_dominance.py features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg delta dominance study"
```

Expected: commit succeeds if there are uncommitted source changes. If previous task commits already included all source changes, this command may report nothing to commit; that is acceptable.

---

## Self-Review Notes

Spec coverage:

- Hybrid helper module plus integration: Tasks 2 and 4.
- Join on `confirmed_at` to `macro_bucket_index`: Task 2.
- Aligned and absolute dominance columns: Task 2.
- Quartile labels: Task 2.
- Four summary scopes: Task 4.
- Existing success win condition: Task 4 uses `_group_success_context_stats` unchanged.
- Missing parquet warning and required-column validation: Task 3.
- End-to-end report integration: Task 5.
- Real output verification: Task 6.

No placeholders remain. Function names and column names match the approved spec.
