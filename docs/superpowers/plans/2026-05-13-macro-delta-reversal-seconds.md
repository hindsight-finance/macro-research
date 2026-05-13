# Macro Delta Reversal Seconds Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the macro delta reversal study with clean primary predictor aliases, 15:59 5-second target windows, and robust non-mean distribution summaries.

**Architecture:** Modify `features/macro_delta_reversal.py` so the existing 1-minute daily study can be enriched by the macro 5-second volume-delta table. Summary generation becomes target-aware, emitting sign, raw-decile, imbalance-decile, tail, and macro-pre59 context rows for primary predictor/target pairs. Tests remain in-memory Polars fixtures and temporary parquet files.

**Tech Stack:** Python, Polars, pytest, parquet outputs, project virtualenv via `.venv/bin/python`.

---

## File Structure

- Modify: `features/macro_delta_reversal.py`
  - Add 5-second input path/constants, schema validation, primary predictor aliases, 15:59 target-window aggregation, target-aware summary helpers, and writer/CLI loading of the 5-second input.
- Modify: `test/test_macro_delta_reversal.py`
  - Add fixture helpers for macro 5-second rows and tests for target boundaries, aliases, pairwise summaries, robust distribution summaries, context rows, and writer persistence.
- Runtime outputs, not necessarily committed:
  - `outputs/nq_macro_delta_reversal.parquet`
  - `outputs/nq_macro_delta_reversal_summary.parquet`

---

### Task 1: Add 5-second schema validation and writer/load signature support

**Files:**
- Modify: `features/macro_delta_reversal.py`
- Modify: `test/test_macro_delta_reversal.py`

- [ ] **Step 1: Write the failing tests**

Add imports/constants to `test/test_macro_delta_reversal.py` if not present:

```python
from features.macro_delta_reversal import (
    build_macro_delta_reversal,
    summarize_macro_delta_reversal,
    write_macro_delta_reversal,
)
```

Append this helper near `_m`:

```python

def _s(date: str, bucket: int, delta: int, classified: int | None = None, total: int | None = None) -> dict:
    classified_size = abs(delta) if classified is None else classified
    total_size = classified_size if total is None else total
    return {
        "datetime_utc": None,
        "trade_date_et": date,
        "macro_bucket_index": bucket,
        "volume_delta": delta,
        "classified_size": classified_size,
        "total_size": total_size,
    }


def _macro_5s_rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("trade_date_et").cast(pl.Date))
```

Append these tests:

```python

def test_build_macro_delta_reversal_requires_macro_5s_schema_when_provided():
    globex = _globex_rows([_g("2025-01-02", 930, 10)])
    macro = _macro_rows([_m("2025-01-02", 59, -5)])
    bad_5s = pl.DataFrame({"trade_date_et": ["2025-01-02"]}).with_columns(pl.col("trade_date_et").cast(pl.Date))

    with pytest.raises(ValueError, match="Missing macro 5-second volume-delta columns"):
        build_macro_delta_reversal(globex, macro, bad_5s)


def test_write_macro_delta_reversal_requires_macro_5s_path_argument(tmp_path: Path):
    globex_path = tmp_path / "globex.parquet"
    macro_path = tmp_path / "macro.parquet"
    macro_5s_path = tmp_path / "macro_5s.parquet"
    output_path = tmp_path / "study.parquet"
    summary_path = tmp_path / "summary.parquet"

    _globex_rows([_g("2025-01-02", 930, 10)]).write_parquet(globex_path)
    _macro_rows([_m("2025-01-02", 59, -5)]).write_parquet(macro_path)
    _macro_5s_rows([_s("2025-01-02", 118, -2)]).write_parquet(macro_5s_path)

    result = write_macro_delta_reversal(globex_path, macro_path, macro_5s_path, output_path, summary_path)

    assert result == (output_path, summary_path)
    assert pl.read_parquet(output_path).height == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_requires_macro_5s_schema_when_provided test/test_macro_delta_reversal.py::test_write_macro_delta_reversal_requires_macro_5s_path_argument -q
```

Expected: FAIL because `build_macro_delta_reversal` and `write_macro_delta_reversal` do not accept macro 5-second input yet.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, add constants near existing paths and required columns:

```python
MACRO_5S_INPUT_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")

MACRO_5S_REQUIRED_COLUMNS = {
    "trade_date_et",
    "macro_bucket_index",
    "volume_delta",
    "classified_size",
    "total_size",
}
```

Update `_validate_inputs` to accept optional macro 5-second input:

```python

def _validate_inputs(
    globex_1m: pl.DataFrame,
    macro_1m: pl.DataFrame,
    macro_5s: pl.DataFrame | None = None,
) -> None:
    globex_missing = _missing_columns(globex_1m, GLOBEX_REQUIRED_COLUMNS)
    if globex_missing:
        raise ValueError(f"Missing Globex volume-delta columns: {globex_missing}")
    macro_missing = _missing_columns(macro_1m, MACRO_REQUIRED_COLUMNS)
    if macro_missing:
        raise ValueError(f"Missing macro volume-delta columns: {macro_missing}")
    if macro_5s is not None:
        macro_5s_missing = _missing_columns(macro_5s, MACRO_5S_REQUIRED_COLUMNS)
        if macro_5s_missing:
            raise ValueError(f"Missing macro 5-second volume-delta columns: {macro_5s_missing}")
```

Update `build_macro_delta_reversal` signature and validation call:

```python

def build_macro_delta_reversal(
    globex_1m: pl.DataFrame,
    macro_1m: pl.DataFrame,
    macro_5s: pl.DataFrame | None = None,
) -> pl.DataFrame:
    _validate_inputs(globex_1m, macro_1m, macro_5s)
```

Leave the rest of the body unchanged for this task.

Update `load_volume_delta_inputs`:

```python

def load_volume_delta_inputs(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_path: str | Path = MACRO_1M_INPUT_PATH,
    macro_5s_path: str | Path = MACRO_5S_INPUT_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(globex_path), pl.read_parquet(macro_path), pl.read_parquet(macro_5s_path)
```

Update `write_macro_delta_reversal` signature/body:

```python

def write_macro_delta_reversal(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_path: str | Path = MACRO_1M_INPUT_PATH,
    macro_5s_path: str | Path = MACRO_5S_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    globex_1m, macro_1m, macro_5s = load_volume_delta_inputs(globex_path, macro_path, macro_5s_path)
    study = build_macro_delta_reversal(globex_1m, macro_1m, macro_5s)
    summary = summarize_macro_delta_reversal(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output
```

Update `main()` to check `MACRO_5S_INPUT_PATH`:

```python
    if not MACRO_5S_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_5S_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_requires_macro_5s_schema_when_provided test/test_macro_delta_reversal.py::test_write_macro_delta_reversal_requires_macro_5s_path_argument -q
```

Expected: PASS.

- [ ] **Step 5: Run all current macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS. If older tests call `write_macro_delta_reversal(globex_path, macro_path, output_path, summary_path)`, update those tests to pass a `macro_5s_path` third argument.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: load macro delta reversal 5s input"
```

---

### Task 2: Add primary predictor aliases

**Files:**
- Modify: `features/macro_delta_reversal.py`
- Modify: `test/test_macro_delta_reversal.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_build_macro_delta_reversal_adds_primary_predictor_aliases():
    globex = _globex_rows(
        [
            _g("2025-01-02", 0, 3, 6, 6),
            _g("2025-01-02", 930, 7, 14, 14),
        ]
    )
    macro = _macro_rows(
        [
            _m("2025-01-02", 50, -2, 4, 4),
            _m("2025-01-02", 59, -5, 10, 10),
        ]
    )

    out = build_macro_delta_reversal(globex, macro)
    row = out.row(0, named=True)

    assert row["eth_rth_pre59_volume_delta"] == row["day_pre_macro_volume_delta"] == 10
    assert row["eth_rth_pre59_delta_imbalance"] == pytest.approx(row["day_pre_macro_delta_imbalance"])
    assert row["eth_rth_macro_pre59_volume_delta"] == row["day_plus_macro_pre59_volume_delta"] == 8
    assert row["rth_macro_pre59_volume_delta"] == row["rth_plus_macro_pre59_volume_delta"] == 5
    assert row["eth_rth_pre59_sign"] == 1
    assert row["eth_rth_macro_pre59_sign"] == 1
    assert row["rth_macro_pre59_sign"] == 1
    assert row["eth_rth_macro_pre59_opposes_k359"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_adds_primary_predictor_aliases -q
```

Expected: FAIL because alias columns are absent.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, add constants near `PREDICTORS`:

```python
PRIMARY_PREDICTOR_ALIASES = {
    "eth_rth_pre59": "day_pre_macro",
    "eth_rth_macro_pre59": "day_plus_macro_pre59",
    "rth_macro_pre59": "rth_plus_macro_pre59",
}

PRIMARY_PREDICTORS = list(PRIMARY_PREDICTOR_ALIASES.keys())
```

Extend `PREDICTORS` to include the primary aliases after existing predictors:

```python
PREDICTORS = [
    "eth_pre_rth",
    "rth_pre_macro",
    "day_pre_macro",
    "macro_pre59",
    "rth_plus_macro_pre59",
    "day_plus_macro_pre59",
    "eth_rth_pre59",
    "eth_rth_macro_pre59",
    "rth_macro_pre59",
]
```

Add helper before `_add_signs_and_relationships`:

```python

def _add_primary_predictor_aliases(frame: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for alias, source in PRIMARY_PREDICTOR_ALIASES.items():
        for suffix in ["volume_delta", "classified_size", "total_size", "delta_imbalance"]:
            exprs.append(pl.col(f"{source}_{suffix}").alias(f"{alias}_{suffix}"))
    return frame.with_columns(exprs)
```

In `build_macro_delta_reversal`, add aliases before signs:

```python
    out = _add_combined_window(out, "rth_pre_macro", "macro_pre59", "rth_plus_macro_pre59")
    out = _add_combined_window(out, "day_pre_macro", "macro_pre59", "day_plus_macro_pre59")
    out = _add_primary_predictor_aliases(out.rename({"trade_date_et": "date"}))
    return _add_signs_and_relationships(out).sort("date")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_adds_primary_predictor_aliases -q
```

Expected: PASS.

- [ ] **Step 5: Run all current macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: add macro delta primary predictor aliases"
```

---

### Task 3: Add 15:59 5-second target-window aggregation

**Files:**
- Modify: `features/macro_delta_reversal.py`
- Modify: `test/test_macro_delta_reversal.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_build_macro_delta_reversal_aggregates_359_5s_target_windows():
    globex = _globex_rows([_g("2025-01-02", 930, 10, 20, 20)])
    macro = _macro_rows([_m("2025-01-02", 59, -12, 24, 24)])
    macro_5s = _macro_5s_rows(
        [
            _s("2025-01-02", 107, 999, 999, 999),
            _s("2025-01-02", 108, 1, 2, 2),
            _s("2025-01-02", 109, 2, 4, 4),
            _s("2025-01-02", 110, -1, 2, 3),
            _s("2025-01-02", 111, -2, 4, 5),
            _s("2025-01-02", 112, 3, 6, 6),
            _s("2025-01-02", 113, -3, 6, 6),
            _s("2025-01-02", 114, -4, 8, 8),
            _s("2025-01-02", 115, -5, 10, 10),
            _s("2025-01-02", 116, -6, 12, 12),
            _s("2025-01-02", 117, -7, 14, 14),
            _s("2025-01-02", 118, -8, 16, 16),
            _s("2025-01-02", 119, -9, 18, 18),
        ]
    )

    out = build_macro_delta_reversal(globex, macro, macro_5s)
    row = out.row(0, named=True)

    assert row["k359_00_59_volume_delta"] == -39
    assert row["k359_00_59_classified_size"] == 102
    assert row["k359_00_59_total_size"] == 104
    assert row["k359_00_59_delta_imbalance"] == pytest.approx(-39 / 102)
    assert row["k359_00_59_sign"] == -1
    assert row["k359_00_29_volume_delta"] == 0
    assert row["k359_00_29_sign"] == 0
    assert row["k359_30_59_volume_delta"] == -39
    assert row["k359_45_59_volume_delta"] == -24
    assert row["k359_50_59_volume_delta"] == -17
    assert row["k359_bucket_108_volume_delta"] == 1
    assert row["k359_bucket_119_volume_delta"] == -9
    assert "k359_bucket_107_volume_delta" not in out.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_aggregates_359_5s_target_windows -q
```

Expected: FAIL because 5-second target columns are absent.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, add constants near predictor constants:

```python
TARGET_WINDOWS_5S = {
    "k359_00_59": (108, 119),
    "k359_00_29": (108, 113),
    "k359_30_59": (114, 119),
    "k359_45_59": (117, 119),
    "k359_50_59": (118, 119),
}

TARGET_WINDOWS = ["k359", *TARGET_WINDOWS_5S.keys(), *[f"k359_bucket_{bucket}" for bucket in range(108, 120)]]
```

Add helper functions before `build_macro_delta_reversal`:

```python

def _aggregate_target_window_5s(macro_5s: pl.DataFrame, start: int, end: int, prefix: str) -> pl.DataFrame:
    return (
        macro_5s.filter(pl.col("macro_bucket_index").is_between(start, end))
        .group_by("trade_date_et")
        .agg(
            pl.col("volume_delta").sum().alias(f"{prefix}_volume_delta"),
            pl.col("classified_size").sum().alias(f"{prefix}_classified_size"),
            pl.col("total_size").sum().alias(f"{prefix}_total_size"),
        )
        .with_columns(
            _safe_ratio_expr(pl.col(f"{prefix}_volume_delta"), pl.col(f"{prefix}_classified_size")).alias(
                f"{prefix}_delta_imbalance"
            )
        )
    )


def _join_359_5s_targets(frame: pl.DataFrame, macro_5s: pl.DataFrame | None) -> pl.DataFrame:
    if macro_5s is None:
        return frame

    out = frame
    for prefix, (start, end) in TARGET_WINDOWS_5S.items():
        out = out.join(_aggregate_target_window_5s(macro_5s, start, end, prefix), on="trade_date_et", how="left")
    for bucket in range(108, 120):
        prefix = f"k359_bucket_{bucket}"
        out = out.join(_aggregate_target_window_5s(macro_5s, bucket, bucket, prefix), on="trade_date_et", how="left")
    return out
```

Update `_add_signs_and_relationships` sign names:

```python
    target_names = [target for target in TARGET_WINDOWS if f"{target}_volume_delta" in frame.columns]
    sign_names = [*PREDICTORS, *target_names]
```

Keep relationship flags versus the original full-minute `k359` unchanged for existing compatibility.

In `build_macro_delta_reversal`, join targets before renaming date:

```python
    out = _add_combined_window(out, "rth_pre_macro", "macro_pre59", "rth_plus_macro_pre59")
    out = _add_combined_window(out, "day_pre_macro", "macro_pre59", "day_plus_macro_pre59")
    out = _join_359_5s_targets(out, macro_5s)
    out = _add_primary_predictor_aliases(out.rename({"trade_date_et": "date"}))
    return _add_signs_and_relationships(out).sort("date")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_aggregates_359_5s_target_windows -q
```

Expected: PASS.

- [ ] **Step 5: Run all macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: add macro delta 359 5s targets"
```

---

### Task 4: Add target-aware pairwise sign summaries

**Files:**
- Modify: `features/macro_delta_reversal.py`
- Modify: `test/test_macro_delta_reversal.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_summarize_macro_delta_reversal_adds_target_aware_sign_rows():
    globex = _globex_rows(
        [
            _g("2025-01-02", 930, 10, 20, 20),
            _g("2025-01-03", 930, -10, 20, 20),
        ]
    )
    macro = _macro_rows(
        [
            _m("2025-01-02", 59, -5, 10, 10),
            _m("2025-01-03", 59, 5, 10, 10),
        ]
    )
    macro_5s = _macro_5s_rows(
        [
            _s("2025-01-02", 118, -2, 4, 4),
            _s("2025-01-02", 119, -3, 6, 6),
            _s("2025-01-03", 118, 2, 4, 4),
            _s("2025-01-03", 119, 3, 6, 6),
        ]
    )
    study = build_macro_delta_reversal(globex, macro, macro_5s)

    summary = summarize_macro_delta_reversal(study)
    row = summary.filter(
        (pl.col("summary_type") == "target_sign")
        & (pl.col("predictor") == "rth_macro_pre59")
        & (pl.col("target_window") == "k359_50_59")
    ).row(0, named=True)

    assert row["n_days"] == 2
    assert row["n_signal_days"] == 2
    assert row["opposite_count"] == 2
    assert row["opposite_rate"] == pytest.approx(1.0)
    assert row["same_count"] == 0
    assert row["zero_target_count"] == 0
    assert row["median_target_delta_when_predictor_positive"] == pytest.approx(-5.0)
    assert row["median_target_delta_when_predictor_negative"] == pytest.approx(5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_target_aware_sign_rows -q
```

Expected: FAIL because target-aware rows are absent.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, add helper functions above `summarize_macro_delta_reversal`:

```python

def _available_target_windows(study: pl.DataFrame) -> list[str]:
    return [target for target in TARGET_WINDOWS if f"{target}_volume_delta" in study.columns]


def _target_values_for_predictor_sign(study: pl.DataFrame, predictor: str, target: str, sign: int) -> pl.DataFrame:
    return study.filter(pl.col(f"{predictor}_sign") == sign).select(pl.col(f"{target}_volume_delta").alias("target_delta"))


def _scalar_or_none(frame: pl.DataFrame, expr: pl.Expr) -> float | None:
    value = frame.select(expr).item()
    return None if value is None else float(value)


def _target_pair_sign_row(study: pl.DataFrame, predictor: str, target: str) -> dict:
    pred_sign = pl.col(f"{predictor}_sign")
    target_sign = pl.col(f"{target}_sign")
    has_signal_expr = (pred_sign != 0) & (target_sign != 0)
    signal = study.filter(has_signal_expr)
    n_signal_days = signal.height
    opposite_count = study.filter(has_signal_expr & (pred_sign == -target_sign)).height
    same_count = study.filter(has_signal_expr & (pred_sign == target_sign)).height
    pos = _target_values_for_predictor_sign(study, predictor, target, 1)
    neg = _target_values_for_predictor_sign(study, predictor, target, -1)
    corr = study.select(pl.corr(f"{predictor}_volume_delta", f"{target}_volume_delta")).item()
    return {
        "summary_type": "target_sign",
        "predictor": predictor,
        "target_window": target,
        "predictor_decile": None,
        "tail": None,
        "condition": None,
        "n_days": study.height,
        "n_signal_days": n_signal_days,
        "opposite_count": opposite_count,
        "opposite_rate": _rate(opposite_count, n_signal_days),
        "same_count": same_count,
        "same_rate": _rate(same_count, n_signal_days),
        "zero_predictor_count": study.filter(pred_sign == 0).height,
        "zero_target_count": study.filter(target_sign == 0).height,
        "mean_predictor_delta": study.select(pl.col(f"{predictor}_volume_delta").mean()).item(),
        "median_predictor_delta": study.select(pl.col(f"{predictor}_volume_delta").median()).item(),
        "mean_target_delta": study.select(pl.col(f"{target}_volume_delta").mean()).item(),
        "median_target_delta": study.select(pl.col(f"{target}_volume_delta").median()).item(),
        "mean_target_delta_when_predictor_positive": _scalar_or_none(pos, pl.col("target_delta").mean()),
        "mean_target_delta_when_predictor_negative": _scalar_or_none(neg, pl.col("target_delta").mean()),
        "median_target_delta_when_predictor_positive": _scalar_or_none(pos, pl.col("target_delta").median()),
        "median_target_delta_when_predictor_negative": _scalar_or_none(neg, pl.col("target_delta").median()),
        "target_p25_when_predictor_positive": _scalar_or_none(pos, pl.col("target_delta").quantile(0.25)),
        "target_p75_when_predictor_positive": _scalar_or_none(pos, pl.col("target_delta").quantile(0.75)),
        "target_p25_when_predictor_negative": _scalar_or_none(neg, pl.col("target_delta").quantile(0.25)),
        "target_p75_when_predictor_negative": _scalar_or_none(neg, pl.col("target_delta").quantile(0.75)),
        "pearson_corr_predictor_vs_target_delta": corr,
    }
```

Modify `summarize_macro_delta_reversal` so after existing legacy rows are appended, it also appends target-aware rows:

```python
    for predictor in PRIMARY_PREDICTORS:
        for target in _available_target_windows(study):
            rows.append(_target_pair_sign_row(study, predictor, target))
```

This can be placed before `return pl.DataFrame(rows)`. Do not remove existing legacy `summary_type == "sign"` rows yet.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_target_aware_sign_rows -q
```

Expected: PASS.

- [ ] **Step 5: Run all macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS. If Polars cannot infer mixed summary schema because older and newer rows have different keys, add missing keys with `None` to legacy rows or build rows through a normalizing helper.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: add target aware delta reversal summaries"
```

---

### Task 5: Add robust raw-delta, imbalance-decile, and tail summaries

**Files:**
- Modify: `features/macro_delta_reversal.py`
- Modify: `test/test_macro_delta_reversal.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_summarize_macro_delta_reversal_adds_raw_and_imbalance_decile_rows():
    globex = _globex_rows([_g(f"2025-01-{day:02d}", 930, day, day * 10, day * 10) for day in range(2, 14)])
    macro = _macro_rows([_m(f"2025-01-{day:02d}", 59, -day, day * 2, day * 2) for day in range(2, 14)])
    macro_5s = _macro_5s_rows([_s(f"2025-01-{day:02d}", 118, -day, day * 2, day * 2) for day in range(2, 14)])
    study = build_macro_delta_reversal(globex, macro, macro_5s)

    summary = summarize_macro_delta_reversal(study)
    raw = summary.filter(
        (pl.col("summary_type") == "target_raw_decile")
        & (pl.col("predictor") == "rth_macro_pre59")
        & (pl.col("target_window") == "k359_50_59")
    )
    imb = summary.filter(
        (pl.col("summary_type") == "target_imbalance_decile")
        & (pl.col("predictor") == "rth_macro_pre59")
        & (pl.col("target_window") == "k359_50_59")
    )

    assert raw.height == 10
    assert raw.select("predictor_decile").to_series().to_list() == list(range(1, 11))
    assert raw.select(pl.col("n_days").sum()).item() == 12
    assert raw.select(pl.col("median_target_delta").max()).item() < 0
    assert imb.height == 10
    assert imb.select(pl.col("n_days").sum()).item() == 12


def test_summarize_macro_delta_reversal_adds_positive_and_negative_tail_rows():
    globex = _globex_rows([_g(f"2025-01-{day:02d}", 930, day) for day in range(2, 32)])
    macro = _macro_rows([_m(f"2025-01-{day:02d}", 59, -day) for day in range(2, 32)])
    macro_5s = _macro_5s_rows([_s(f"2025-01-{day:02d}", 118, -day) for day in range(2, 32)])
    study = build_macro_delta_reversal(globex, macro, macro_5s)

    summary = summarize_macro_delta_reversal(study)
    tails = summary.filter(
        (pl.col("summary_type") == "target_tail")
        & (pl.col("predictor") == "rth_macro_pre59")
        & (pl.col("target_window") == "k359_50_59")
    )

    assert set(tails.select("tail").to_series().to_list()) == {
        "positive_top_20",
        "positive_top_10",
        "negative_bottom_20",
        "negative_bottom_10",
    }
    positive_top_20 = tails.filter(pl.col("tail") == "positive_top_20").row(0, named=True)
    assert positive_top_20["n_days"] >= 5
    assert positive_top_20["opposite_rate"] == pytest.approx(1.0)
    assert positive_top_20["median_target_delta"] < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_raw_and_imbalance_decile_rows test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_positive_and_negative_tail_rows -q
```

Expected: FAIL because target robust rows are absent.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, add helper functions above `summarize_macro_delta_reversal`:

```python

def _base_target_summary_row(
    study: pl.DataFrame,
    predictor: str,
    target: str,
    summary_type: str,
    subset: pl.DataFrame,
    predictor_decile: int | None = None,
    tail: str | None = None,
    condition: str | None = None,
) -> dict:
    pred_sign = pl.col(f"{predictor}_sign")
    target_sign = pl.col(f"{target}_sign")
    signal = subset.filter((pred_sign != 0) & (target_sign != 0))
    n_signal_days = signal.height
    opposite_count = subset.filter((pred_sign != 0) & (target_sign != 0) & (pred_sign == -target_sign)).height
    same_count = subset.filter((pred_sign != 0) & (target_sign != 0) & (pred_sign == target_sign)).height
    return {
        "summary_type": summary_type,
        "predictor": predictor,
        "target_window": target,
        "predictor_decile": predictor_decile,
        "tail": tail,
        "condition": condition,
        "n_days": subset.height,
        "n_signal_days": n_signal_days,
        "opposite_count": opposite_count,
        "opposite_rate": _rate(opposite_count, n_signal_days),
        "same_count": same_count,
        "same_rate": _rate(same_count, n_signal_days),
        "zero_predictor_count": subset.filter(pred_sign == 0).height,
        "zero_target_count": subset.filter(target_sign == 0).height,
        "mean_predictor_delta": subset.select(pl.col(f"{predictor}_volume_delta").mean()).item() if subset.height else None,
        "median_predictor_delta": subset.select(pl.col(f"{predictor}_volume_delta").median()).item() if subset.height else None,
        "mean_target_delta": subset.select(pl.col(f"{target}_volume_delta").mean()).item() if subset.height else None,
        "median_target_delta": subset.select(pl.col(f"{target}_volume_delta").median()).item() if subset.height else None,
        "target_p25": subset.select(pl.col(f"{target}_volume_delta").quantile(0.25)).item() if subset.height else None,
        "target_p75": subset.select(pl.col(f"{target}_volume_delta").quantile(0.75)).item() if subset.height else None,
        "pearson_corr_predictor_vs_target_delta": subset.select(pl.corr(f"{predictor}_volume_delta", f"{target}_volume_delta")).item() if subset.height else None,
    }


def _target_decile_rows(study: pl.DataFrame, predictor: str, target: str, value_suffix: str, summary_type: str) -> list[dict]:
    value_col = f"{predictor}_{value_suffix}"
    non_null = study.filter(pl.col(value_col).is_not_null())
    unique_count = non_null.select(pl.col(value_col).n_unique()).item() if not non_null.is_empty() else 0
    if non_null.height < 10 or unique_count < 10:
        return []
    deciled = non_null.with_columns(
        ((pl.col(value_col).rank(method="ordinal") - 1) * 10 / non_null.height)
        .floor()
        .cast(pl.Int64)
        .clip(0, 9)
        .add(1)
        .alias("predictor_decile")
    )
    rows = []
    for decile in range(1, 11):
        subset = deciled.filter(pl.col("predictor_decile") == decile)
        rows.append(_base_target_summary_row(study, predictor, target, summary_type, subset, predictor_decile=decile))
    return rows


def _target_tail_rows(study: pl.DataFrame, predictor: str, target: str) -> list[dict]:
    value_col = f"{predictor}_volume_delta"
    rows = []
    positive = study.filter(pl.col(value_col) > 0)
    negative = study.filter(pl.col(value_col) < 0)
    tail_specs = [
        ("positive_top_20", positive, 0.80, ">="),
        ("positive_top_10", positive, 0.90, ">="),
        ("negative_bottom_20", negative, 0.20, "<="),
        ("negative_bottom_10", negative, 0.10, "<="),
    ]
    for label, frame, quantile, op in tail_specs:
        if frame.is_empty():
            continue
        threshold = frame.select(pl.col(value_col).quantile(quantile)).item()
        subset = frame.filter(pl.col(value_col) >= threshold) if op == ">=" else frame.filter(pl.col(value_col) <= threshold)
        rows.append(_base_target_summary_row(study, predictor, target, "target_tail", subset, tail=label))
    return rows
```

In `summarize_macro_delta_reversal`, inside the loop over `PRIMARY_PREDICTORS` and available targets, append robust rows after the sign row:

```python
            rows.extend(_target_decile_rows(study, predictor, target, "volume_delta", "target_raw_decile"))
            rows.extend(_target_decile_rows(study, predictor, target, "delta_imbalance", "target_imbalance_decile"))
            rows.extend(_target_tail_rows(study, predictor, target))
```

If schema inference fails because row dictionaries differ, add a `_normalize_summary_rows(rows)` helper that unions all keys and fills missing values with `None` before `pl.DataFrame` construction:

```python

def _normalize_summary_rows(rows: list[dict]) -> list[dict]:
    keys = sorted({key for row in rows for key in row})
    return [{key: row.get(key) for key in keys} for row in rows]
```

Then return:

```python
    return pl.DataFrame(_normalize_summary_rows(rows))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_raw_and_imbalance_decile_rows test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_positive_and_negative_tail_rows -q
```

Expected: PASS.

- [ ] **Step 5: Run all macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: add robust macro delta reversal summaries"
```

---

### Task 6: Add macro pre-59 context rows

**Files:**
- Modify: `features/macro_delta_reversal.py`
- Modify: `test/test_macro_delta_reversal.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_summarize_macro_delta_reversal_adds_macro_pre59_context_rows():
    globex = _globex_rows(
        [
            _g("2025-01-02", 930, 10),
            _g("2025-01-03", 930, 10),
            _g("2025-01-04", 930, -10),
            _g("2025-01-05", 930, -10),
        ]
    )
    macro = _macro_rows(
        [
            _m("2025-01-02", 50, 2),
            _m("2025-01-02", 59, -5),
            _m("2025-01-03", 50, -2),
            _m("2025-01-03", 59, -5),
            _m("2025-01-04", 50, -2),
            _m("2025-01-04", 59, 5),
            _m("2025-01-05", 50, 2),
            _m("2025-01-05", 59, 5),
        ]
    )
    macro_5s = _macro_5s_rows(
        [
            _s("2025-01-02", 118, -5),
            _s("2025-01-03", 118, -5),
            _s("2025-01-04", 118, 5),
            _s("2025-01-05", 118, 5),
        ]
    )
    study = build_macro_delta_reversal(globex, macro, macro_5s)

    summary = summarize_macro_delta_reversal(study)
    same = summary.filter(
        (pl.col("summary_type") == "macro_pre59_context")
        & (pl.col("condition") == "macro_pre59_same_as_rth_pre_macro")
        & (pl.col("target_window") == "k359_50_59")
    ).row(0, named=True)
    opposes = summary.filter(
        (pl.col("summary_type") == "macro_pre59_context")
        & (pl.col("condition") == "macro_pre59_opposes_rth_pre_macro")
        & (pl.col("target_window") == "k359_50_59")
    ).row(0, named=True)

    assert same["n_days"] == 2
    assert same["opposite_rate"] == pytest.approx(1.0)
    assert opposes["n_days"] == 2
    assert opposes["opposite_rate"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_macro_pre59_context_rows -q
```

Expected: FAIL because macro pre-59 context rows are absent.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, add this helper above `summarize_macro_delta_reversal`:

```python

def _macro_pre59_context_rows(study: pl.DataFrame, target: str) -> list[dict]:
    specs = [
        ("macro_pre59_same_as_eth_rth_pre59", "eth_rth_pre59", True),
        ("macro_pre59_opposes_eth_rth_pre59", "eth_rth_pre59", False),
        ("macro_pre59_same_as_rth_pre_macro", "rth_pre_macro", True),
        ("macro_pre59_opposes_rth_pre_macro", "rth_pre_macro", False),
    ]
    rows = []
    for condition, base_predictor, same_direction in specs:
        base_sign = pl.col(f"{base_predictor}_sign")
        macro_sign = pl.col("macro_pre59_sign")
        valid = (base_sign != 0) & (macro_sign != 0)
        direction_filter = macro_sign == base_sign if same_direction else macro_sign == -base_sign
        subset = study.filter(valid & direction_filter)
        unresolved_predictor = "rth_macro_pre59" if base_predictor == "rth_pre_macro" else "eth_rth_macro_pre59"
        rows.append(
            _base_target_summary_row(
                study,
                unresolved_predictor,
                target,
                "macro_pre59_context",
                subset,
                condition=condition,
            )
        )
    return rows
```

In `summarize_macro_delta_reversal`, after robust rows are added for all predictor/target pairs, add:

```python
    for target in _available_target_windows(study):
        rows.extend(_macro_pre59_context_rows(study, target))
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_macro_pre59_context_rows -q
```

Expected: PASS.

- [ ] **Step 5: Run all macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: add macro pre59 context summaries"
```

---

### Task 7: Runtime verification and output inspection

**Files:**
- Modify only if required by failures: `features/macro_delta_reversal.py`, `test/test_macro_delta_reversal.py`

- [ ] **Step 1: Run targeted tests**

```bash
.venv/bin/python -m pytest test/test_volume_delta.py test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

```bash
.venv/bin/python -m pytest test -q
```

Expected: PASS, or document unrelated pre-existing failures exactly.

- [ ] **Step 3: Ensure output inputs are present in the worktree if needed**

If running inside a worktree where `outputs/` is ignored and source parquet files are absent, copy the required existing outputs from the main workspace into the worktree:

```bash
mkdir -p outputs
cp ../../outputs/nq_globex_volume_delta_1m.parquet outputs/nq_globex_volume_delta_1m.parquet
cp ../../outputs/nq_macro_volume_delta_1m.parquet outputs/nq_macro_volume_delta_1m.parquet
cp ../../outputs/nq_macro_volume_delta_5s.parquet outputs/nq_macro_volume_delta_5s.parquet
```

- [ ] **Step 4: Run the feature**

```bash
.venv/bin/python -m features.macro_delta_reversal
```

Expected:

```text
[OK] Wrote macro delta reversal → outputs/nq_macro_delta_reversal.parquet
[OK] Wrote macro delta reversal summary → outputs/nq_macro_delta_reversal_summary.parquet
```

- [ ] **Step 5: Inspect key summary outputs**

Run:

```bash
.venv/bin/python - <<'PY'
import polars as pl
study = pl.read_parquet('outputs/nq_macro_delta_reversal.parquet')
summary = pl.read_parquet('outputs/nq_macro_delta_reversal_summary.parquet')
print('study', study.shape)
print('summary', summary.shape)
print(study.select([
    'date',
    'eth_rth_pre59_volume_delta',
    'eth_rth_macro_pre59_volume_delta',
    'rth_macro_pre59_volume_delta',
    'k359_50_59_volume_delta',
]).head())
print(summary.group_by('summary_type').len().sort('summary_type'))
print(summary.filter(
    (pl.col('summary_type') == 'target_sign')
    & (pl.col('predictor') == 'eth_rth_macro_pre59')
    & (pl.col('target_window') == 'k359_50_59')
).select([
    'predictor', 'target_window', 'n_days', 'n_signal_days', 'opposite_rate',
    'median_target_delta_when_predictor_positive',
    'median_target_delta_when_predictor_negative',
    'pearson_corr_predictor_vs_target_delta',
]))
PY
```

Expected:

- Study is non-empty.
- Summary contains `target_sign`, `target_raw_decile`, `target_imbalance_decile`, `target_tail`, and `macro_pre59_context` rows.
- Primary predictor alias columns exist.
- `k359_50_59_volume_delta` exists.

- [ ] **Step 6: Check git status**

```bash
git status --short
```

Expected: source/test changes are committed; generated `outputs/*.parquet` may be ignored/untracked and should not be committed unless explicitly requested.

- [ ] **Step 7: Commit final fixes if any**

If Step 1–5 required source/test fixes:

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "fix: verify macro delta seconds extension runtime"
```

---

## Final Verification

- [ ] **Step 1: Run targeted tests fresh**

```bash
.venv/bin/python -m pytest test/test_volume_delta.py test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests fresh**

```bash
.venv/bin/python -m pytest test -q
```

Expected: PASS.

- [ ] **Step 3: Run runtime feature fresh**

```bash
.venv/bin/python -m features.macro_delta_reversal
```

Expected: writes both output parquet files successfully.
