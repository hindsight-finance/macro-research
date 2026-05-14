from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR

TICK_INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_tick_range_context.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_tick_range_context_summary.parquet")

REQUIRED_TICK_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks"}
CANDLE_SPECS = {"k350": 50, "k359": 59}
WINDOW_SPECS = {f"00_{end * 5 + 4:02d}": (0, end * 5 + 4) for end in range(12)}
NAMED_WINDOWS = {
    "first_5s": (0, 4),
    "first_10s": (0, 9),
    "first_30s": (0, 29),
    "last_30s": (30, 59),
    "full_candle": (0, 59),
}
THRESHOLDS = [25.0, 50.0, 75.0, 90.0]


def _columns(frame: pl.DataFrame | pl.LazyFrame | pl.Schema) -> list[str]:
    if isinstance(frame, pl.Schema):
        return list(frame.names())
    if isinstance(frame, pl.LazyFrame):
        return list(frame.collect_schema().names())
    return frame.columns


def _missing_columns(frame: pl.DataFrame | pl.LazyFrame | pl.Schema, required: set[str]) -> list[str]:
    return sorted(required.difference(_columns(frame)))


def _validate_tick_columns(frame: pl.DataFrame | pl.LazyFrame | pl.Schema) -> None:
    missing = _missing_columns(frame, REQUIRED_TICK_COLUMNS)
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def _prepare_ticks(ticks: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    _validate_tick_columns(ticks)
    ts_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ)
    selected = ticks.select("ts_event", "intra_ts_rank", "price_ticks")
    prepared = (
        selected.with_columns(
            pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC")),
            pl.col("intra_ts_rank").cast(pl.Int64),
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .with_columns(datetime_et=ts_et)
        .with_columns(
            date=pl.col("datetime_et").dt.date(),
            hour_et=pl.col("datetime_et").dt.hour(),
            minute_et=pl.col("datetime_et").dt.minute(),
            second_et=pl.col("datetime_et").dt.second(),
        )
        .filter((pl.col("hour_et") == 15) & pl.col("minute_et").is_between(50, 59))
        .sort(["date", "ts_event", "intra_ts_rank"])
    )
    if isinstance(prepared, pl.LazyFrame):
        return prepared.collect(engine="streaming")
    return prepared


def _range_row(frame: pl.DataFrame, prefix: str) -> dict:
    if frame.is_empty():
        return {
            f"{prefix}_tick_count": 0,
            f"{prefix}_open": None,
            f"{prefix}_high": None,
            f"{prefix}_low": None,
            f"{prefix}_close": None,
            f"{prefix}_range_points": None,
        }
    high = frame.select(pl.col("price").max()).item()
    low = frame.select(pl.col("price").min()).item()
    return {
        f"{prefix}_tick_count": frame.height,
        f"{prefix}_open": frame.item(0, "price"),
        f"{prefix}_high": high,
        f"{prefix}_low": low,
        f"{prefix}_close": frame.item(frame.height - 1, "price"),
        f"{prefix}_range_points": high - low,
    }


def _pct(numer: float | None, denom: float | None) -> float | None:
    return (numer / denom * 100.0) if numer is not None and denom else None


def _add_metrics(row: dict) -> dict:
    window_range = row["window_range_points"]
    candle_open = row["candle_open"]
    candle_range = row["candle_range_points"]
    macro_range = row["macro_range_points"]
    row["range_raw_pct_of_open"] = _pct(window_range, candle_open)
    row["range_pct_of_candle"] = _pct(window_range, candle_range)
    row["range_pct_of_macro"] = _pct(window_range, macro_range)

    window_high = row["window_high"]
    window_low = row["window_low"]
    candle_high = row["candle_high"]
    candle_low = row["candle_low"]
    macro_high = row["macro_high"]
    macro_low = row["macro_low"]

    if None not in (window_high, window_low, candle_high, candle_low):
        row["candle_additive_high_extension_points"] = max(0.0, candle_high - window_high)
        row["candle_additive_low_extension_points"] = max(0.0, window_low - candle_low)
        row["candle_additive_total_extension_points"] = (
            row["candle_additive_high_extension_points"] + row["candle_additive_low_extension_points"]
        )
    else:
        row["candle_additive_high_extension_points"] = None
        row["candle_additive_low_extension_points"] = None
        row["candle_additive_total_extension_points"] = None

    if None not in (window_high, window_low, macro_high, macro_low):
        row["macro_additive_high_extension_points"] = max(0.0, macro_high - window_high)
        row["macro_additive_low_extension_points"] = max(0.0, window_low - macro_low)
        row["macro_additive_total_extension_points"] = (
            row["macro_additive_high_extension_points"] + row["macro_additive_low_extension_points"]
        )
    else:
        row["macro_additive_high_extension_points"] = None
        row["macro_additive_low_extension_points"] = None
        row["macro_additive_total_extension_points"] = None

    row["candle_additive_total_extension_pct_of_candle"] = _pct(
        row["candle_additive_total_extension_points"], candle_range
    )
    row["macro_additive_total_extension_pct_of_macro"] = _pct(
        row["macro_additive_total_extension_points"], macro_range
    )
    return row


def _add_k359_metrics(row: dict, candle: str, pre359_high: float | None, pre359_low: float | None) -> dict:
    row["k359_range_pct_of_macro"] = (
        _pct(row["candle_range_points"], row["macro_range_points"]) if candle == "k359" else None
    )
    if candle == "k359" and None not in (pre359_high, pre359_low, row["candle_high"], row["candle_low"]):
        hi_ext = max(0.0, row["candle_high"] - pre359_high)
        lo_ext = max(0.0, pre359_low - row["candle_low"])
        total = hi_ext + lo_ext
        row["k359_macro_additive_high_extension_from_pre359_points"] = hi_ext
        row["k359_macro_additive_low_extension_from_pre359_points"] = lo_ext
        row["k359_macro_additive_total_extension_from_pre359_points"] = total
        row["k359_macro_additive_total_extension_from_pre359_pct_of_macro"] = _pct(total, row["macro_range_points"])
    else:
        row["k359_macro_additive_high_extension_from_pre359_points"] = None
        row["k359_macro_additive_low_extension_from_pre359_points"] = None
        row["k359_macro_additive_total_extension_from_pre359_points"] = None
        row["k359_macro_additive_total_extension_from_pre359_pct_of_macro"] = None
    return row


def build_macro_tick_range_context(ticks: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    work = _prepare_ticks(ticks)
    rows: list[dict] = []
    all_windows = {**WINDOW_SPECS, **NAMED_WINDOWS}
    complete_dates = (
        work.group_by("date")
        .agg(pl.col("minute_et").filter(pl.col("minute_et").is_in([50, 59])).n_unique().alias("required_candles"))
        .filter(pl.col("required_candles") == 2)
        .select("date")
    )
    work = work.join(complete_dates, on="date", how="inner")
    for date in work["date"].unique().sort().to_list():
        day = work.filter(pl.col("date") == date)
        macro_stats = _range_row(day, "macro")
        pre359 = day.filter(pl.col("minute_et").is_between(50, 58))
        pre359_high = pre359.select(pl.col("price").max()).item() if not pre359.is_empty() else None
        pre359_low = pre359.select(pl.col("price").min()).item() if not pre359.is_empty() else None
        for candle, minute in CANDLE_SPECS.items():
            candle_frame = day.filter(pl.col("minute_et") == minute)
            candle_stats = _range_row(candle_frame, "candle")
            for window, (start_second, end_second) in all_windows.items():
                window_frame = candle_frame.filter(pl.col("second_et").is_between(start_second, end_second))
                row = {
                    "date": date,
                    "candle": candle,
                    "window": window,
                    "window_start_second": start_second,
                    "window_end_second": end_second,
                    **_range_row(window_frame, "window"),
                    **candle_stats,
                    **macro_stats,
                }
                row = _add_metrics(row)
                row = _add_k359_metrics(row, candle, pre359_high, pre359_low)
                rows.append(row)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["date", "candle", "window_start_second", "window_end_second", "window"]
    )


def _rate(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


def _scalar(subset: pl.DataFrame, expr: pl.Expr) -> float | None:
    return subset.select(expr).item() if subset.height else None


def _baseline_row(candle: str, window: str, subset: pl.DataFrame) -> dict:
    valid = subset.filter(pl.col("window_range_points").is_not_null())
    return {
        "summary_type": "window_baseline",
        "candle": candle,
        "window": window,
        "n_days": valid.height,
        "median_range_points": _scalar(valid, pl.col("window_range_points").median()),
        "mean_range_points": _scalar(valid, pl.col("window_range_points").mean()),
        "median_range_raw_pct_of_open": _scalar(valid, pl.col("range_raw_pct_of_open").median()),
        "mean_range_raw_pct_of_open": _scalar(valid, pl.col("range_raw_pct_of_open").mean()),
        "median_range_pct_of_candle": _scalar(valid, pl.col("range_pct_of_candle").median()),
        "mean_range_pct_of_candle": _scalar(valid, pl.col("range_pct_of_candle").mean()),
        "median_range_pct_of_macro": _scalar(valid, pl.col("range_pct_of_macro").median()),
        "mean_range_pct_of_macro": _scalar(valid, pl.col("range_pct_of_macro").mean()),
        "median_candle_additive_total_extension_points": _scalar(
            valid, pl.col("candle_additive_total_extension_points").median()
        ),
        "median_macro_additive_total_extension_points": _scalar(
            valid, pl.col("macro_additive_total_extension_points").median()
        ),
        "median_k359_macro_additive_total_extension_from_pre359_points": _scalar(
            valid, pl.col("k359_macro_additive_total_extension_from_pre359_points").median()
        ),
    }


def _threshold_row(
    summary_type: str,
    candle: str,
    window: str,
    subset: pl.DataFrame,
    metric: str,
    threshold: float,
) -> dict:
    valid = subset.filter(pl.col(metric).is_not_null())
    hits = valid.filter(pl.col(metric) >= threshold).height
    return {
        "summary_type": summary_type,
        "candle": candle,
        "window": window,
        "threshold": threshold,
        "n_days": valid.height,
        "hit_count": hits,
        "hit_rate": _rate(hits, valid.height),
        "median_metric": _scalar(valid, pl.col(metric).median()),
    }


def _decile_rows(summary_type: str, candle: str, window: str, subset: pl.DataFrame, metric: str) -> list[dict]:
    valid = subset.filter(pl.col(metric).is_not_null()).sort([metric, "date"])
    unique_count = valid.select(pl.col(metric).n_unique()).item() if valid.height else 0
    if valid.height < 10 or unique_count < 10:
        return []
    deciled = valid.with_columns(
        (((pl.int_range(pl.len()) * 10) / pl.len()).floor().cast(pl.Int64).clip(0, 9) + 1).alias("decile")
    )
    rows = []
    for decile in range(1, 11):
        d = deciled.filter(pl.col("decile") == decile)
        rows.append(
            {
                "summary_type": summary_type,
                "candle": candle,
                "window": window,
                "decile_metric": metric,
                "decile": decile,
                "n_days": d.height,
                "median_metric": _scalar(d, pl.col(metric).median()),
                "median_range_points": _scalar(d, pl.col("window_range_points").median()),
                "median_range_pct_of_candle": _scalar(d, pl.col("range_pct_of_candle").median()),
                "median_range_pct_of_macro": _scalar(d, pl.col("range_pct_of_macro").median()),
            }
        )
    return rows


def _normalize_summary_rows(rows: list[dict]) -> list[dict]:
    keys = sorted({key for row in rows for key in row})
    return [{key: row.get(key) for key in keys} for row in rows]


def summarize_macro_tick_range_context(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    if study.is_empty():
        return pl.DataFrame(_normalize_summary_rows(rows), infer_schema_length=None)
    for key, subset in study.group_by(["candle", "window"], maintain_order=True):
        candle, window = key
        rows.append(_baseline_row(candle, window, subset))
        for threshold in THRESHOLDS:
            rows.append(_threshold_row("threshold_pct_of_candle", candle, window, subset, "range_pct_of_candle", threshold))
            rows.append(_threshold_row("threshold_pct_of_macro", candle, window, subset, "range_pct_of_macro", threshold))
            if candle == "k359" and window == "full_candle":
                rows.append(
                    _threshold_row(
                        "threshold_k359_range_pct_of_macro",
                        candle,
                        window,
                        subset,
                        "k359_range_pct_of_macro",
                        threshold,
                    )
                )
        rows.extend(_decile_rows("decile_range_raw_pct_of_open", candle, window, subset, "range_raw_pct_of_open"))
        rows.extend(_decile_rows("decile_range_pct_of_candle", candle, window, subset, "range_pct_of_candle"))
        rows.extend(_decile_rows("decile_range_pct_of_macro", candle, window, subset, "range_pct_of_macro"))
    return pl.DataFrame(_normalize_summary_rows(rows), infer_schema_length=None)


def write_macro_tick_range_context(
    input_path: str | Path = TICK_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    study = build_macro_tick_range_context(pl.scan_parquet(input_path))
    summary = summarize_macro_tick_range_context(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not TICK_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {TICK_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary = write_macro_tick_range_context()
    print(f"[OK] Wrote macro tick range context -> {output}")
    print(f"[OK] Wrote macro tick range context summary -> {summary}")


if __name__ == "__main__":
    main()
