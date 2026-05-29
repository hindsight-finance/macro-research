from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

from utils import data_sources
from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR, scan_source

TICK_INPUT_PATH = data_sources.tick_data_url()
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

STUDY_SCHEMA = {
    "date": pl.Date,
    "candle": pl.Utf8,
    "window": pl.Utf8,
    "window_start_second": pl.Int64,
    "window_end_second": pl.Int64,
    "window_tick_count": pl.Int64,
    "window_open": pl.Float64,
    "window_high": pl.Float64,
    "window_low": pl.Float64,
    "window_close": pl.Float64,
    "window_range_points": pl.Float64,
    "candle_tick_count": pl.Int64,
    "candle_open": pl.Float64,
    "candle_high": pl.Float64,
    "candle_low": pl.Float64,
    "candle_close": pl.Float64,
    "candle_range_points": pl.Float64,
    "macro_tick_count": pl.Int64,
    "macro_open": pl.Float64,
    "macro_high": pl.Float64,
    "macro_low": pl.Float64,
    "macro_close": pl.Float64,
    "macro_range_points": pl.Float64,
    "range_raw_pct_of_open": pl.Float64,
    "range_pct_of_candle": pl.Float64,
    "range_pct_of_macro": pl.Float64,
    "candle_additive_high_extension_points": pl.Float64,
    "candle_additive_low_extension_points": pl.Float64,
    "candle_additive_total_extension_points": pl.Float64,
    "macro_additive_high_extension_points": pl.Float64,
    "macro_additive_low_extension_points": pl.Float64,
    "macro_additive_total_extension_points": pl.Float64,
    "candle_additive_total_extension_pct_of_candle": pl.Float64,
    "macro_additive_total_extension_pct_of_macro": pl.Float64,
    "k359_range_pct_of_macro": pl.Float64,
    "k359_macro_additive_high_extension_from_pre359_points": pl.Float64,
    "k359_macro_additive_low_extension_from_pre359_points": pl.Float64,
    "k359_macro_additive_total_extension_from_pre359_points": pl.Float64,
    "k359_macro_additive_total_extension_from_pre359_pct_of_macro": pl.Float64,
}

SUMMARY_SCHEMA = {
    "summary_type": pl.Utf8,
    "candle": pl.Utf8,
    "window": pl.Utf8,
    "n_days": pl.Int64,
    "median_range_points": pl.Float64,
    "mean_range_points": pl.Float64,
    "median_range_raw_pct_of_open": pl.Float64,
    "mean_range_raw_pct_of_open": pl.Float64,
    "median_range_pct_of_candle": pl.Float64,
    "mean_range_pct_of_candle": pl.Float64,
    "median_range_pct_of_macro": pl.Float64,
    "mean_range_pct_of_macro": pl.Float64,
    "median_candle_additive_total_extension_points": pl.Float64,
    "median_macro_additive_total_extension_points": pl.Float64,
    "median_k359_macro_additive_total_extension_from_pre359_points": pl.Float64,
    "threshold": pl.Float64,
    "hit_count": pl.Int64,
    "hit_rate": pl.Float64,
    "median_metric": pl.Float64,
    "decile_metric": pl.Utf8,
    "decile": pl.Int64,
}



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


def _empty_study_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=STUDY_SCHEMA)


def _empty_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SUMMARY_SCHEMA)


def _prepare_ticks(ticks: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    _validate_tick_columns(ticks)
    lazy = ticks.lazy() if isinstance(ticks, pl.DataFrame) else ticks
    ts_utc = pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC"))
    return (
        lazy.select("ts_event", "intra_ts_rank", "price_ticks")
        .with_columns(
            ts_utc.alias("ts_event"),
            pl.col("intra_ts_rank").cast(pl.Int64),
            (pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR).alias("price"),
        )
        .with_columns(datetime_et=pl.col("ts_event").dt.convert_time_zone(MARKET_TZ))
        .with_columns(
            date=pl.col("datetime_et").dt.date(),
            hour_et=pl.col("datetime_et").dt.hour(),
            minute_et=pl.col("datetime_et").dt.minute(),
            second_et=pl.col("datetime_et").dt.second(),
        )
        .filter((pl.col("hour_et") == 15) & pl.col("minute_et").is_between(50, 59))
    )


def _range_exprs(prefix: str) -> list[pl.Expr]:
    return [
        pl.len().cast(pl.Int64).alias(f"{prefix}_tick_count"),
        pl.col("price").first().alias(f"{prefix}_open"),
        pl.col("price").max().alias(f"{prefix}_high"),
        pl.col("price").min().alias(f"{prefix}_low"),
        pl.col("price").last().alias(f"{prefix}_close"),
        (pl.col("price").max() - pl.col("price").min()).alias(f"{prefix}_range_points"),
    ]


def _complete_dates(work: pl.LazyFrame) -> pl.LazyFrame:
    return (
        work.group_by("date")
        .agg(pl.col("minute_et").n_unique().alias("macro_minute_count"))
        .filter(pl.col("macro_minute_count") == 10)
        .select("date")
    )


def _candle_window_grid() -> pl.DataFrame:
    rows = []
    all_windows = {**WINDOW_SPECS, **NAMED_WINDOWS}
    for candle, minute in CANDLE_SPECS.items():
        for window, (start_second, end_second) in all_windows.items():
            rows.append(
                {
                    "candle": candle,
                    "minute_et": minute,
                    "window": window,
                    "window_start_second": start_second,
                    "window_end_second": end_second,
                }
            )
    return pl.DataFrame(rows)


def _pct_expr(numer: str, denom: str) -> pl.Expr:
    return (
        pl.when(pl.col(numer).is_not_null() & pl.col(denom).is_not_null() & (pl.col(denom) != 0))
        .then(pl.col(numer) / pl.col(denom) * 100.0)
        .otherwise(None)
    )


def _all_not_null(*cols: str) -> pl.Expr:
    return pl.all_horizontal([pl.col(col).is_not_null() for col in cols])


def _add_lazy_metrics(frame: pl.LazyFrame) -> pl.LazyFrame:
    return (
        frame.with_columns(
            _pct_expr("window_range_points", "candle_open").alias("range_raw_pct_of_open"),
            _pct_expr("window_range_points", "candle_range_points").alias("range_pct_of_candle"),
            _pct_expr("window_range_points", "macro_range_points").alias("range_pct_of_macro"),
            pl.when(_all_not_null("window_high", "candle_high"))
            .then((pl.col("candle_high") - pl.col("window_high")).clip(0.0))
            .otherwise(None)
            .alias("candle_additive_high_extension_points"),
            pl.when(_all_not_null("window_low", "candle_low"))
            .then((pl.col("window_low") - pl.col("candle_low")).clip(0.0))
            .otherwise(None)
            .alias("candle_additive_low_extension_points"),
            pl.when(_all_not_null("window_high", "macro_high"))
            .then((pl.col("macro_high") - pl.col("window_high")).clip(0.0))
            .otherwise(None)
            .alias("macro_additive_high_extension_points"),
            pl.when(_all_not_null("window_low", "macro_low"))
            .then((pl.col("window_low") - pl.col("macro_low")).clip(0.0))
            .otherwise(None)
            .alias("macro_additive_low_extension_points"),
        )
        .with_columns(
            (
                pl.col("candle_additive_high_extension_points")
                + pl.col("candle_additive_low_extension_points")
            ).alias("candle_additive_total_extension_points"),
            (
                pl.col("macro_additive_high_extension_points")
                + pl.col("macro_additive_low_extension_points")
            ).alias("macro_additive_total_extension_points"),
        )
        .with_columns(
            _pct_expr("candle_additive_total_extension_points", "candle_range_points").alias(
                "candle_additive_total_extension_pct_of_candle"
            ),
            _pct_expr("macro_additive_total_extension_points", "macro_range_points").alias(
                "macro_additive_total_extension_pct_of_macro"
            ),
            pl.when(pl.col("candle") == "k359")
            .then(_pct_expr("candle_range_points", "macro_range_points"))
            .otherwise(None)
            .alias("k359_range_pct_of_macro"),
            pl.when((pl.col("candle") == "k359") & _all_not_null("candle_high", "pre359_high"))
            .then((pl.col("candle_high") - pl.col("pre359_high")).clip(0.0))
            .otherwise(None)
            .alias("k359_macro_additive_high_extension_from_pre359_points"),
            pl.when((pl.col("candle") == "k359") & _all_not_null("candle_low", "pre359_low"))
            .then((pl.col("pre359_low") - pl.col("candle_low")).clip(0.0))
            .otherwise(None)
            .alias("k359_macro_additive_low_extension_from_pre359_points"),
        )
        .with_columns(
            (
                pl.col("k359_macro_additive_high_extension_from_pre359_points")
                + pl.col("k359_macro_additive_low_extension_from_pre359_points")
            ).alias("k359_macro_additive_total_extension_from_pre359_points")
        )
        .with_columns(
            _pct_expr("k359_macro_additive_total_extension_from_pre359_points", "macro_range_points").alias(
                "k359_macro_additive_total_extension_from_pre359_pct_of_macro"
            )
        )
    )


def _window_stats(work: pl.LazyFrame) -> pl.LazyFrame:
    frames = []
    all_windows = {**WINDOW_SPECS, **NAMED_WINDOWS}
    for candle, minute in CANDLE_SPECS.items():
        for window, (start_second, end_second) in all_windows.items():
            frames.append(
                work.filter((pl.col("minute_et") == minute) & pl.col("second_et").is_between(start_second, end_second))
                .sort(["date", "ts_event", "intra_ts_rank"])
                .with_columns(
                    pl.lit(candle).alias("candle"),
                    pl.lit(window).alias("window"),
                    pl.lit(start_second).cast(pl.Int64).alias("window_start_second"),
                    pl.lit(end_second).cast(pl.Int64).alias("window_end_second"),
                )
                .group_by(["date", "candle", "window", "window_start_second", "window_end_second"])
                .agg(_range_exprs("window"))
            )
    return pl.concat(frames, how="vertical")


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
    complete_dates = _complete_dates(work)
    work = work.join(complete_dates, on="date", how="inner")

    ordered = work.sort(["date", "ts_event", "intra_ts_rank"])
    macro_stats = ordered.group_by("date").agg(_range_exprs("macro"))
    candle_stats = (
        ordered.filter(pl.col("minute_et").is_in(list(CANDLE_SPECS.values())))
        .with_columns(
            pl.when(pl.col("minute_et") == CANDLE_SPECS["k350"])
            .then(pl.lit("k350"))
            .otherwise(pl.lit("k359"))
            .alias("candle")
        )
        .group_by(["date", "candle"])
        .agg(_range_exprs("candle"))
    )
    pre359_stats = (
        ordered.filter(pl.col("minute_et").is_between(50, 58))
        .group_by("date")
        .agg(
            pl.col("price").max().alias("pre359_high"),
            pl.col("price").min().alias("pre359_low"),
        )
    )

    grid = complete_dates.join(_candle_window_grid().lazy(), how="cross")
    study_lazy = (
        grid.join(
            _window_stats(ordered),
            on=["date", "candle", "window", "window_start_second", "window_end_second"],
            how="left",
        )
        .join(candle_stats, on=["date", "candle"], how="left")
        .join(macro_stats, on="date", how="left")
        .join(pre359_stats, on="date", how="left")
        .with_columns(pl.col("window_tick_count").fill_null(0).cast(pl.Int64))
    )
    study_lazy = _add_lazy_metrics(study_lazy).select(list(STUDY_SCHEMA))
    out = study_lazy.sort(["date", "candle", "window_start_second", "window_end_second", "window"]).collect(
        engine="streaming"
    )
    if out.is_empty():
        return _empty_study_frame()
    return out


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
    keys = list(SUMMARY_SCHEMA)
    return [{key: row.get(key) for key in keys} for row in rows]


def _summary_frame(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return _empty_summary_frame()
    return pl.DataFrame(_normalize_summary_rows(rows), schema=SUMMARY_SCHEMA, infer_schema_length=None)


def summarize_macro_tick_range_context(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    if study.is_empty():
        return _empty_summary_frame()
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
    return _summary_frame(rows)


def write_macro_tick_range_context(
    input_path: str | Path = TICK_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    study = build_macro_tick_range_context(scan_source(input_path))
    summary = summarize_macro_tick_range_context(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not data_sources.source_exists(TICK_INPUT_PATH):
        print(f"[ERROR] Input not found: {TICK_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary = write_macro_tick_range_context()
    print(f"[OK] Wrote macro tick range context -> {output}")
    print(f"[OK] Wrote macro tick range context summary -> {summary}")


if __name__ == "__main__":
    main()
