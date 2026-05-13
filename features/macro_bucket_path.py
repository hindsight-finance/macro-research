from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

MACRO_5S_INPUT_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_bucket_path.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_bucket_path_summary.parquet")

MACRO_5S_REQUIRED_COLUMNS = {
    "trade_date_et",
    "macro_bucket_index",
    "volume_delta",
    "classified_size",
    "total_size",
}

CANDLE_SPECS = {
    "k350": (0, 11),
    "k359": (108, 119),
}
RELATIVE_BUCKETS = list(range(12))
CUMULATIVE_WINDOWS = {f"cum_00_{end * 5 + 4:02d}": (0, end) for end in RELATIVE_BUCKETS}
NAMED_WINDOWS = {
    "early_5s": (0, 0),
    "early_10s": (0, 1),
    "early_30s": (0, 5),
    "late_30s": (6, 11),
    "full": (0, 11),
}


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _validate_inputs(macro_5s: pl.DataFrame) -> None:
    missing = _missing_columns(macro_5s, MACRO_5S_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Missing macro 5-second volume-delta columns: {missing}")


def _safe_ratio_expr(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator != 0).then(numerator / denominator).otherwise(None)


def _sign_expr(column: str) -> pl.Expr:
    return pl.when(pl.col(column) > 0).then(1).when(pl.col(column) < 0).then(-1).otherwise(0)


def _add_window_columns(frame: pl.DataFrame, prefix: str, start: int, end: int) -> pl.DataFrame:
    delta_cols = [pl.col(f"b{bucket}_volume_delta") for bucket in range(start, end + 1)]
    classified_cols = [pl.col(f"b{bucket}_classified_size") for bucket in range(start, end + 1)]
    total_cols = [pl.col(f"b{bucket}_total_size") for bucket in range(start, end + 1)]
    delta_present = pl.sum_horizontal([col.is_not_null().cast(pl.Int64) for col in delta_cols])
    classified_present = pl.sum_horizontal([col.is_not_null().cast(pl.Int64) for col in classified_cols])
    total_present = pl.sum_horizontal([col.is_not_null().cast(pl.Int64) for col in total_cols])
    return frame.with_columns(
        pl.when(delta_present > 0).then(pl.sum_horizontal(delta_cols)).otherwise(None).alias(f"{prefix}_volume_delta"),
        pl.when(classified_present > 0).then(pl.sum_horizontal(classified_cols)).otherwise(None).alias(f"{prefix}_classified_size"),
        pl.when(total_present > 0).then(pl.sum_horizontal(total_cols)).otherwise(None).alias(f"{prefix}_total_size"),
    ).with_columns(
        _safe_ratio_expr(pl.col(f"{prefix}_volume_delta"), pl.col(f"{prefix}_classified_size")).alias(
            f"{prefix}_delta_imbalance"
        ),
        _sign_expr(f"{prefix}_volume_delta").alias(f"{prefix}_sign"),
    )


def _add_path_windows(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for prefix, (start, end) in CUMULATIVE_WINDOWS.items():
        out = _add_window_columns(out, prefix, start, end)
    for prefix, (start, end) in NAMED_WINDOWS.items():
        out = _add_window_columns(out, prefix, start, end)
    return out


def _first_max_index_expr(value_prefix: str, max_col: str) -> pl.Expr:
    expr = None
    for bucket in RELATIVE_BUCKETS:
        if value_prefix == "bucket":
            is_max = pl.col(f"b{bucket}_volume_delta").fill_null(0).abs() == pl.col(max_col)
        else:
            is_max = pl.col(f"cum_00_{bucket * 5 + 4:02d}_volume_delta").fill_null(0).abs() == pl.col(max_col)
        expr = pl.when(is_max).then(bucket) if expr is None else expr.when(is_max).then(bucket)
    return pl.when(pl.col(max_col) > 0).then(expr.otherwise(None)).otherwise(None)


def _add_path_diagnostics(frame: pl.DataFrame) -> pl.DataFrame:
    abs_bucket_exprs = [pl.col(f"b{bucket}_volume_delta").fill_null(0).abs() for bucket in RELATIVE_BUCKETS]
    cum_cols = [f"cum_00_{bucket * 5 + 4:02d}_volume_delta" for bucket in RELATIVE_BUCKETS]
    abs_cum_values = [pl.col(col).fill_null(0).abs() for col in cum_cols]

    out = frame.with_columns(
        pl.sum_horizontal(abs_bucket_exprs).alias("sum_abs_bucket_delta"),
        pl.max_horizontal(abs_bucket_exprs).alias("max_abs_bucket_delta"),
        pl.max_horizontal(abs_cum_values).alias("peak_abs_cum_delta"),
    ).with_columns(
        _safe_ratio_expr(pl.col("full_volume_delta"), pl.col("sum_abs_bucket_delta")).alias("path_efficiency"),
        _safe_ratio_expr(pl.col("early_10s_volume_delta").abs(), pl.col("sum_abs_bucket_delta")).alias(
            "early_10s_abs_flow_share"
        ),
        _first_max_index_expr("bucket", "max_abs_bucket_delta").alias("max_abs_bucket_index"),
        _first_max_index_expr("cum", "peak_abs_cum_delta").alias("peak_abs_cum_bucket_index"),
    )

    early_sign = pl.col("early_10s_sign")
    favorable = [pl.col(col).fill_null(0) * early_sign for col in cum_cols]
    out = out.with_columns(
        pl.when(early_sign != 0).then(pl.max_horizontal(favorable)).otherwise(None).alias("max_favorable_cum_delta"),
        pl.when(early_sign != 0).then(pl.min_horizontal(favorable)).otherwise(None).alias("max_adverse_cum_delta"),
    )

    flip_struct = pl.struct([f"cum_00_{bucket * 5 + 4:02d}_sign" for bucket in RELATIVE_BUCKETS])
    return out.with_columns(
        flip_struct.map_elements(_count_last_nonzero_sign_flips, return_dtype=pl.Int64).alias("cum_sign_flip_count")
    )


def _count_last_nonzero_sign_flips(values: dict[str, int]) -> int:
    flips = 0
    last = 0
    for value in values.values():
        sign = int(value or 0)
        if sign == 0:
            continue
        if last != 0 and sign != last:
            flips += 1
        last = sign
    return flips


def _add_continuation_flags(frame: pl.DataFrame) -> pl.DataFrame:
    comparisons = {
        "30s": "early_30s",
        "late30": "late_30s",
        "full": "full",
    }
    exprs = []
    early_sign = pl.col("early_10s_sign")
    for label, target in comparisons.items():
        target_sign = pl.col(f"{target}_sign")
        has_signal = (early_sign != 0) & (target_sign != 0)
        exprs.extend(
            [
                (has_signal & (early_sign == target_sign)).alias(f"early_10s_continues_to_{label}"),
                (has_signal & (early_sign == -target_sign)).alias(f"early_10s_fades_to_{label}"),
            ]
        )
    return frame.with_columns(exprs)


def _build_candle_rows(macro_5s: pl.DataFrame, candle: str, start: int, end: int) -> pl.DataFrame:
    filtered = macro_5s.filter(pl.col("macro_bucket_index").is_between(start, end)).with_columns(
        pl.lit(candle).alias("candle"),
        (pl.col("macro_bucket_index") - start).cast(pl.Int8).alias("relative_bucket"),
    )
    base = filtered.group_by("trade_date_et", "candle").agg(
        pl.len().alias("bucket_count"),
        (pl.len() == 12).alias("complete_candle"),
    )
    out = base
    for bucket in RELATIVE_BUCKETS:
        one = filtered.filter(pl.col("relative_bucket") == bucket).select(
            "trade_date_et",
            "candle",
            pl.col("volume_delta").alias(f"b{bucket}_volume_delta"),
            pl.col("classified_size").alias(f"b{bucket}_classified_size"),
            pl.col("total_size").alias(f"b{bucket}_total_size"),
        )
        one = one.with_columns(
            _safe_ratio_expr(pl.col(f"b{bucket}_volume_delta"), pl.col(f"b{bucket}_classified_size")).alias(
                f"b{bucket}_delta_imbalance"
            )
        )
        out = out.join(one, on=["trade_date_et", "candle"], how="left")
    sign_exprs = [_sign_expr(f"b{bucket}_volume_delta").alias(f"b{bucket}_sign") for bucket in RELATIVE_BUCKETS]
    out = out.with_columns(sign_exprs).rename({"trade_date_et": "date"})
    out = _add_path_windows(out)
    out = _add_path_diagnostics(out)
    return _add_continuation_flags(out)


def _sum_preserve_all_null(column: str) -> pl.Expr:
    return pl.when(pl.col(column).null_count() == pl.len()).then(None).otherwise(pl.col(column).sum()).alias(column)


def _aggregate_duplicate_buckets(macro_5s: pl.DataFrame) -> pl.DataFrame:
    return macro_5s.group_by("trade_date_et", "macro_bucket_index", maintain_order=True).agg(
        _sum_preserve_all_null("volume_delta"),
        _sum_preserve_all_null("classified_size"),
        _sum_preserve_all_null("total_size"),
    )


def _decile_values_for_column(frame: pl.DataFrame, value_col: str, output_col: str) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for candle in CANDLE_SPECS:
        subset = frame.filter((pl.col("candle") == candle) & pl.col(value_col).is_not_null()).select(
            "date", "candle", value_col
        )
        unique_count = subset.select(pl.col(value_col).n_unique()).item() if subset.height else 0
        if subset.height < 10 or unique_count < 10:
            values = frame.filter(pl.col("candle") == candle).select("date", "candle").with_columns(
                pl.lit(None, dtype=pl.Int64).alias(output_col)
            )
        else:
            values = (
                subset.sort([value_col, "date"])
                .with_row_index("_rank0")
                .with_columns(((pl.col("_rank0") * 10 / pl.len()).floor().cast(pl.Int64).clip(0, 9) + 1).alias(output_col))
                .select("date", "candle", output_col)
            )
            missing = frame.filter((pl.col("candle") == candle) & pl.col(value_col).is_null()).select("date", "candle").with_columns(
                pl.lit(None, dtype=pl.Int64).alias(output_col)
            )
            values = pl.concat([values, missing], how="vertical")
        parts.append(values)
    return pl.concat(parts, how="vertical") if parts else pl.DataFrame({"date": [], "candle": [], output_col: []})


def _add_conviction_categories(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame.with_columns(pl.col("early_10s_volume_delta").abs().alias("early_10s_abs_delta"))
    for value_col, output_col in [
        ("early_10s_volume_delta", "early_10s_raw_decile"),
        ("early_10s_delta_imbalance", "early_10s_imbalance_decile"),
        ("early_10s_abs_delta", "early_10s_abs_decile"),
    ]:
        out = out.join(_decile_values_for_column(out, value_col, output_col), on=["date", "candle"], how="left")
    return out.with_columns(
        pl.when(pl.col("early_10s_sign") == 0)
        .then(pl.lit("neutral"))
        .when(pl.col("early_10s_raw_decile").is_in([1, 2]))
        .then(pl.lit("strong_negative"))
        .when(pl.col("early_10s_raw_decile").is_in([3, 4]))
        .then(pl.lit("weak_negative"))
        .when(pl.col("early_10s_raw_decile").is_in([5, 6]))
        .then(pl.lit("neutral"))
        .when(pl.col("early_10s_raw_decile").is_in([7, 8]))
        .then(pl.lit("weak_positive"))
        .when(pl.col("early_10s_raw_decile").is_in([9, 10]))
        .then(pl.lit("strong_positive"))
        .when(pl.col("early_10s_sign") < 0)
        .then(pl.lit("weak_negative"))
        .when(pl.col("early_10s_sign") > 0)
        .then(pl.lit("weak_positive"))
        .otherwise(None)
        .alias("early_10s_category"),
        pl.when(pl.col("early_10s_abs_decile").is_in([9, 10]))
        .then(pl.lit("high_abs_conviction"))
        .when(pl.col("early_10s_abs_decile").is_between(4, 8))
        .then(pl.lit("mid_abs_conviction"))
        .when(pl.col("early_10s_abs_decile").is_between(1, 3))
        .then(pl.lit("low_abs_conviction"))
        .otherwise(None)
        .alias("early_10s_abs_category"),
    )


def build_macro_bucket_path(macro_5s: pl.DataFrame) -> pl.DataFrame:
    _validate_inputs(macro_5s)
    macro_5s = _aggregate_duplicate_buckets(macro_5s)
    frames = [_build_candle_rows(macro_5s, candle, start, end) for candle, (start, end) in CANDLE_SPECS.items()]
    out = pl.concat(frames, how="diagonal_relaxed").sort(["date", "candle"])
    return _add_conviction_categories(out)


def _rate(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


def _scalar(frame: pl.DataFrame, expr: pl.Expr) -> float | int | None:
    if frame.is_empty():
        return None
    return frame.select(expr).item()


def _sign_count(subset: pl.DataFrame, column: str, sign: int) -> int:
    if sign > 0:
        return subset.filter(pl.col(column) > 0).height
    if sign < 0:
        return subset.filter(pl.col(column) < 0).height
    return subset.filter(pl.col(column) == 0).height


def _base_summary_row(subset: pl.DataFrame, summary_type: str, candle: str, **labels: object) -> dict:
    complete = subset.filter(pl.col("complete_candle"))
    signal_30s = complete.filter((pl.col("early_10s_sign") != 0) & (pl.col("early_30s_sign") != 0))
    signal_late30 = complete.filter((pl.col("early_10s_sign") != 0) & (pl.col("late_30s_sign") != 0))
    signal_full = complete.filter((pl.col("early_10s_sign") != 0) & (pl.col("full_sign") != 0))
    n_signal_30s = signal_30s.height
    n_signal_late30 = signal_late30.height
    n_signal_full = signal_full.height
    continue_30s = complete.filter(pl.col("early_10s_continues_to_30s")).height
    fade_30s = complete.filter(pl.col("early_10s_fades_to_30s")).height
    continue_late30 = complete.filter(pl.col("early_10s_continues_to_late30")).height
    fade_late30 = complete.filter(pl.col("early_10s_fades_to_late30")).height
    continue_full = complete.filter(pl.col("early_10s_continues_to_full")).height
    fade_full = complete.filter(pl.col("early_10s_fades_to_full")).height
    early_pos = complete.filter(pl.col("early_10s_sign") > 0).height
    early_neg = complete.filter(pl.col("early_10s_sign") < 0).height
    early_zero = complete.filter(pl.col("early_10s_sign") == 0).height
    full_pos = complete.filter(pl.col("full_sign") > 0).height
    full_neg = complete.filter(pl.col("full_sign") < 0).height
    full_zero = complete.filter(pl.col("full_sign") == 0).height
    n_complete = complete.height
    row = {
        "summary_type": summary_type,
        "candle": candle,
        "n_days": subset.height,
        "n_complete_days": n_complete,
        "n_signal_days": n_signal_full,
        "n_signal_30s_days": n_signal_30s,
        "n_signal_late30_days": n_signal_late30,
        "n_signal_full_days": n_signal_full,
        "early_10s_positive_count": early_pos,
        "early_10s_negative_count": early_neg,
        "early_10s_zero_count": early_zero,
        "early_10s_positive_rate": _rate(early_pos, n_complete),
        "early_10s_negative_rate": _rate(early_neg, n_complete),
        "early_10s_zero_rate": _rate(early_zero, n_complete),
        "full_positive_count": full_pos,
        "full_negative_count": full_neg,
        "full_zero_count": full_zero,
        "full_positive_rate": _rate(full_pos, n_complete),
        "full_negative_rate": _rate(full_neg, n_complete),
        "full_zero_rate": _rate(full_zero, n_complete),
        "continue_to_30s_count": continue_30s,
        "continue_to_30s_rate": _rate(continue_30s, n_signal_30s),
        "fade_to_30s_count": fade_30s,
        "fade_to_30s_rate": _rate(fade_30s, n_signal_30s),
        "continue_to_late30_count": continue_late30,
        "continue_to_late30_rate": _rate(continue_late30, n_signal_late30),
        "fade_to_late30_count": fade_late30,
        "fade_to_late30_rate": _rate(fade_late30, n_signal_late30),
        "continue_to_full_count": continue_full,
        "continue_to_full_rate": _rate(continue_full, n_signal_full),
        "fade_to_full_count": fade_full,
        "fade_to_full_rate": _rate(fade_full, n_signal_full),
        "mean_early_10s_delta": _scalar(complete, pl.col("early_10s_volume_delta").mean()),
        "median_early_10s_delta": _scalar(complete, pl.col("early_10s_volume_delta").median()),
        "mean_late_30s_delta": _scalar(complete, pl.col("late_30s_volume_delta").mean()),
        "median_late_30s_delta": _scalar(complete, pl.col("late_30s_volume_delta").median()),
        "mean_full_delta": _scalar(complete, pl.col("full_volume_delta").mean()),
        "median_full_delta": _scalar(complete, pl.col("full_volume_delta").median()),
        "full_p25": _scalar(complete, pl.col("full_volume_delta").quantile(0.25)),
        "full_p75": _scalar(complete, pl.col("full_volume_delta").quantile(0.75)),
        "mean_path_efficiency": _scalar(complete, pl.col("path_efficiency").mean()),
        "median_path_efficiency": _scalar(complete, pl.col("path_efficiency").median()),
        "mean_early_10s_abs_flow_share": _scalar(complete, pl.col("early_10s_abs_flow_share").mean()),
        "median_early_10s_abs_flow_share": _scalar(complete, pl.col("early_10s_abs_flow_share").median()),
        "mean_cum_sign_flip_count": _scalar(complete, pl.col("cum_sign_flip_count").mean()),
        "median_cum_sign_flip_count": _scalar(complete, pl.col("cum_sign_flip_count").median()),
    }
    row.update(labels)
    return row


def _normalize_summary_rows(rows: list[dict]) -> list[dict]:
    keys = sorted({key for row in rows for key in row})
    return [{key: row.get(key) for key in keys} for row in rows]


def summarize_macro_bucket_path(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for candle in sorted(study["candle"].unique().to_list()):
        candle_df = study.filter(pl.col("candle") == candle)
        rows.append(_base_summary_row(candle_df, "candle_baseline", candle))
        for category in ["strong_negative", "weak_negative", "neutral", "weak_positive", "strong_positive"]:
            rows.append(
                _base_summary_row(
                    candle_df.filter(pl.col("early_10s_category") == category),
                    "early_10s_category",
                    candle,
                    early_10s_category=category,
                )
            )
        for category in ["low_abs_conviction", "mid_abs_conviction", "high_abs_conviction"]:
            rows.append(
                _base_summary_row(
                    candle_df.filter(pl.col("early_10s_abs_category") == category),
                    "early_10s_abs_category",
                    candle,
                    early_10s_abs_category=category,
                )
            )
        for decile in range(1, 11):
            rows.append(
                _base_summary_row(
                    candle_df.filter(pl.col("early_10s_raw_decile") == decile),
                    "early_10s_raw_decile",
                    candle,
                    early_10s_raw_decile=decile,
                )
            )
            rows.append(
                _base_summary_row(
                    candle_df.filter(pl.col("early_10s_imbalance_decile") == decile),
                    "early_10s_imbalance_decile",
                    candle,
                    early_10s_imbalance_decile=decile,
                )
            )
            rows.append(
                _base_summary_row(
                    candle_df.filter(pl.col("early_10s_abs_decile") == decile),
                    "early_10s_abs_decile",
                    candle,
                    early_10s_abs_decile=decile,
                )
            )
    return pl.DataFrame(_normalize_summary_rows(rows), infer_schema_length=None)


def load_macro_5s_input(path: str | Path = MACRO_5S_INPUT_PATH) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_macro_bucket_path(
    input_path: str | Path = MACRO_5S_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    macro_5s = load_macro_5s_input(input_path)
    study = build_macro_bucket_path(macro_5s)
    summary = summarize_macro_bucket_path(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not MACRO_5S_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_5S_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary_output = write_macro_bucket_path()
    print(f"[OK] Wrote macro bucket path -> {output}")
    print(f"[OK] Wrote macro bucket path summary -> {summary_output}")


if __name__ == "__main__":
    main()
