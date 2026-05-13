from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

GLOBEX_1M_INPUT_PATH = Path("outputs/nq_globex_volume_delta_1m.parquet")
MACRO_1M_INPUT_PATH = Path("outputs/nq_macro_volume_delta_1m.parquet")
MACRO_5S_INPUT_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_delta_reversal.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_delta_reversal_summary.parquet")

GLOBEX_REQUIRED_COLUMNS = {
    "trade_date_et",
    "session_minute_index",
    "volume_delta",
    "classified_size",
    "total_size",
}
MACRO_REQUIRED_COLUMNS = {
    "trade_date_et",
    "macro_minute_index",
    "volume_delta",
    "classified_size",
    "total_size",
}
MACRO_5S_REQUIRED_COLUMNS = {
    "trade_date_et",
    "macro_bucket_index",
    "volume_delta",
    "classified_size",
    "total_size",
}

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

PRIMARY_PREDICTOR_ALIASES = {
    "eth_rth_pre59": "day_pre_macro",
    "eth_rth_macro_pre59": "day_plus_macro_pre59",
    "rth_macro_pre59": "rth_plus_macro_pre59",
}

PRIMARY_PREDICTORS = list(PRIMARY_PREDICTOR_ALIASES.keys())


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


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


def _safe_ratio_expr(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator != 0).then(numerator / denominator).otherwise(None)


def _aggregate_window(frame: pl.DataFrame, index_col: str, start: int, end: int, prefix: str) -> pl.DataFrame:
    return (
        frame.filter(pl.col(index_col).is_between(start, end))
        .group_by("trade_date_et")
        .agg(
            pl.col("volume_delta").sum().alias(f"{prefix}_volume_delta"),
            pl.col("classified_size").sum().alias(f"{prefix}_classified_size"),
            pl.col("total_size").sum().alias(f"{prefix}_total_size"),
        )
        .with_columns(
            _safe_ratio_expr(
                pl.col(f"{prefix}_volume_delta"),
                pl.col(f"{prefix}_classified_size"),
            ).alias(f"{prefix}_delta_imbalance")
        )
    )


def _extract_k359(macro_1m: pl.DataFrame) -> pl.DataFrame:
    return (
        macro_1m.filter(pl.col("macro_minute_index") == 59)
        .group_by("trade_date_et")
        .agg(
            pl.col("volume_delta").sum().alias("k359_volume_delta"),
            pl.col("classified_size").sum().alias("k359_classified_size"),
            pl.col("total_size").sum().alias("k359_total_size"),
        )
        .with_columns(
            _safe_ratio_expr(pl.col("k359_volume_delta"), pl.col("k359_classified_size")).alias(
                "k359_delta_imbalance"
            )
        )
    )


def _add_combined_window(frame: pl.DataFrame, left: str, right: str, output: str) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col(f"{left}_volume_delta").fill_null(0) + pl.col(f"{right}_volume_delta").fill_null(0)).alias(
            f"{output}_volume_delta"
        ),
        (pl.col(f"{left}_classified_size").fill_null(0) + pl.col(f"{right}_classified_size").fill_null(0)).alias(
            f"{output}_classified_size"
        ),
        (pl.col(f"{left}_total_size").fill_null(0) + pl.col(f"{right}_total_size").fill_null(0)).alias(
            f"{output}_total_size"
        ),
    ).with_columns(
        _safe_ratio_expr(pl.col(f"{output}_volume_delta"), pl.col(f"{output}_classified_size")).alias(
            f"{output}_delta_imbalance"
        )
    )


def _sign_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) > 0)
        .then(1)
        .when(pl.col(column) < 0)
        .then(-1)
        .otherwise(0)
    )


def _add_primary_predictor_aliases(frame: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for alias, source in PRIMARY_PREDICTOR_ALIASES.items():
        for suffix in ["volume_delta", "classified_size", "total_size", "delta_imbalance"]:
            exprs.append(pl.col(f"{source}_{suffix}").alias(f"{alias}_{suffix}"))
    return frame.with_columns(exprs)


def _add_signs_and_relationships(frame: pl.DataFrame) -> pl.DataFrame:
    sign_names = [*PREDICTORS, "k359"]
    out = frame.with_columns([_sign_expr(f"{name}_volume_delta").alias(f"{name}_sign") for name in sign_names])

    relationship_exprs: list[pl.Expr] = []
    for predictor in PREDICTORS:
        pred_sign = pl.col(f"{predictor}_sign")
        target_sign = pl.col("k359_sign")
        has_signal = (pred_sign != 0) & (target_sign != 0)
        relationship_exprs.extend(
            [
                has_signal.alias(f"{predictor}_has_signal"),
                (has_signal & (pred_sign == -target_sign)).alias(f"{predictor}_opposes_k359"),
                (has_signal & (pred_sign == target_sign)).alias(f"{predictor}_same_as_k359"),
            ]
        )

    out = out.with_columns(relationship_exprs)
    return out.with_columns(
        (
            (pl.col("macro_pre59_sign") != 0)
            & (pl.col("rth_pre_macro_sign") != 0)
            & (pl.col("macro_pre59_sign") == -pl.col("rth_pre_macro_sign"))
        ).alias("macro_pre59_opposes_rth_pre_macro"),
        (
            (pl.col("macro_pre59_sign") != 0)
            & (pl.col("day_pre_macro_sign") != 0)
            & (pl.col("macro_pre59_sign") == -pl.col("day_pre_macro_sign"))
        ).alias("macro_pre59_opposes_day_pre_macro"),
        (
            (pl.col("k359_sign") != 0)
            & (pl.col("rth_plus_macro_pre59_sign") != 0)
            & (pl.col("k359_sign") == -pl.col("rth_plus_macro_pre59_sign"))
        ).alias("k359_opposes_rth_plus_macro_pre59"),
        (
            (pl.col("k359_sign") != 0)
            & (pl.col("day_plus_macro_pre59_sign") != 0)
            & (pl.col("k359_sign") == -pl.col("day_plus_macro_pre59_sign"))
        ).alias("k359_opposes_day_plus_macro_pre59"),
    )


def build_macro_delta_reversal(
    globex_1m: pl.DataFrame,
    macro_1m: pl.DataFrame,
    macro_5s: pl.DataFrame | None = None,
) -> pl.DataFrame:
    _validate_inputs(globex_1m, macro_1m, macro_5s)

    target = _extract_k359(macro_1m)
    eth = _aggregate_window(globex_1m, "session_minute_index", 0, 929, "eth_pre_rth")
    rth = _aggregate_window(globex_1m, "session_minute_index", 930, 1309, "rth_pre_macro")
    day = _aggregate_window(globex_1m, "session_minute_index", 0, 1309, "day_pre_macro")
    macro_pre59 = _aggregate_window(macro_1m, "macro_minute_index", 50, 58, "macro_pre59")

    out = (
        target.join(eth, on="trade_date_et", how="left")
        .join(rth, on="trade_date_et", how="left")
        .join(day, on="trade_date_et", how="left")
        .join(macro_pre59, on="trade_date_et", how="left")
    )
    out = _add_combined_window(out, "rth_pre_macro", "macro_pre59", "rth_plus_macro_pre59")
    out = _add_combined_window(out, "day_pre_macro", "macro_pre59", "day_plus_macro_pre59")
    out = _add_primary_predictor_aliases(out.rename({"trade_date_et": "date"}))
    return _add_signs_and_relationships(out).sort("date")


def _rate(numer: int, denom: int) -> float | None:
    return (numer / denom) if denom else None


def _mean_for_sign(study: pl.DataFrame, predictor: str, sign: int) -> float | None:
    values = study.filter(pl.col(f"{predictor}_sign") == sign).select(pl.col("k359_volume_delta").mean()).item()
    return None if values is None else float(values)


def _median_for_sign(study: pl.DataFrame, predictor: str, sign: int) -> float | None:
    values = study.filter(pl.col(f"{predictor}_sign") == sign).select(pl.col("k359_volume_delta").median()).item()
    return None if values is None else float(values)


def _decile_rows(study: pl.DataFrame, predictor: str) -> list[dict]:
    value_col = f"{predictor}_volume_delta"
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
    for record in deciled.group_by("predictor_decile").agg(
        pl.len().alias("n_days"),
        pl.col(value_col).mean().alias("mean_predictor_delta"),
        pl.col("k359_volume_delta").mean().alias("mean_k359_delta"),
        pl.col(f"{predictor}_opposes_k359").sum().alias("opposite_count"),
        pl.col(f"{predictor}_has_signal").sum().alias("n_signal_days"),
    ).sort("predictor_decile").to_dicts():
        n_signal_days = int(record["n_signal_days"])
        opposite_count = int(record["opposite_count"])
        rows.append(
            {
                "summary_type": "decile",
                "predictor": predictor,
                "predictor_decile": int(record["predictor_decile"]),
                "n_days": int(record["n_days"]),
                "n_signal_days": n_signal_days,
                "opposite_count": opposite_count,
                "opposite_rate": _rate(opposite_count, n_signal_days),
                "same_count": None,
                "same_rate": None,
                "zero_predictor_count": None,
                "zero_k359_count": None,
                "mean_predictor_delta": float(record["mean_predictor_delta"]),
                "median_predictor_delta": None,
                "mean_k359_delta": float(record["mean_k359_delta"]),
                "median_k359_delta": None,
                "mean_k359_delta_when_predictor_positive": None,
                "mean_k359_delta_when_predictor_negative": None,
                "median_k359_delta_when_predictor_positive": None,
                "median_k359_delta_when_predictor_negative": None,
                "pearson_corr_predictor_vs_k359_delta": None,
            }
        )
    return rows


def summarize_macro_delta_reversal(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    n_days = study.height
    for predictor in PREDICTORS:
        signal = study.filter(pl.col(f"{predictor}_has_signal"))
        n_signal_days = signal.height
        opposite_count = study.filter(pl.col(f"{predictor}_opposes_k359")).height
        same_count = study.filter(pl.col(f"{predictor}_same_as_k359")).height
        corr = study.select(pl.corr(f"{predictor}_volume_delta", "k359_volume_delta")).item()
        rows.append(
            {
                "summary_type": "sign",
                "predictor": predictor,
                "predictor_decile": None,
                "n_days": n_days,
                "n_signal_days": n_signal_days,
                "opposite_count": opposite_count,
                "opposite_rate": _rate(opposite_count, n_signal_days),
                "same_count": same_count,
                "same_rate": _rate(same_count, n_signal_days),
                "zero_predictor_count": study.filter(pl.col(f"{predictor}_sign") == 0).height,
                "zero_k359_count": study.filter(pl.col("k359_sign") == 0).height,
                "mean_predictor_delta": study.select(pl.col(f"{predictor}_volume_delta").mean()).item(),
                "median_predictor_delta": study.select(pl.col(f"{predictor}_volume_delta").median()).item(),
                "mean_k359_delta": study.select(pl.col("k359_volume_delta").mean()).item(),
                "median_k359_delta": study.select(pl.col("k359_volume_delta").median()).item(),
                "mean_k359_delta_when_predictor_positive": _mean_for_sign(study, predictor, 1),
                "mean_k359_delta_when_predictor_negative": _mean_for_sign(study, predictor, -1),
                "median_k359_delta_when_predictor_positive": _median_for_sign(study, predictor, 1),
                "median_k359_delta_when_predictor_negative": _median_for_sign(study, predictor, -1),
                "pearson_corr_predictor_vs_k359_delta": corr,
            }
        )
        rows.extend(_decile_rows(study, predictor))
    return pl.DataFrame(rows)


def load_volume_delta_inputs(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_path: str | Path = MACRO_1M_INPUT_PATH,
    macro_5s_path: str | Path = MACRO_5S_INPUT_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(globex_path), pl.read_parquet(macro_path), pl.read_parquet(macro_5s_path)


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


def main() -> None:
    if not GLOBEX_1M_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {GLOBEX_1M_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not MACRO_1M_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_1M_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not MACRO_5S_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_5S_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary_output = write_macro_delta_reversal()
    print(f"[OK] Wrote macro delta reversal → {output}")
    print(f"[OK] Wrote macro delta reversal summary → {summary_output}")


if __name__ == "__main__":
    main()
