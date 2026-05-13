from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

GLOBEX_1M_INPUT_PATH = Path("outputs/nq_globex_volume_delta_1m.parquet")
MACRO_5S_INPUT_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_1550_delta_impulse.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_1550_delta_impulse_summary.parquet")

GLOBEX_REQUIRED_COLUMNS = {
    "trade_date_et",
    "session_minute_index",
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

PREDICTORS = ["eth_only_pre350", "rth_only_pre350", "eth_rth_pre350"]
TARGET_WINDOWS_5S = {
    "k350_00_09": (0, 1),
    "k350_00_04": (0, 0),
    "k350_05_09": (1, 1),
    "k350_00_29": (0, 5),
    "k350_00_59": (0, 11),
}
TARGET_WINDOWS = [*TARGET_WINDOWS_5S.keys(), *[f"k350_bucket_{bucket}" for bucket in range(0, 12)]]
ROBUST_TARGET_WINDOWS = [*TARGET_WINDOWS_5S.keys()]


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _validate_inputs(globex_1m: pl.DataFrame, macro_5s: pl.DataFrame) -> None:
    globex_missing = _missing_columns(globex_1m, GLOBEX_REQUIRED_COLUMNS)
    if globex_missing:
        raise ValueError(f"Missing Globex volume-delta columns: {globex_missing}")
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
            _safe_ratio_expr(pl.col(f"{prefix}_volume_delta"), pl.col(f"{prefix}_classified_size")).alias(
                f"{prefix}_delta_imbalance"
            )
        )
    )


def _aggregate_target_window_5s(macro_5s: pl.DataFrame, start: int, end: int, prefix: str) -> pl.DataFrame:
    return _aggregate_window(macro_5s, "macro_bucket_index", start, end, prefix)


def _sign_expr(column: str) -> pl.Expr:
    return pl.when(pl.col(column) > 0).then(1).when(pl.col(column) < 0).then(-1).otherwise(0)


def _add_signs(frame: pl.DataFrame) -> pl.DataFrame:
    names = [*PREDICTORS, *TARGET_WINDOWS]
    return frame.with_columns([_sign_expr(f"{name}_volume_delta").alias(f"{name}_sign") for name in names])


def _add_primary_relationships(frame: pl.DataFrame) -> pl.DataFrame:
    relationship_exprs: list[pl.Expr] = []
    target = "k350_00_09"
    target_sign = pl.col(f"{target}_sign")
    for predictor in PREDICTORS:
        pred_sign = pl.col(f"{predictor}_sign")
        has_signal = (pred_sign != 0) & (target_sign != 0)
        relationship_exprs.extend(
            [
                has_signal.alias(f"{predictor}_has_signal"),
                (has_signal & (pred_sign == -target_sign)).alias(f"{predictor}_opposes_{target}"),
                (has_signal & (pred_sign == target_sign)).alias(f"{predictor}_same_as_{target}"),
            ]
        )
    return frame.with_columns(relationship_exprs)


def build_macro_1550_delta_impulse(globex_1m: pl.DataFrame, macro_5s: pl.DataFrame) -> pl.DataFrame:
    _validate_inputs(globex_1m, macro_5s)
    eth = _aggregate_window(globex_1m, "session_minute_index", 0, 929, "eth_only_pre350")
    rth = _aggregate_window(globex_1m, "session_minute_index", 930, 1309, "rth_only_pre350")
    eth_rth = _aggregate_window(globex_1m, "session_minute_index", 0, 1309, "eth_rth_pre350")
    out = None
    for prefix, (start, end) in TARGET_WINDOWS_5S.items():
        target = _aggregate_target_window_5s(macro_5s, start, end, prefix)
        out = target if out is None else out.join(target, on="trade_date_et", how="full", coalesce=True)
    for bucket in range(0, 12):
        prefix = f"k350_bucket_{bucket}"
        target = _aggregate_target_window_5s(macro_5s, bucket, bucket, prefix)
        out = out.join(target, on="trade_date_et", how="full", coalesce=True)

    if out is None:
        out = pl.DataFrame({"trade_date_et": []}, schema={"trade_date_et": pl.Date})

    out = (
        out.join(eth, on="trade_date_et", how="left")
        .join(rth, on="trade_date_et", how="left")
        .join(eth_rth, on="trade_date_et", how="left")
        .rename({"trade_date_et": "date"})
    )
    out = _add_signs(out)
    out = _add_primary_relationships(out)
    return out.sort("date")


def _rate(numer: int, denom: int) -> float | None:
    return (numer / denom) if denom else None


def _available_target_windows(study: pl.DataFrame) -> list[str]:
    return [target for target in TARGET_WINDOWS if f"{target}_volume_delta" in study.columns]


def _scalar_or_none(frame: pl.DataFrame, expr: pl.Expr) -> float | None:
    if frame.is_empty():
        return None
    value = frame.select(expr).item()
    return None if value is None else float(value)


def _target_values_for_predictor_sign(study: pl.DataFrame, predictor: str, target: str, sign: int) -> pl.DataFrame:
    return study.filter(pl.col(f"{predictor}_sign") == sign).select(pl.col(f"{target}_volume_delta").alias("target_delta"))


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


def summarize_macro_1550_delta_impulse(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for predictor in PREDICTORS:
        for target in _available_target_windows(study):
            rows.append(_target_pair_sign_row(study, predictor, target))
    return pl.DataFrame(rows, infer_schema_length=None)


def load_volume_delta_inputs(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_5s_path: str | Path = MACRO_5S_INPUT_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(globex_path), pl.read_parquet(macro_5s_path)


def write_macro_1550_delta_impulse(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_5s_path: str | Path = MACRO_5S_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    globex_1m, macro_5s = load_volume_delta_inputs(globex_path, macro_5s_path)
    study = build_macro_1550_delta_impulse(globex_1m, macro_5s)
    summary = summarize_macro_1550_delta_impulse(study)
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
    if not MACRO_5S_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_5S_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary_output = write_macro_1550_delta_impulse()
    print(f"[OK] Wrote macro 15:50 delta impulse -> {output}")
    print(f"[OK] Wrote macro 15:50 delta impulse summary -> {summary_output}")


if __name__ == "__main__":
    main()
