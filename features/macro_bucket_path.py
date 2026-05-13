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
    return out.with_columns(sign_exprs).rename({"trade_date_et": "date"})


def build_macro_bucket_path(macro_5s: pl.DataFrame) -> pl.DataFrame:
    _validate_inputs(macro_5s)
    frames = [_build_candle_rows(macro_5s, candle, start, end) for candle, (start, end) in CANDLE_SPECS.items()]
    return pl.concat(frames, how="diagonal_relaxed").sort(["date", "candle"])


def summarize_macro_bucket_path(study: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for candle in sorted(study["candle"].unique().to_list()):
        subset = study.filter(pl.col("candle") == candle)
        rows.append({"summary_type": "candle_baseline", "candle": candle, "n_days": subset.height})
    return pl.DataFrame(rows, infer_schema_length=None)


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
