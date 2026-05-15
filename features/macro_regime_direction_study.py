from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl


DEFAULT_REGIME_PATH = Path("outputs/trend_modeling/cache/nq_trend_modeling_table_regime_3scalar.parquet")
DEFAULT_MACRO_PATH = Path("outputs/nq_macro_outcomes.parquet")
DEFAULT_DELTA_PATH = Path("outputs/nq_globex_volume_delta_1m.parquet")
DEFAULT_OUTPUT_PATH = Path("outputs/nq_macro_regime_direction_study.parquet")
DEFAULT_FIGURE_DIR = Path("outputs/figs/macro_regime_direction")

REGIME_WINDOWS = {
    "1pm-3pm": "13_15",
    "3pm-3:50pm": "15_1550",
}
REGIME_SCORE_COLUMNS = ("trend_score", "containment_score", "chop_score")
DELTA_WINDOWS = {
    "13_15": ("13:00:00", "15:00:00"),
    "15_1550": ("15:00:00", "15:50:00"),
    "1550_16": ("15:50:00", "16:00:00"),
}
MACRO_OUTCOME_COLUMNS = (
    "macro_dir_points",
    "macro_range_pct",
    "skew_ratio",
    "close_in_range",
    "macro_high_time",
    "macro_low_time",
    "postclose_range_pct",
)
MARKET_TZ = "America/New_York"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "outputs" / "nq_macro_outcomes.parquet"
        if candidate.exists():
            return parent
    return Path.cwd()


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


def classify_macro_direction(macro_dir_points: float | int | None) -> str | None:
    if macro_dir_points is None:
        return None
    if macro_dir_points > 0:
        return "bullish"
    if macro_dir_points < 0:
        return "bearish"
    return "flat"


def classify_delta_sign(volume_delta: float | int | None) -> str | None:
    if volume_delta is None:
        return None
    if volume_delta > 0:
        return "buy"
    if volume_delta < 0:
        return "sell"
    return "neutral"


def _read_frame(frame_or_path: pl.DataFrame | str | Path) -> pl.DataFrame:
    if isinstance(frame_or_path, pl.DataFrame):
        return frame_or_path
    return pl.read_parquet(_resolve_path(frame_or_path))


def _build_regime_wide(regime: pl.DataFrame) -> pl.DataFrame:
    required = {"trade_date", "session_name", *REGIME_SCORE_COLUMNS}
    missing = required.difference(regime.columns)
    if missing:
        raise ValueError(f"Regime frame missing columns: {sorted(missing)}")

    out: pl.DataFrame | None = None
    for session_name, suffix in REGIME_WINDOWS.items():
        cols = [pl.col("trade_date").alias("date")]
        cols.extend(pl.col(column).alias(f"{column}_{suffix}") for column in REGIME_SCORE_COLUMNS)
        if "feature_status" in regime.columns:
            cols.append(pl.col("feature_status").alias(f"feature_status_{suffix}"))

        window = regime.filter(pl.col("session_name") == session_name).select(cols)
        out = window if out is None else out.join(window, on="date", how="full", coalesce=True)

    if out is None:
        return pl.DataFrame({"date": []}, schema={"date": pl.Date})
    return out.sort("date")


def _add_macro_direction(macro: pl.DataFrame) -> pl.DataFrame:
    if "date" not in macro.columns or "macro_dir_points" not in macro.columns:
        raise ValueError("Macro frame must contain date and macro_dir_points")
    keep = ["date", *[column for column in MACRO_OUTCOME_COLUMNS if column in macro.columns]]
    return macro.select(keep).with_columns(
        macro_dir_sign=pl.col("macro_dir_points").sign().fill_null(0).cast(pl.Int8),
        macro_direction=(
            pl.when(pl.col("macro_dir_points") > 0)
            .then(pl.lit("bullish"))
            .when(pl.col("macro_dir_points") < 0)
            .then(pl.lit("bearish"))
            .otherwise(pl.lit("flat"))
        ),
    )


def _aggregate_delta_window(delta: pl.DataFrame, label: str, start_time: str, end_time: str) -> pl.DataFrame:
    required = {"datetime_utc", "trade_date_et", "volume_delta", "classified_size", "total_size", "tick_delta"}
    missing = required.difference(delta.columns)
    if missing:
        raise ValueError(f"Delta frame missing columns: {sorted(missing)}")

    base = delta.with_columns(
        et_time=pl.col("datetime_utc").dt.convert_time_zone(MARKET_TZ).dt.strftime("%H:%M:%S"),
    )
    return (
        base.filter((pl.col("et_time") >= start_time) & (pl.col("et_time") < end_time))
        .group_by(pl.col("trade_date_et").alias("date"))
        .agg(
            pl.col("volume_delta").sum().alias(f"delta_{label}_volume_delta"),
            pl.col("tick_delta").sum().alias(f"delta_{label}_tick_delta"),
            pl.col("classified_size").sum().alias(f"delta_{label}_classified_size"),
            pl.col("total_size").sum().alias(f"delta_{label}_total_size"),
            pl.col("classified_share").mean().alias(f"delta_{label}_classified_share")
            if "classified_share" in delta.columns
            else (pl.col("classified_size").sum() / pl.col("total_size").sum()).alias(f"delta_{label}_classified_share"),
        )
        .with_columns(
            (pl.col(f"delta_{label}_volume_delta") / pl.col(f"delta_{label}_classified_size")).alias(
                f"delta_{label}_imbalance"
            ),
            pl.when(pl.col(f"delta_{label}_volume_delta") > 0)
            .then(pl.lit("buy"))
            .when(pl.col(f"delta_{label}_volume_delta") < 0)
            .then(pl.lit("sell"))
            .otherwise(pl.lit("neutral"))
            .alias(f"delta_{label}_delta_sign"),
        )
    )


def _build_delta_wide(delta: pl.DataFrame | None) -> pl.DataFrame | None:
    if delta is None:
        return None
    out: pl.DataFrame | None = None
    for label, (start_time, end_time) in DELTA_WINDOWS.items():
        window = _aggregate_delta_window(delta, label=label, start_time=start_time, end_time=end_time)
        out = window if out is None else out.join(window, on="date", how="full", coalesce=True)
    return out.sort("date") if out is not None else None


def _tertile_expr(column: str, low_cut: float | None, high_cut: float | None) -> pl.Expr:
    if low_cut is None or high_cut is None:
        return pl.lit(None, dtype=pl.Utf8)
    return (
        pl.when(pl.col(column).is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pl.col(column) <= low_cut)
        .then(pl.lit("low"))
        .when(pl.col(column) <= high_cut)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
    )


def _with_tertile_bucket(frame: pl.DataFrame, column: str, bucket_column: str) -> pl.DataFrame:
    values = frame.select(pl.col(column).drop_nulls()).to_series()
    if values.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(bucket_column))
    low_cut = float(values.quantile(1 / 3, interpolation="linear"))
    high_cut = float(values.quantile(2 / 3, interpolation="linear"))
    return frame.with_columns(_tertile_expr(column, low_cut, high_cut).alias(bucket_column))


def _add_buckets(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for score in REGIME_SCORE_COLUMNS:
        for suffix in REGIME_WINDOWS.values():
            column = f"{score}_{suffix}"
            if column in out.columns:
                out = _with_tertile_bucket(out, column, f"{column}_bucket")

    for label in DELTA_WINDOWS:
        delta_col = f"delta_{label}_volume_delta"
        if delta_col in out.columns:
            out = out.with_columns(pl.col(delta_col).abs().alias(f"delta_{label}_abs_volume_delta"))
            out = _with_tertile_bucket(out, f"delta_{label}_abs_volume_delta", f"delta_{label}_magnitude_bucket")
    return out


def build_macro_regime_direction_study(
    regime: pl.DataFrame | str | Path,
    macro: pl.DataFrame | str | Path,
    delta: pl.DataFrame | str | Path | None = None,
) -> pl.DataFrame:
    regime_df = _read_frame(regime)
    macro_df = _read_frame(macro)
    delta_df = _read_frame(delta) if delta is not None else None

    study = _add_macro_direction(macro_df).join(_build_regime_wide(regime_df), on="date", how="inner")
    delta_wide = _build_delta_wide(delta_df)
    if delta_wide is not None:
        study = study.join(delta_wide, on="date", how="left")

    delta_size_columns = [f"delta_{label}_classified_size" for label in DELTA_WINDOWS if f"delta_{label}_classified_size" in study.columns]
    if delta_size_columns:
        delta_available = pl.any_horizontal([pl.col(column).is_not_null() for column in delta_size_columns])
    else:
        delta_available = pl.lit(False)

    return _add_buckets(study.with_columns(delta_available=delta_available)).sort("date")


def _direction_rate_expr(direction: str) -> pl.Expr:
    return (pl.col("macro_direction") == direction).mean().alias(f"{direction}_rate")


def _baseline(study: pl.DataFrame) -> pl.DataFrame:
    return study.select(
        pl.lit("all").alias("scope"),
        pl.len().alias("sample_size"),
        _direction_rate_expr("bullish"),
        _direction_rate_expr("bearish"),
        _direction_rate_expr("flat"),
        pl.col("macro_dir_points").mean().alias("avg_macro_dir_points"),
        pl.col("macro_range_pct").mean().alias("avg_macro_range_pct") if "macro_range_pct" in study.columns else pl.lit(None).alias("avg_macro_range_pct"),
    )


def _summarize_bucket(study: pl.DataFrame, column: str, baseline: Mapping[str, float]) -> pl.DataFrame:
    return (
        study.filter(pl.col(column).is_not_null())
        .group_by(column)
        .agg(
            pl.len().alias("sample_size"),
            _direction_rate_expr("bullish"),
            _direction_rate_expr("bearish"),
            _direction_rate_expr("flat"),
            pl.col("macro_dir_points").mean().alias("avg_macro_dir_points"),
            pl.col("macro_range_pct").mean().alias("avg_macro_range_pct") if "macro_range_pct" in study.columns else pl.lit(None).alias("avg_macro_range_pct"),
        )
        .rename({column: "bucket"})
        .with_columns(
            pl.lit(column).alias("cohort"),
            (pl.col("bullish_rate") - baseline["bullish_rate"]).alias("bullish_lift"),
            (pl.col("bearish_rate") - baseline["bearish_rate"]).alias("bearish_lift"),
        )
        .select(
            "cohort",
            "bucket",
            "sample_size",
            "bullish_rate",
            "bearish_rate",
            "flat_rate",
            "bullish_lift",
            "bearish_lift",
            "avg_macro_dir_points",
            "avg_macro_range_pct",
        )
        .sort(["cohort", "bucket"])
    )


def _numeric_columns(study: pl.DataFrame, candidates: Sequence[str]) -> list[str]:
    numeric_types = {pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    schema = study.schema
    return [column for column in candidates if column in schema and schema[column] in numeric_types]


def _correlations(study: pl.DataFrame) -> pl.DataFrame:
    feature_candidates = []
    for score in REGIME_SCORE_COLUMNS:
        feature_candidates.extend(f"{score}_{suffix}" for suffix in REGIME_WINDOWS.values())
    for label in DELTA_WINDOWS:
        feature_candidates.extend(
            [
                f"delta_{label}_volume_delta",
                f"delta_{label}_imbalance",
                f"delta_{label}_tick_delta",
            ]
        )
    features = _numeric_columns(study, feature_candidates)
    outcomes = _numeric_columns(study, MACRO_OUTCOME_COLUMNS)

    rows: list[dict] = []
    for feature in features:
        for outcome in outcomes:
            pair = study.select(feature, outcome).drop_nulls()
            sample_size = pair.height
            pearson = pair.select(pl.corr(feature, outcome)).item() if sample_size > 1 else None
            ranked = pair.with_columns(pl.col(feature).rank().alias("_feature_rank"), pl.col(outcome).rank().alias("_outcome_rank"))
            spearman = ranked.select(pl.corr("_feature_rank", "_outcome_rank")).item() if sample_size > 1 else None
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "pearson": pearson,
                    "spearman": spearman,
                    "sample_size": sample_size,
                }
            )
    return pl.DataFrame(rows)


def _combo_summary(study: pl.DataFrame, left: str, right: str, baseline: Mapping[str, float]) -> pl.DataFrame:
    if left not in study.columns or right not in study.columns:
        return pl.DataFrame(
            schema={
                "left_cohort": pl.Utf8,
                "right_cohort": pl.Utf8,
                "left_bucket": pl.Utf8,
                "right_bucket": pl.Utf8,
                "sample_size": pl.UInt32,
                "bullish_rate": pl.Float64,
                "bearish_rate": pl.Float64,
                "bullish_lift": pl.Float64,
                "bearish_lift": pl.Float64,
            }
        )
    return (
        study.filter(pl.col(left).is_not_null() & pl.col(right).is_not_null())
        .group_by(left, right)
        .agg(
            pl.len().alias("sample_size"),
            _direction_rate_expr("bullish"),
            _direction_rate_expr("bearish"),
        )
        .rename({left: "left_bucket", right: "right_bucket"})
        .with_columns(
            pl.lit(left).alias("left_cohort"),
            pl.lit(right).alias("right_cohort"),
            (pl.col("bullish_rate") - baseline["bullish_rate"]).alias("bullish_lift"),
            (pl.col("bearish_rate") - baseline["bearish_rate"]).alias("bearish_lift"),
        )
        .select(
            "left_cohort",
            "right_cohort",
            "left_bucket",
            "right_bucket",
            "sample_size",
            "bullish_rate",
            "bearish_rate",
            "bullish_lift",
            "bearish_lift",
        )
        .sort(["left_bucket", "right_bucket"])
    )


def summarize_macro_regime_direction_study(study: pl.DataFrame) -> dict[str, pl.DataFrame]:
    baseline = _baseline(study)
    base_row = baseline.row(0, named=True)
    bucket_columns = [
        column
        for column in study.columns
        if column.endswith("_bucket") or column.endswith("_delta_sign")
    ]
    single_parts = [_summarize_bucket(study, column, base_row) for column in bucket_columns]
    single_bucket = pl.concat(single_parts, how="vertical") if single_parts else pl.DataFrame()
    combo = _combo_summary(
        study,
        "trend_score_13_15_bucket",
        "trend_score_15_1550_bucket",
        base_row,
    )
    return {
        "baseline": baseline,
        "single_bucket": single_bucket,
        "window_combo": combo,
        "correlations": _correlations(study),
    }


def _write_summary_csvs(summaries: Mapping[str, pl.DataFrame], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in summaries.items():
        frame.write_csv(figure_dir / f"{name}.csv")


def _plot_trend_heatmap(combo: pl.DataFrame, figure_dir: Path) -> None:
    if combo.is_empty():
        return
    order = ["low", "mid", "high"]
    pivot = combo.pivot(index="left_bucket", on="right_bucket", values="bullish_lift", aggregate_function="mean")
    values = []
    for left in order:
        row_values = []
        row = pivot.filter(pl.col("left_bucket") == left)
        for right in order:
            value = row.item(0, right) if row.height and right in row.columns else math.nan
            row_values.append(float(value) if value is not None else math.nan)
        values.append(row_values)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(order)), order)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("3pm-3:50pm trend tertile")
    ax.set_ylabel("1pm-3pm trend tertile")
    ax.set_title("Bullish lift vs baseline")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "trend_window_cohort_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_lift_bars(single_bucket: pl.DataFrame, figure_dir: Path) -> None:
    focus = single_bucket.filter(pl.col("cohort").is_in(["trend_score_13_15_bucket", "trend_score_15_1550_bucket"]))
    if focus.is_empty():
        return
    labels = [f"{row['cohort']}:{row['bucket']}" for row in focus.iter_rows(named=True)]
    values = focus["bullish_lift"].to_list()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(values)), values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylabel("Bullish lift")
    ax.set_title("Trend regime bucket lift")
    fig.tight_layout()
    fig.savefig(figure_dir / "trend_bucket_bullish_lift.png", dpi=150)
    plt.close(fig)


def _plot_scatter(study: pl.DataFrame, figure_dir: Path) -> None:
    if "trend_score_15_1550" not in study.columns or "macro_dir_points" not in study.columns:
        return
    pair = study.select("trend_score_15_1550", "macro_dir_points").drop_nulls()
    if pair.is_empty():
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(pair["trend_score_15_1550"].to_list(), pair["macro_dir_points"].to_list(), alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("3pm-3:50pm trend score")
    ax.set_ylabel("Macro dir points")
    ax.set_title("Prior trend score vs macro direction")
    fig.tight_layout()
    fig.savefig(figure_dir / "trend_score_vs_macro_dir_points.png", dpi=150)
    plt.close(fig)


def write_macro_regime_direction_study(
    regime_path: str | Path = DEFAULT_REGIME_PATH,
    macro_path: str | Path = DEFAULT_MACRO_PATH,
    delta_path: str | Path | None = DEFAULT_DELTA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict[str, Path]:
    output = _resolve_path(output_path)
    figures = _resolve_path(figure_dir)
    study = build_macro_regime_direction_study(regime_path, macro_path, delta_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)

    summaries = summarize_macro_regime_direction_study(study)
    _write_summary_csvs(summaries, figures)
    _plot_trend_heatmap(summaries["window_combo"], figures)
    _plot_lift_bars(summaries["single_bucket"], figures)
    _plot_scatter(study, figures)

    return {"study": output, "figure_dir": figures}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build descriptive macro-regime direction study.")
    parser.add_argument("--regime-path", default=str(DEFAULT_REGIME_PATH))
    parser.add_argument("--macro-path", default=str(DEFAULT_MACRO_PATH))
    parser.add_argument("--delta-path", default=str(DEFAULT_DELTA_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    delta_path = args.delta_path if args.delta_path else None
    wrote = write_macro_regime_direction_study(
        regime_path=args.regime_path,
        macro_path=args.macro_path,
        delta_path=delta_path,
        output_path=args.output_path,
        figure_dir=args.figure_dir,
    )
    print(f"Wrote study table: {wrote['study']}")
    print(f"Wrote summaries/figures: {wrote['figure_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
