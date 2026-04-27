from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    from features.macro_extreme_timing import summarize_macro_extreme_timing
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from features.macro_extreme_timing import summarize_macro_extreme_timing

DATASET_PATH = Path("outputs/nq_macro_extreme_timing.parquet")
OUT_DIR = Path("outputs/figs/macro_extreme_timing")
EXPECTED_KEY_MINUTES = [50, 54, 55, 59]
SECONDS = list(range(60))


@dataclass(frozen=True)
class MacroExtremeTimingVizSummary:
    path: Path
    dataset_name: str
    frequency: pl.DataFrame
    quantiles: pl.DataFrame
    total_observations: int
    total_days: int


def _long_extremes(df: pl.DataFrame) -> pl.DataFrame:
    high = df.select(
        "date",
        "macro_trend_state",
        "macro_minute_index",
        pl.lit("high").alias("extreme"),
        pl.col("candle_high_time").alias("second"),
    )
    low = df.select(
        "date",
        "macro_trend_state",
        "macro_minute_index",
        pl.lit("low").alias("extreme"),
        pl.col("candle_low_time").alias("second"),
    )
    return pl.concat([high, low], how="vertical")


def _frequency_table(df: pl.DataFrame) -> pl.DataFrame:
    long = _long_extremes(df)
    combos = long.select("macro_trend_state", "macro_minute_index", "extreme").unique()
    seconds = pl.DataFrame({"second": SECONDS})
    grid = combos.join(seconds, how="cross")
    counts = long.group_by("macro_trend_state", "macro_minute_index", "extreme", "second").len(name="count")
    return (
        grid.join(counts, on=["macro_trend_state", "macro_minute_index", "extreme", "second"], how="left")
        .with_columns(pl.col("count").fill_null(0))
        .sort("macro_trend_state", "macro_minute_index", "extreme", "second")
    )


def summarize_macro_extreme_timing_dataset(path: str | Path) -> MacroExtremeTimingVizSummary:
    dataset_path = Path(path)
    df = pl.read_parquet(dataset_path)
    required = {"date", "macro_trend_state", "macro_minute_index", "candle_high_time", "candle_low_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return MacroExtremeTimingVizSummary(
        path=dataset_path,
        dataset_name=dataset_path.stem,
        frequency=_frequency_table(df),
        quantiles=summarize_macro_extreme_timing(df),
        total_observations=df.height,
        total_days=df.select(pl.col("date").n_unique()).item(),
    )


def _plot_histograms(df: pl.DataFrame, dataset_name: str, out_path: Path) -> None:
    trend_states = [s for s in ["bullish", "bearish", "neutral"] if s in df.select("macro_trend_state").to_series().unique().to_list()]
    key_minutes = [m for m in EXPECTED_KEY_MINUTES if m in df.select("macro_minute_index").to_series().unique().to_list()]
    if not trend_states or not key_minutes:
        return

    fig, axes = plt.subplots(len(key_minutes), len(trend_states), figsize=(5 * len(trend_states), 3.2 * len(key_minutes)), squeeze=False)
    bins = np.arange(0, 62, 2)
    for r, minute in enumerate(key_minutes):
        for c, trend in enumerate(trend_states):
            ax = axes[r][c]
            sub = df.filter((pl.col("macro_minute_index") == minute) & (pl.col("macro_trend_state") == trend))
            high = sub.select("candle_high_time").to_series().to_numpy()
            low = sub.select("candle_low_time").to_series().to_numpy()
            ax.hist(high, bins=bins, alpha=0.58, label="High", color="#08519c")
            ax.hist(low, bins=bins, alpha=0.48, label="Low", color="#cb181d")
            ax.set_title(f"15:{minute:02d} {trend} (n={sub.height})")
            ax.set_xlim(0, 59)
            ax.set_xlabel("Second in minute")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.22)
            if r == 0 and c == 0:
                ax.legend()
    fig.suptitle(f"{dataset_name}: high/low first-touch timing")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_heatmap(frequency: pl.DataFrame, dataset_name: str, out_path: Path) -> None:
    work = frequency.filter(pl.col("macro_trend_state").is_in(["bullish", "bearish"]))
    if work.is_empty():
        work = frequency
    labels = []
    rows = []
    for row in work.select("macro_trend_state", "macro_minute_index", "extreme").unique().sort(
        "macro_trend_state", "macro_minute_index", "extreme"
    ).iter_rows(named=True):
        sub = work.filter(
            (pl.col("macro_trend_state") == row["macro_trend_state"])
            & (pl.col("macro_minute_index") == row["macro_minute_index"])
            & (pl.col("extreme") == row["extreme"])
        ).sort("second")
        values = sub.select("count").to_series().to_numpy().astype(float)
        total = values.sum()
        rows.append(values / total if total else values)
        labels.append(f"{row['macro_trend_state']} 15:{int(row['macro_minute_index']):02d} {row['extreme']}")

    if not rows:
        return

    matrix = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(13, max(4, 0.32 * len(labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_title(f"{dataset_name}: second-level extreme timing frequency")
    ax.set_xlabel("Second in minute")
    ax.set_ylabel("Context")
    ax.set_xticks(list(range(0, 60, 5)))
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Share of context observations")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _directional_extreme_stats(df: pl.DataFrame) -> pl.DataFrame:
    work = df.filter(pl.col("macro_trend_state").is_in(["bullish", "bearish"])).with_columns(
        directional_extreme=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.lit("high")).otherwise(pl.lit("low")),
        directional_extreme_time=pl.when(pl.col("macro_trend_state") == "bullish")
        .then(pl.col("candle_high_time"))
        .otherwise(pl.col("candle_low_time")),
    )
    if work.is_empty():
        return pl.DataFrame(
            schema={
                "macro_trend_state": pl.String,
                "macro_minute_index": pl.UInt8,
                "directional_extreme": pl.String,
                "sample_size": pl.UInt32,
                "mean_time": pl.Float64,
                "median_time": pl.Float64,
                "p25_time": pl.Float64,
                "p75_time": pl.Float64,
                "p90_time": pl.Float64,
                "late_extreme_pct": pl.Float64,
            }
        )
    return (
        work.group_by("macro_trend_state", "macro_minute_index", "directional_extreme")
        .agg(
            pl.len().alias("sample_size"),
            pl.col("directional_extreme_time").mean().alias("mean_time"),
            pl.col("directional_extreme_time").median().alias("median_time"),
            pl.col("directional_extreme_time").quantile(0.25).alias("p25_time"),
            pl.col("directional_extreme_time").quantile(0.75).alias("p75_time"),
            pl.col("directional_extreme_time").quantile(0.90).alias("p90_time"),
            ((pl.col("directional_extreme_time") >= 45).mean() * 100.0).alias("late_extreme_pct"),
        )
        .sort("macro_trend_state", "macro_minute_index")
    )


def _plot_ecdf(df: pl.DataFrame, dataset_name: str, out_path: Path) -> None:
    trend_states = [s for s in ["bullish", "bearish"] if s in df.select("macro_trend_state").to_series().unique().to_list()]
    key_minutes = [m for m in EXPECTED_KEY_MINUTES if m in df.select("macro_minute_index").to_series().unique().to_list()]
    if not trend_states or not key_minutes:
        return
    fig, axes = plt.subplots(len(key_minutes), len(trend_states), figsize=(5 * len(trend_states), 3.0 * len(key_minutes)), squeeze=False)
    for r, minute in enumerate(key_minutes):
        for c, trend in enumerate(trend_states):
            ax = axes[r][c]
            sub = df.filter((pl.col("macro_minute_index") == minute) & (pl.col("macro_trend_state") == trend))
            for col, label, color in [("candle_high_time", "High", "#08519c"), ("candle_low_time", "Low", "#cb181d")]:
                values = np.sort(sub.select(col).to_series().to_numpy().astype(float))
                if len(values):
                    y = np.arange(1, len(values) + 1) / len(values)
                    ax.step(values, y, where="post", label=label, color=color, linewidth=1.8)
            ax.set_title(f"15:{minute:02d} {trend} (n={sub.height})")
            ax.set_xlim(0, 59)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Second in minute")
            ax.set_ylabel("Cumulative share")
            ax.grid(True, alpha=0.25)
            if r == 0 and c == 0:
                ax.legend()
    fig.suptitle(f"{dataset_name}: ECDF of first-touch timing")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_violin(df: pl.DataFrame, dataset_name: str, out_path: Path) -> None:
    contexts = []
    data = []
    colors = []
    for trend in ["bullish", "bearish"]:
        for minute in EXPECTED_KEY_MINUTES:
            sub = df.filter((pl.col("macro_trend_state") == trend) & (pl.col("macro_minute_index") == minute))
            for col, extreme, color in [("candle_high_time", "H", "#08519c"), ("candle_low_time", "L", "#cb181d")]:
                values = sub.select(col).to_series().to_numpy().astype(float)
                if len(values):
                    contexts.append(f"{trend[:4]}\n15:{minute:02d}\n{extreme}")
                    data.append(values)
                    colors.append(color)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(data)), 6))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    ax.set_title(f"{dataset_name}: timing variance by candle/context")
    ax.set_ylabel("Second in minute")
    ax.set_ylim(0, 59)
    ax.set_xticks(np.arange(1, len(contexts) + 1))
    ax.set_xticklabels(contexts, rotation=0, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_high_low_scatter(df: pl.DataFrame, dataset_name: str, out_path: Path) -> None:
    trend_states = [s for s in ["bullish", "bearish"] if s in df.select("macro_trend_state").to_series().unique().to_list()]
    if not trend_states:
        return
    fig, axes = plt.subplots(len(EXPECTED_KEY_MINUTES), len(trend_states), figsize=(5 * len(trend_states), 3.8 * len(EXPECTED_KEY_MINUTES)), squeeze=False)
    for r, minute in enumerate(EXPECTED_KEY_MINUTES):
        for c, trend in enumerate(trend_states):
            ax = axes[r][c]
            sub = df.filter((pl.col("macro_minute_index") == minute) & (pl.col("macro_trend_state") == trend))
            ax.scatter(
                sub.select("candle_low_time").to_series().to_numpy(),
                sub.select("candle_high_time").to_series().to_numpy(),
                s=14,
                alpha=0.38,
                color="#525252",
            )
            ax.plot([0, 59], [0, 59], color="#de2d26", linewidth=1.0, linestyle="--")
            ax.set_title(f"15:{minute:02d} {trend} (n={sub.height})")
            ax.set_xlim(0, 59)
            ax.set_ylim(0, 59)
            ax.set_xlabel("Low second")
            ax.set_ylabel("High second")
            ax.grid(True, alpha=0.22)
    fig.suptitle(f"{dataset_name}: high/low formation relationship")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_gap_distribution(df: pl.DataFrame, dataset_name: str, out_path: Path) -> None:
    trend_states = [s for s in ["bullish", "bearish"] if s in df.select("macro_trend_state").to_series().unique().to_list()]
    if not trend_states:
        return
    fig, axes = plt.subplots(len(EXPECTED_KEY_MINUTES), len(trend_states), figsize=(5 * len(trend_states), 3.0 * len(EXPECTED_KEY_MINUTES)), squeeze=False)
    bins = np.arange(0, 62, 2)
    for r, minute in enumerate(EXPECTED_KEY_MINUTES):
        for c, trend in enumerate(trend_states):
            ax = axes[r][c]
            sub = df.filter((pl.col("macro_minute_index") == minute) & (pl.col("macro_trend_state") == trend))
            values = sub.select("candle_extreme_gap_seconds").to_series().to_numpy()
            ax.hist(values, bins=bins, color="#756bb1", alpha=0.72)
            ax.set_title(f"15:{minute:02d} {trend} gap (n={sub.height})")
            ax.set_xlim(0, 59)
            ax.set_xlabel("Seconds between high/low")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.22)
    fig.suptitle(f"{dataset_name}: high-low gap distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _write_summary_files(summary: MacroExtremeTimingVizSummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.frequency.write_csv(out_dir / f"{summary.dataset_name}_frequency.csv")
    summary.quantiles.write_csv(out_dir / f"{summary.dataset_name}_quantiles.csv")
    raw = pl.read_parquet(summary.path)
    _directional_extreme_stats(raw).write_csv(out_dir / f"{summary.dataset_name}_directional_extreme_stats.csv")
    _plot_histograms(raw, summary.dataset_name, out_dir / f"{summary.dataset_name}_histograms.png")
    _plot_heatmap(summary.frequency, summary.dataset_name, out_dir / f"{summary.dataset_name}_heatmap.png")
    _plot_ecdf(raw, summary.dataset_name, out_dir / f"{summary.dataset_name}_ecdf.png")
    _plot_violin(raw, summary.dataset_name, out_dir / f"{summary.dataset_name}_violin.png")
    _plot_high_low_scatter(raw, summary.dataset_name, out_dir / f"{summary.dataset_name}_high_low_scatter.png")
    _plot_gap_distribution(raw, summary.dataset_name, out_dir / f"{summary.dataset_name}_extreme_gap_distribution.png")


def process_dataset(path: str | Path, out_dir: str | Path = OUT_DIR) -> MacroExtremeTimingVizSummary:
    summary = summarize_macro_extreme_timing_dataset(path)
    _write_summary_files(summary, Path(out_dir))
    return summary


def main(paths: Iterable[str | Path] = (DATASET_PATH,), out_dir: str | Path = OUT_DIR) -> None:
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rollup_rows = []
    for path in paths:
        summary = process_dataset(path, output_root)
        rollup_rows.append(
            {
                "dataset": summary.dataset_name,
                "path": str(summary.path),
                "total_days": summary.total_days,
                "total_observations": summary.total_observations,
            }
        )
        print(f"[OK] {summary.dataset_name}: {summary.total_days} days, outputs -> {output_root}")
    pl.DataFrame(rollup_rows).write_csv(output_root / "macro_extreme_timing_rollup.csv")


if __name__ == "__main__":
    main()
