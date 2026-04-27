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


def _write_summary_files(summary: MacroExtremeTimingVizSummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.frequency.write_csv(out_dir / f"{summary.dataset_name}_frequency.csv")
    summary.quantiles.write_csv(out_dir / f"{summary.dataset_name}_quantiles.csv")
    raw = pl.read_parquet(summary.path)
    _plot_histograms(raw, summary.dataset_name, out_dir / f"{summary.dataset_name}_histograms.png")
    _plot_heatmap(summary.frequency, summary.dataset_name, out_dir / f"{summary.dataset_name}_heatmap.png")


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
