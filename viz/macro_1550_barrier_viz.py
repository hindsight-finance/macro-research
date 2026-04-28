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
    from features.macro_1550_barrier import summarize_macro_1550_barrier_study
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from features.macro_1550_barrier import summarize_macro_1550_barrier_study

DATASET_PATH = Path("outputs/nq_macro_1550_barrier.parquet")
OUT_DIR = Path("outputs/figs/macro_1550_barrier")


@dataclass(frozen=True)
class BarrierVizSummary:
    path: Path
    dataset_name: str
    summary: pl.DataFrame
    edge_cases: pl.DataFrame
    total_days: int


def summarize_barrier_dataset(path: str | Path) -> BarrierVizSummary:
    dataset_path = Path(path)
    df = pl.read_parquet(dataset_path)
    required = {"date", "bullish_edge_case", "macro_low_minute_index", "macro_low_after_1550_points"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return BarrierVizSummary(
        path=dataset_path,
        dataset_name=dataset_path.stem,
        summary=summarize_macro_1550_barrier_study(df),
        edge_cases=df.filter(pl.col("bullish_edge_case")).sort("date"),
        total_days=df.height,
    )


def _plot_rates(summary: pl.DataFrame, dataset_name: str, out_path: Path) -> None:
    if summary.is_empty():
        return
    overall = summary.filter(pl.col("scope") == "bullish_macro")
    if overall.is_empty():
        return
    row = overall.row(0, named=True)
    labels = ["15:50 low\nfirst 10s", "15:50 low\nmacro low", "barrier\nholds", "edge\ncase"]
    values = [
        row["low_1550_first10_pct"],
        row["low_1550_is_macro_low_pct"],
        row["barrier_holds_pct"],
        row["edge_case_pct"],
    ]
    colors = ["#6baed6", "#3182bd", "#31a354", "#de2d26"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.78)
    ax.set_title(f"{dataset_name}: bullish 15:50 low barrier")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        if value is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_edge_depth(edge_cases: pl.DataFrame, dataset_name: str, out_path: Path) -> None:
    if edge_cases.is_empty():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No edge cases", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    values = edge_cases.select("macro_low_after_1550_points").to_series().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(30, max(5, len(values) // 4)), color="#de2d26", alpha=0.72)
    ax.set_title(f"{dataset_name}: edge-case depth below 15:50 low")
    ax.set_xlabel("Points below 15:50 low")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary_files(summary: BarrierVizSummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.summary.write_csv(out_dir / f"{summary.dataset_name}_summary.csv")
    summary.edge_cases.write_csv(out_dir / f"{summary.dataset_name}_edge_cases.csv")
    _plot_rates(summary.summary, summary.dataset_name, out_dir / f"{summary.dataset_name}_rates.png")
    _plot_edge_depth(summary.edge_cases, summary.dataset_name, out_dir / f"{summary.dataset_name}_edge_depth.png")


def process_dataset(path: str | Path, out_dir: str | Path = OUT_DIR) -> BarrierVizSummary:
    summary = summarize_barrier_dataset(path)
    _write_summary_files(summary, Path(out_dir))
    return summary


def main(paths: Iterable[str | Path] = (DATASET_PATH,), out_dir: str | Path = OUT_DIR) -> None:
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        summary = process_dataset(path, output_root)
        rows.append({"dataset": summary.dataset_name, "path": str(summary.path), "total_days": summary.total_days})
        print(f"[OK] {summary.dataset_name}: {summary.total_days} days, outputs -> {output_root}")
    pl.DataFrame(rows).write_csv(output_root / "macro_1550_barrier_rollup.csv")


if __name__ == "__main__":
    main()
