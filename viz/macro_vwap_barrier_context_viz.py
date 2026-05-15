from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

CONTEXT_INPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context.parquet")
SUMMARY_INPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context_summary.parquet")
OUTPUT_DIR = Path("outputs/figs/macro_vwap_barrier_context")


def _write_csv(df: pl.DataFrame, path: Path, wrote: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    wrote.append(path)


def _save_hist(values: list[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=min(30, max(1, len(values))), color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("points")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_ecdf(values: list[float], path: Path, title: str) -> None:
    sorted_values = sorted(values)
    y = [(idx + 1) / len(sorted_values) for idx in range(len(sorted_values))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted_values, y, color="#f58518")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("ECDF")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_scatter(df: pl.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    small = df.select(x, y).drop_nulls()
    fig, ax = plt.subplots(figsize=(7, 5))
    if small.height:
        ax.scatter(small[x].to_list(), small[y].to_list(), s=18, alpha=0.65)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_violin(df: pl.DataFrame, group_col: str, value_col: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if group_col in df.columns and value_col in df.columns:
        groups = df.select(group_col).drop_nulls().unique().sort(group_col).to_series().to_list()
        data = [df.filter(pl.col(group_col) == group).select(value_col).drop_nulls().to_series().to_list() for group in groups]
        data = [values for values in data if values]
        labels = [group for group in groups if df.filter(pl.col(group_col) == group).select(value_col).drop_nulls().height]
        if data:
            ax.violinplot(data, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1), labels, rotation=25, ha="right")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _wrong_side_bucket_expr() -> pl.Expr:
    value = pl.when(pl.col("vwap_dist_at_1550_close_points").is_not_null()).then((-pl.col("vwap_dist_at_1550_close_points")).clip(0.0)).otherwise(None)
    return (
        pl.when(value.is_null() | (value <= 0)).then(pl.lit("no_wrong_side_close"))
        .when(value <= 0.25).then(pl.lit("wrong_le_1tick"))
        .when(value <= 2.0).then(pl.lit("wrong_1tick_to_2pts"))
        .when(value <= 5.0).then(pl.lit("wrong_2_to_5pts"))
        .otherwise(pl.lit("wrong_gt_5pts"))
    )


def _save_heatmap(df: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    if {"barrier_first10", "vwap_10s_constructive"} <= set(df.columns):
        counts = (
            df.group_by("barrier_first10", "vwap_10s_constructive")
            .agg(pl.len().alias("n"))
            .sort("barrier_first10", "vwap_10s_constructive")
        )
        x_labels = [str(x) for x in counts.select("vwap_10s_constructive").unique().sort("vwap_10s_constructive").to_series().to_list()]
        y_labels = [str(x) for x in counts.select("barrier_first10").unique().sort("barrier_first10").to_series().to_list()]
        matrix = [[0 for _ in x_labels] for _ in y_labels]
        for row in counts.iter_rows(named=True):
            yi = y_labels.index(str(row["barrier_first10"]))
            xi = x_labels.index(str(row["vwap_10s_constructive"]))
            matrix[yi][xi] = row["n"]
        im = ax.imshow(matrix, cmap="Blues")
        for yi, values in enumerate(matrix):
            for xi, value in enumerate(values):
                ax.text(xi, yi, str(value), ha="center", va="center", color="black")
        ax.set_xticks(range(len(x_labels)), x_labels, rotation=25, ha="right")
        ax.set_yticks(range(len(y_labels)), y_labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Barrier first10 by VWAP 10s context")
    ax.set_xlabel("vwap_10s_constructive")
    ax.set_ylabel("barrier_first10")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def process_dataset(
    context_path: str | Path = CONTEXT_INPUT_PATH,
    summary_path: str | Path = SUMMARY_INPUT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> list[Path]:
    context = pl.read_parquet(context_path)
    summary = pl.read_parquet(summary_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wrote: list[Path] = []

    _write_csv(summary.sort("scope", "target_name", "bucket"), out / "summary_by_scope.csv", wrote)
    _write_csv(
        context.select(
            pl.col("worst_wrong_side_dist_points").quantile(0.10).alias("p10_worst_wrong_side_dist_points"),
            pl.col("worst_wrong_side_dist_points").quantile(0.25).alias("p25_worst_wrong_side_dist_points"),
            pl.col("worst_wrong_side_dist_points").median().alias("median_worst_wrong_side_dist_points"),
            pl.col("worst_wrong_side_dist_points").quantile(0.75).alias("p75_worst_wrong_side_dist_points"),
            pl.col("worst_wrong_side_dist_points").quantile(0.90).alias("p90_worst_wrong_side_dist_points"),
            pl.col("wrong_side_share_1550").quantile(0.10).alias("p10_wrong_side_share_1550"),
            pl.col("wrong_side_share_1550").median().alias("median_wrong_side_share_1550"),
            pl.col("wrong_side_share_1550").quantile(0.90).alias("p90_wrong_side_share_1550"),
        ),
        out / "wrong_side_quantiles.csv",
        wrote,
    )
    _write_csv(
        context.group_by("vwap_context_10s_to_1555").agg(
            pl.len().alias("sample_size"),
            pl.col("target_1555_1559_points").mean().alias("avg_target_1555_1559_points"),
            pl.col("target_1555_1559_points").median().alias("median_target_1555_1559_points"),
        ).sort("vwap_context_10s_to_1555"),
        out / "vwap_1555_decision_summary.csv",
        wrote,
    )
    _write_csv(
        summary.filter(pl.col("scope").is_in(["wrong_side_close_bucket", "vwap_1555_decision", "barrier_1555_context"])),
        out / "target_quantiles_by_bucket.csv",
        wrote,
    )

    for col, filename, title in [
        ("vwap_10s_dist_points", "vwap_10s_dist_hist.png", "VWAP distance at 15:50:10"),
        ("vwap_1555_dist_points", "vwap_1555_dist_hist.png", "VWAP distance at 15:55"),
    ]:
        values = context.select(col).drop_nulls().to_series().to_list()
        if values:
            path = out / filename
            _save_hist(values, path, title)
            wrote.append(path)
    for col, filename, title in [
        ("worst_wrong_side_dist_points", "wrong_side_dist_ecdf.png", "Worst wrong-side VWAP distance ECDF"),
        ("wrong_side_share_1550", "wrong_side_share_ecdf.png", "Wrong-side VWAP share ECDF"),
    ]:
        values = context.select(col).drop_nulls().to_series().to_list()
        if values:
            path = out / filename
            _save_ecdf(values, path, title)
            wrote.append(path)

    violin_context = context
    if "vwap_dist_at_1550_close_points" in violin_context.columns:
        violin_context = violin_context.with_columns(_wrong_side_bucket_expr().alias("wrong_side_close_bucket"))
    elif "wrong_side_close_bucket" not in violin_context.columns:
        violin_context = violin_context.with_columns(pl.lit("all").alias("wrong_side_close_bucket"))
    for group_col, value_col, filename, title in [
        ("wrong_side_close_bucket", "target_1550_10s_1559_points", "target_by_wrong_side_bucket_violin.png", "Target by wrong-side close bucket"),
        ("vwap_context_10s_to_1555", "target_1555_1559_points", "target_by_1555_context_violin.png", "Target by 15:55 VWAP context"),
    ]:
        path = out / filename
        _save_violin(violin_context, group_col, value_col, path, title)
        wrote.append(path)

    heatmap_path = out / "barrier_vwap_heatmap.png"
    _save_heatmap(context, heatmap_path)
    wrote.append(heatmap_path)

    scatter_path = out / "vwap_1555_scatter_target.png"
    _save_scatter(context, "vwap_1555_dist_points", "target_1555_1559_points", scatter_path, "15:55 VWAP distance vs 15:55-15:59 target")
    wrote.append(scatter_path)
    return wrote


def main() -> None:
    wrote = process_dataset()
    for path in wrote:
        print(f"[OK] Wrote macro VWAP barrier context viz -> {path}")


if __name__ == "__main__":
    main()
