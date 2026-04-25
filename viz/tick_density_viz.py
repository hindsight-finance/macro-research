from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

DATASET_PATHS = [
    Path("outputs/nq_macro_tick_density.parquet"),
    Path("outputs/nq_macro_tick_density_1550_5s.parquet"),
    Path("outputs/nq_macro_tick_density_1554_5s.parquet"),
    Path("outputs/nq_macro_tick_density_1555_5s.parquet"),
    Path("outputs/nq_macro_tick_density_1559_5s.parquet"),
]
OUT_DIR = Path("outputs/figs/tick_density")
NORMALITY_ALPHA = 0.05


@dataclass(frozen=True)
class TickDensitySummary:
    path: Path
    dataset_name: str
    index_column: str
    band_stats: pd.DataFrame
    normality: pd.DataFrame
    overall_normality: str
    total_observations: int
    total_days: int


def _detect_index_column(df: pd.DataFrame) -> str:
    if "macro_minute_index" in df.columns:
        return "macro_minute_index"
    if "bucket_index" in df.columns:
        return "bucket_index"
    raise ValueError("Expected macro_minute_index or bucket_index column")


def _expected_index_values(index_column: str) -> list[int]:
    if index_column == "macro_minute_index":
        return [*range(40, 60), *range(0, 11)]
    if index_column == "bucket_index":
        return list(range(12))
    raise ValueError(f"Unsupported index column: {index_column}")


def _classify_overall_normality(normality: pd.DataFrame) -> str:
    known = normality["is_normal"].dropna()
    if known.empty:
        return "insufficient_data"
    truthy = known.astype(bool)
    if truthy.all():
        return "all_normal"
    if (~truthy).all():
        return "all_non_normal"
    return "mixed"


def summarize_metric_by_index(df: pd.DataFrame, index_column: str, metric_column: str) -> pd.DataFrame:
    if metric_column not in df.columns:
        raise ValueError(f"Missing {metric_column} column")
    expected_index = _expected_index_values(index_column)
    return (
        df.groupby(index_column)[metric_column]
        .agg(
            **{
                f"mean_{metric_column}": "mean",
                f"p25_{metric_column}": lambda s: s.quantile(0.25),
                f"p75_{metric_column}": lambda s: s.quantile(0.75),
                f"median_{metric_column}": "median",
                f"std_{metric_column}": "std",
                f"sample_size_{metric_column}": "size",
            }
        )
        .reindex(expected_index)
        .reset_index()
    )


def summarize_tick_density_dataset(path: str | Path, alpha: float = NORMALITY_ALPHA) -> TickDensitySummary:
    dataset_path = Path(path)
    df = pd.read_parquet(dataset_path)
    if "tick_count" not in df.columns:
        raise ValueError(f"Missing tick_count column: {dataset_path}")
    if "date_utc" not in df.columns:
        raise ValueError(f"Missing date_utc column: {dataset_path}")

    index_column = _detect_index_column(df)
    work = df[["date_utc", index_column, "tick_count"]].copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"], format="mixed")
    work["tick_count"] = pd.to_numeric(work["tick_count"])
    band_stats = summarize_metric_by_index(work, index_column, "tick_count")

    rows: list[dict] = []
    grouped = {idx: group for idx, group in work.groupby(index_column, sort=True)}
    for idx in _expected_index_values(index_column):
        group = grouped.get(idx)
        if group is None:
            rows.append(
                {
                    index_column: idx,
                    "sample_size": 0,
                    "test_name": "missing_index",
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "alpha": alpha,
                    "is_normal": pd.NA,
                }
            )
            continue

        values = group["tick_count"].dropna().astype(float)
        n = len(values)
        if n < 3:
            rows.append(
                {
                    index_column: idx,
                    "sample_size": n,
                    "test_name": "insufficient_data",
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "alpha": alpha,
                    "is_normal": pd.NA,
                }
            )
            continue

        statistic, p_value = stats.shapiro(values.to_numpy())
        rows.append(
            {
                index_column: idx,
                "sample_size": n,
                "test_name": "shapiro",
                "statistic": statistic,
                "p_value": p_value,
                "alpha": alpha,
                "is_normal": bool(p_value >= alpha),
            }
        )

    normality = pd.DataFrame(rows)
    return TickDensitySummary(
        path=dataset_path,
        dataset_name=dataset_path.stem,
        index_column=index_column,
        band_stats=band_stats,
        normality=normality,
        overall_normality=_classify_overall_normality(normality),
        total_observations=len(work),
        total_days=work["date_utc"].nunique(),
    )



def _plot_x_positions_and_labels(index_values: pd.Series) -> tuple[list[int], list[str]]:
    labels = [str(int(value)) for value in index_values.tolist()]
    positions = list(range(len(labels)))
    return positions, labels

def _label_for_index_column(index_column: str) -> str:
    return "Macro minute index" if index_column == "macro_minute_index" else "5-second bucket index"


def _plot_metric_bands(
    stats_df: pd.DataFrame,
    index_column: str,
    metric_column: str,
    title: str,
    ylabel: str,
    out_path: Path,
    subtitle: str | None = None,
) -> None:
    x_positions, x_labels = _plot_x_positions_and_labels(stats_df[index_column])
    mean_col = f"mean_{metric_column}"
    p25_col = f"p25_{metric_column}"
    p75_col = f"p75_{metric_column}"

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(
        x_positions,
        stats_df[p25_col].to_numpy(dtype=float),
        stats_df[p75_col].to_numpy(dtype=float),
        color="#9ecae1",
        alpha=0.45,
        label="P25-P75 band",
    )
    ax.plot(x_positions, stats_df[mean_col].to_numpy(dtype=float), color="#08519c", linewidth=2.2, label="Mean")
    ax.plot(x_positions, stats_df[p25_col].to_numpy(dtype=float), color="#3182bd", linestyle="--", linewidth=1.4, label="P25")
    ax.plot(x_positions, stats_df[p75_col].to_numpy(dtype=float), color="#3182bd", linestyle="--", linewidth=1.4, label="P75")
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.set_xlabel(_label_for_index_column(index_column))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(min(x_positions), max(x_positions))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary_files(summary: TickDensitySummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / f"{summary.dataset_name}_hist.png"
    if hist_path.exists():
        hist_path.unlink()

    _plot_metric_bands(
        summary.band_stats,
        summary.index_column,
        "tick_count",
        f"{summary.dataset_name} cross-day tick-count bands",
        "Tick count",
        out_dir / f"{summary.dataset_name}_bands.png",
        subtitle=f"Overall normality: {summary.overall_normality}",
    )
    summary.band_stats.to_csv(out_dir / f"{summary.dataset_name}_band_stats.csv", index=False)
    summary.normality.to_csv(out_dir / f"{summary.dataset_name}_normality.csv", index=False)

    if summary.index_column == "macro_minute_index":
        raw_df = pd.read_parquet(summary.path)
        total_size_stats = summarize_metric_by_index(raw_df, summary.index_column, "total_size")
        _plot_metric_bands(
            total_size_stats,
            summary.index_column,
            "total_size",
            f"{summary.dataset_name} cross-day total-size bands",
            "Total size",
            out_dir / f"{summary.dataset_name}_total_size_bands.png",
        )
        total_size_stats.to_csv(out_dir / f"{summary.dataset_name}_total_size_band_stats.csv", index=False)


def process_dataset(path: str | Path, out_dir: str | Path = OUT_DIR, alpha: float = NORMALITY_ALPHA) -> TickDensitySummary:
    dataset_path = Path(path)
    summary = summarize_tick_density_dataset(dataset_path, alpha=alpha)
    _write_summary_files(summary, Path(out_dir))
    return summary


def main(paths: Iterable[str | Path] = DATASET_PATHS, out_dir: str | Path = OUT_DIR) -> None:
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rollup_rows: list[dict] = []
    for path in paths:
        summary = process_dataset(path, output_root)
        rollup_rows.append(
            {
                "dataset": summary.dataset_name,
                "path": str(summary.path),
                "index_column": summary.index_column,
                "total_days": summary.total_days,
                "total_observations": summary.total_observations,
                "overall_normality": summary.overall_normality,
                "normal_buckets": int((summary.normality["is_normal"] == True).sum()),
                "non_normal_buckets": int((summary.normality["is_normal"] == False).sum()),
                "insufficient_buckets": int(summary.normality["is_normal"].isna().sum()),
            }
        )
        print(
            f"[OK] {summary.dataset_name}: {summary.total_days} days, "
            f"{summary.overall_normality}, outputs -> {output_root}"
        )

    pd.DataFrame(rollup_rows).to_csv(output_root / "tick_density_normality_rollup.csv", index=False)


if __name__ == "__main__":
    main()
