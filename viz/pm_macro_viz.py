import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

INTERACTION_PATH = Path("../outputs/pm_macro_interaction.parquet")
OUT_DIR = INTERACTION_PATH.parent / "figs"
OUT_DIR.mkdir(exist_ok=True)


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(INTERACTION_PATH).with_columns(
        macro_range_pct_of_price=(pl.col("macro_range_points") / pl.col("macro_open")) * 100.0,
    )
    return df.with_columns(
        inclusive_high_time_from_15=pl.when(pl.col("macro_high") > pl.col("hr3_high"))
        .then(pl.col("macro_high_time_min_from_15"))
        .when(pl.col("macro_high") == pl.col("hr3_high"))
        .then(pl.min_horizontal("macro_high_time_min_from_15", "threepm_high_time_min_from_15"))
        .otherwise(pl.col("threepm_high_time_min_from_15")),
        inclusive_low_time_from_15=pl.when(pl.col("macro_low") < pl.col("hr3_low"))
        .then(pl.col("macro_low_time_min_from_15"))
        .when(pl.col("macro_low") == pl.col("hr3_low"))
        .then(pl.min_horizontal("macro_low_time_min_from_15", "threepm_low_time_min_from_15"))
        .otherwise(pl.col("threepm_low_time_min_from_15")),
    ).with_columns(
        inclusive_high_bucket_5m=(pl.col("inclusive_high_time_from_15") // 5) * 5,
        inclusive_low_bucket_5m=(pl.col("inclusive_low_time_from_15") // 5) * 5,
    )


def plot_bucket_hist(df: pl.DataFrame, col: str, title: str, outfile: Path):
    counts = df.group_by(col).len().drop_nulls(col).sort(col)
    if counts.is_empty():
        print(f"{title}: no data, skipping")
        return
    x = counts[col].cast(pl.Int64).to_numpy()
    y = counts["len"].to_numpy() / counts["len"].sum() * 100.0
    fig, ax = plt.subplots()
    ax.bar(x, y, width=5, align="edge")
    ax.set_xlabel("Minutes from 15:00 (bucket start)")
    ax.set_ylabel("% of days")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:02d}-{v+5:02d}" for v in x], rotation=45)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved: {outfile}")


def make_histograms(df: pl.DataFrame):
    bull = df.filter(pl.col("macro_dir_sign") > 0)
    bear = df.filter(pl.col("macro_dir_sign") < 0)
    plot_bucket_hist(bull, "inclusive_high_bucket_5m", "Bullish macro – HIGH of 3pm+macro window (5m buckets)", OUT_DIR / "pm_macro_bull_high_inclusive_hist.png")
    plot_bucket_hist(bull, "inclusive_low_bucket_5m", "Bullish macro – LOW of 3pm+macro window (5m buckets)", OUT_DIR / "pm_macro_bull_low_inclusive_hist.png")
    plot_bucket_hist(bear, "inclusive_high_bucket_5m", "Bearish macro – HIGH of 3pm+macro window (5m buckets)", OUT_DIR / "pm_macro_bear_high_inclusive_hist.png")
    plot_bucket_hist(bear, "inclusive_low_bucket_5m", "Bearish macro – LOW of 3pm+macro window (5m buckets)", OUT_DIR / "pm_macro_bear_low_inclusive_hist.png")


def breakout_stats(df: pl.DataFrame):
    n = df.height
    took_high = pl.col("macro_took_threepm_high")
    took_low = pl.col("macro_took_threepm_low")
    metrics = df.select(
        took_any=(took_high | took_low).sum(),
        took_both=pl.col("macro_took_threepm_both").sum(),
        no_break=(~(took_high | took_low)).sum(),
        one_side_only=((took_high | took_low) & ~pl.col("macro_took_threepm_both")).sum(),
    ).row(0, named=True)
    pct = lambda x: (x / n) * 100.0 if n > 0 else np.nan
    print("\n=== Macro breakout vs 3pm range ===")
    print(f"Total days: {n}")
    print(f"Macro broke at least one side of 3pm range: {pct(metrics['took_any']):5.2f}%")
    print(f"  – only one side:                        {pct(metrics['one_side_only']):5.2f}%")
    print(f"  – both sides:                           {pct(metrics['took_both']):5.2f}%")
    print(f"Macro did NOT break 3pm range:            {pct(metrics['no_break']):5.2f}%")


def direction_stats(df: pl.DataFrame):
    valid = df.filter((pl.col("pm_dir_sign") != 0) & (pl.col("macro_dir_sign") != 0))
    cont = valid.filter(pl.col("pm_dir_sign") == pl.col("macro_dir_sign"))
    against = valid.filter(pl.col("pm_dir_sign") == -pl.col("macro_dir_sign"))
    n_valid = valid.height
    pct = lambda x, denom: (x / denom) * 100.0 if denom > 0 else np.nan
    print("\n=== Macro vs PM direction ===")
    print(f"Days with both PM and Macro having non-zero direction: {n_valid}")
    print(f"  Macro continued PM direction: {pct(cont.height, n_valid):5.2f}%")
    print(f"  Macro went against PM dir:    {pct(against.height, n_valid):5.2f}%")


def main():
    df = load_data()
    print(f"Loaded {df.height} rows from {INTERACTION_PATH}")
    make_histograms(df)
    breakout_stats(df)
    direction_stats(df)


if __name__ == "__main__":
    main()
