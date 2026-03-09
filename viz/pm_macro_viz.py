import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===========================================================
# Hardcoded path to interaction parquet
# Adjust relative/absolute path as needed
# ===========================================================
INTERACTION_PATH = Path("../outputs/pm_macro_interaction.parquet")
OUT_DIR = INTERACTION_PATH.parent / "figs"
OUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INTERACTION_PATH)

    # macro move as % of price (use macro_open as reference level)
    df["macro_range_pct_of_price"] = (df["macro_range_points"] / df["macro_open"]) * 100.0

    # --- inclusive high/low time over 3pm + macro window (15:00–16:00) ---

    # minutes from 15:00 for each component already in the parquet
    # threepm_high_time_min_from_15
    # threepm_low_time_min_from_15
    # macro_high_time_min_from_15
    # macro_low_time_min_from_15

    # inclusive HIGH time
    inclusive_high = np.where(
        df["macro_high"] > df["hr3_high"],
        df["macro_high_time_min_from_15"],
        df["threepm_high_time_min_from_15"],
    )

    # handle exact equality: pick earlier of the two
    eq_high_mask = df["macro_high"] == df["hr3_high"]
    if eq_high_mask.any():
        earlier_high = df.loc[eq_high_mask, ["macro_high_time_min_from_15",
                                             "threepm_high_time_min_from_15"]].min(axis=1)
        inclusive_high[eq_high_mask.to_numpy()] = earlier_high.to_numpy()

    df["inclusive_high_time_from_15"] = inclusive_high

    # inclusive LOW time
    inclusive_low = np.where(
        df["macro_low"] < df["hr3_low"],
        df["macro_low_time_min_from_15"],
        df["threepm_low_time_min_from_15"],
    )

    eq_low_mask = df["macro_low"] == df["hr3_low"]
    if eq_low_mask.any():
        earlier_low = df.loc[eq_low_mask, ["macro_low_time_min_from_15",
                                           "threepm_low_time_min_from_15"]].min(axis=1)
        inclusive_low[eq_low_mask.to_numpy()] = earlier_low.to_numpy()

    df["inclusive_low_time_from_15"] = inclusive_low

    # 5m buckets for inclusive highs/lows
    df["inclusive_high_bucket_5m"] = (df["inclusive_high_time_from_15"] // 5) * 5
    df["inclusive_low_bucket_5m"] = (df["inclusive_low_time_from_15"] // 5) * 5

    return df



# -----------------------------------------------------------
# Histograms: 3pm high/low formation in 5m buckets
# -----------------------------------------------------------
def plot_bucket_hist(df: pd.DataFrame, col: str, title: str, outfile: Path):
    counts = df[col].value_counts().sort_index()
    if counts.empty:
        print(f"{title}: no data, skipping")
        return

    # bucket starts (0,5,10,...) and % of days in each bucket
    x = counts.index.astype(int)
    y = counts.values / counts.values.sum() * 100.0

    fig, ax = plt.subplots()
    ax.bar(x, y, width=5, align="edge")
    ax.set_xlabel("Minutes from 15:00 (bucket start)")
    ax.set_ylabel("% of days")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    # nicer x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:02d}-{v+5:02d}" for v in x], rotation=45)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved: {outfile}")


def make_histograms(df: pd.DataFrame):
    bull = df[df["macro_dir_sign"] > 0]
    bear = df[df["macro_dir_sign"] < 0]

    # Bullish macro: inclusive high / low formation
    plot_bucket_hist(
        bull,
        "inclusive_high_bucket_5m",
        "Bullish macro – HIGH of 3pm+macro window (5m buckets)",
        OUT_DIR / "pm_macro_bull_high_inclusive_hist.png",
    )
    plot_bucket_hist(
        bull,
        "inclusive_low_bucket_5m",
        "Bullish macro – LOW of 3pm+macro window (5m buckets)",
        OUT_DIR / "pm_macro_bull_low_inclusive_hist.png",
    )

    # Bearish macro: inclusive high / low formation
    plot_bucket_hist(
        bear,
        "inclusive_high_bucket_5m",
        "Bearish macro – HIGH of 3pm+macro window (5m buckets)",
        OUT_DIR / "pm_macro_bear_high_inclusive_hist.png",
    )
    plot_bucket_hist(
        bear,
        "inclusive_low_bucket_5m",
        "Bearish macro – LOW of 3pm+macro window (5m buckets)",
        OUT_DIR / "pm_macro_bear_low_inclusive_hist.png",
    )



# -----------------------------------------------------------
# Breakout stats vs 3pm range
# -----------------------------------------------------------
def breakout_stats(df: pd.DataFrame):
    n = len(df)

    took_high = df["macro_took_threepm_high"]
    took_low = df["macro_took_threepm_low"]
    took_both = df["macro_took_threepm_both"]
    took_any = took_high | took_low
    took_none = ~took_any
    took_one_side_only = took_any & ~took_both

    def pct(x: int) -> float:
        return (x / n) * 100.0 if n > 0 else np.nan

    print("\n=== Macro breakout vs 3pm range ===")
    print(f"Total days: {n}")
    print(f"Macro broke at least one side of 3pm range: {pct(took_any.sum()):5.2f}%")
    print(f"  – only one side:                        {pct(took_one_side_only.sum()):5.2f}%")
    print(f"  – both sides:                           {pct(took_both.sum()):5.2f}%")
    print(f"Macro did NOT break 3pm range:            {pct(took_none.sum()):5.2f}%")

    # Average macro % move by group
    groups = {
        "any_break": df[took_any],
        "both_sides": df[took_both],
        "no_break": df[took_none],
        "one_side_only": df[took_one_side_only],
    }

    print("\nAverage macro move as % of price (using macro_open):")
    for name, g in groups.items():
        if len(g) == 0:
            avg_move = np.nan
        else:
            avg_move = g["macro_range_pct_of_price"].mean()
        print(f"  {name:14s}: {avg_move:5.3f} %  (n={len(g)})")


# -----------------------------------------------------------
# Directional continuation stats: macro vs PM
# -----------------------------------------------------------
def direction_stats(df: pd.DataFrame):
    pm_sign = df["pm_dir_sign"]
    macro_sign = df["macro_dir_sign"]

    # Cases where both have a clear non-zero sign
    mask_valid = (pm_sign != 0) & (macro_sign != 0)
    valid = df[mask_valid]
    n_valid = len(valid)

    cont_mask = mask_valid & (pm_sign == macro_sign)
    against_mask = mask_valid & (pm_sign == -macro_sign)

    # anything else in mask_valid but not cont/against is weird, but we can treat as "other"
    other_mask = mask_valid & ~(cont_mask | against_mask)

    def pct(x: int, denom: int) -> float:
        return (x / denom) * 100.0 if denom > 0 else np.nan

    print("\n=== Macro vs PM direction ===")
    print(f"Days with both PM and Macro having non-zero direction: {n_valid}")
    print(f"  Macro continued PM direction: {pct(cont_mask.sum(), n_valid):5.2f}%")
    print(f"  Macro went against PM dir:    {pct(against_mask.sum(), n_valid):5.2f}%")
    print(f"  Other (odd cases):            {pct(other_mask.sum(), n_valid):5.2f}%")

    # You can also look at average macro move in each case:
    print("\nAverage macro % move by directional relationship:")
    for label, m in [
        ("continue", cont_mask),
        ("against", against_mask),
        ("other", other_mask),
    ]:
        g = df[m]
        if len(g) == 0:
            avg_move = np.nan
        else:
            avg_move = g["macro_range_pct_of_price"].mean()
        print(f"  {label:8s}: {avg_move:5.3f} %  (n={len(g)})")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows from {INTERACTION_PATH}")

    make_histograms(df)
    breakout_stats(df)
    direction_stats(df)


if __name__ == "__main__":
    main()
