import pandas as pd
import numpy as np

# =====================================================================
# HARD-CODED INPUTS / OUTPUT
# =====================================================================
PM_PATH = "../outputs/pm_3pm.parquet"
MACRO_PATH = "../outputs/macro_outcomes.parquet"
OUT_PATH = "../outputs/pm_macro_interaction.parquet"
# =====================================================================


def build_pm_macro_interaction(print_summary: bool = True):
    """
    Build daily features describing interaction between 3pm hour (hr3)
    and macro close session, using:
        - pm_hr3.parquet
        - macro_outcomes.parquet

    Outputs:
      - 3pm range macro exclusive (percent of 3pm+macro combined range)
      - 3pm range macro inclusive (percent)
      - macro range (percent of 3pm+macro combined range)
      - 3pm dir, pm dir, macro dir
      - flags for macro taking 3pm high / low
      - 5m histograms of 3pm high/low formation by macro direction
    """

    pm = pd.read_parquet(PM_PATH)
    macro = pd.read_parquet(MACRO_PATH)
    df = pm.merge(macro, on="date", how="inner", suffixes=("", "_macro"))

    # -----------------------------------------------------------------
    # Combined 3pm (15:00–15:49) + macro (15:50–16:00) range
    # -----------------------------------------------------------------
    combined_low = df[["hr3_low", "macro_low"]].min(axis=1)
    combined_high = df[["hr3_high", "macro_high"]].max(axis=1)
    combined_range = combined_high - combined_low
    safe_den = combined_range.replace(0, np.nan)

    df["threepm_macro_combined_range_points"] = combined_range

    # 3pm range macro exclusive (% of combined)
    df["threepm_range_macro_excl_pct"] = (df["hr3_range"] / safe_den) * 100

    # 3pm range macro inclusive (100% by definition)
    df["threepm_range_macro_incl_pct"] = np.where(
        combined_range.notna() & (combined_range != 0), 100.0, np.nan
    )

    # Macro range (% of combined 3pm+macro)
    df["macro_range_pct_of_threepm_macro"] = (
        df["macro_range_points"] / safe_den
    ) * 100

    # -----------------------------------------------------------------
    # Directions
    # -----------------------------------------------------------------
    df["pm_dir_points"] = df["pm_dir"]
    df["threepm_dir_points"] = df["hr3_dir"]
    df["macro_dir_points"] = df["macro_dir_points"]

    # use 0 for missing direction so we can store as int8
    df["pm_dir_sign"] = (
        np.sign(df["pm_dir_points"])
        .fillna(0)
        .astype("int8")
    )

    df["threepm_dir_sign"] = (
        np.sign(df["threepm_dir_points"])
        .fillna(0)
        .astype("int8")
    )

    df["macro_dir_sign"] = (
        np.sign(df["macro_dir_points"])
        .fillna(0)
        .astype("int8")
    )


    # -----------------------------------------------------------------
    # Macro taking 3pm high/low
    # -----------------------------------------------------------------
    df["macro_took_threepm_high"] = df["macro_high"] > df["hr3_high"]
    df["macro_took_threepm_low"] = df["macro_low"] < df["hr3_low"]
    df["macro_took_threepm_both"] = (
        df["macro_took_threepm_high"] & df["macro_took_threepm_low"]
    )

    # -----------------------------------------------------------------
    # Timing and 5m buckets
    # -----------------------------------------------------------------
    df["threepm_high_time_min_from_15"] = df["hr3_high_time"]
    df["threepm_low_time_min_from_15"] = df["hr3_low_time"]
    df["macro_high_time_min_from_15"] = df["macro_high_time"] + 50
    df["macro_low_time_min_from_15"] = df["macro_low_time"] + 50

    df["threepm_high_bucket_5m"] = (df["threepm_high_time_min_from_15"] // 5) * 5
    df["threepm_low_bucket_5m"] = (df["threepm_low_time_min_from_15"] // 5) * 5

    # -----------------------------------------------------------------
    # Save and summary
    # -----------------------------------------------------------------
    df.to_parquet(OUT_PATH, index=False)
    print(f"✅ Saved pm–macro interaction features to: {OUT_PATH}")
    print(f"Rows: {len(df)}")

    if print_summary:
        print_summary_stats(df)


def print_summary_stats(df: pd.DataFrame):
    """Print 5-minute histograms of 3pm high/low times conditioned on macro direction."""
    bull = df[df["macro_dir_sign"] > 0]
    bear = df[df["macro_dir_sign"] < 0]

    def _print_hist(label, series):
        counts = series.value_counts().sort_index()
        if counts.empty:
            print(f"{label}: (no data)")
            return

        print(f"\n{label}:")
        for bucket, cnt in counts.items():
            # bucket may be float (e.g. 0.0, 5.0), so cast to int for pretty printing
            b = int(bucket)
            print(f"  {b:02d}–{b+5:02d}  -> {cnt}")


    # Bullish macro
    _print_hist("Bullish macro – when 3pm HIGH forms", bull["threepm_high_bucket_5m"])
    _print_hist("Bullish macro – when 3pm LOW forms", bull["threepm_low_bucket_5m"])

    # Bearish macro
    _print_hist("Bearish macro – when 3pm HIGH forms", bear["threepm_high_bucket_5m"])
    _print_hist("Bearish macro – when 3pm LOW forms", bear["threepm_low_bucket_5m"])

    # Mean range shares
    print("\nMean 3pm/macro range shares (% of combined 3pm+macro range):")
    print(
        df[["threepm_range_macro_excl_pct", "macro_range_pct_of_threepm_macro"]]
        .mean()
        .round(2)
    )


if __name__ == "__main__":
    build_pm_macro_interaction(print_summary=True)
