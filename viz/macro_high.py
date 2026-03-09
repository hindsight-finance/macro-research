import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load data
# -------------------------
df = pd.read_parquet("../outputs/nq_macro_outcomes.parquet")

# macro_high_time / macro_low_time are already ints: minutes since t0
# Ensure numeric with NaNs handled
df["macro_high_time"] = pd.to_numeric(df["macro_high_time"], errors="coerce")
df["macro_low_time"]  = pd.to_numeric(df["macro_low_time"], errors="coerce")

# -------------------------
# Define macro direction
# -------------------------
# > 0  = bullish macro (close > open)
# < 0  = bearish macro (close < open)
# == 0 will be dropped
df = df[df["macro_dir_points"] != 0].copy()
df["macro_dir"] = np.where(df["macro_dir_points"] > 0, "bullish", "bearish")

bullish = df[df["macro_dir"] == "bullish"]
bearish = df[df["macro_dir"] == "bearish"]

# -------------------------
# Helper: plot histograms
# -------------------------
def plot_macro_time_hist(df_slice, low_col, high_col, title):
    # Filter valid values
    lows  = df_slice[low_col].dropna()
    highs = df_slice[high_col].dropna()

    # Choose bin edges to cover whole macro window, e.g. 0–10 min
    # Adjust if your macro window is a different length
    min_val = int(min(lows.min(), highs.min()))
    max_val = int(max(lows.max(), highs.max()))
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # 1-min bins, centered on ints

    plt.figure(figsize=(8, 4))
    plt.hist(lows,  bins=bins, alpha=0.6, edgecolor="black", label="Low time")
    plt.hist(highs, bins=bins, alpha=0.6, edgecolor="black", label="High time")

    plt.title(title)
    plt.xlabel("Minutes since macro open (t0)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(min_val, max_val + 1))
    plt.tight_layout()
    plt.show()

# Bullish: when do lows vs highs form?
plot_macro_time_hist(
    bullish,
    low_col="macro_low_time",
    high_col="macro_high_time",
    title="Bullish macros – macro low vs high time (minutes since t0)"
)

# Bearish: when do highs vs lows form?
plot_macro_time_hist(
    bearish,
    low_col="macro_low_time",
    high_col="macro_high_time",
    title="Bearish macros – macro low vs high time (minutes since t0)"
)

# -------------------------
# Clustering percentages
# -------------------------
# Injection windows mapped to offsets from macro open:
#  - [15:49–15:51]  → [-1, 0, 1]
#  - [15:54–15:56]  → [ 4, 5, 6]
#  - [15:58–16:00]  → [ 8, 9,10]

windows = {
    "cluster_1_near_15:50": (-1, 1),
    "cluster_2_near_15:55": (4, 6),
    "cluster_3_near_15:59": (8, 10),
}

def pct_in_window(df_slice, col, start, end):
    vals = df_slice[col].dropna()
    mask = (vals >= start) & (vals <= end)
    if len(vals) == 0:
        return np.nan
    return mask.mean() * 100.0

print("\n=== Bullish macros – low formation clustering ===")
for name, (lo, hi) in windows.items():
    p = pct_in_window(bullish, "macro_low_time", lo, hi)
    print(f"{name} [{lo},{hi}] min from t0: {p:.2f}%")

print("\n=== Bullish macros – high formation clustering ===")
for name, (lo, hi) in windows.items():
    p = pct_in_window(bullish, "macro_high_time", lo, hi)
    print(f"{name} [{lo},{hi}] min from t0: {p:.2f}%")

print("\n=== Bearish macros – high formation clustering ===")
for name, (lo, hi) in windows.items():
    p = pct_in_window(bearish, "macro_high_time", lo, hi)
    print(f"{name} [{lo},{hi}] min from t0: {p:.2f}%")

print("\n=== Bearish macros – low formation clustering ===")
for name, (lo, hi) in windows.items():
    p = pct_in_window(bearish, "macro_low_time", lo, hi)
    print(f"{name} [{lo},{hi}] min from t0: {p:.2f}%")
