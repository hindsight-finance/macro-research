# features/trend/test.py
"""
Test ADX calculation on NQ 1-minute bars for 3:00-3:50pm window.
"""
import pandas as pd
from pathlib import Path
from adx_calc import calculate_adx_full_from_df

# Load NQ data
data_path = Path(__file__).parent / "NQ.csv"
df = pd.read_csv(data_path, parse_dates=["DateTime_ET"])

# Filter for 3pm - 3:50pm (15:00 - 15:50)
df["time"] = df["DateTime_ET"].dt.time
mask = (df["time"] >= pd.Timestamp("15:00").time()) & (df["time"] < pd.Timestamp("15:50").time())
bars_1m = df[mask].copy()

print(f"Loaded {len(bars_1m)} bars between 3:00-3:50pm")
print(f"Date range: {bars_1m['DateTime_ET'].min()} to {bars_1m['DateTime_ET'].max()}")
print()

# Calculate ADX with period=12
adx_result = calculate_adx_full_from_df(
    bars_1m,
    period=12,
    high_col="High",
    low_col="Low",
    close_col="Close"
)

# Combine with original data for display
result = pd.concat([
    bars_1m[["DateTime_ET", "Open", "High", "Low", "Close"]].reset_index(drop=True),
    adx_result.reset_index(drop=True)
], axis=1)

print("ADX Results (period=12):")
print(result.to_string())
