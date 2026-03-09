# features/trend/test.py
"""
Test DX calculation on NQ 1-minute bars for 3:50-4:00pm window.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import di_indicators
from adx_calc import calculate_dx_full_from_df
from di_persistence import calculate_di_persistence, calculate_di_persistence_avg
from di_crossovers import count_di_crossovers, calculate_crossover_penalty

# Load NQ data
data_path = Path(__file__).parent / "NQ.csv"
df = pd.read_csv(data_path, parse_dates=["DateTime_ET"])

# Filter for 3:50pm - 4:00pm (15:50 - 16:00)
df["time"] = df["DateTime_ET"].dt.time
mask = (df["time"] >= pd.Timestamp("15:45").time()) & (df["time"] < pd.Timestamp("16:00").time())
bars_1m = df[mask].copy()

print(f"Loaded {len(bars_1m)} bars between 3:50-4:00pm")
print(f"Date range: {bars_1m['DateTime_ET'].min()} to {bars_1m['DateTime_ET'].max()}")
print()

# Calculate DX with period=5
dx_result = calculate_dx_full_from_df(
    bars_1m,
    period=5,
    high_col="High",
    low_col="Low",
    close_col="Close"
)

# Combine with original data for display
result = pd.concat([
    bars_1m[["DateTime_ET", "Open", "High", "Low", "Close"]].reset_index(drop=True),
    dx_result.reset_index(drop=True)
], axis=1)

print("DX Results (period=5):")
print(result.to_string())
print()

# Test DI Persistence
print("=" * 80)
print("DI Persistence Test")
print("=" * 80)

# Extract +DI and -DI, filter out NaN values
plus_di = dx_result["+DI"].values
minus_di = dx_result["-DI"].values

# Find valid (non-NaN) indices
valid_mask = ~(np.isnan(plus_di) | np.isnan(minus_di))
plus_di_valid = plus_di[valid_mask]
minus_di_valid = minus_di[valid_mask]

if len(plus_di_valid) > 0:
    print(f"Valid DI bars: {len(plus_di_valid)} out of {len(plus_di)} total bars")
    print()
    
    # Calculate persistence scores
    persistence_max = calculate_di_persistence(plus_di_valid, minus_di_valid)
    persistence_avg = calculate_di_persistence_avg(plus_di_valid, minus_di_valid)
    
    print(f"Max Consecutive Persistence Score: {persistence_max:.4f}")
    print(f"  - Interpretation: {persistence_max*100:.1f}% of bars in longest run")
    print()
    print(f"Average Run Persistence Score: {persistence_avg:.4f}")
    print(f"  - Interpretation: Average run length is {persistence_avg*100:.1f}% of total bars")
    print()
    
    # Show dominance breakdown
    dominant = plus_di_valid > minus_di_valid
    plus_dominant_count = np.sum(dominant)
    minus_dominant_count = len(dominant) - plus_dominant_count
    equal_count = np.sum(plus_di_valid == minus_di_valid)
    
    print(f"Dominance Breakdown:")
    print(f"  +DI dominant: {plus_dominant_count} bars ({plus_dominant_count/len(dominant)*100:.1f}%)")
    print(f"  -DI dominant: {minus_dominant_count} bars ({minus_dominant_count/len(dominant)*100:.1f}%)")
    if equal_count > 0:
        print(f"  Equal: {equal_count} bars ({equal_count/len(dominant)*100:.1f}%)")
else:
    print("No valid DI values found (all NaN)")

# Test DI Crossovers
print()
print("=" * 80)
print("DI Crossovers Test")
print("=" * 80)

if len(plus_di_valid) > 0:
    # Calculate crossover metrics
    crossover_count = count_di_crossovers(plus_di_valid, minus_di_valid)
    penalty_default = calculate_crossover_penalty(plus_di_valid, minus_di_valid)
    penalty_custom = calculate_crossover_penalty(plus_di_valid, minus_di_valid, max_expected=len(plus_di_valid) / 3)
    
    print(f"Valid DI bars: {len(plus_di_valid)} out of {len(plus_di)} total bars")
    print()
    
    print(f"Crossover Count: {crossover_count}")
    print(f"  - Interpretation: Directional leadership flipped {crossover_count} times")
    print(f"  - Crossover rate: {crossover_count/len(plus_di_valid)*100:.1f}% of bars")
    print()
    
    print(f"Crossover Penalty Score (default max_expected): {penalty_default:.4f}")
    print(f"  - Interpretation: {penalty_default*100:.1f}% smooth (1.0 = no crossovers, 0.0 = many crossovers)")
    print()
    
    print(f"Crossover Penalty Score (custom max_expected={len(plus_di_valid)/3:.1f}): {penalty_custom:.4f}")
    print(f"  - Interpretation: {penalty_custom*100:.1f}% smooth with stricter threshold")
    print()
    
    # Show crossover details
    di_difference = plus_di_valid - minus_di_valid
    sign_changes = []
    for i in range(1, len(di_difference)):
        if (di_difference[i-1] > 0 and di_difference[i] < 0) or \
           (di_difference[i-1] < 0 and di_difference[i] > 0):
            sign_changes.append(i)
    
    if len(sign_changes) > 0:
        print(f"Crossover Locations (bar indices in valid array): {sign_changes}")
        print(f"  - First crossover at bar {sign_changes[0]}, last at bar {sign_changes[-1]}")
    else:
        print("No crossovers detected - consistent directional leadership")
else:
    print("No valid DI values found (all NaN)")
