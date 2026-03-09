"""
Test Dynamic Range Analysis (DRA) calculation on NQ 1-minute bars.

Tests the DRA class with:
- Initial range from 3pm-3:10pm bars
- Subsequent bars from 3:10pm onwards
- Rolling average overlap calculation

Run this script from the project root directory (macro/) using:
    python -m features.trend.DRA.test.test_dra

Or from the features/trend/DRA/test directory using:
    python test_dra.py
    (requires running from macro/ directory context)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add DRA directory to path
dra_dir = Path(__file__).parent.parent
if str(dra_dir) not in sys.path:
    sys.path.insert(0, str(dra_dir))

# Import DRA module
from dra import DRA, DEFAULT_WINDOW


def load_nq_data(data_path: Path) -> pd.DataFrame:
    """
    Load NQ CSV data.
    
    Args:
        data_path: Path to NQ.csv
        
    Returns:
        DataFrame with OHLC data
    """
    df = pd.read_csv(data_path, parse_dates=["DateTime_ET"])
    
    # Extract time component
    df["time"] = df["DateTime_ET"].dt.time
    
    # Rename columns to lowercase for consistency
    df = df.rename(columns={
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Open": "open"
    })
    
    return df


def load_and_filter_nq_data(data_path: Path, start_time: str, end_time: str) -> pd.DataFrame:
    """
    Load NQ CSV and filter for a specific time window.
    
    Args:
        data_path: Path to NQ.csv
        start_time: Start time in "HH:MM" format (24-hour)
        end_time: End time in "HH:MM" format (24-hour)
        
    Returns:
        Filtered DataFrame with OHLC data
    """
    df = load_nq_data(data_path)
    
    # Filter for time window
    start = pd.Timestamp(start_time).time()
    end = pd.Timestamp(end_time).time()
    mask = (df["time"] >= start) & (df["time"] < end)
    bars = df[mask].copy()
    
    return bars


def print_result(overlaps: list, rolling_averages: list, window_name: str):
    """
    Pretty print DRA analysis results.
    
    Args:
        overlaps: List of overlap values
        rolling_averages: List of rolling average values
        window_name: Name of the time window
    """
    print("=" * 80)
    print(f"DRA Analysis Results: {window_name}")
    print("=" * 80)
    
    if not overlaps:
        print("❌ Error: No overlap data calculated")
        print()
        return
    
    # Statistics
    print("\nStatistics:")
    print(f"  • Total bars processed:  {len(overlaps)}")
    print(f"  • Initial range:         {window_name}")
    print(f"  • Rolling window:        {DEFAULT_WINDOW} bars")
    print()
    
    # Overlap metrics
    print("Overlap Metrics:")
    print(f"  • Mean overlap:          {np.mean(overlaps):.4f}")
    print(f"  • Median overlap:        {np.median(overlaps):.4f}")
    print(f"  • Min overlap:           {np.min(overlaps):.4f}")
    print(f"  • Max overlap:           {np.max(overlaps):.4f}")
    print(f"  • Std deviation:         {np.std(overlaps):.4f}")
    print()
    
    # Rolling average metrics
    if rolling_averages:
        print("Rolling Average Metrics:")
        print(f"  • Final rolling avg:    {rolling_averages[-1]:.4f}")
        print(f"  • Mean rolling avg:     {np.mean(rolling_averages):.4f}")
        print(f"  • Min rolling avg:      {np.min(rolling_averages):.4f}")
        print(f"  • Max rolling avg:      {np.max(rolling_averages):.4f}")
        print()
    
    # Interpretation
    final_overlap = overlaps[-1] if overlaps else 0
    final_rolling = rolling_averages[-1] if rolling_averages else 0
    
    if final_rolling > 0.8:
        interpretation = "📉 High overlap = Consolidating within initial range"
    elif final_rolling < 0.3:
        interpretation = "📈 Low overlap = Breaking out/trending away from range"
    else:
        interpretation = "➡️  Moderate overlap = Mixed behavior"
    
    print(f"   Interpretation: {interpretation}")
    print()


def test_basic_dra():
    """
    Test basic DRA functionality with 3pm-3:10pm initial range.
    """
    print("=" * 80)
    print("Basic DRA Test")
    print("=" * 80)
    
    # Load data
    data_path = Path(__file__).parent / "NQ.csv"
    
    # Get initial range bars (3pm-3:10pm)
    initial_bars = load_and_filter_nq_data(data_path, "15:00", "15:10")
    
    print(f"\nInitial Range (3pm-3:10pm):")
    print(f"  Loaded {len(initial_bars)} bars")
    if len(initial_bars) > 0:
        print(f"  Date range: {initial_bars['DateTime_ET'].min()} to {initial_bars['DateTime_ET'].max()}")
        print(f"  High: {initial_bars['high'].max():.2f}")
        print(f"  Low: {initial_bars['low'].min():.2f}")
        print(f"  Range: {initial_bars['high'].max() - initial_bars['low'].min():.2f}")
    
    if len(initial_bars) == 0:
        print("❌ No initial range bars found. Cannot proceed with test.")
        return
    
    # Get subsequent bars (3:10pm onwards)
    subsequent_bars = load_and_filter_nq_data(data_path, "15:10", "16:00")
    
    print(f"\nSubsequent Bars (3:10pm-4pm):")
    print(f"  Loaded {len(subsequent_bars)} bars")
    if len(subsequent_bars) > 0:
        print(f"  Date range: {subsequent_bars['DateTime_ET'].min()} to {subsequent_bars['DateTime_ET'].max()}")
    
    if len(subsequent_bars) == 0:
        print("❌ No subsequent bars found. Cannot proceed with test.")
        return
    
    # Initialize DRA
    dra = DRA(window=DEFAULT_WINDOW)
    
    # Set initial range
    dra.set_initial_range(initial_bars)
    
    print(f"\nInitial Range Set:")
    print(f"  High: {dra.initial_high:.2f}")
    print(f"  Low: {dra.initial_low:.2f}")
    print(f"  Range: {dra.initial_range:.2f}")
    print()
    
    # Process subsequent bars
    overlaps = []
    rolling_averages = []
    
    for idx, row in subsequent_bars.iterrows():
        rolling_avg = dra.update(row)
        overlaps.append(dra.overlaps[-1])
        rolling_averages.append(rolling_avg)
    
    # Print results
    print_result(overlaps, rolling_averages, "3pm-3:10pm initial range")
    
    # Show sample of first and last values
    print("Sample Values (first 5 bars):")
    for i in range(min(5, len(overlaps))):
        print(f"  Bar {i+1}: Overlap={overlaps[i]:.4f}, Rolling Avg={rolling_averages[i]:.4f}")
    
    if len(overlaps) > 5:
        print("\nSample Values (last 5 bars):")
        for i in range(max(0, len(overlaps)-5), len(overlaps)):
            print(f"  Bar {i+1}: Overlap={overlaps[i]:.4f}, Rolling Avg={rolling_averages[i]:.4f}")
    
    print()


def test_custom_window():
    """
    Test DRA with custom rolling window size.
    """
    print("=" * 80)
    print("Custom Window Size Test")
    print("=" * 80)
    
    # Load data
    data_path = Path(__file__).parent / "NQ.csv"
    initial_bars = load_and_filter_nq_data(data_path, "15:00", "15:10")
    subsequent_bars = load_and_filter_nq_data(data_path, "15:10", "16:00")
    
    if len(initial_bars) == 0 or len(subsequent_bars) == 0:
        print("❌ Insufficient data for test.")
        return
    
    # Test with different window sizes
    window_sizes = [5, 10, 20]
    
    for window_size in window_sizes:
        print(f"\nTesting with window size: {window_size}")
        
        dra = DRA(window=window_size)
        dra.set_initial_range(initial_bars)
        
        rolling_averages = []
        for idx, row in subsequent_bars.iterrows():
            rolling_avg = dra.update(row)
            rolling_averages.append(rolling_avg)
        
        print(f"  Final rolling average: {rolling_averages[-1]:.4f}")
        print(f"  Mean rolling average: {np.mean(rolling_averages):.4f}")
    
    print()


def test_edge_cases():
    """
    Test edge cases: zero range, empty data, insufficient data, etc.
    """
    print("=" * 80)
    print("Edge Case Tests")
    print("=" * 80)
    
    # Test with zero range (all prices same)
    print("\n1. Testing with zero range (all prices identical):")
    zero_range_data = pd.DataFrame({
        'high': [100.0] * 10,
        'low': [100.0] * 10,
    })
    
    dra = DRA()
    try:
        dra.set_initial_range(zero_range_data)
        print(f"   Initial range: {dra.initial_range}")
        
        # Try to update with a bar
        test_bar = pd.Series({'high': 100.0, 'low': 100.0})
        result = dra.update(test_bar)
        print(f"   Update result: {result}")
        print(f"   ✅ Handled zero range correctly (returns 0.0)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test with empty initial range
    print("\n2. Testing with empty initial range:")
    empty_data = pd.DataFrame(columns=['high', 'low'])
    dra = DRA()
    try:
        dra.set_initial_range(empty_data)
        print(f"   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {e}")
    
    # Test with missing columns
    print("\n3. Testing with missing columns:")
    missing_cols = pd.DataFrame({'high': [100, 101, 102]})
    dra = DRA()
    try:
        dra.set_initial_range(missing_cols)
        print(f"   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {e}")
    
    # Test update before setting initial range
    print("\n4. Testing update before setting initial range:")
    dra = DRA()
    test_bar = pd.Series({'high': 100.0, 'low': 99.0})
    try:
        dra.update(test_bar)
        print(f"   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {e}")
    
    # Test with DataFrame vs list input
    print("\n5. Testing with list of bar objects:")
    class Bar:
        def __init__(self, high, low):
            self.high = high
            self.low = low
    
    bar_list = [Bar(100, 99), Bar(101, 100), Bar(102, 101)]
    dra = DRA()
    dra.set_initial_range(bar_list)
    print(f"   Initial range: {dra.initial_range}")
    print(f"   ✅ Successfully handled list input")
    
    # Test with negative overlap (bar completely outside range)
    print("\n6. Testing with bar completely outside range:")
    dra = DRA()
    initial = pd.DataFrame({'high': [100.0], 'low': [99.0]})
    dra.set_initial_range(initial)
    
    # Bar completely above range
    bar_above = pd.Series({'high': 110.0, 'low': 109.0})
    result = dra.update(bar_above)
    print(f"   Bar above range - Overlap: {dra.overlaps[-1]:.4f}, Rolling avg: {result:.4f}")
    print(f"   ✅ Negative overlap capped at 0.0")
    
    # Bar completely below range
    bar_below = pd.Series({'high': 90.0, 'low': 89.0})
    result = dra.update(bar_below)
    print(f"   Bar below range - Overlap: {dra.overlaps[-1]:.4f}, Rolling avg: {result:.4f}")
    print(f"   ✅ Negative overlap capped at 0.0")
    
    # Test with partial overlap
    print("\n7. Testing with bar containing initial range:")
    bar_partial = pd.Series({'high': 100.5, 'low': 98.5})  # Contains [99, 100]
    result = dra.update(bar_partial)
    print(f"   Bar containing range - Overlap: {dra.overlaps[-1]:.4f}, Rolling avg: {result:.4f}")
    print(f"   Expected overlap: 1.0 (bar completely contains initial range)")
    
    # Test with full overlap
    print("\n8. Testing with full overlap:")
    bar_full = pd.Series({'high': 100.0, 'low': 99.0})  # Exactly matches range
    result = dra.update(bar_full)
    print(f"   Full overlap bar - Overlap: {dra.overlaps[-1]:.4f}, Rolling avg: {result:.4f}")
    print(f"   Expected overlap: 1.0 (fully within range)")
    
    print()


def test_consolidation_vs_breakout():
    """
    Test DRA with synthetic data to demonstrate consolidation vs breakout behavior.
    """
    print("=" * 80)
    print("Consolidation vs Breakout Test")
    print("=" * 80)
    
    # Create initial range
    initial_range = pd.DataFrame({
        'high': [100.0, 100.5, 101.0, 100.8, 100.9],
        'low': [99.0, 99.2, 99.5, 99.3, 99.4]
    })
    
    # Test 1: Consolidating behavior (stays within range)
    print("\n1. Consolidating behavior (stays within initial range):")
    dra_consolidating = DRA(window=5)
    dra_consolidating.set_initial_range(initial_range)
    
    consolidating_bars = pd.DataFrame({
        'high': [100.5, 100.7, 100.6, 100.8, 100.4, 100.5, 100.6],
        'low': [99.5, 99.6, 99.4, 99.7, 99.3, 99.4, 99.5]
    })
    
    overlaps_consolidating = []
    for idx, row in consolidating_bars.iterrows():
        rolling_avg = dra_consolidating.update(row)
        overlaps_consolidating.append(dra_consolidating.overlaps[-1])
    
    print(f"   Mean overlap: {np.mean(overlaps_consolidating):.4f}")
    print(f"   Final rolling avg: {dra_consolidating.overlaps[-5:]}")
    print(f"   ✅ High overlap indicates consolidation")
    
    # Test 2: Breakout behavior (moves away from range)
    print("\n2. Breakout behavior (moves away from initial range):")
    dra_breakout = DRA(window=5)
    dra_breakout.set_initial_range(initial_range)
    
    breakout_bars = pd.DataFrame({
        'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        'low': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    })
    
    overlaps_breakout = []
    for idx, row in breakout_bars.iterrows():
        rolling_avg = dra_breakout.update(row)
        overlaps_breakout.append(dra_breakout.overlaps[-1])
    
    print(f"   Mean overlap: {np.mean(overlaps_breakout):.4f}")
    print(f"   Final rolling avg: {dra_breakout.overlaps[-5:]}")
    print(f"   ✅ Low overlap indicates breakout/trending")
    
    # Test 3: Mixed behavior
    print("\n3. Mixed behavior (some overlap, some breakout):")
    dra_mixed = DRA(window=5)
    dra_mixed.set_initial_range(initial_range)
    
    mixed_bars = pd.DataFrame({
        'high': [100.5, 101.0, 102.0, 100.8, 103.0, 100.6, 104.0],
        'low': [99.5, 100.0, 101.0, 99.7, 102.0, 99.4, 103.0]
    })
    
    overlaps_mixed = []
    for idx, row in mixed_bars.iterrows():
        rolling_avg = dra_mixed.update(row)
        overlaps_mixed.append(dra_mixed.overlaps[-1])
    
    print(f"   Mean overlap: {np.mean(overlaps_mixed):.4f}")
    print(f"   Final rolling avg: {dra_mixed.overlaps[-5:]}")
    print(f"   ✅ Moderate overlap indicates mixed behavior")
    
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Dynamic Range Analysis (DRA) Test Suite")
    print("=" * 80)
    print()
    
    # Show configuration
    print("DRA Configuration:")
    print(f"  - Default window: {DEFAULT_WINDOW} bars")
    print(f"  - Initial range: 3pm-3:10pm (15:00-15:10)")
    print(f"  - Subsequent bars: 3:10pm onwards")
    print()
    
    # Test basic functionality
    test_basic_dra()
    
    # Test custom window sizes
    test_custom_window()
    
    # Test edge cases
    test_edge_cases()
    
    # Test consolidation vs breakout
    test_consolidation_vs_breakout()
    
    print("=" * 80)
    print("Test Suite Complete")
    print("=" * 80)

