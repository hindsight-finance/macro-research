# features/trend/ATR Range/Testing/test_atr.py
r"""
Test ATR/Range Ratio calculation on NQ 1-minute bars for all time windows.

Tests the analyze_session function with:
- 1pm-3pm window (5m bars, ATR period=10)
- 3pm-3:50pm window (2m bars, ATR period=8)
- 3:50-4pm window (1m bars, ATR period=3)

Run this script from the project root directory (macro/) using:
    python -m features.trend.ATR\ Range.Testing.test_atr

Or from the features/trend/ATR Range/Testing directory using:
    python test_atr.py
    (requires running from macro/ directory context)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add ATR Range directory to path (handles space in directory name)
atr_range_dir = Path(__file__).parent.parent
if str(atr_range_dir) not in sys.path:
    sys.path.insert(0, str(atr_range_dir))

# Import ATR module (using direct import since we added the directory to path)
from atr import analyze_session, calculate_atr, calculate_ratio, SESSIONS


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
    df = pd.read_csv(data_path, parse_dates=["DateTime_ET"])
    
    # Extract time component
    df["time"] = df["DateTime_ET"].dt.time
    
    # Filter for time window
    start = pd.Timestamp(start_time).time()
    end = pd.Timestamp(end_time).time()
    mask = (df["time"] >= start) & (df["time"] < end)
    bars = df[mask].copy()
    
    # Rename columns to lowercase for consistency
    bars = bars.rename(columns={
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Open": "open"
    })
    
    return bars


def print_result(result: dict, window_name: str):
    """
    Pretty print ATR/Range analysis results.
    
    Args:
        result: Result dictionary from analyze_session
        window_name: Name of the time window
    """
    print("=" * 80)
    print(f"ATR/Range Analysis Results: {window_name}")
    print("=" * 80)
    
    if result['signal'] == 'NO_RANGE':
        print(f"❌ Error: No range detected (zero price range)")
        print()
        return
    
    # Main signal
    signal = result['signal']
    signal_emoji = {
        'TRENDING': '📈',
        'CONSOLIDATING': '📉',
        'NEUTRAL': '➡️',
        'NO_RANGE': '❌'
    }
    emoji = signal_emoji.get(signal, '❓')
    
    print(f"\n{emoji} Signal: {signal}")
    print()
    
    # Metrics
    print("Metrics:")
    print(f"  • ATR/Range Ratio:    {result['raw_ratio']:.4f}")
    print(f"  • Median ATR:         {result['median_atr']:.4f}")
    print(f"  • Total Range:        {result['total_range']:.4f}")
    print()
    
    # Interpretation
    ratio = result['raw_ratio']
    if ratio < 0.5:
        interpretation = "✅ Trending (low ATR relative to range)"
    elif ratio > 0.8:
        interpretation = "⚠️  Consolidating (high ATR relative to range)"
    else:
        interpretation = "➡️  Neutral (moderate ATR relative to range)"
    
    print(f"   Interpretation: {interpretation}")
    print()


def run_single_window(window_name: str, start_time: str, end_time: str):
    """
    Test ATR/Range analysis for a single time window.
    
    Args:
        window_name: Window identifier ("1pm-3pm", "3pm-3:50pm", "3:50-4pm")
        start_time: Start time in "HH:MM" format
        end_time: End time in "HH:MM" format
    """
    # Load data
    data_path = Path(__file__).parent / "NQ.csv"
    bars = load_and_filter_nq_data(data_path, start_time, end_time)
    
    print(f"\n{'='*80}")
    print(f"Testing Window: {window_name}")
    print(f"Time Range: {start_time} - {end_time}")
    print(f"Loaded {len(bars)} bars")
    if len(bars) > 0:
        print(f"Date range: {bars['DateTime_ET'].min()} to {bars['DateTime_ET'].max()}")
    print(f"{'='*80}\n")
    
    # Calculate ATR/Range analysis
    result = analyze_session(bars, window_name)
    
    # Print results
    print_result(result, window_name)
    
    return result


def test_all_windows():
    """
    Test ATR/Range analysis for all configured windows.
    """
    data_path = Path(__file__).parent / "NQ.csv"
    
    # Define time windows (matching SESSIONS config)
    windows = {
        "1pm-3pm": ("13:00", "15:00"),
        "3pm-3:50pm": ("15:00", "15:50"),
        "3:50-4pm": ("15:50", "16:00")
    }
    
    # Load all data
    bars_dict = {}
    for window_name, (start, end) in windows.items():
        bars = load_and_filter_nq_data(data_path, start, end)
        bars_dict[window_name] = bars
        print(f"Loaded {len(bars)} bars for {window_name}")
    
    print()
    
    # Calculate for all windows
    results = {}
    for window_name, bars in bars_dict.items():
        results[window_name] = analyze_session(bars, window_name)
    
    # Print all results
    for window_name, result in results.items():
        print_result(result, window_name)
    
    # Summary comparison
    print("=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    print(f"{'Window':<20} {'Signal':<15} {'Ratio':<12} {'Median ATR':<12} {'Total Range':<12}")
    print("-" * 80)
    
    for window_name, result in results.items():
        if result['signal'] != 'NO_RANGE':
            print(f"{window_name:<20} {result['signal']:<15} "
                  f"{result['raw_ratio']:<12.4f} {result['median_atr']:<12.4f} "
                  f"{result['total_range']:<12.4f}")
        else:
            print(f"{window_name:<20} {'NO_RANGE':<15}")
    
    print()


def test_edge_cases():
    """
    Test edge cases: insufficient data, empty data, zero range, etc.
    """
    print("=" * 80)
    print("Edge Case Tests")
    print("=" * 80)
    
    # Test with zero range (all prices same)
    print("\n1. Testing with zero range (all prices identical):")
    zero_range_data = pd.DataFrame({
        'high': [100.0] * 10,
        'low': [100.0] * 10,
        'close': [100.0] * 10
    })
    result = analyze_session(zero_range_data, "3:50-4pm")
    print(f"   Signal: {result['signal']}")
    print(f"   Raw ratio: {result['raw_ratio']}")
    
    # Test with trending data (low ratio expected)
    print("\n2. Testing with trending data (low ATR/Range ratio):")
    trending_data = pd.DataFrame({
        'high': np.linspace(100, 110, 24) + np.random.randn(24) * 0.1,
        'low': np.linspace(99, 109, 24) + np.random.randn(24) * 0.1,
        'close': np.linspace(99.5, 109.5, 24) + np.random.randn(24) * 0.1,
    })
    result = analyze_session(trending_data, "1pm-3pm")
    print(f"   Signal: {result['signal']}")
    print(f"   Raw ratio: {result['raw_ratio']:.4f}")
    print(f"   Expected: TRENDING (ratio < 0.5)")
    
    # Test with consolidating data (high ratio expected)
    print("\n3. Testing with consolidating data (high ATR/Range ratio):")
    consolidating_data = pd.DataFrame({
        'high': 100 + np.random.randn(24) * 2,
        'low': 99 + np.random.randn(24) * 2,
        'close': 99.5 + np.random.randn(24) * 2,
    })
    result = analyze_session(consolidating_data, "1pm-3pm")
    print(f"   Signal: {result['signal']}")
    print(f"   Raw ratio: {result['raw_ratio']:.4f}")
    print(f"   Expected: CONSOLIDATING (ratio > 0.8)")
    
    # Test with neutral data (medium ratio)
    print("\n4. Testing with neutral data (medium ATR/Range ratio):")
    neutral_data = pd.DataFrame({
        'high': np.linspace(100, 102, 24) + np.random.randn(24) * 0.5,
        'low': np.linspace(99, 101, 24) + np.random.randn(24) * 0.5,
        'close': np.linspace(99.5, 101.5, 24) + np.random.randn(24) * 0.5,
    })
    result = analyze_session(neutral_data, "1pm-3pm")
    print(f"   Signal: {result['signal']}")
    print(f"   Raw ratio: {result['raw_ratio']:.4f}")
    print(f"   Expected: NEUTRAL (0.5 <= ratio <= 0.8)")
    
    # Test with very few bars
    print("\n5. Testing with minimal data (3 bars):")
    minimal_data = pd.DataFrame({
        'high': [100, 101, 102],
        'low': [99, 100, 101],
        'close': [99.5, 100.5, 101.5]
    })
    result = analyze_session(minimal_data, "3:50-4pm")
    print(f"   Signal: {result['signal']}")
    print(f"   Raw ratio: {result['raw_ratio']}")
    
    # Test with invalid session name
    print("\n6. Testing with invalid session name:")
    try:
        result = analyze_session(trending_data, "invalid-session")
        print(f"   Unexpected success: {result}")
    except KeyError as e:
        print(f"   ✅ Correctly raised KeyError: {e}")
    
    print()


def test_atr_calculation():
    """
    Test the underlying ATR calculation function.
    """
    print("=" * 80)
    print("ATR Calculation Tests")
    print("=" * 80)
    
    # Create test data
    dates = pd.date_range('2024-01-01 13:00', periods=10, freq='5min')
    test_data = pd.DataFrame({
        'datetime': dates,
        'high': [100, 101, 102, 101, 100, 99, 98, 99, 100, 101],
        'low': [99, 100, 101, 100, 99, 98, 97, 98, 99, 100],
        'close': [99.5, 100.5, 101.5, 100.5, 99.5, 98.5, 97.5, 98.5, 99.5, 100.5]
    })
    
    # Test ATR calculation
    print("\n1. Testing ATR calculation with period=5:")
    atr = calculate_atr(test_data, period=5)
    print(f"   ATR values: {atr.values}")
    print(f"   Mean ATR: {atr.mean():.4f}")
    print(f"   Median ATR: {atr.median():.4f}")
    
    # Test ratio calculation
    print("\n2. Testing ratio calculation:")
    ratio, median_atr, total_range = calculate_ratio(test_data, atr_period=5)
    print(f"   Raw ratio: {ratio:.4f}")
    print(f"   Median ATR: {median_atr:.4f}")
    print(f"   Total range: {total_range:.4f}")
    
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("ATR/Range Ratio Test Suite")
    print("=" * 80)
    print()
    
    # Show configuration
    print("Session Configurations:")
    for session_name, config in SESSIONS.items():
        print(f"  {session_name}:")
        print(f"    - Bar size: {config['bar_size']}")
        print(f"    - Total bars: {config['total_bars']}")
        print(f"    - ATR period: {config['atr_period']}")
    print()
    
    # Test individual windows
    print("\n" + "=" * 80)
    print("Individual Window Tests")
    print("=" * 80)
    
    run_single_window("1pm-3pm", "13:00", "15:00")
    run_single_window("3pm-3:50pm", "15:00", "15:50")
    run_single_window("3:50-4pm", "15:50", "16:00")
    
    # Test all windows at once
    print("\n" + "=" * 80)
    print("Batch Processing Test (All Windows)")
    print("=" * 80)
    test_all_windows()
    
    # Test ATR calculation
    test_atr_calculation()
    
    # Test edge cases
    test_edge_cases()
    
    print("=" * 80)
    print("Test Suite Complete")
    print("=" * 80)

