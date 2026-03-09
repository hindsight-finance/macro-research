"""
Test Swing Point Density calculation on NQ 1-minute bars for all time windows.

Tests the swing point density detection with:
- Early Afternoon (1pm-3pm)
- Late Afternoon (3pm-3:50pm)
- Close (3:50pm-4pm)

Run this script from the project root directory (macro/) using:
    python -m features.trend."Swing Point Density".test.test_spd

Or from the features/trend/Swing Point Density/test directory using:
    python test_spd.py
    (requires running from macro/ directory context)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import time

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add Swing Point Density directory to path (handles space in directory name)
spd_dir = Path(__file__).parent.parent
if str(spd_dir) not in sys.path:
    sys.path.insert(0, str(spd_dir))

# Import SPD module
from spd import (
    detect_swing_highs,
    detect_swing_lows,
    filter_bars_by_time,
    classify_density,
    get_swing_density,
    analyze_all_windows,
    WINDOWS,
    THRESHOLDS
)


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
    
    return df


def df_to_bars(df: pd.DataFrame) -> list:
    """
    Convert DataFrame to list of dictionaries expected by spd functions.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        List of bar dictionaries
    """
    bars = []
    for _, row in df.iterrows():
        bars.append({
            'time': row['time'],
            'timestamp': row['DateTime_ET'],  # Also include timestamp for flexibility
            'high': row['High'],
            'low': row['Low'],
            'open': row['Open'],
            'close': row['Close'],
            'volume': row['Volume']
        })
    return bars


def load_and_filter_nq_data(data_path: Path, start_time: str, end_time: str) -> list:
    """
    Load NQ CSV and filter for a specific time window, converting to bar format.
    
    Args:
        data_path: Path to NQ.csv
        start_time: Start time in "HH:MM" format (24-hour)
        end_time: End time in "HH:MM" format (24-hour)
        
    Returns:
        List of bar dictionaries within time window
    """
    df = load_nq_data(data_path)
    
    # Filter for time window
    start = pd.Timestamp(start_time).time()
    end = pd.Timestamp(end_time).time()
    mask = (df["time"] >= start) & (df["time"] < end)
    filtered_df = df[mask].copy()
    
    return df_to_bars(filtered_df)


def print_result(result: dict, window_name: str):
    """
    Pretty print swing density analysis results.
    
    Args:
        result: Result dictionary from get_swing_density or analyze_all_windows
        window_name: Name of the time window
    """
    print("=" * 80)
    print(f"Swing Point Density Analysis: {window_name}")
    print("=" * 80)
    
    if result.get('classification') == 'insufficient_data':
        print(f"❌ Error: Insufficient data ({result.get('bars_analyzed', 0)} bars)")
        print()
        return
    
    # Main classification
    classification = result.get('classification', 'unknown')
    classification_emoji = {
        'trending': '📈',
        'mixed': '➡️',
        'chop': '📉',
        'insufficient_data': '❌'
    }
    emoji = classification_emoji.get(classification, '❓')
    
    print(f"\n{emoji} Classification: {classification.upper()}")
    print()
    
    # Metrics
    print("Metrics:")
    print(f"  • Total Swing Points:  {result.get('count', 0)}")
    print(f"  • Swing Highs:         {result.get('swing_high_count', 0)}")
    print(f"  • Swing Lows:          {result.get('swing_low_count', 0)}")
    print(f"  • Bars Analyzed:       {result.get('bars_analyzed', 0)}")
    print()
    
    # Interpretation
    count = result.get('count', 0)
    if count < THRESHOLDS['trending']:
        interpretation = "✅ Trending (low swing density = smooth price action)"
    elif count > THRESHOLDS['chop']:
        interpretation = "⚠️  Choppy/Consolidation (high swing density = choppy price action)"
    else:
        interpretation = "➡️  Mixed (moderate swing density)"
    
    print(f"   Interpretation: {interpretation}")
    print()


def test_single_window(window_name: str, start_time: str, end_time: str):
    """
    Test swing density analysis for a single time window.
    
    Args:
        window_name: Window identifier
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
        print(f"First bar time: {bars[0]['time']}")
        print(f"Last bar time: {bars[-1]['time']}")
    print(f"{'='*80}\n")
    
    # Calculate swing density
    start = pd.Timestamp(start_time).time()
    end = pd.Timestamp(end_time).time()
    result = get_swing_density(bars, start, end)
    
    # Print results
    print_result(result, window_name)
    
    return result


def test_all_windows():
    """
    Test swing density analysis for all configured windows.
    """
    data_path = Path(__file__).parent / "NQ.csv"
    
    # Load all data for the day
    df = load_nq_data(data_path)
    bars = df_to_bars(df)
    
    print(f"\n{'='*80}")
    print("Testing All Configured Windows")
    print(f"{'='*80}")
    print(f"Total bars loaded: {len(bars)}")
    if len(bars) > 0:
        print(f"Date range: {bars[0]['timestamp']} to {bars[-1]['timestamp']}")
    print()
    
    # Analyze all windows
    results = analyze_all_windows(bars)
    
    # Print all results
    for window_name, result in results.items():
        print_result(result, result.get('window_name', window_name))
    
    # Summary comparison
    print("=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    print(f"{'Window':<30} {'Classification':<15} {'Swing Count':<12} {'Highs':<8} {'Lows':<8} {'Bars':<8}")
    print("-" * 80)
    
    for window_name, result in results.items():
        window_display = result.get('window_name', window_name)
        classification = result.get('classification', 'unknown')
        count = result.get('count', 0)
        highs = result.get('swing_high_count', 0)
        lows = result.get('swing_low_count', 0)
        bars_count = result.get('bars_analyzed', 0)
        
        print(f"{window_display:<30} {classification.upper():<15} {count:<12} {highs:<8} {lows:<8} {bars_count:<8}")
    
    print()


def test_swing_detection():
    """
    Test the underlying swing detection functions.
    """
    print("=" * 80)
    print("Swing Detection Function Tests")
    print("=" * 80)
    
    # Create test data with known swing points
    test_bars = [
        {'high': 100, 'low': 99},   # 0
        {'high': 101, 'low': 100},  # 1 - swing high (100 < 101 > 100)
        {'high': 100, 'low': 99},   # 2
        {'high': 99, 'low': 98},    # 3 - swing low (99 > 98 < 99)
        {'high': 99, 'low': 99},    # 4
        {'high': 102, 'low': 101},  # 5 - swing high (100 < 102 > 99)
        {'high': 99, 'low': 97},    # 6 - swing low (99 > 97 < 99)
        {'high': 99, 'low': 98},    # 7
    ]
    
    print("\n1. Testing swing high detection:")
    swing_highs = detect_swing_highs(test_bars)
    print(f"   Detected swing highs at indices: {swing_highs}")
    print(f"   Expected: [1, 5]")
    print(f"   ✅ Pass" if swing_highs == [1, 5] else "   ❌ Fail")
    
    print("\n2. Testing swing low detection:")
    swing_lows = detect_swing_lows(test_bars)
    print(f"   Detected swing lows at indices: {swing_lows}")
    print(f"   Expected: [3, 6]")
    print(f"   ✅ Pass" if swing_lows == [3, 6] else "   ❌ Fail")
    
    # Test with real data
    print("\n3. Testing with NQ data sample:")
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)
    sample_bars = df_to_bars(df.head(20))
    
    swing_highs = detect_swing_highs(sample_bars)
    swing_lows = detect_swing_lows(sample_bars)
    
    print(f"   Sample size: {len(sample_bars)} bars")
    print(f"   Swing highs detected: {len(swing_highs)}")
    print(f"   Swing lows detected: {len(swing_lows)}")
    print(f"   Total swings: {len(swing_highs) + len(swing_lows)}")
    
    print()


def test_classification():
    """
    Test the classification function.
    """
    print("=" * 80)
    print("Classification Function Tests")
    print("=" * 80)
    
    test_cases = [
        (0, 'trending'),
        (3, 'trending'),
        (4, 'trending'),  # Just below threshold
        (5, 'mixed'),     # At threshold
        (6, 'mixed'),
        (7, 'mixed'),
        (8, 'mixed'),     # At threshold
        (9, 'chop'),      # Above threshold
        (15, 'chop'),
    ]
    
    print("\nTesting classification thresholds:")
    print(f"Trending threshold: < {THRESHOLDS['trending']}")
    print(f"Chop threshold: > {THRESHOLDS['chop']}")
    print()
    
    all_passed = True
    for count, expected in test_cases:
        result = classify_density(count)
        passed = result == expected
        status = "✅" if passed else "❌"
        print(f"   {status} Count={count:2d} -> {result:10s} (expected {expected})")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n   ✅ All classification tests passed!")
    else:
        print("\n   ❌ Some classification tests failed!")
    
    print()


def test_edge_cases():
    """
    Test edge cases: insufficient data, empty data, etc.
    """
    print("=" * 80)
    print("Edge Case Tests")
    print("=" * 80)
    
    # Test with insufficient data (< 3 bars)
    print("\n1. Testing with insufficient data (2 bars):")
    minimal_bars = [
        {'high': 100, 'low': 99, 'time': time(13, 0)},
        {'high': 101, 'low': 100, 'time': time(13, 1)},
    ]
    result = get_swing_density(minimal_bars, time(13, 0), time(13, 2))
    print(f"   Classification: {result['classification']}")
    print(f"   Count: {result['count']}")
    print(f"   ✅ Pass" if result['classification'] == 'insufficient_data' else "   ❌ Fail")
    
    # Test with empty data
    print("\n2. Testing with empty data:")
    empty_bars = []
    result = get_swing_density(empty_bars, time(13, 0), time(15, 0))
    print(f"   Classification: {result['classification']}")
    print(f"   Count: {result['count']}")
    print(f"   ✅ Pass" if result['classification'] == 'insufficient_data' else "   ❌ Fail")
    
    # Test with exactly 3 bars (minimum for swing detection)
    print("\n3. Testing with exactly 3 bars:")
    three_bars = [
        {'high': 100, 'low': 99, 'time': time(13, 0)},
        {'high': 101, 'low': 100, 'time': time(13, 1)},
        {'high': 100, 'low': 99, 'time': time(13, 2)},
    ]
    result = get_swing_density(three_bars, time(13, 0), time(13, 3))
    print(f"   Classification: {result['classification']}")
    print(f"   Count: {result['count']}")
    print(f"   Bars analyzed: {result['bars_analyzed']}")
    
    # Test with trending data (low swing count)
    print("\n4. Testing with trending data (low swing density):")
    trending_bars = []
    base_price = 100
    for i in range(20):
        trending_bars.append({
            'high': base_price + i * 0.5 + 0.1,
            'low': base_price + i * 0.5 - 0.1,
            'time': time(13, i)
        })
    result = get_swing_density(trending_bars, time(13, 0), time(13, 20))
    print(f"   Classification: {result['classification']}")
    print(f"   Count: {result['count']}")
    print(f"   Expected: trending (count < {THRESHOLDS['trending']})")
    
    # Test with choppy data (high swing count)
    print("\n5. Testing with choppy data (high swing density):")
    choppy_bars = []
    for i in range(20):
        # Create alternating pattern
        direction = 1 if i % 2 == 0 else -1
        choppy_bars.append({
            'high': 100 + direction * 0.5 + 0.1,
            'low': 100 + direction * 0.5 - 0.1,
            'time': time(13, i)
        })
    result = get_swing_density(choppy_bars, time(13, 0), time(13, 20))
    print(f"   Classification: {result['classification']}")
    print(f"   Count: {result['count']}")
    print(f"   Expected: chop (count > {THRESHOLDS['chop']})")
    
    # Test time filtering
    print("\n6. Testing time filtering:")
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)
    all_bars = df_to_bars(df)
    
    # Filter for a specific window
    filtered = filter_bars_by_time(all_bars, time(13, 0), time(15, 0))
    print(f"   Total bars: {len(all_bars)}")
    print(f"   Filtered bars (1pm-3pm): {len(filtered)}")
    if len(filtered) > 0:
        print(f"   First filtered bar time: {filtered[0]['time']}")
        print(f"   Last filtered bar time: {filtered[-1]['time']}")
    
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Swing Point Density Test Suite")
    print("=" * 80)
    print()
    
    # Show configuration
    print("Window Configurations:")
    for window_name, config in WINDOWS.items():
        print(f"  {config['name']}:")
        print(f"    - Start: {config['start']}")
        print(f"    - End: {config['end']}")
    print()
    
    print("Classification Thresholds:")
    print(f"  - Trending: < {THRESHOLDS['trending']} swings")
    print(f"  - Mixed: {THRESHOLDS['trending']} - {THRESHOLDS['chop']} swings")
    print(f"  - Chop: > {THRESHOLDS['chop']} swings")
    print()
    
    # Test swing detection functions
    test_swing_detection()
    
    # Test classification
    test_classification()
    
    # Test individual windows
    print("\n" + "=" * 80)
    print("Individual Window Tests")
    print("=" * 80)
    
    test_single_window("Early Afternoon", "13:00", "15:00")
    test_single_window("Late Afternoon", "15:00", "15:50")
    test_single_window("Close", "15:50", "16:00")
    
    # Test all windows at once
    print("\n" + "=" * 80)
    print("Batch Processing Test (All Windows)")
    print("=" * 80)
    test_all_windows()
    
    # Test edge cases
    test_edge_cases()
    
    print("=" * 80)
    print("Test Suite Complete")
    print("=" * 80)

