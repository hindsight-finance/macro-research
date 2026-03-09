# features/trend/test_trend_quality.py
"""
Test Trend Quality calculation on NQ 1-minute bars for all time windows.

Tests the calculate_trend_quality function with:
- 1pm-3pm window (5m bars, ADX, period=12)
- 3pm-3:50pm window (2m bars, ADX, period=12)
- 3:50pm-4pm window (1m bars, DX, period=5)

Run this script from the project root directory (macro/) using:
    python -m features.trend.test_trend_quality

Or from the features/trend directory using:
    python test_trend_quality.py
    (requires running from macro/ directory context)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import trend_quality module
from features.trend.ADX.trend_quality import calculate_trend_quality, calculate_all_windows, WINDOW_CONFIGS


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
    
    # Rename columns to lowercase for consistency (trend_quality expects 'high', 'low', 'close')
    bars = bars.rename(columns={
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Open": "open"
    })
    
    return bars


def print_result(result: dict, window_name: str):
    """
    Pretty print trend quality results.
    
    Args:
        result: Result dictionary from calculate_trend_quality
        window_name: Name of the time window
    """
    print("=" * 80)
    print(f"Trend Quality Results: {window_name}")
    print("=" * 80)
    
    if result['quality_score'] is None:
        print(f"❌ Error: {result['metadata'].get('error', 'Unknown error')}")
        print()
        return
    
    # Main score
    score = result['quality_score']
    print(f"\n📊 Overall Trend Quality Score: {score:.4f}")
    
    # Score interpretation
    if score >= 0.7:
        interpretation = "✅ Strong, clean trend"
    elif score >= 0.5:
        interpretation = "⚠️  Moderate trend"
    elif score >= 0.3:
        interpretation = "⚠️  Weak trend"
    else:
        interpretation = "❌ Choppy/consolidating"
    
    print(f"   Interpretation: {interpretation}")
    print()
    
    # Component breakdown
    components = result['components']
    print("Component Scores:")
    print(f"  • Strength (ADX/DX):     {components['strength']:.4f} (raw: {components['strength_raw']:.2f})")
    print(f"  • Persistence:          {components['persistence']:.4f}")
    print(f"  • Crossover (smooth):    {components['crossover']:.4f}")
    print(f"  • Dominant DI:           {components['dominant_di'].upper()}")
    print()
    
    # Metadata
    metadata = result['metadata']
    print("Metadata:")
    print(f"  • Total bars:            {metadata['total_bars']}")
    print(f"  • Valid bars:            {metadata['valid_bars']}")
    print(f"  • Period:                {metadata['period']}")
    print(f"  • Indicator type:        {metadata['indicator_type']}")
    print(f"  • Window:                 {metadata['window']}")
    print()


def test_single_window(window_name: str, start_time: str, end_time: str):
    """
    Test trend quality for a single time window.
    
    Args:
        window_name: Window identifier ("1pm-3pm", "3pm-3:50pm", "3:50pm-4pm")
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
    
    # Calculate trend quality
    result = calculate_trend_quality(bars, window_name)
    
    # Print results
    print_result(result, window_name)
    
    return result


def test_all_windows():
    """
    Test trend quality for all configured windows.
    """
    data_path = Path(__file__).parent / "NQ.csv"
    
    # Define time windows
    windows = {
        "1pm-3pm": ("13:00", "15:00"),
        "3pm-3:50pm": ("15:00", "15:50"),
        "3:50pm-4pm": ("15:50", "16:00")
    }
    
    # Load all data
    bars_dict = {}
    for window_name, (start, end) in windows.items():
        bars = load_and_filter_nq_data(data_path, start, end)
        bars_dict[window_name] = bars
        print(f"Loaded {len(bars)} bars for {window_name}")
    
    print()
    
    # Calculate for all windows
    results = calculate_all_windows(bars_dict)
    
    # Print all results
    for window_name, result in results.items():
        print_result(result, window_name)
    
    # Summary comparison
    print("=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    print(f"{'Window':<20} {'Quality Score':<15} {'Strength':<12} {'Persistence':<12} {'Crossover':<12}")
    print("-" * 80)
    
    for window_name, result in results.items():
        if result['quality_score'] is not None:
            comp = result['components']
            print(f"{window_name:<20} {result['quality_score']:<15.4f} "
                  f"{comp['strength']:<12.4f} {comp['persistence']:<12.4f} "
                  f"{comp['crossover']:<12.4f}")
        else:
            print(f"{window_name:<20} {'ERROR':<15}")
    
    print()


def test_edge_cases():
    """
    Test edge cases: insufficient data, empty data, etc.
    """
    print("=" * 80)
    print("Edge Case Tests")
    print("=" * 80)
    
    # Test with insufficient data
    print("\n1. Testing with insufficient data (< min_bars_required):")
    small_bars = pd.DataFrame({
        'high': [100, 101, 102],
        'low': [99, 100, 101],
        'close': [100.5, 101.5, 102.5]
    })
    result = calculate_trend_quality(small_bars, "3:50pm-4pm")
    print(f"   Result: {result['quality_score']}")
    print(f"   Error: {result['metadata'].get('error', 'None')}")
    
    # Test with empty DataFrame
    print("\n2. Testing with empty DataFrame:")
    empty_bars = pd.DataFrame(columns=['high', 'low', 'close'])
    result = calculate_trend_quality(empty_bars, "3:50pm-4pm")
    print(f"   Result: {result['quality_score']}")
    print(f"   Error: {result['metadata'].get('error', 'None')}")
    
    # Test with None
    print("\n3. Testing with None DataFrame:")
    result = calculate_trend_quality(None, "3:50pm-4pm")
    print(f"   Result: {result['quality_score']}")
    print(f"   Error: {result['metadata'].get('error', 'None')}")
    
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Trend Quality Test Suite")
    print("=" * 80)
    print()
    
    # Show configuration
    print("Window Configurations:")
    for window_name, config in WINDOW_CONFIGS.items():
        print(f"  {window_name}:")
        print(f"    - Bar size: {config['bar_size']}")
        print(f"    - Period: {config['period']}")
        print(f"    - Use DX: {config['use_dx']}")
        print(f"    - Weights: {config['weights']}")
        print(f"    - Min bars: {config['min_bars_required']}")
    print()
    
    # Test individual windows
    print("\n" + "=" * 80)
    print("Individual Window Tests")
    print("=" * 80)
    
    test_single_window("1pm-3pm", "13:00", "15:00")
    test_single_window("3pm-3:50pm", "15:00", "15:50")
    test_single_window("3:50pm-4pm", "15:50", "16:00")
    
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

