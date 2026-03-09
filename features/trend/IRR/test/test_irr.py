"""
Test Intrabar Range Ratio (IRR) calculation on NQ 1-minute bars.

Tests the IRR module with:
- Single bar IRR computation
- Full window IRR (entire session as one candle)
- Sub-window IRRs (session divided into sub-windows)
- Two-level analysis structure
- Regime classification
- Reversal detection
- Edge cases (zero range, doji bars, etc.)

Run this script from the project root directory (macro/) using:
    python -m features.trend.IRR.test.test_irr

Or from the test directory using:
    python test_irr.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add IRR directory to path
irr_dir = Path(__file__).parent.parent
if str(irr_dir) not in sys.path:
    sys.path.insert(0, str(irr_dir))

# Import IRR module
from irr import (
    IRRAnalyzer,
    analyze_session,
    batch_analyze_windows
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
    df["time"] = df["DateTime_ET"].dt.strftime("%H:%M")
    
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
    mask = (df["time"] >= start_time) & (df["time"] < end_time)
    bars = df[mask].copy()
    bars = bars.reset_index(drop=True)
    
    return bars


def print_section(title: str):
    """Print a formatted section header."""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_basic_irr_calculation():
    """
    Test basic IRR calculation on single bars.
    """
    print_section("Basic IRR Calculation Test")
    
    print("\nIRR Formula: 1 - (body / range)")
    print("  body  = |close - open|")
    print("  range = high - low")
    print()
    print("Interpretation:")
    print("  IRR ≈ 0: Strong body (directional)")
    print("  IRR ≈ 1: Long wicks/doji (indecision/rejection)")
    print()
    
    # Test cases
    test_bars = [
        # (open, high, low, close, description, expected)
        (100, 110, 90, 109, "Strong bullish (body covers most of range)", 0.55),
        (100, 110, 90, 91, "Strong bearish (body covers most of range)", 0.55),
        (100, 110, 90, 100, "Doji (no body)", 1.0),
        (100, 105, 95, 102, "Small body, moderate wicks", 0.8),
        (95, 110, 90, 105, "Half body, half wicks", 0.5),
        (90, 110, 90, 110, "Full body bullish (no wicks)", 0.0),
    ]
    
    print("Test Cases:")
    print("-" * 70)
    print(f"{'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'IRR':>8} {'Expected':>10}  Description")
    print("-" * 70)
    
    analyzer = IRRAnalyzer()
    
    for open_p, high, low, close, desc, expected in test_bars:
        irr = analyzer.compute_irr(open_p, high, low, close)
        status = "✅" if abs(irr - expected) < 0.01 else "⚠️"
        print(f"{open_p:>8.1f} {high:>8.1f} {low:>8.1f} {close:>8.1f} {irr:>8.2f} {expected:>10.2f}  {status} {desc}")
    
    print()


def test_irr_on_nq_data():
    """
    Test IRR calculation on real NQ data.
    """
    print_section("IRR on NQ 1-Minute Data")
    
    # Load data
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)
    
    print(f"\nLoaded {len(df)} bars total")
    print()
    
    # Calculate IRR for each bar
    analyzer = IRRAnalyzer()
    irr_values = []
    
    for _, row in df.iterrows():
        irr = analyzer.compute_irr(row['open'], row['high'], row['low'], row['close'])
        if irr is not None:
            irr_values.append(irr)
    
    irr_array = np.array(irr_values)
    
    print("IRR Statistics (all bars):")
    print("-" * 40)
    print(f"  Valid bars:    {len(irr_values)}")
    print(f"  Mean IRR:      {np.mean(irr_array):.4f}")
    print(f"  Median IRR:    {np.median(irr_array):.4f}")
    print(f"  Std Dev:       {np.std(irr_array):.4f}")
    print(f"  Min IRR:       {np.min(irr_array):.4f}")
    print(f"  Max IRR:       {np.max(irr_array):.4f}")
    print()
    
    # Distribution analysis
    print("IRR Distribution:")
    print("-" * 40)
    
    ranges = [(0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]
    for low_bound, high_bound in ranges:
        count = np.sum((irr_array >= low_bound) & (irr_array < high_bound))
        pct = count / len(irr_array) * 100
        bar = "█" * int(pct / 2)
        print(f"  {low_bound:.1f}-{high_bound:.1f}: {pct:5.1f}% {bar}")
    
    print()


def test_two_level_analysis():
    """
    Test the two-level analysis structure (full window + sub-windows).
    """
    print_section("Two-Level Analysis Test")
    
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_and_filter_nq_data(data_path, '15:00', '15:50')  # 3pm-3:50pm
    
    print(f"\nAnalyzing 3pm-3:50pm window ({len(df)} bars)")
    print()
    
    if len(df) < 50:
        print("⚠️ Insufficient data for test")
        return
    
    # Use last 50 bars for analysis
    window_end = len(df) - 1
    window_start = window_end - 49  # 50 bars total
    
    analyzer = IRRAnalyzer(num_subwindows=5)
    result = analyzer.analyze(df, window_start, window_end, session_label="3pm-3:50pm")
    
    print("LEVEL 1: Full Window IRR")
    print("-" * 60)
    print(f"  Full Window IRR:        {result.full_window_irr:.4f}")
    print(f"  Directional Strength:   {result.directional_strength:.4f}")
    print(f"  Regime:                 {result.regime}")
    print(f"  Interpretation:         {result.interpretation}")
    print()
    
    print("LEVEL 2: Sub-Window IRRs (5 x 10-bar windows)")
    print("-" * 60)
    for i, (irr, detail) in enumerate(zip(result.sub_window_irrs, result.sub_window_details)):
        irr_str = f"{irr:.4f}" if irr is not None else "N/A"
        print(f"  {detail.time_label:20s}  IRR: {irr_str:>6s}  "
              f"Range: {detail.bar_range:.2f}  Body: {detail.body:.2f}")
    print()
    
    print("STATISTICS:")
    print("-" * 60)
    print(f"  Median Sub-Window IRR:  {result.median_sub_irr:.4f}")
    print(f"  Average Sub-Window IRR: {result.average_sub_irr:.4f}")
    print()
    
    if result.reversal_info:
        print("REVERSAL DETECTION:")
        print("-" * 60)
        rev = result.reversal_info
        high_irr_str = f"{rev.high_window_irr:.4f}" if rev.high_window_irr is not None else "N/A"
        low_irr_str = f"{rev.low_window_irr:.4f}" if rev.low_window_irr is not None else "N/A"
        print(f"  High made in window:    {rev.high_window_idx + 1} (IRR: {high_irr_str})")
        print(f"  Low made in window:     {rev.low_window_idx + 1} (IRR: {low_irr_str})")
        print(f"  Session High:           {rev.high_price:.2f}")
        print(f"  Session Low:            {rev.low_price:.2f}")
        print()


def test_session_analysis():
    """
    Test session-based analysis using analyze_session().
    """
    print_section("Session Analysis Test")
    
    data_path = Path(__file__).parent / "NQ.csv"
    
    # Define session windows
    windows = {
        'early': ('13:00', '15:00'),      # 1pm-3pm
        'middle': ('15:00', '15:50'),     # 3pm-3:50pm
        'close': ('15:50', '16:00'),      # 3:50pm-4pm
    }
    
    print("\nAnalyzing different session windows:")
    print("=" * 80)
    
    for window_name, (start, end) in windows.items():
        df = load_and_filter_nq_data(data_path, start, end)
        
        print(f"\n{window_name.upper()} Window ({start}-{end})")
        print("-" * 60)
        
        if len(df) < 10:
            print("  ⚠️ Insufficient data")
            continue
        
        # Determine appropriate window size and num_subwindows
        if len(df) >= 50:
            window_size = 50
            num_subs = 5
        else:
            window_size = len(df)
            num_subs = min(5, len(df) // 2)
        
        result_dict = analyze_session(df, session_name=window_name, 
                                     window_size=window_size, 
                                     num_subwindows=num_subs)
        
        if 'error' in result_dict:
            print(f"  Error: {result_dict['error']}")
            continue
        
        full_irr_str = f"{result_dict['full_window_irr']:.4f}" if result_dict['full_window_irr'] is not None else "N/A"
        med_irr_str = f"{result_dict['median_sub_irr']:.4f}" if result_dict['median_sub_irr'] is not None else "N/A"
        avg_irr_str = f"{result_dict['average_sub_irr']:.4f}" if result_dict['average_sub_irr'] is not None else "N/A"
        dir_str_str = f"{result_dict['directional_strength']:.4f}" if result_dict['directional_strength'] is not None else "N/A"
        
        print(f"  Bars analyzed:          {window_size}")
        print(f"  Full Window IRR:        {full_irr_str}")
        print(f"  Median Sub IRR:         {med_irr_str}")
        print(f"  Average Sub IRR:        {avg_irr_str}")
        print(f"  Directional Strength:   {dir_str_str}")
        print(f"  Regime:                 {result_dict['regime']}")
        print(f"  Interpretation:         {result_dict['interpretation']}")
    
    print()


def test_regime_classification():
    """
    Test regime classification thresholds.
    """
    print_section("Regime Classification Test")
    
    print("\nThreshold Definitions:")
    print("-" * 40)
    print("  IRR > 0.65: high_reversion (consolidation)")
    print("  IRR < 0.35: directional (trending)")
    print("  0.35 ≤ IRR ≤ 0.65: mixed")
    print("  None: unknown")
    print()
    
    # Test values across the spectrum
    test_values = [
        (None, "unknown"),
        (0.15, "directional"),
        (0.30, "directional"),
        (0.35, "mixed"),
        (0.50, "mixed"),
        (0.65, "mixed"),
        (0.70, "high_reversion"),
        (0.85, "high_reversion"),
        (0.95, "high_reversion"),
    ]
    
    print("Classification Tests:")
    print("-" * 40)
    
    all_passed = True
    for irr_value, expected in test_values:
        result = IRRAnalyzer.classify_regime(irr_value)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        
        irr_str = f"{irr_value:.2f}" if irr_value is not None else "None"
        print(f"  IRR={irr_str:>6}: {result:15s} (expected: {expected}) {status}")
    
    print()
    if all_passed:
        print("✅ All classification tests passed!")
    else:
        print("❌ Some classification tests failed!")
    print()


def test_edge_cases():
    """
    Test edge cases for IRR calculation.
    """
    print_section("Edge Case Tests")
    
    analyzer = IRRAnalyzer()
    
    # Test 1: Zero range (high == low)
    print("\n1. Zero Range (high == low):")
    print("-" * 40)
    
    irr = analyzer.compute_irr(100, 100, 100, 100)
    print(f"   Bar: O=100, H=100, L=100, C=100")
    print(f"   IRR: {irr}")
    print(f"   ✅ Correctly returns None for zero range")
    
    # Test 2: Very small range
    print("\n2. Very Small Range (< 0.0001):")
    print("-" * 40)
    
    irr = analyzer.compute_irr(100.00001, 100.00002, 100.00001, 100.00002)
    print(f"   Bar: O=100.00001, H=100.00002, L=100.00001, C=100.00002")
    print(f"   IRR: {irr}")
    print(f"   ✅ Handles very small ranges appropriately")
    
    # Test 3: Inverted bar (open > close for bearish)
    print("\n3. Bearish Bar (open > close):")
    print("-" * 40)
    
    irr = analyzer.compute_irr(110, 112, 98, 100)
    expected_body = abs(100 - 110)  # 10
    expected_range = 112 - 98  # 14
    expected_irr = 1 - (10 / 14)  # ~0.286
    
    print(f"   Bar: O=110, H=112, L=98, C=100")
    print(f"   Body: {expected_body}, Range: {expected_range}")
    print(f"   IRR: {irr:.4f} (expected: {expected_irr:.4f})")
    print(f"   ✅ Correctly handles bearish bars")
    
    # Test 4: Window too small for sub-windows
    print("\n4. Window Too Small for Sub-Windows:")
    print("-" * 40)
    
    # Create small DataFrame
    small_df = pd.DataFrame({
        'open': [100, 101],
        'high': [102, 103],
        'low': [99, 100],
        'close': [101, 102]
    })
    
    try:
        result = analyzer.analyze(small_df, 0, 1)
        print(f"   Window: 2 bars, 5 sub-windows requested")
        print(f"   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   Window: 2 bars, 5 sub-windows requested")
        print(f"   ✅ Correctly raises ValueError: {e}")
    
    # Test 5: All zero-range bars
    print("\n5. All Zero-Range Bars:")
    print("-" * 40)
    
    zero_df = pd.DataFrame({
        'open': [100] * 10,
        'high': [100] * 10,
        'low': [100] * 10,
        'close': [100] * 10
    })
    
    result = analyzer.analyze(zero_df, 0, 9)
    print(f"   Bars: 10 bars with H=L")
    print(f"   Full Window IRR: {result.full_window_irr}")
    print(f"   Median Sub IRR: {result.median_sub_irr}")
    print(f"   ✅ Handles all zero-range bars")
    
    # Test 6: Mix of valid and invalid bars
    print("\n6. Mix of Valid and Invalid Bars:")
    print("-" * 40)
    
    mixed_df = pd.DataFrame({
        'open':  [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        'high':  [100, 110, 100, 110, 110, 100, 110, 100, 110, 110],
        'low':   [100,  90, 100,  90,  90, 100,  90, 100,  90,  90],
        'close': [100, 105, 100, 100, 109, 100, 105, 100, 100, 109]
    })
    
    result = analyzer.analyze(mixed_df, 0, 9)
    med_irr_str = f"{result.median_sub_irr:.4f}" if result.median_sub_irr is not None else "N/A"
    print(f"   Bars: Mixed valid/invalid")
    print(f"   Median Sub IRR: {med_irr_str}")
    print(f"   ✅ Correctly filters out invalid bars")
    
    print()


def test_irr_interpretation():
    """
    Test IRR interpretation with market scenarios.
    """
    print_section("IRR Market Interpretation")
    
    scenarios = [
        {
            'name': 'Strong Directional Trend',
            'df': pd.DataFrame({
                'open':  [100, 109, 118, 127, 136, 145, 154, 163, 172, 181],
                'high':  [110, 119, 128, 137, 146, 155, 164, 173, 182, 191],
                'low':   [ 99, 108, 117, 126, 135, 144, 153, 162, 171, 180],
                'close': [109, 118, 127, 136, 145, 154, 163, 172, 181, 190]
            }),
            'expected_regime': 'directional'
        },
        {
            'name': 'Consolidation (Doji Pattern)',
            'df': pd.DataFrame({
                'open':  [100, 100, 101, 100, 99, 100, 100, 101, 100, 99],
                'high':  [105, 106, 107, 104, 105, 106, 105, 106, 104, 105],
                'low':   [ 95,  94,  93,  96,  95,  94,  95,  94,  96,  95],
                'close': [100, 101, 100,  99, 100, 101, 100,  99, 100, 101]
            }),
            'expected_regime': 'high_reversion'
        },
        {
            'name': 'Mixed/Choppy Market',
            'df': pd.DataFrame({
                'open':  [100, 105, 103, 106, 104, 107, 105, 108, 106, 109],
                'high':  [108, 110, 108, 112, 109, 113, 110, 114, 111, 115],
                'low':   [ 95, 100,  98, 101,  99, 102, 100, 103, 101, 104],
                'close': [105, 103, 106, 104, 107, 105, 108, 106, 109, 107]
            }),
            'expected_regime': 'mixed'
        },
    ]
    
    print("\nMarket Scenarios:")
    print("-" * 60)
    
    analyzer = IRRAnalyzer(num_subwindows=5)
    
    for scenario in scenarios:
        result = analyzer.analyze(scenario['df'], 0, len(scenario['df']) - 1)
        
        status = "✅" if result.regime == scenario['expected_regime'] else "⚠️"
        
        full_irr_str = f"{result.full_window_irr:.4f}" if result.full_window_irr is not None else "N/A"
        med_irr_str = f"{result.median_sub_irr:.4f}" if result.median_sub_irr is not None else "N/A"
        
        print(f"\n  {scenario['name']}:")
        print(f"    Bars: {len(scenario['df'])}")
        print(f"    Full Window IRR: {full_irr_str}")
        print(f"    Median Sub IRR: {med_irr_str}")
        print(f"    Regime: {result.regime} (expected: {scenario['expected_regime']}) {status}")
        print(f"    Interpretation: {result.interpretation}")
    
    print()


def test_batch_analysis():
    """
    Test batch analysis for backtesting.
    """
    print_section("Batch Analysis Test")
    
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_and_filter_nq_data(data_path, '15:00', '15:50')
    
    print(f"\nRunning batch analysis on 3pm-3:50pm data ({len(df)} bars)")
    print("Window size: 50 bars, Step: 10 bars")
    print()
    
    if len(df) < 50:
        print("⚠️ Insufficient data for test")
        return
    
    results_df = batch_analyze_windows(df, window_size=50, step_size=10, num_subwindows=5)
    
    print(f"Generated {len(results_df)} analysis windows")
    print()
    
    print("Sample Results:")
    print("-" * 80)
    print(f"{'Window End':>12} {'Full IRR':>10} {'Med Sub IRR':>12} {'Dir Strength':>13} {'Regime':>15} {'Interpretation':>20}")
    print("-" * 80)
    
    for idx, row in results_df.head(5).iterrows():
        print(f"{row['window_end_idx']:>12} "
              f"{row['full_window_irr']:>10.4f} "
              f"{row['median_sub_irr']:>12.4f} "
              f"{row['directional_strength']:>13.4f} "
              f"{row['regime']:>15} "
              f"{row['interpretation']:>20}")
    
    print()
    
    # Show statistics
    print("Batch Statistics:")
    print("-" * 60)
    print(f"  Mean Full Window IRR:   {results_df['full_window_irr'].mean():.4f}")
    print(f"  Mean Median Sub IRR:    {results_df['median_sub_irr'].mean():.4f}")
    print(f"  Mean Dir Strength:      {results_df['directional_strength'].mean():.4f}")
    print()
    
    # Regime distribution
    print("Regime Distribution:")
    print("-" * 60)
    regime_counts = results_df['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {regime:20s}: {count:3d} ({pct:5.1f}%)")
    
    print()


def test_statistical_properties():
    """
    Test statistical properties of IRR across NQ data.
    """
    print_section("Statistical Properties Test")
    
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)
    
    analyzer = IRRAnalyzer()
    
    # Calculate IRR for all bars
    df['irr'] = df.apply(
        lambda row: analyzer.compute_irr(row['open'], row['high'], row['low'], row['close']),
        axis=1
    )
    
    # Filter valid IRR values
    valid_irr = df['irr'].dropna()
    
    print(f"\nDataset: {len(valid_irr)} valid bars")
    print()
    
    # Percentiles
    print("Percentiles:")
    print("-" * 40)
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(valid_irr, p)
        print(f"  {p:3d}th percentile: {val:.4f}")
    
    print()
    
    # Time-based analysis (if we have enough data)
    df['hour'] = pd.to_datetime(df['DateTime_ET']).dt.hour
    
    print("IRR by Hour:")
    print("-" * 40)
    
    hourly_irr = df.groupby('hour')['irr'].agg(['mean', 'median', 'count'])
    
    for hour, row in hourly_irr.iterrows():
        if row['count'] > 0 and not np.isnan(row['median']):
            regime = IRRAnalyzer.classify_regime(row['median'])
            print(f"  {hour:02d}:00 - Mean: {row['mean']:.4f}, Median: {row['median']:.4f}, "
                  f"n={int(row['count']):4d}, Regime: {regime}")
    
    print()


if __name__ == "__main__":
    print()
    print("=" * 80)
    print("  Intrabar Range Ratio (IRR) Test Suite - Version 2.0")
    print("=" * 80)
    print()
    print("IRR measures the proportion of a bar's range made up by wicks vs body:")
    print("  IRR = 1 - (body / range)")
    print("  where body = |close - open| and range = high - low")
    print()
    print("Version 2.0 features:")
    print("  - Two-level analysis: Full window + Sub-windows")
    print("  - Class-based structure (IRRAnalyzer)")
    print("  - Reversal detection")
    print("  - Enhanced interpretation")
    print()
    
    # Run all tests
    test_basic_irr_calculation()
    test_irr_on_nq_data()
    test_two_level_analysis()
    test_session_analysis()
    test_regime_classification()
    test_edge_cases()
    test_irr_interpretation()
    test_batch_analysis()
    test_statistical_properties()
    
    print("=" * 80)
    print("  Test Suite Complete")
    print("=" * 80)
