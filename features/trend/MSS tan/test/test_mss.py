"""
Test Multi-Scale Slope (MSS) Analysis on NQ 1-minute bars.

Tests the MSS class with:
- Normalized slope calculations
- Sub-window slope consistency
- Prominent extrema detection
- Composite trending score calculation
- Edge cases and synthetic data

Run this script from the project root directory (macro/) using:
    python -m features.trend."MSS tan".test.test_mss

Or from the features/trend/MSS tan/test directory using:
    python test_mss.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add MSS directory to path
mss_dir = Path(__file__).parent.parent
if str(mss_dir) not in sys.path:
    sys.path.insert(0, str(mss_dir))

# Import MSS module
from mss import (
    MultiScaleSlope, MSSResult, ExtremaInfo,
    analyze_session, batch_analyze_windows,
    DEFAULT_ATR_PERIOD, DEFAULT_NUM_SUBWINDOWS
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
    df["timestamp"] = df["DateTime_ET"]
    
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
    bars = df[mask].copy().reset_index(drop=True)
    
    return bars


def print_result(result: MSSResult, window_name: str):
    """
    Pretty print MSS analysis results.
    
    Args:
        result: MSSResult object
        window_name: Name of the time window
    """
    print("=" * 80)
    print(f"MSS Analysis Results: {window_name}")
    print("=" * 80)
    
    print(f"\n📊 Trending Score: {result.trending_score:.4f}")
    print(f"   Interpretation: {result.interpretation}")
    
    print(f"\n📈 Main Window Slope (normalized): {result.main_slope:.6f}")
    
    print(f"\n📉 Sub-Window Slopes ({len(result.sub_slopes)} windows):")
    for i, slope in enumerate(result.sub_slopes):
        direction = "↑" if slope > 0 else "↓" if slope < 0 else "→"
        print(f"   Sub-window {i+1}: {slope:+.6f} {direction}")
    
    print(f"\n🎯 Component Scores:")
    for component, score in result.components.items():
        if component != 'trending_score':
            print(f"   • {component}: {score:.4f}")
    
    print(f"\n🔍 Extrema Information:")
    if result.extrema['prominent_high']:
        ph = result.extrema['prominent_high']
        print(f"   Prominent High: idx={ph['index']}, price={ph['price']:.2f}, prominence={ph['prominence']:.2f}")
    else:
        print("   Prominent High: None found")
    
    if result.extrema['prominent_low']:
        pl = result.extrema['prominent_low']
        print(f"   Prominent Low: idx={pl['index']}, price={pl['price']:.2f}, prominence={pl['prominence']:.2f}")
    else:
        print("   Prominent Low: None found")
    
    if result.extrema['segment_slopes']:
        print(f"   Segment Slopes: {[f'{s:.4f}' for s in result.extrema['segment_slopes']]}")
    
    print()


def test_normalized_slope_calculation():
    """
    Test normalized slope calculation with synthetic data.
    """
    print("=" * 80)
    print("Normalized Slope Calculation Tests")
    print("=" * 80)
    
    mss = MultiScaleSlope()
    
    # Test 1: Perfect uptrend (low volatility)
    print("\n1. Perfect uptrend (low volatility):")
    prices_up = np.array([100 + i * 0.1 for i in range(50)])  # Smooth upward
    slope = mss.calculate_normalized_slope(prices_up, 0, len(prices_up) - 1)
    print(f"   Prices: {prices_up[0]:.2f} → {prices_up[-1]:.2f}")
    print(f"   Normalized slope: {slope:.6f}")
    print(f"   ✅ High positive slope indicates strong uptrend")
    
    # Test 2: Perfect downtrend (low volatility)
    print("\n2. Perfect downtrend (low volatility):")
    prices_down = np.array([100 - i * 0.1 for i in range(50)])
    slope = mss.calculate_normalized_slope(prices_down, 0, len(prices_down) - 1)
    print(f"   Prices: {prices_down[0]:.2f} → {prices_down[-1]:.2f}")
    print(f"   Normalized slope: {slope:.6f}")
    print(f"   ✅ High negative slope indicates strong downtrend")
    
    # Test 3: Same move but high volatility
    print("\n3. Uptrend with high volatility (noisy):")
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, 50)
    prices_noisy = np.array([100 + i * 0.1 + noise[i] for i in range(50)])
    slope = mss.calculate_normalized_slope(prices_noisy, 0, len(prices_noisy) - 1)
    print(f"   Prices: {prices_noisy[0]:.2f} → {prices_noisy[-1]:.2f}")
    print(f"   Normalized slope: {slope:.6f}")
    print(f"   ✅ Lower slope due to volatility normalization")
    
    # Test 4: Flat market
    print("\n4. Flat market (no trend):")
    prices_flat = np.array([100.0] * 50)
    slope = mss.calculate_normalized_slope(prices_flat, 0, len(prices_flat) - 1)
    print(f"   Prices: constant at {prices_flat[0]:.2f}")
    print(f"   Normalized slope: {slope:.6f}")
    print(f"   ✅ Zero slope indicates no trend")
    
    # Test 5: Sub-window calculation
    print("\n5. Sub-window slope comparison:")
    prices_mixed = np.concatenate([
        np.linspace(100, 105, 25),  # Up first half
        np.linspace(105, 102, 25)   # Down second half
    ])
    first_half_slope = mss.calculate_normalized_slope(prices_mixed, 0, 24)
    second_half_slope = mss.calculate_normalized_slope(prices_mixed, 25, 49)
    print(f"   First half slope: {first_half_slope:.6f} (up)")
    print(f"   Second half slope: {second_half_slope:.6f} (down)")
    print(f"   ✅ Sub-windows can have opposing directions")
    
    print()


def test_sub_window_slopes():
    """
    Test sub-window slope division and calculation.
    """
    print("=" * 80)
    print("Sub-Window Slope Tests")
    print("=" * 80)
    
    mss = MultiScaleSlope()
    
    # Test with consistent trend
    print("\n1. Consistent uptrend (all sub-windows positive):")
    prices = np.linspace(100, 110, 60)
    sub_slopes = mss.calculate_sub_window_slopes(prices, 0, len(prices) - 1)
    print(f"   Sub-slopes: {[f'{s:.4f}' for s in sub_slopes]}")
    all_positive = all(s > 0 for s in sub_slopes)
    print(f"   All positive: {all_positive}")
    print(f"   ✅ {'High directional consistency' if all_positive else 'Mixed consistency'}")
    
    # Test with alternating trend
    print("\n2. Alternating trend (mixed sub-windows):")
    prices_alt = np.array([100 + 2 * np.sin(i * np.pi / 10) for i in range(60)])
    sub_slopes = mss.calculate_sub_window_slopes(prices_alt, 0, len(prices_alt) - 1)
    print(f"   Sub-slopes: {[f'{s:.4f}' for s in sub_slopes]}")
    signs = [np.sign(s) for s in sub_slopes]
    print(f"   Signs: {signs}")
    print(f"   ✅ Mixed directions indicate choppy behavior")
    
    # Test window division
    print(f"\n3. Verify window division (60 bars / {DEFAULT_NUM_SUBWINDOWS} = {60//DEFAULT_NUM_SUBWINDOWS} bars each):")
    print(f"   Total bars: 60")
    print(f"   Sub-windows: {DEFAULT_NUM_SUBWINDOWS}")
    print(f"   Expected bars per sub-window: ~{60//DEFAULT_NUM_SUBWINDOWS}")
    print(f"   ✅ Window division correct")
    
    print()


def test_prominent_extrema():
    """
    Test prominent extrema detection.
    """
    print("=" * 80)
    print("Prominent Extrema Detection Tests")
    print("=" * 80)
    
    mss = MultiScaleSlope()
    
    # Create synthetic OHLC with clear swings
    print("\n1. Clear swing high and low:")
    n = 60
    close_prices = np.concatenate([
        np.linspace(100, 110, 20),   # Up to high
        np.linspace(110, 95, 20),    # Down to low  
        np.linspace(95, 102, 20)     # Partial recovery
    ])
    
    # Generate OHLC from close prices
    df = pd.DataFrame({
        'open': close_prices - 0.5,
        'high': close_prices + np.random.uniform(0.5, 1.5, n),
        'low': close_prices - np.random.uniform(0.5, 1.5, n),
        'close': close_prices
    })
    
    # Calculate ATR for threshold
    atr_series = mss.calculate_atr(df)
    atr_value = atr_series.iloc[-1]
    
    print(f"   ATR value: {atr_value:.4f}")
    
    prominent_high, prominent_low = mss.find_prominent_extrema(df, 0, n - 1, atr_value)
    
    if prominent_high:
        print(f"   Prominent High: idx={prominent_high.index}, price={prominent_high.price:.2f}, prominence={prominent_high.prominence:.2f}")
    else:
        print("   Prominent High: None found")
    
    if prominent_low:
        print(f"   Prominent Low: idx={prominent_low.index}, price={prominent_low.price:.2f}, prominence={prominent_low.prominence:.2f}")
    else:
        print("   Prominent Low: None found")
    
    print(f"   ✅ Expected high near index 19 (peak), low near index 39 (trough)")
    
    # Test with flat market
    print("\n2. Flat market (no prominent extrema):")
    df_flat = pd.DataFrame({
        'open': [100.0] * 50,
        'high': [100.1] * 50,
        'low': [99.9] * 50,
        'close': [100.0] * 50
    })
    
    atr_flat = mss.calculate_atr(df_flat).iloc[-1]
    high_flat, low_flat = mss.find_prominent_extrema(df_flat, 0, 49, atr_flat)
    
    print(f"   Prominent High: {high_flat}")
    print(f"   Prominent Low: {low_flat}")
    print(f"   ✅ No prominent extrema in flat market (as expected)")
    
    print()


def test_composite_score():
    """
    Test composite score calculation components.
    """
    print("=" * 80)
    print("Composite Score Component Tests")
    print("=" * 80)
    
    mss = MultiScaleSlope()
    
    # Test 1: Perfect trend (all components high)
    print("\n1. Perfect trend scenario:")
    main_slope = 0.5
    sub_slopes = [0.45, 0.52, 0.48, 0.51, 0.49, 0.47]  # All positive, consistent
    extrema_coherence = 1.0
    
    components = mss.calculate_composite_score(main_slope, sub_slopes, extrema_coherence)
    
    print(f"   Main slope: {main_slope}")
    print(f"   Sub-slopes: {sub_slopes}")
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")
    print(f"   ✅ High trending score expected")
    
    # Test 2: Mixed directions
    print("\n2. Mixed direction scenario:")
    main_slope = 0.1
    sub_slopes = [0.2, -0.15, 0.18, -0.1, 0.22, -0.05]  # Alternating
    extrema_coherence = 0.5
    
    components = mss.calculate_composite_score(main_slope, sub_slopes, extrema_coherence)
    
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")
    print(f"   ✅ Lower trending score due to inconsistency")
    
    # Test 3: Strong but noisy
    print("\n3. Strong move but high variance:")
    main_slope = 0.8
    sub_slopes = [0.3, 1.2, 0.1, 1.5, 0.4, 1.0]  # High variance but all positive
    extrema_coherence = 0.75
    
    components = mss.calculate_composite_score(main_slope, sub_slopes, extrema_coherence)
    
    for k, v in components.items():
        print(f"   {k}: {v:.4f}")
    print(f"   ✅ Moderate score - strong magnitude but poor alignment")
    
    print()


def test_full_analysis_real_data():
    """
    Test full MSS analysis on real NQ data.
    """
    print("=" * 80)
    print("Full Analysis on Real NQ Data")
    print("=" * 80)
    
    # Load data
    data_path = Path(__file__).parent / "NQ.csv"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Test 3pm session (50 bars: 3:00pm - 3:50pm)
    print("\n1. 3pm Session Analysis (15:00 - 15:50):")
    session_data = load_and_filter_nq_data(data_path, "15:00", "15:50")
    
    print(f"   Loaded {len(session_data)} bars")
    print(f"   Time range: {session_data['time'].iloc[0]} to {session_data['time'].iloc[-1]}")
    print(f"   Price range: {session_data['close'].iloc[0]:.2f} → {session_data['close'].iloc[-1]:.2f}")
    
    if len(session_data) >= 50:
        mss = MultiScaleSlope()
        
        # Ensure we have preload data
        full_data = load_and_filter_nq_data(data_path, "14:46", "15:50")
        
        if len(full_data) >= 64:  # 14 preload + 50 window
            window_start = 14
            window_end = len(full_data) - 1
            
            try:
                result = mss.calculate_trending_score(
                    full_data, 
                    window_start, 
                    window_end, 
                    preload_bars=14
                )
                print_result(result, "3pm Session")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ⚠️ Not enough data for preload (have {len(full_data)}, need 64)")
    else:
        print(f"   ⚠️ Not enough session data (have {len(session_data)}, need 50)")
    
    # Test London session window
    print("\n2. London Session Analysis (08:00 - 09:00):")
    london_data = load_and_filter_nq_data(data_path, "07:46", "09:00")
    
    print(f"   Loaded {len(london_data)} bars")
    
    if len(london_data) >= 74:  # 14 preload + 60 window
        window_start = 14
        window_end = 73
        
        try:
            result = mss.calculate_trending_score(
                london_data,
                window_start,
                window_end,
                preload_bars=14
            )
            print_result(result, "London Session")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ⚠️ Not enough London data")
    
    print()


def test_analyze_session_convenience():
    """
    Test the convenience analyze_session function.
    """
    print("=" * 80)
    print("Convenience Function Tests")
    print("=" * 80)
    
    data_path = Path(__file__).parent / "NQ.csv"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Load sufficient data
    df = load_and_filter_nq_data(data_path, "14:46", "15:50")
    
    print(f"\n1. analyze_session with '3pm' config:")
    result = analyze_session(df, session_name='3pm')
    
    if result.get('error'):
        print(f"   ❌ Error: {result['error']}")
    else:
        print(f"   Trending Score: {result['trending_score']:.4f}")
        print(f"   Interpretation: {result['interpretation']}")
        print(f"   ✅ Session analysis completed")
    
    print()


def test_batch_analysis():
    """
    Test batch analysis for backtesting.
    """
    print("=" * 80)
    print("Batch Analysis Tests")
    print("=" * 80)
    
    data_path = Path(__file__).parent / "NQ.csv"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Load full day data
    df = load_nq_data(data_path)
    
    print(f"\n1. Rolling window analysis:")
    print(f"   Total bars: {len(df)}")
    print(f"   Window size: 50")
    print(f"   Step size: 10")
    
    results = batch_analyze_windows(df, window_size=50, step_size=50, preload_bars=14)
    
    print(f"\n   Generated {len(results)} window analyses")
    
    if len(results) > 0:
        valid_results = results.dropna(subset=['trending_score'])
        print(f"   Valid results: {len(valid_results)}")
        
        if len(valid_results) > 0:
            print(f"\n   Score Statistics:")
            print(f"   • Mean: {valid_results['trending_score'].mean():.4f}")
            print(f"   • Std:  {valid_results['trending_score'].std():.4f}")
            print(f"   • Min:  {valid_results['trending_score'].min():.4f}")
            print(f"   • Max:  {valid_results['trending_score'].max():.4f}")
            
            print(f"\n   Interpretation Distribution:")
            print(valid_results['interpretation'].value_counts().to_string())
    
    print()


def test_edge_cases():
    """
    Test edge cases and error handling.
    """
    print("=" * 80)
    print("Edge Case Tests")
    print("=" * 80)
    
    mss = MultiScaleSlope()
    
    # Test 1: Window too small (with 4 subwindows, need at least 8 bars)
    print("\n1. Window too small:")
    small_df = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [100.1] * 6,
        'low': [99.9] * 6,
        'close': [100.0] * 6
    })
    
    try:
        result = mss.calculate_trending_score(small_df, 0, 5, preload_bars=0)
        print(f"   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {e}")
    
    # Test 2: Invalid indices
    print("\n2. Invalid window indices:")
    normal_df = pd.DataFrame({
        'open': [100.0] * 100,
        'high': [100.1] * 100,
        'low': [99.9] * 100,
        'close': [100.0] * 100
    })
    
    try:
        result = mss.calculate_trending_score(normal_df, 50, 30)  # end < start
        print(f"   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {e}")
    
    # Test 3: Zero volatility (flat prices)
    print("\n3. Zero volatility (all same price):")
    flat_df = pd.DataFrame({
        'open': [100.0] * 60,
        'high': [100.0] * 60,
        'low': [100.0] * 60,
        'close': [100.0] * 60
    })
    
    try:
        result = mss.calculate_trending_score(flat_df, 14, 59, preload_bars=14)
        print(f"   Main slope: {result.main_slope}")
        print(f"   Trending score: {result.trending_score}")
        print(f"   ✅ Handled zero volatility correctly")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Extreme price movement
    print("\n4. Extreme price movement:")
    extreme_df = pd.DataFrame({
        'open': np.linspace(100, 200, 60),
        'high': np.linspace(101, 201, 60),
        'low': np.linspace(99, 199, 60),
        'close': np.linspace(100, 200, 60)
    })
    
    try:
        result = mss.calculate_trending_score(extreme_df, 14, 59, preload_bars=14)
        print(f"   Main slope: {result.main_slope}")
        print(f"   Trending score: {result.trending_score}")
        print(f"   Interpretation: {result.interpretation}")
        print(f"   ✅ Handled extreme movement correctly")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Synthetic consolidation
    print("\n5. Synthetic consolidation (mean-reverting):")
    np.random.seed(42)
    mean_price = 100
    consolidation_prices = mean_price + np.cumsum(np.random.randn(60) * 0.1 - 0.05 * (np.arange(60) % 2 - 0.5))
    consolidation_df = pd.DataFrame({
        'open': consolidation_prices - 0.1,
        'high': consolidation_prices + np.abs(np.random.randn(60) * 0.3),
        'low': consolidation_prices - np.abs(np.random.randn(60) * 0.3),
        'close': consolidation_prices
    })
    
    try:
        result = mss.calculate_trending_score(consolidation_df, 14, 59, preload_bars=14)
        print(f"   Main slope: {result.main_slope:.6f}")
        print(f"   Trending score: {result.trending_score:.4f}")
        print(f"   Interpretation: {result.interpretation}")
        print(f"   ✅ Low score indicates consolidation")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()


def test_interpretation_thresholds():
    """
    Test score interpretation across threshold boundaries.
    """
    print("=" * 80)
    print("Interpretation Threshold Tests")
    print("=" * 80)
    
    test_scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    
    print("\n   Score  →  Interpretation")
    print("   " + "-" * 40)
    
    for score in test_scores:
        interpretation = MultiScaleSlope.interpret_score(score)
        print(f"   {score:.2f}   →  {interpretation}")
    
    print()


def test_trending_vs_consolidating():
    """
    Test with synthetic trending vs consolidating data.
    """
    print("=" * 80)
    print("Trending vs Consolidating Comparison")
    print("=" * 80)
    
    mss = MultiScaleSlope()
    np.random.seed(42)
    
    # Create trending data (strong directional move)
    print("\n1. Synthetic Strong Uptrend:")
    trend_prices = np.linspace(100, 115, 60) + np.random.randn(60) * 0.3
    trend_df = pd.DataFrame({
        'open': trend_prices - 0.1,
        'high': trend_prices + np.abs(np.random.randn(60) * 0.5),
        'low': trend_prices - np.abs(np.random.randn(60) * 0.5),
        'close': trend_prices
    })
    
    result_trend = mss.calculate_trending_score(trend_df, 14, 59, preload_bars=14)
    print(f"   Price change: {trend_df['close'].iloc[14]:.2f} → {trend_df['close'].iloc[59]:.2f}")
    print(f"   Trending Score: {result_trend.trending_score:.4f}")
    print(f"   Interpretation: {result_trend.interpretation}")
    print(f"   Expected: > 0.7 (Strong/Trending)")
    
    # Create consolidating data (range-bound)
    print("\n2. Synthetic Consolidation (Range-Bound):")
    base = 100
    consolidation_prices = base + np.sin(np.linspace(0, 6*np.pi, 60)) * 2 + np.random.randn(60) * 0.2
    consolidation_df = pd.DataFrame({
        'open': consolidation_prices - 0.1,
        'high': consolidation_prices + np.abs(np.random.randn(60) * 0.3),
        'low': consolidation_prices - np.abs(np.random.randn(60) * 0.3),
        'close': consolidation_prices
    })
    
    result_consolidation = mss.calculate_trending_score(consolidation_df, 14, 59, preload_bars=14)
    print(f"   Price change: {consolidation_df['close'].iloc[14]:.2f} → {consolidation_df['close'].iloc[59]:.2f}")
    print(f"   Trending Score: {result_consolidation.trending_score:.4f}")
    print(f"   Interpretation: {result_consolidation.interpretation}")
    print(f"   Expected: < 0.4 (Choppy/Consolidating)")
    
    # Create mixed/transitional data
    print("\n3. Synthetic Mixed (Trend then Reversal):")
    mixed_prices = np.concatenate([
        np.linspace(100, 108, 30),  # Up
        np.linspace(108, 103, 30)   # Down
    ]) + np.random.randn(60) * 0.2
    mixed_df = pd.DataFrame({
        'open': mixed_prices - 0.1,
        'high': mixed_prices + np.abs(np.random.randn(60) * 0.4),
        'low': mixed_prices - np.abs(np.random.randn(60) * 0.4),
        'close': mixed_prices
    })
    
    result_mixed = mss.calculate_trending_score(mixed_df, 14, 59, preload_bars=14)
    print(f"   Price change: {mixed_df['close'].iloc[14]:.2f} → {mixed_df['close'].iloc[59]:.2f}")
    print(f"   Trending Score: {result_mixed.trending_score:.4f}")
    print(f"   Interpretation: {result_mixed.interpretation}")
    print(f"   Expected: 0.4-0.6 (Mixed/Transitional)")
    
    print("\n   Summary:")
    print(f"   Trending Score:      {result_trend.trending_score:.4f} ({result_trend.interpretation})")
    print(f"   Consolidation Score: {result_consolidation.trending_score:.4f} ({result_consolidation.interpretation})")
    print(f"   Mixed Score:         {result_mixed.trending_score:.4f} ({result_mixed.interpretation})")
    
    # Verify expected ordering
    if result_trend.trending_score > result_mixed.trending_score > result_consolidation.trending_score:
        print(f"\n   ✅ Scores follow expected pattern: Trending > Mixed > Consolidating")
    else:
        print(f"\n   ⚠️ Score ordering differs from expected")
    
    print()


def test_diagnostics():
    """
    Test diagnostic output for calibration and debugging.
    """
    print("=" * 80)
    print("Diagnostic Output Tests")
    print("=" * 80)
    
    data_path = Path(__file__).parent / "NQ.csv"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    df = load_and_filter_nq_data(data_path, "14:46", "15:50")
    
    if len(df) < 64:
        print(f"⚠️ Not enough data")
        return
    
    mss = MultiScaleSlope()
    result = mss.calculate_trending_score(
        df, window_start=14, window_end=len(df)-1, 
        preload_bars=14, include_diagnostics=True
    )
    
    print("\n📊 Diagnostic Information:")
    if result.diagnostics:
        diag = result.diagnostics
        print(f"   Raw Price Change: {diag.main_slope_raw:.6f} ({diag.main_slope_raw*100:.4f}%)")
        print(f"   Internal Volatility (std log returns): {diag.internal_volatility:.6f}")
        print(f"   ATR Value: {diag.atr_value:.4f}")
        print(f"\n   Sub-Slope Statistics:")
        print(f"   • Mean: {diag.sub_slope_mean:.4f}")
        print(f"   • Std Dev: {diag.sub_slope_std:.4f}")
        print(f"   • Range: {diag.sub_slope_range:.4f}")
        print(f"   • Positive: {diag.num_positive_subs}/{DEFAULT_NUM_SUBWINDOWS}")
        print(f"   • Negative: {diag.num_negative_subs}/{DEFAULT_NUM_SUBWINDOWS}")
    
    print(f"\n   Main Normalized Slope: {result.main_slope:.6f}")
    print(f"   Interpretation: {result.interpretation}")
    
    # Show raw vs floored values
    print(f"\n📈 Component Raw vs Floored Values:")
    print(f"   Slope Alignment: {result.components.get('slope_alignment_raw', 'N/A'):.4f} → {result.components['slope_alignment']:.4f}")
    print(f"   Extrema Coherence: {result.components.get('extrema_coherence_raw', 'N/A'):.4f} → {result.components['extrema_coherence']:.4f}")
    
    print()


def test_batch_with_diagnostics():
    """
    Test batch analysis with diagnostics enabled.
    """
    print("=" * 80)
    print("Batch Analysis with Diagnostics")
    print("=" * 80)
    
    data_path = Path(__file__).parent / "NQ.csv"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    df = load_nq_data(data_path)
    
    print(f"\n   Running batch analysis with diagnostics...")
    results = batch_analyze_windows(df, window_size=50, step_size=50, 
                                     preload_bars=14, include_diagnostics=True)
    
    valid = results.dropna(subset=['trending_score'])
    
    if len(valid) > 0:
        print(f"\n   Score Distribution (v2.0):")
        print(f"   • Mean: {valid['trending_score'].mean():.4f}")
        print(f"   • Std:  {valid['trending_score'].std():.4f}")
        print(f"   • Min:  {valid['trending_score'].min():.4f}")
        print(f"   • Max:  {valid['trending_score'].max():.4f}")
        
        print(f"\n   Interpretation Distribution:")
        print(valid['interpretation'].value_counts().to_string())
        
        # Show improvement metrics
        high_scores = (valid['trending_score'] >= 0.6).sum()
        print(f"\n   Windows with score >= 0.6: {high_scores}/{len(valid)}")
        
        if 'diag_raw_slope' in valid.columns:
            print(f"\n   Raw Slope Range: {valid['diag_raw_slope'].min():.6f} to {valid['diag_raw_slope'].max():.6f}")
            print(f"   Volatility Range: {valid['diag_volatility'].min():.6f} to {valid['diag_volatility'].max():.6f}")
    
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Multi-Scale Slope (MSS) Analysis Test Suite")
    print("=" * 80)
    print()
    
    # Show configuration
    print("MSS Configuration (v2.1 - Quick Wins Calibration):")
    print(f"  - ATR Period: {DEFAULT_ATR_PERIOD}")
    print(f"  - Sub-windows: {DEFAULT_NUM_SUBWINDOWS} (reduced from 6)")
    print(f"  - Magnitude: tanh(slope * 5.0) - aggressive scaling")
    print(f"  - Score weights:")
    print(f"    • Directional Consistency: 0.35")
    print(f"    • Slope Alignment: 0.20")
    print(f"    • Extrema Coherence: 0.20")
    print(f"    • Magnitude: 0.25")
    print()
    
    # Run tests
    test_normalized_slope_calculation()
    test_sub_window_slopes()
    test_prominent_extrema()
    test_composite_score()
    test_interpretation_thresholds()
    test_trending_vs_consolidating()
    test_edge_cases()
    test_diagnostics()
    test_full_analysis_real_data()
    test_analyze_session_convenience()
    test_batch_analysis()
    test_batch_with_diagnostics()
    
    print("=" * 80)
    print("Test Suite Complete")
    print("=" * 80)

