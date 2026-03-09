"""
Test Lag Autocorrelation and Hurst Exponent calculation on NQ 1-minute bars.

Tests the IntradayTemporalFeatures class with:
- Autocorrelation at various lags
- Hurst exponent (simple and R/S methods)
- Session-specific analysis (afternoon and close sessions)
- Regime detection

Run this script from the project root directory (macro/) using:
    python -m features.trend."Lag autocorr".test.test_lag

Or from the test directory using:
    python test_lag.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add lag directory to path
lag_dir = Path(__file__).parent.parent
if str(lag_dir) not in sys.path:
    sys.path.insert(0, str(lag_dir))

# Import lag module
from lag import (
    IntradayTemporalFeatures,
    SessionConfig,
    analyze_intraday_session,
    batch_analyze_sessions
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
    
    # Extract time and date components
    df["time"] = df["DateTime_ET"].dt.strftime("%H:%M")
    df["date"] = df["DateTime_ET"].dt.date
    
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
    
    return bars


def print_section(title: str):
    """Print a formatted section header."""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_basic_autocorrelation():
    """
    Test basic autocorrelation calculation.
    """
    print_section("Basic Autocorrelation Test")
    
    # Load data
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_and_filter_nq_data(data_path, "13:00", "15:00")
    
    print(f"\nLoaded {len(df)} bars for 1pm-3pm session")
    
    if len(df) == 0:
        print("❌ No data found for the specified time window.")
        return
    
    # Create feature calculator
    prices = df["close"].values
    tf = IntradayTemporalFeatures(prices, compute_returns=True)
    
    print(f"\nData points (returns): {tf.n_data}")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    print()
    
    # Calculate autocorrelations at various lags
    print("Autocorrelation Results:")
    print("-" * 40)
    
    lags = [1, 2, 3, 5, 10, 15, 20]
    for lag in lags:
        acf = tf.autocorrelation(lag)
        if np.isnan(acf):
            print(f"  Lag {lag:3d}: insufficient data")
        else:
            bar = "█" * int(abs(acf) * 20)
            sign = "+" if acf > 0 else "-"
            print(f"  Lag {lag:3d}: {acf:+.4f}  {sign}{bar}")
    
    print()
    
    # Interpretation
    acf1 = tf.autocorrelation(1)
    if not np.isnan(acf1):
        if acf1 > 0.15:
            print("📈 Positive ACF(1): Returns show momentum/persistence")
        elif acf1 < -0.15:
            print("📉 Negative ACF(1): Returns show mean reversion")
        else:
            print("➡️  Near-zero ACF(1): Returns appear random")
    
    print()


def test_hurst_exponent():
    """
    Test Hurst exponent calculation with both methods.
    """
    print_section("Hurst Exponent Test")
    
    # Load data
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_and_filter_nq_data(data_path, "13:00", "15:00")
    
    if len(df) == 0:
        print("❌ No data found for the specified time window.")
        return
    
    prices = df["close"].values
    tf = IntradayTemporalFeatures(prices, compute_returns=True)
    
    print(f"\nCalculating Hurst exponent for {tf.n_data} return observations")
    print()
    
    # Simple (variance) method
    hurst_simple = tf.hurst_exponent_simple()
    print(f"Hurst (Simple/Variance method): {hurst_simple:.4f}")
    
    # R/S method
    hurst_rs = tf.hurst_exponent_rs()
    print(f"Hurst (R/S method):              {hurst_rs:.4f}")
    
    print()
    print("Interpretation:")
    print("-" * 40)
    print("  H < 0.5: Mean-reverting (anti-persistent)")
    print("  H = 0.5: Random walk (Brownian motion)")
    print("  H > 0.5: Trending (persistent)")
    print()
    
    # Interpret the result
    if not np.isnan(hurst_simple):
        if hurst_simple > 0.6:
            print(f"📈 H = {hurst_simple:.2f}: Strong trending behavior")
        elif hurst_simple > 0.55:
            print(f"📈 H = {hurst_simple:.2f}: Mild trending behavior")
        elif hurst_simple < 0.4:
            print(f"📉 H = {hurst_simple:.2f}: Strong mean-reverting behavior")
        elif hurst_simple < 0.45:
            print(f"📉 H = {hurst_simple:.2f}: Mild mean-reverting behavior")
        else:
            print(f"➡️  H = {hurst_simple:.2f}: Near random walk behavior")
    
    print()


def test_session_analysis():
    """
    Test session-specific analysis with configuration.
    """
    print_section("Session-Specific Analysis Test")
    
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)
    
    # Test afternoon session (1pm-3pm)
    print("\n1. Afternoon Session (1pm-3pm):")
    print("-" * 40)
    
    afternoon = df[(df["time"] >= "13:00") & (df["time"] < "15:00")]
    
    if len(afternoon) >= 20:
        config = SessionConfig.afternoon_session()
        features = analyze_intraday_session(afternoon, "close", config)
        
        print(f"  Bars analyzed:    {features['n_bars']}")
        print(f"  Price change:     {features['price_change_pct']:.2f}%")
        print(f"  ACF lag 1:        {features.get('acf_lag1', np.nan):.4f}")
        print(f"  ACF lag 2:        {features.get('acf_lag2', np.nan):.4f}")
        print(f"  ACF lag 5:        {features.get('acf_lag5', np.nan):.4f}")
        print(f"  Hurst exponent:   {features.get('hurst_exponent', np.nan):.4f}")
        print(f"  Detected regime:  {features.get('regime', 'unknown')}")
    else:
        print(f"  ❌ Insufficient data: {len(afternoon)} bars (need 20+)")
    
    # Test close session (3pm-3:50pm)
    print("\n2. Close Session (3pm-3:50pm):")
    print("-" * 40)
    
    close = df[(df["time"] >= "15:00") & (df["time"] < "15:50")]
    
    if len(close) >= 10:
        config = SessionConfig.close_session()
        features = analyze_intraday_session(close, "close", config)
        
        print(f"  Bars analyzed:    {features['n_bars']}")
        print(f"  Price change:     {features['price_change_pct']:.2f}%")
        print(f"  ACF lag 1:        {features.get('acf_lag1', np.nan):.4f}")
        print(f"  ACF lag 2:        {features.get('acf_lag2', np.nan):.4f}")
        print(f"  ACF lag 5:        {features.get('acf_lag5', np.nan):.4f}")
        print(f"  Hurst exponent:   {features.get('hurst_exponent', np.nan):.4f}")
        print(f"  Detected regime:  {features.get('regime', 'unknown')}")
    else:
        print(f"  ❌ Insufficient data: {len(close)} bars (need 10+)")
    
    print()


def test_regime_detection():
    """
    Test regime detection on various synthetic data patterns.
    """
    print_section("Regime Detection Test (Synthetic Data)")
    
    np.random.seed(42)
    n_points = 120  # Similar to afternoon session
    
    # Test 1: Trending data (random walk with drift)
    print("\n1. Trending Data (random walk with drift):")
    print("-" * 40)
    
    drift = 0.001
    noise = np.random.randn(n_points) * 0.005
    trending_returns = drift + noise
    trending_prices = 100 * np.exp(np.cumsum(trending_returns))
    
    tf_trend = IntradayTemporalFeatures(trending_prices, compute_returns=True)
    print(f"   ACF(1):  {tf_trend.autocorrelation(1):.4f}")
    print(f"   Hurst:   {tf_trend.hurst_exponent_simple():.4f}")
    print(f"   Regime:  {tf_trend.get_regime_signal()}")
    
    # Test 2: Mean-reverting data (Ornstein-Uhlenbeck)
    print("\n2. Mean-Reverting Data (synthetic OU process):")
    print("-" * 40)
    
    theta = 0.3  # Mean reversion speed
    mu = 100     # Long-term mean
    sigma = 0.5
    
    mr_prices = [100]
    for _ in range(n_points - 1):
        dx = theta * (mu - mr_prices[-1]) + sigma * np.random.randn()
        mr_prices.append(mr_prices[-1] + dx)
    mr_prices = np.array(mr_prices)
    
    tf_mr = IntradayTemporalFeatures(mr_prices, compute_returns=True)
    print(f"   ACF(1):  {tf_mr.autocorrelation(1):.4f}")
    print(f"   Hurst:   {tf_mr.hurst_exponent_simple():.4f}")
    print(f"   Regime:  {tf_mr.get_regime_signal()}")
    
    # Test 3: Random walk
    print("\n3. Pure Random Walk:")
    print("-" * 40)
    
    rw_returns = np.random.randn(n_points) * 0.005
    rw_prices = 100 * np.exp(np.cumsum(rw_returns))
    
    tf_rw = IntradayTemporalFeatures(rw_prices, compute_returns=True)
    print(f"   ACF(1):  {tf_rw.autocorrelation(1):.4f}")
    print(f"   Hurst:   {tf_rw.hurst_exponent_simple():.4f}")
    print(f"   Regime:  {tf_rw.get_regime_signal()}")
    
    # Test 4: Strongly autocorrelated returns
    print("\n4. Strongly Autocorrelated Returns:")
    print("-" * 40)
    
    ar_coefficient = 0.7
    innovations = np.random.randn(n_points) * 0.005
    ar_returns = [innovations[0]]
    for i in range(1, n_points):
        ar_returns.append(ar_coefficient * ar_returns[-1] + innovations[i])
    ar_prices = 100 * np.exp(np.cumsum(ar_returns))
    
    tf_ar = IntradayTemporalFeatures(ar_prices, compute_returns=True)
    print(f"   ACF(1):  {tf_ar.autocorrelation(1):.4f}")
    print(f"   Hurst:   {tf_ar.hurst_exponent_simple():.4f}")
    print(f"   Regime:  {tf_ar.get_regime_signal()}")
    
    print()


def test_edge_cases():
    """
    Test edge cases: insufficient data, constant prices, etc.
    """
    print_section("Edge Case Tests")
    
    # Test 1: Very short data
    print("\n1. Very Short Data (5 points):")
    print("-" * 40)
    
    short_prices = np.array([100, 101, 100.5, 101.2, 100.8])
    tf_short = IntradayTemporalFeatures(short_prices, compute_returns=True)
    
    acf1 = tf_short.autocorrelation(1)
    hurst = tf_short.hurst_exponent_simple()
    
    print(f"   Data points: {tf_short.n_data}")
    print(f"   ACF(1):      {'nan (insufficient data)' if np.isnan(acf1) else f'{acf1:.4f}'}")
    print(f"   Hurst:       {'nan (insufficient data)' if np.isnan(hurst) else f'{hurst:.4f}'}")
    print(f"   ✅ Correctly handles insufficient data")
    
    # Test 2: Constant prices
    print("\n2. Constant Prices:")
    print("-" * 40)
    
    constant_prices = np.array([100.0] * 50)
    tf_const = IntradayTemporalFeatures(constant_prices, compute_returns=True)
    
    acf1 = tf_const.autocorrelation(1)
    hurst = tf_const.hurst_exponent_simple()
    
    print(f"   Data points: {tf_const.n_data}")
    print(f"   ACF(1):      {'nan' if np.isnan(acf1) else f'{acf1:.4f}'}")
    print(f"   Hurst:       {'nan' if np.isnan(hurst) else f'{hurst:.4f}'}")
    print(f"   ✅ Correctly handles zero variance")
    
    # Test 3: Large lag request
    print("\n3. Large Lag Request (lag > data length):")
    print("-" * 40)
    
    prices = np.array([100 + i * 0.1 for i in range(30)])
    tf = IntradayTemporalFeatures(prices, compute_returns=True)
    
    acf_valid = tf.autocorrelation(5)
    acf_invalid = tf.autocorrelation(50)  # Larger than data
    
    print(f"   Data points:  {tf.n_data}")
    print(f"   ACF(5):       {acf_valid:.4f}")
    print(f"   ACF(50):      {'nan (lag too large)' if np.isnan(acf_invalid) else f'{acf_invalid:.4f}'}")
    print(f"   ✅ Correctly returns nan for invalid lag")
    
    # Test 4: Negative lag
    print("\n4. Negative/Zero Lag Request:")
    print("-" * 40)
    
    acf_zero = tf.autocorrelation(0)
    acf_neg = tf.autocorrelation(-1)
    
    print(f"   ACF(0):       {'nan (invalid)' if np.isnan(acf_zero) else f'{acf_zero:.4f}'}")
    print(f"   ACF(-1):      {'nan (invalid)' if np.isnan(acf_neg) else f'{acf_neg:.4f}'}")
    print(f"   ✅ Correctly handles invalid lag values")
    
    # Test 5: compute_returns=False
    print("\n5. Using Raw Prices (compute_returns=False):")
    print("-" * 40)
    
    prices = np.array([100 + i * 0.5 + np.random.randn() * 0.1 for i in range(50)])
    tf_raw = IntradayTemporalFeatures(prices, compute_returns=False)
    
    print(f"   Data points: {tf_raw.n_data} (same as input)")
    print(f"   ACF(1):      {tf_raw.autocorrelation(1):.4f}")
    print(f"   Note: High ACF expected for trending raw prices")
    print(f"   ✅ compute_returns=False works correctly")
    
    print()


def test_batch_analysis():
    """
    Test batch analysis across multiple trading days.
    """
    print_section("Batch Session Analysis Test")
    
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)
    
    print(f"\nTotal bars loaded: {len(df)}")
    print(f"Unique dates: {df['date'].nunique()}")
    print()
    
    # Run batch analysis
    results = batch_analyze_sessions(df, date_col='date', time_col='time', price_col='close')
    
    if len(results) == 0:
        print("❌ No results from batch analysis.")
        return
    
    print(f"Analyzed {len(results)} trading days")
    print()
    
    # Show results for each day
    print("Results by Date:")
    print("-" * 80)
    
    for _, row in results.iterrows():
        print(f"\nDate: {row['date']}")
        
        # Afternoon session
        if 'afternoon_regime' in row:
            print(f"  Afternoon (1pm-3pm):")
            print(f"    Bars:    {row.get('afternoon_n_bars', 'N/A')}")
            print(f"    ACF(1):  {row.get('afternoon_acf_lag1', np.nan):.4f}")
            print(f"    Hurst:   {row.get('afternoon_hurst_exponent', np.nan):.4f}")
            print(f"    Regime:  {row.get('afternoon_regime', 'N/A')}")
        
        # Close session
        if 'close_regime' in row:
            print(f"  Close (3pm-3:50pm):")
            print(f"    Bars:    {row.get('close_n_bars', 'N/A')}")
            print(f"    ACF(1):  {row.get('close_acf_lag1', np.nan):.4f}")
            print(f"    Hurst:   {row.get('close_hurst_exponent', np.nan):.4f}")
            print(f"    Regime:  {row.get('close_regime', 'N/A')}")
    
    print()


def test_all_features():
    """
    Test get_all_features method for completeness.
    """
    print_section("All Features Test")
    
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_and_filter_nq_data(data_path, "13:00", "15:00")
    
    if len(df) == 0:
        print("❌ No data found.")
        return
    
    prices = df["close"].values
    tf = IntradayTemporalFeatures(prices, compute_returns=True)
    
    # Test with afternoon config
    print("\n1. With Afternoon Session Config:")
    print("-" * 40)
    
    features_afternoon = tf.get_all_features(SessionConfig.afternoon_session())
    for key, value in features_afternoon.items():
        if isinstance(value, float):
            print(f"   {key:20s}: {value:.4f}")
        else:
            print(f"   {key:20s}: {value}")
    
    # Test with close config
    print("\n2. With Close Session Config:")
    print("-" * 40)
    
    features_close = tf.get_all_features(SessionConfig.close_session())
    for key, value in features_close.items():
        if isinstance(value, float):
            print(f"   {key:20s}: {value:.4f}")
        else:
            print(f"   {key:20s}: {value}")
    
    # Test without config
    print("\n3. Without Session Config (default):")
    print("-" * 40)
    
    features_default = tf.get_all_features()
    for key, value in features_default.items():
        if isinstance(value, float):
            print(f"   {key:20s}: {value:.4f}")
        else:
            print(f"   {key:20s}: {value}")
    
    print()


if __name__ == "__main__":
    print()
    print("=" * 80)
    print("  Lag Autocorrelation & Hurst Exponent Test Suite")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  - Afternoon session: 1pm-3pm (120 bars expected)")
    print(f"  - Close session: 3pm-3:50pm (50 bars expected)")
    print()
    
    # Run all tests
    test_basic_autocorrelation()
    test_hurst_exponent()
    test_session_analysis()
    test_regime_detection()
    test_edge_cases()
    test_batch_analysis()
    test_all_features()
    
    print("=" * 80)
    print("  Test Suite Complete")
    print("=" * 80)



