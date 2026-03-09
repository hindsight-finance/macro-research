import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SessionConfig:
    """Configuration for intraday session analysis"""
    name: str
    n_bars: int  # Expected number of bars in session
    
    @classmethod
    def afternoon_session(cls):
        """1pm-3pm session: 120 bars"""
        return cls(name="afternoon", n_bars=120)
    
    @classmethod
    def close_session(cls):
        """3pm-3:50pm session: 50 bars"""
        return cls(name="close", n_bars=50)


class IntradayTemporalFeatures:
    """
    Calculate temporal features optimized for intraday minute-bar data.
    Handles small sample sizes (50-120 bars) typical of intraday sessions.
    """
    
    def __init__(self, prices: np.ndarray, compute_returns: bool = True):
        """
        Args:
            prices: 1D array of prices (e.g., close prices for each minute)
            compute_returns: If True, works with log returns instead of prices
        """
        self.prices = np.array(prices)
        self.n = len(self.prices)
        
        if compute_returns and self.n > 1:
            # Use log returns for better stationarity
            self.data = np.diff(np.log(self.prices))
        else:
            self.data = self.prices
            
        self.n_data = len(self.data)
    
    def autocorrelation(self, lag: int) -> float:
        """
        Calculate autocorrelation at specified lag using efficient numpy method.
        Optimized for small sample sizes.
        
        Args:
            lag: Time lag in bars (1 = 1 minute for 1min data)
            
        Returns:
            Autocorrelation coefficient [-1, 1] or np.nan if insufficient data
        """
        if lag >= self.n_data or lag < 1:
            return np.nan
        
        # Need at least 10 observations for meaningful correlation
        if self.n_data < 10 or self.n_data - lag < 5:
            return np.nan
        
        # Demean the series
        x = self.data - np.mean(self.data)
        
        # Calculate correlation between x[t] and x[t-lag]
        c0 = np.dot(x, x) / self.n_data  # Variance
        
        if c0 == 0:
            return np.nan
        
        c_lag = np.dot(x[lag:], x[:-lag]) / self.n_data  # Autocovariance at lag
        
        return c_lag / c0
    
    def autocorrelations(self, lags: list = [1, 2, 5]) -> Dict[str, float]:
        """
        Calculate multiple autocorrelation values efficiently.
        
        Returns:
            Dictionary with keys like 'acf_lag1', 'acf_lag2', 'acf_lag5'
        """
        result = {}
        for lag in lags:
            result[f'acf_lag{lag}'] = self.autocorrelation(lag)
        return result
    
    def hurst_exponent_simple(self) -> float:
        """
        Calculate Hurst exponent using simplified method suitable for small samples.
        Uses variance method which is more stable for 50-120 bar sequences.
        
        Returns:
            Hurst exponent [0, 1] or np.nan if insufficient data
        """
        if self.n_data < 20:  # Need minimum data
            return np.nan
        
        # Use adaptive lag range based on data length
        max_lag = min(self.n_data // 4, 20)
        
        if max_lag < 3:
            return np.nan
        
        lags = range(2, max_lag + 1)
        variances = []
        valid_lags = []
        
        for lag in lags:
            if lag >= self.n_data:
                break
                
            # Calculate variance of lag-differenced series
            diffs = self.data[lag:] - self.data[:-lag]
            
            if len(diffs) < 3:
                break
                
            var = np.var(diffs, ddof=1)
            
            if var > 0 and np.isfinite(var):
                variances.append(var)
                valid_lags.append(lag)
        
        if len(variances) < 3:
            return np.nan
        
        # Fit power law: Var(lag) ~ lag^(2H)
        log_lags = np.log(valid_lags)
        log_vars = np.log(variances)
        
        # Linear regression
        coeffs = np.polyfit(log_lags, log_vars, 1)
        hurst = coeffs[0] / 2  # Divide by 2 because Var ~ lag^(2H)
        
        # Clip to valid range and handle edge cases
        return np.clip(hurst, 0.0, 1.0)
    
    def hurst_exponent_rs(self) -> float:
        """
        Rescaled Range (R/S) method for Hurst - more robust but needs more data.
        Use this for 120-bar sessions, use simple method for 50-bar.
        
        Returns:
            Hurst exponent [0, 1] or np.nan if insufficient data
        """
        if self.n_data < 30:
            return np.nan
        
        # Adaptive window sizing for small samples
        min_window = max(10, self.n_data // 10)
        max_window = self.n_data // 2
        
        # Use fewer windows for small datasets
        windows = []
        if max_window >= min_window * 2:
            w = min_window
            while w <= max_window and len(windows) < 5:
                windows.append(w)
                w = int(w * 1.5)  # Gentler growth than powers of 2
        
        if len(windows) < 2:
            return np.nan
        
        rs_values = []
        
        for window in windows:
            num_chunks = self.n_data // window
            
            if num_chunks == 0:
                continue
            
            chunk_rs = []
            
            for i in range(num_chunks):
                chunk = self.data[i * window:(i + 1) * window]
                
                mean = np.mean(chunk)
                Y = chunk - mean
                Z = np.cumsum(Y)
                
                R = np.max(Z) - np.min(Z)
                S = np.std(chunk, ddof=1)
                
                if S > 0 and R > 0:
                    chunk_rs.append(R / S)
            
            if chunk_rs:
                rs_values.append(np.mean(chunk_rs))
        
        if len(rs_values) < 2:
            return np.nan
        
        # Fit log-log regression
        log_windows = np.log(windows[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        valid = np.isfinite(log_windows) & np.isfinite(log_rs)
        
        if np.sum(valid) < 2:
            return np.nan
        
        coeffs = np.polyfit(log_windows[valid], log_rs[valid], 1)
        hurst = coeffs[0]
        
        return np.clip(hurst, 0.0, 1.0)
    
    def get_all_features(self, session_config: Optional[SessionConfig] = None) -> Dict[str, float]:
        """
        Calculate all temporal features optimized for session length.
        
        Args:
            session_config: Configuration for the session type
            
        Returns:
            Dictionary with all features
        """
        features = {}
        
        # Autocorrelations at standard lags
        features.update(self.autocorrelations([1, 2, 5]))
        
        # Choose Hurst method based on data length
        if session_config and session_config.name == "close":
            # Use simpler method for shorter close session
            hurst = self.hurst_exponent_simple()
        else:
            # Try R/S for longer afternoon session, fallback to simple
            hurst = self.hurst_exponent_rs()
            if np.isnan(hurst):
                hurst = self.hurst_exponent_simple()
        
        features['hurst_exponent'] = hurst
        
        return features
    
    def get_regime_signal(self) -> str:
        """
        Quick interpretation helper for regime detection.
        
        Returns:
            String indicating detected regime
        """
        acf1 = self.autocorrelation(1)
        hurst = self.hurst_exponent_simple()
        
        if np.isnan(acf1) or np.isnan(hurst):
            return "insufficient_data"
        
        if hurst > 0.6 and acf1 > 0.2:
            return "trending"
        elif hurst < 0.4 and acf1 < -0.1:
            return "mean_reverting"
        elif abs(acf1) < 0.1 and 0.45 <= hurst <= 0.55:
            return "random_walk"
        elif hurst > 0.5:
            return "weak_trend"
        else:
            return "weak_reversion"


def analyze_intraday_session(
    df: pd.DataFrame,
    price_col: str = 'close',
    session_config: Optional[SessionConfig] = None
) -> Dict[str, float]:
    """
    Analyze a single intraday session from a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data (one row per minute)
        price_col: Column name for price (default 'close')
        session_config: Session configuration
        
    Returns:
        Dictionary of temporal features
    """
    prices = df[price_col].values
    
    tf = IntradayTemporalFeatures(prices, compute_returns=True)
    features = tf.get_all_features(session_config)
    
    # Add regime signal
    features['regime'] = tf.get_regime_signal()
    
    # Add metadata
    features['n_bars'] = len(prices)
    features['price_change_pct'] = ((prices[-1] - prices[0]) / prices[0]) * 100
    
    return features


def batch_analyze_sessions(
    df: pd.DataFrame,
    date_col: str = 'date',
    time_col: str = 'time',
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Analyze multiple days of intraday data, computing features for each session.
    
    Args:
        df: DataFrame with intraday data
        date_col: Column with date
        time_col: Column with time (as string like "13:00" or datetime)
        price_col: Price column to analyze
        
    Returns:
        DataFrame with one row per day, columns for each feature set
    """
    results = []
    
    # Group by date
    for date, day_data in df.groupby(date_col):
        row = {'date': date}
        
        # Afternoon session: 1pm-3pm (13:00-15:00)
        afternoon = day_data[
            (day_data[time_col] >= '13:00') & 
            (day_data[time_col] < '15:00')
        ]
        
        if len(afternoon) >= 20:  # Minimum bars for meaningful analysis
            afternoon_features = analyze_intraday_session(
                afternoon, 
                price_col, 
                SessionConfig.afternoon_session()
            )
            for k, v in afternoon_features.items():
                row[f'afternoon_{k}'] = v
        
        # Close session: 3pm-3:50pm (15:00-15:50)
        close = day_data[
            (day_data[time_col] >= '15:00') & 
            (day_data[time_col] < '15:50')
        ]
        
        if len(close) >= 10:
            close_features = analyze_intraday_session(
                close,
                price_col,
                SessionConfig.close_session()
            )
            for k, v in close_features.items():
                row[f'close_{k}'] = v
        
        results.append(row)
    
    return pd.DataFrame(results)