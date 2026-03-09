"""
Multi-Scale Slope (MSS) Analysis - Trend Detection via Hierarchical Slope Consistency

This metric analyzes price windows at multiple scales to determine trend quality by examining:
1. Main window slope (normalized for volatility)
2. Six equal sub-window slopes
3. Extrema-based path slopes
4. Consistency and alignment across all scales

The metric is scale-invariant and comparable across different sessions, making it
particularly useful for detecting trend exhaustion and false breakouts.

Version 2.0 - Improved calibration for real market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


# Configuration constants
DEFAULT_ATR_PERIOD = 14
DEFAULT_NUM_SUBWINDOWS = 4  # Reduced from 6 for better per-window statistics
DEFAULT_PROMINENCE_LOOKBACK = 3
DEFAULT_AMPLITUDE_THRESHOLD_FACTOR = 0.5

# Composite score weights (rebalanced for real data)
WEIGHT_DIRECTIONAL_CONSISTENCY = 0.35
WEIGHT_SLOPE_ALIGNMENT = 0.20
WEIGHT_EXTREMA_COHERENCE = 0.20
WEIGHT_MAGNITUDE = 0.25

# Floor values to prevent complete collapse
FLOOR_SLOPE_ALIGNMENT = 0.1
FLOOR_EXTREMA_COHERENCE = 0.2


@dataclass
class ExtremaInfo:
    """Information about a prominent extremum (high or low)."""
    index: int
    price: float
    prominence: float
    is_high: bool


@dataclass
class Diagnostics:
    """Diagnostic information for debugging and calibration."""
    main_slope_raw: float  # Before normalization (percent change)
    internal_volatility: float  # Std dev of log returns
    sub_slope_mean: float
    sub_slope_std: float
    sub_slope_range: float
    num_positive_subs: int
    num_negative_subs: int
    atr_value: float
    
    def to_dict(self) -> Dict:
        return {
            'main_slope_raw': round(self.main_slope_raw, 6),
            'internal_volatility': round(self.internal_volatility, 6),
            'sub_slope_mean': round(self.sub_slope_mean, 6),
            'sub_slope_std': round(self.sub_slope_std, 6),
            'sub_slope_range': round(self.sub_slope_range, 6),
            'num_positive_subs': self.num_positive_subs,
            'num_negative_subs': self.num_negative_subs,
            'atr_value': round(self.atr_value, 4)
        }


@dataclass
class MSSResult:
    """Complete result of Multi-Scale Slope analysis."""
    trending_score: float
    main_slope: float
    sub_slopes: List[float]
    components: Dict[str, float]
    extrema: Dict[str, Optional[Dict]]
    interpretation: str
    diagnostics: Optional[Diagnostics] = None
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary format."""
        result = {
            'trending_score': self.trending_score,
            'main_slope': self.main_slope,
            'sub_slopes': self.sub_slopes,
            'components': self.components,
            'extrema': self.extrema,
            'interpretation': self.interpretation
        }
        if self.diagnostics:
            result['diagnostics'] = self.diagnostics.to_dict()
        return result


class MultiScaleSlope:
    """
    Multi-Scale Slope (MSS) Analysis for trend/consolidation detection.
    
    Analyzes hierarchical slope consistency across multiple time scales to determine
    whether price action is trending or consolidating. The metric examines:
    
    1. Main window normalized slope (volatility-adjusted)
    2. Sub-window slopes (6 equal divisions)
    3. Extrema-based path coherence
    4. Weighted composite score
    
    Version 2.0 improvements:
    - Improved slope alignment formula that doesn't collapse to zero
    - Continuous extrema coherence scoring
    - Adaptive magnitude scoring for real market data
    - Floor values to prevent score collapse
    - Diagnostic output for calibration
    
    Attributes:
        atr_period: Period for ATR calculation (default: 14)
        num_subwindows: Number of sub-window divisions (default: 6)
        prominence_lookback: Bars on each side for extrema detection (default: 3)
        amplitude_threshold_factor: ATR multiplier for swing amplitude (default: 0.5)
    """
    
    def __init__(
        self,
        atr_period: int = DEFAULT_ATR_PERIOD,
        num_subwindows: int = DEFAULT_NUM_SUBWINDOWS,
        prominence_lookback: int = DEFAULT_PROMINENCE_LOOKBACK,
        amplitude_threshold_factor: float = DEFAULT_AMPLITUDE_THRESHOLD_FACTOR
    ):
        """
        Initialize Multi-Scale Slope analyzer.
        
        Args:
            atr_period: Period for ATR calculation
            num_subwindows: Number of sub-windows for hierarchical analysis
            prominence_lookback: Bars to check on each side for extrema
            amplitude_threshold_factor: ATR multiplier for minimum swing amplitude
        """
        self.atr_period = atr_period
        self.num_subwindows = num_subwindows
        self.prominence_lookback = prominence_lookback
        self.amplitude_threshold_factor = amplitude_threshold_factor
    
    def calculate_atr(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR) using standard formula.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (uses instance default if None)
            
        Returns:
            Series of ATR values
        """
        if period is None:
            period = self.atr_period
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the rolling mean of True Range
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_normalized_slope(
        self,
        prices: np.ndarray,
        start_idx: int,
        end_idx: int,
        return_components: bool = False
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Calculate volatility-normalized slope for a price window.
        
        Formula: (percent_change / num_bars) / std_dev(log_returns)
        
        This gives a risk-adjusted slope: "percentage per bar per unit of volatility"
        - High value = strong directional move with low internal chop (trending)
        - Low value = weak move OR high internal chop (consolidating)
        
        Args:
            prices: Array of prices (typically close prices)
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)
            return_components: If True, also return raw slope and volatility
            
        Returns:
            Normalized slope value, or tuple of (normalized, raw, volatility) if return_components
        """
        if end_idx <= start_idx or end_idx >= len(prices):
            if return_components:
                return 0.0, 0.0, 0.0
            return 0.0
        
        window_prices = prices[start_idx:end_idx + 1]
        num_bars = len(window_prices) - 1
        
        if num_bars < 1:
            if return_components:
                return 0.0, 0.0, 0.0
            return 0.0
        
        start_price = window_prices[0]
        end_price = window_prices[-1]
        
        # Avoid division by zero
        if start_price == 0 or np.isnan(start_price):
            if return_components:
                return 0.0, 0.0, 0.0
            return 0.0
        
        # Calculate percent change (raw slope)
        percent_change = (end_price - start_price) / start_price
        raw_slope = percent_change / num_bars
        
        # Calculate log returns
        log_returns = np.diff(np.log(window_prices))
        
        # Handle zero or near-zero volatility
        if len(log_returns) < 2:
            if return_components:
                return raw_slope, raw_slope, 0.0
            return raw_slope
        
        std_dev = np.std(log_returns, ddof=1)
        
        # Prevent division by zero while preserving direction
        epsilon = 1e-10
        if std_dev < epsilon:
            std_dev = epsilon
        
        normalized_slope = raw_slope / std_dev
        
        if return_components:
            return normalized_slope, percent_change, std_dev
        return normalized_slope
    
    def calculate_sub_window_slopes(
        self,
        prices: np.ndarray,
        window_start: int,
        window_end: int
    ) -> List[float]:
        """
        Divide window into sub-windows and calculate normalized slope for each.
        
        Args:
            prices: Array of prices
            window_start: Start index of main window
            window_end: End index of main window (inclusive)
            
        Returns:
            List of normalized slopes for each sub-window
        """
        window_size = window_end - window_start + 1
        sub_window_size = window_size // self.num_subwindows
        
        if sub_window_size < 2:
            return [0.0] * self.num_subwindows
        
        sub_slopes = []
        
        for i in range(self.num_subwindows):
            sub_start = window_start + (i * sub_window_size)
            
            # Last sub-window takes any remaining bars
            if i == self.num_subwindows - 1:
                sub_end = window_end
            else:
                sub_end = sub_start + sub_window_size - 1
            
            slope = self.calculate_normalized_slope(prices, sub_start, sub_end)
            sub_slopes.append(slope)
        
        return sub_slopes
    
    def find_prominent_extrema(
        self,
        df: pd.DataFrame,
        window_start: int,
        window_end: int,
        atr_value: float
    ) -> Tuple[Optional[ExtremaInfo], Optional[ExtremaInfo]]:
        """
        Find the most prominent high and low within the window.
        
        Prominence criteria:
        - A high must be highest within N bars on EACH side
        - A low must be lowest within N bars on EACH side
        - Must exceed amplitude threshold: > amplitude_threshold_factor * ATR
        - Among all qualifying swings, select the most prominent
        
        Args:
            df: DataFrame with OHLC data
            window_start: Start index of window
            window_end: End index of window (inclusive)
            atr_value: Current ATR value for amplitude threshold
            
        Returns:
            Tuple of (prominent_high, prominent_low), either can be None
        """
        if window_end >= len(df) or window_start < 0:
            return None, None
        
        window_df = df.iloc[window_start:window_end + 1].copy()
        window_df = window_df.reset_index(drop=True)
        
        if len(window_df) < (2 * self.prominence_lookback + 1):
            return None, None
        
        highs = window_df['high'].values
        lows = window_df['low'].values
        
        amplitude_threshold = self.amplitude_threshold_factor * atr_value
        
        # Find candidate highs
        candidate_highs: List[ExtremaInfo] = []
        for i in range(self.prominence_lookback, len(window_df) - self.prominence_lookback):
            is_local_high = True
            
            for j in range(1, self.prominence_lookback + 1):
                if highs[i] <= highs[i - j]:
                    is_local_high = False
                    break
            
            if is_local_high:
                for j in range(1, self.prominence_lookback + 1):
                    if highs[i] <= highs[i + j]:
                        is_local_high = False
                        break
            
            if is_local_high:
                left_low = min(lows[max(0, i - self.prominence_lookback):i])
                right_low = min(lows[i + 1:min(len(lows), i + self.prominence_lookback + 1)])
                prominence = highs[i] - max(left_low, right_low)
                
                if prominence >= amplitude_threshold:
                    candidate_highs.append(ExtremaInfo(
                        index=window_start + i,
                        price=highs[i],
                        prominence=prominence,
                        is_high=True
                    ))
        
        # Find candidate lows
        candidate_lows: List[ExtremaInfo] = []
        for i in range(self.prominence_lookback, len(window_df) - self.prominence_lookback):
            is_local_low = True
            
            for j in range(1, self.prominence_lookback + 1):
                if lows[i] >= lows[i - j]:
                    is_local_low = False
                    break
            
            if is_local_low:
                for j in range(1, self.prominence_lookback + 1):
                    if lows[i] >= lows[i + j]:
                        is_local_low = False
                        break
            
            if is_local_low:
                left_high = max(highs[max(0, i - self.prominence_lookback):i])
                right_high = max(highs[i + 1:min(len(highs), i + self.prominence_lookback + 1)])
                prominence = min(left_high, right_high) - lows[i]
                
                if prominence >= amplitude_threshold:
                    candidate_lows.append(ExtremaInfo(
                        index=window_start + i,
                        price=lows[i],
                        prominence=prominence,
                        is_high=False
                    ))
        
        prominent_high = max(candidate_highs, key=lambda x: x.prominence) if candidate_highs else None
        prominent_low = max(candidate_lows, key=lambda x: x.prominence) if candidate_lows else None
        
        return prominent_high, prominent_low
    
    def calculate_extrema_path_coherence(
        self,
        prices: np.ndarray,
        main_slope: float,
        window_start: int,
        window_end: int,
        prominent_high: Optional[ExtremaInfo],
        prominent_low: Optional[ExtremaInfo]
    ) -> Tuple[float, List[float]]:
        """
        Calculate extrema path coherence score with continuous scoring.
        
        Path construction:
        - If main_slope > 0 (uptrend): Start → Prominent High → Prominent Low → End
        - If main_slope < 0 (downtrend): Start → Prominent Low → Prominent High → End
        
        Version 2.0 improvements:
        - Continuous scoring based on slope magnitude, not just direction
        - Neutral fallback (0.5) when no extrema found instead of penalty
        - Uses tanh for smooth 0-1 mapping
        
        Args:
            prices: Array of close prices
            main_slope: Main window normalized slope (determines direction)
            window_start: Start index of window
            window_end: End index of window
            prominent_high: Most prominent high in window (or None)
            prominent_low: Most prominent low in window (or None)
            
        Returns:
            Tuple of (coherence_score, segment_slopes)
        """
        segment_slopes = []
        
        # If no extrema found, return neutral score (not penalizing)
        if prominent_high is None and prominent_low is None:
            return 0.5, segment_slopes
        
        # Helper for continuous scoring based on slope alignment
        def continuous_score(slope: float, expected_positive: bool) -> float:
            """Score based on alignment strength, not just direction."""
            if expected_positive:
                # Higher positive slope = higher score
                if slope > 0:
                    return 0.5 * (1 + np.tanh(slope))  # Maps positive slopes to 0.5-1.0
                else:
                    return 0.5 * (1 + np.tanh(slope))  # Maps negative slopes to 0.0-0.5
            else:
                # More negative slope = higher score
                if slope < 0:
                    return 0.5 * (1 - np.tanh(slope))  # Maps negative slopes to 0.5-1.0
                else:
                    return 0.5 * (1 - np.tanh(slope))  # Maps positive slopes to 0.0-0.5
        
        if main_slope > 0:
            # Uptrend path
            if prominent_high is not None and prominent_low is not None:
                first_point = prominent_high if prominent_high.index < prominent_low.index else prominent_low
                second_point = prominent_low if prominent_high.index < prominent_low.index else prominent_high
                
                slope1 = self.calculate_normalized_slope(prices, window_start, first_point.index)
                slope2 = self.calculate_normalized_slope(prices, first_point.index, second_point.index)
                slope3 = self.calculate_normalized_slope(prices, second_point.index, window_end)
                segment_slopes = [slope1, slope2, slope3]
                
                # Continuous scoring for uptrend
                score1 = continuous_score(slope1, expected_positive=True)
                score3 = continuous_score(slope3, expected_positive=True)
                coherence = (score1 + score3) / 2  # Average of entry and exit alignment
                
                return coherence, segment_slopes
            
            elif prominent_high is not None:
                slope1 = self.calculate_normalized_slope(prices, window_start, prominent_high.index)
                slope2 = self.calculate_normalized_slope(prices, prominent_high.index, window_end)
                segment_slopes = [slope1, slope2]
                
                score1 = continuous_score(slope1, expected_positive=True)
                coherence = score1 * 0.75 + 0.25 * 0.5  # Partial info, partial neutral
                return coherence, segment_slopes
            
            elif prominent_low is not None:
                slope1 = self.calculate_normalized_slope(prices, window_start, prominent_low.index)
                slope2 = self.calculate_normalized_slope(prices, prominent_low.index, window_end)
                segment_slopes = [slope1, slope2]
                
                score2 = continuous_score(slope2, expected_positive=True)
                coherence = score2 * 0.75 + 0.25 * 0.5
                return coherence, segment_slopes
        
        else:
            # Downtrend path
            if prominent_high is not None and prominent_low is not None:
                first_point = prominent_low if prominent_low.index < prominent_high.index else prominent_high
                second_point = prominent_high if prominent_low.index < prominent_high.index else prominent_low
                
                slope1 = self.calculate_normalized_slope(prices, window_start, first_point.index)
                slope2 = self.calculate_normalized_slope(prices, first_point.index, second_point.index)
                slope3 = self.calculate_normalized_slope(prices, second_point.index, window_end)
                segment_slopes = [slope1, slope2, slope3]
                
                # Continuous scoring for downtrend
                score1 = continuous_score(slope1, expected_positive=False)
                score3 = continuous_score(slope3, expected_positive=False)
                coherence = (score1 + score3) / 2
                
                return coherence, segment_slopes
            
            elif prominent_low is not None:
                slope1 = self.calculate_normalized_slope(prices, window_start, prominent_low.index)
                slope2 = self.calculate_normalized_slope(prices, prominent_low.index, window_end)
                segment_slopes = [slope1, slope2]
                
                score1 = continuous_score(slope1, expected_positive=False)
                coherence = score1 * 0.75 + 0.25 * 0.5
                return coherence, segment_slopes
            
            elif prominent_high is not None:
                slope1 = self.calculate_normalized_slope(prices, window_start, prominent_high.index)
                slope2 = self.calculate_normalized_slope(prices, prominent_high.index, window_end)
                segment_slopes = [slope1, slope2]
                
                score2 = continuous_score(slope2, expected_positive=False)
                coherence = score2 * 0.75 + 0.25 * 0.5
                return coherence, segment_slopes
        
        return 0.5, segment_slopes
    
    def calculate_composite_score(
        self,
        main_slope: float,
        sub_slopes: List[float],
        extrema_coherence: float
    ) -> Dict[str, float]:
        """
        Calculate composite trending score from all components.
        
        Version 2.0 improvements:
        - Slope alignment uses exponential decay instead of hard cutoff
        - Magnitude uses adaptive scaling based on typical slope ranges
        - Floor values prevent complete collapse of individual components
        - Rebalanced weights for real market data
        
        Components:
        1. Directional Consistency (0.35): Fraction of sub-windows matching main direction
        2. Slope Alignment (0.20): How tightly clustered sub-slopes are (exponential decay)
        3. Extrema Path Coherence (0.20): Whether swing structure aligns with trend
        4. Slope Magnitude (0.25): Absolute strength of main slope (adaptive)
        
        Args:
            main_slope: Normalized slope of main window
            sub_slopes: List of normalized slopes for sub-windows
            extrema_coherence: Coherence score from extrema path analysis
            
        Returns:
            Dictionary with all component scores and final trending_score
        """
        epsilon = 1e-6
        
        # Component 1: Directional Consistency
        if abs(main_slope) < epsilon:
            directional_consistency = sum(1 for s in sub_slopes if abs(s) < epsilon) / len(sub_slopes)
        else:
            main_sign = np.sign(main_slope)
            matching = sum(1 for s in sub_slopes if np.sign(s) == main_sign)
            directional_consistency = matching / len(sub_slopes)
        
        # Component 2: Slope Alignment (FIXED - exponential decay formula)
        # Uses coefficient of variation with dampening
        sub_slopes_array = np.array(sub_slopes)
        sub_std = np.std(sub_slopes_array, ddof=1) if len(sub_slopes) > 1 else 0.0
        sub_mean = np.mean(np.abs(sub_slopes_array))
        
        # Option: Normalize by expected variance scaling (sqrt of num_subwindows)
        # This accounts for the fact that sub-window variance naturally increases with more divisions
        expected_variance_scale = np.sqrt(self.num_subwindows)
        
        if abs(main_slope) > epsilon:
            # Relative variance compared to main slope, scaled by expected variance
            relative_variance = sub_std / (abs(main_slope) * expected_variance_scale)
            # Exponential decay - more forgiving than hard cutoff
            slope_alignment_raw = np.exp(-relative_variance)
        else:
            # If main slope is near zero, use coefficient of variation
            if sub_mean > epsilon:
                cv = sub_std / sub_mean
                slope_alignment_raw = 1 / (1 + cv)
            else:
                slope_alignment_raw = 1.0  # All sub-slopes near zero = aligned
        
        # Apply floor
        slope_alignment = max(FLOOR_SLOPE_ALIGNMENT, slope_alignment_raw)
        
        # Component 3: Extrema Path Coherence (with floor)
        extrema_coherence_floored = max(FLOOR_EXTREMA_COHERENCE, extrema_coherence)
        
        # Component 4: Slope Magnitude (aggressive scaling for real data)
        # Real normalized slopes are typically small (0.01-1.0 range)
        # Use tanh with 5x scaling to amplify signal
        abs_slope = abs(main_slope)
        magnitude_score = np.tanh(abs_slope * 5.0)
        
        # Calculate final weighted score
        trending_score = (
            WEIGHT_DIRECTIONAL_CONSISTENCY * directional_consistency +
            WEIGHT_SLOPE_ALIGNMENT * slope_alignment +
            WEIGHT_EXTREMA_COHERENCE * extrema_coherence_floored +
            WEIGHT_MAGNITUDE * magnitude_score
        )
        
        return {
            'directional_consistency': round(directional_consistency, 4),
            'slope_alignment': round(slope_alignment, 4),
            'slope_alignment_raw': round(slope_alignment_raw, 4),  # Before floor
            'extrema_coherence': round(extrema_coherence_floored, 4),
            'extrema_coherence_raw': round(extrema_coherence, 4),  # Before floor
            'magnitude_score': round(magnitude_score, 4),
            'trending_score': round(trending_score, 4)
        }
    
    @staticmethod
    def interpret_score(score: float) -> str:
        """
        Interpret the trending score.
        
        Args:
            score: Trending score (0-1)
            
        Returns:
            Human-readable interpretation
        """
        if score >= 0.8:
            return "STRONG_TREND"
        elif score >= 0.6:
            return "TRENDING_WITH_NOISE"
        elif score >= 0.4:
            return "MIXED_TRANSITIONAL"
        elif score >= 0.2:
            return "CHOPPY_CONSOLIDATING"
        else:
            return "HIGHLY_CHOPPY"
    
    def calculate_trending_score(
        self,
        df: pd.DataFrame,
        window_start: int,
        window_end: int,
        preload_bars: int = 0,
        include_diagnostics: bool = True
    ) -> MSSResult:
        """
        Calculate complete Multi-Scale Slope trending score for a window.
        
        This is the main entry point for analysis. It:
        1. Calculates ATR using pre-loaded data
        2. Computes main window normalized slope
        3. Calculates sub-window slopes
        4. Finds prominent extrema and path coherence
        5. Combines into composite score
        
        Args:
            df: DataFrame with OHLC data (columns: timestamp, open, high, low, close)
            window_start: Start index of analysis window
            window_end: End index of analysis window (inclusive)
            preload_bars: Number of bars before window_start used for ATR initialization
            include_diagnostics: Whether to include diagnostic information
            
        Returns:
            MSSResult with complete analysis
            
        Raises:
            ValueError: If insufficient data or invalid indices
        """
        # Validate inputs
        if window_start < 0 or window_end >= len(df):
            raise ValueError(f"Invalid window indices: start={window_start}, end={window_end}, df_len={len(df)}")
        
        if window_end <= window_start:
            raise ValueError(f"window_end ({window_end}) must be greater than window_start ({window_start})")
        
        window_size = window_end - window_start + 1
        if window_size < self.num_subwindows * 2:
            raise ValueError(f"Window size ({window_size}) too small for {self.num_subwindows} sub-windows")
        
        # Calculate ATR - include preload bars if available
        atr_start = max(0, window_start - preload_bars)
        atr_df = df.iloc[atr_start:window_end + 1].copy()
        atr_series = self.calculate_atr(atr_df)
        atr_value = atr_series.iloc[-1]
        
        if atr_value == 0 or np.isnan(atr_value):
            atr_value = 1e-6
        
        # Extract prices for slope calculations
        prices = df['close'].values
        
        # Step 1: Calculate main window slope (with components for diagnostics)
        main_slope, raw_pct_change, internal_vol = self.calculate_normalized_slope(
            prices, window_start, window_end, return_components=True
        )
        
        # Step 2: Calculate sub-window slopes
        sub_slopes = self.calculate_sub_window_slopes(prices, window_start, window_end)
        
        # Step 3: Find prominent extrema
        prominent_high, prominent_low = self.find_prominent_extrema(
            df, window_start, window_end, atr_value
        )
        
        # Step 4: Calculate extrema path coherence
        extrema_coherence, segment_slopes = self.calculate_extrema_path_coherence(
            prices, main_slope, window_start, window_end, prominent_high, prominent_low
        )
        
        # Step 5: Calculate composite score
        components = self.calculate_composite_score(main_slope, sub_slopes, extrema_coherence)
        
        # Build extrema info dict
        extrema_info = {
            'prominent_high': {
                'index': prominent_high.index,
                'price': prominent_high.price,
                'prominence': prominent_high.prominence
            } if prominent_high else None,
            'prominent_low': {
                'index': prominent_low.index,
                'price': prominent_low.price,
                'prominence': prominent_low.prominence
            } if prominent_low else None,
            'segment_slopes': segment_slopes
        }
        
        # Build diagnostics
        diagnostics = None
        if include_diagnostics:
            sub_slopes_array = np.array(sub_slopes)
            diagnostics = Diagnostics(
                main_slope_raw=raw_pct_change,
                internal_volatility=internal_vol,
                sub_slope_mean=float(np.mean(sub_slopes_array)),
                sub_slope_std=float(np.std(sub_slopes_array, ddof=1)) if len(sub_slopes) > 1 else 0.0,
                sub_slope_range=float(np.max(sub_slopes_array) - np.min(sub_slopes_array)),
                num_positive_subs=int(np.sum(sub_slopes_array > 0)),
                num_negative_subs=int(np.sum(sub_slopes_array < 0)),
                atr_value=atr_value
            )
        
        interpretation = self.interpret_score(components['trending_score'])
        
        return MSSResult(
            trending_score=components['trending_score'],
            main_slope=round(main_slope, 6),
            sub_slopes=[round(s, 6) for s in sub_slopes],
            components=components,
            extrema=extrema_info,
            interpretation=interpretation,
            diagnostics=diagnostics
        )


def analyze_session(
    df: pd.DataFrame,
    session_name: str = 'default',
    window_size: Optional[int] = None
) -> Dict:
    """
    Analyze a trading session using Multi-Scale Slope analysis.
    
    Convenience function for analyzing complete session windows.
    
    Args:
        df: DataFrame with OHLC data for the session
        session_name: Name of session for configuration (e.g., '3pm', 'london')
        window_size: Override automatic window sizing
        
    Returns:
        Dictionary with analysis results
    """
    session_configs = {
        '3pm': {'window': 50, 'preload': 14},
        'london': {'window': 60, 'preload': 14},
        'default': {'window': 50, 'preload': 14}
    }
    
    config = session_configs.get(session_name, session_configs['default'])
    
    if window_size is None:
        window_size = config['window']
    
    preload = config['preload']
    required_bars = window_size + preload
    
    if len(df) < required_bars:
        return {
            'session': session_name,
            'error': f'Insufficient data: need {required_bars} bars, got {len(df)}',
            'trending_score': None
        }
    
    window_end = len(df) - 1
    window_start = window_end - window_size + 1
    
    mss = MultiScaleSlope()
    
    try:
        result = mss.calculate_trending_score(
            df, window_start, window_end, preload_bars=preload
        )
        
        return {
            'session': session_name,
            **result.to_dict()
        }
        
    except Exception as e:
        return {
            'session': session_name,
            'error': str(e),
            'trending_score': None
        }


def batch_analyze_windows(
    df: pd.DataFrame,
    window_size: int = 50,
    step_size: int = 1,
    preload_bars: int = 14,
    include_diagnostics: bool = False
) -> pd.DataFrame:
    """
    Analyze multiple rolling windows for backtesting.
    
    Args:
        df: DataFrame with OHLC data
        window_size: Size of analysis window
        step_size: Step between windows
        preload_bars: Bars before window for ATR initialization
        include_diagnostics: Include diagnostic columns
        
    Returns:
        DataFrame with trending scores for each window
    """
    results = []
    mss = MultiScaleSlope()
    
    start_idx = preload_bars + window_size - 1
    
    for end_idx in range(start_idx, len(df), step_size):
        window_start = end_idx - window_size + 1
        
        try:
            result = mss.calculate_trending_score(
                df, window_start, end_idx, 
                preload_bars=preload_bars,
                include_diagnostics=include_diagnostics
            )
            
            row = {
                'window_end_idx': end_idx,
                'timestamp': df.iloc[end_idx].get('timestamp', end_idx),
                'trending_score': result.trending_score,
                'main_slope': result.main_slope,
                'directional_consistency': result.components['directional_consistency'],
                'slope_alignment': result.components['slope_alignment'],
                'extrema_coherence': result.components['extrema_coherence'],
                'magnitude_score': result.components['magnitude_score'],
                'interpretation': result.interpretation
            }
            
            if include_diagnostics and result.diagnostics:
                row.update({
                    'diag_raw_slope': result.diagnostics.main_slope_raw,
                    'diag_volatility': result.diagnostics.internal_volatility,
                    'diag_sub_std': result.diagnostics.sub_slope_std,
                    'diag_num_positive': result.diagnostics.num_positive_subs,
                    'diag_num_negative': result.diagnostics.num_negative_subs
                })
            
            results.append(row)
            
        except Exception as e:
            results.append({
                'window_end_idx': end_idx,
                'timestamp': df.iloc[end_idx].get('timestamp', end_idx),
                'trending_score': np.nan,
                'error': str(e)
            })
    
    return pd.DataFrame(results)
