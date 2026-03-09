"""
Intraday Range Ratio (IRR) Analysis - Trend Detection via Body/Range Ratio

This metric analyzes how much of a candle's range is body vs wicks:
- High IRR (>0.65): Long wicks relative to body → indecision, mean reversion, consolidation
- Low IRR (<0.35): Body dominates range → directional conviction, trending
- Mid IRR (0.35-0.65): Mixed signals

The analysis operates at two levels:
1. Full Window IRR: Entire session treated as one candle
2. Sub-Window IRRs: Session divided into N sub-windows for pattern analysis

Version 2.0 - Class-based structure with hierarchical analysis
"""

import numpy as np
import pandas as pd
from statistics import median, mean
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


# Configuration constants
DEFAULT_NUM_SUBWINDOWS = 5
MIN_RANGE_THRESHOLD = 0.0001  # Avoid division by zero

# Regime thresholds
HIGH_IRR_THRESHOLD = 0.65  # Mean reversion / consolidation
LOW_IRR_THRESHOLD = 0.35   # Directional / trending


@dataclass
class SubWindowDetail:
    """Detailed information about a single sub-window."""
    time_label: str
    irr: Optional[float]
    open: float
    high: float
    low: float
    close: float
    bar_range: float
    body: float
    
    def to_dict(self) -> Dict:
        return {
            'time_label': self.time_label,
            'irr': round(self.irr, 4) if self.irr is not None else None,
            'open': round(self.open, 4),
            'high': round(self.high, 4),
            'low': round(self.low, 4),
            'close': round(self.close, 4),
            'bar_range': round(self.bar_range, 4),
            'body': round(self.body, 4)
        }


@dataclass
class ReversalInfo:
    """Information about potential reversal points."""
    high_window_idx: int  # Which sub-window made the session high
    low_window_idx: int   # Which sub-window made the session low
    high_window_irr: Optional[float]  # IRR of the window that made the high
    low_window_irr: Optional[float]   # IRR of the window that made the low
    high_price: float
    low_price: float
    
    def to_dict(self) -> Dict:
        return {
            'high_window_idx': self.high_window_idx,
            'low_window_idx': self.low_window_idx,
            'high_window_irr': round(self.high_window_irr, 4) if self.high_window_irr is not None else None,
            'low_window_irr': round(self.low_window_irr, 4) if self.low_window_irr is not None else None,
            'high_price': round(self.high_price, 4),
            'low_price': round(self.low_price, 4)
        }


@dataclass
class IRRResult:
    """Complete result of IRR analysis."""
    full_window_irr: Optional[float]
    sub_window_irrs: List[Optional[float]]
    median_sub_irr: Optional[float]
    average_sub_irr: Optional[float]
    sub_window_details: List[SubWindowDetail]
    regime: str
    directional_strength: Optional[float]
    reversal_info: Optional[ReversalInfo]
    interpretation: str
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary format."""
        return {
            'full_window_irr': round(self.full_window_irr, 4) if self.full_window_irr is not None else None,
            'sub_window_irrs': [round(irr, 4) if irr is not None else None for irr in self.sub_window_irrs],
            'median_sub_irr': round(self.median_sub_irr, 4) if self.median_sub_irr is not None else None,
            'average_sub_irr': round(self.average_sub_irr, 4) if self.average_sub_irr is not None else None,
            'sub_window_details': [detail.to_dict() for detail in self.sub_window_details],
            'regime': self.regime,
            'directional_strength': round(self.directional_strength, 4) if self.directional_strength is not None else None,
            'reversal_info': self.reversal_info.to_dict() if self.reversal_info else None,
            'interpretation': self.interpretation
        }


class IRRAnalyzer:
    """
    Intraday Range Ratio (IRR) Analyzer for trend/consolidation detection.
    
    Analyzes the ratio of candle body to total range at multiple time scales:
    - Full window analysis treats entire session as one candle
    - Sub-window analysis breaks session into smaller periods for pattern detection
    
    IRR Formula: 1 - |Close - Open| / (High - Low)
    
    Interpretation:
    - IRR near 1.0: All wicks, no body → strong indecision
    - IRR near 0.0: All body, no wicks → strong directional move
    
    Attributes:
        num_subwindows: Number of sub-windows to divide the main window into
        min_range: Minimum range threshold to avoid division by zero
    """
    
    def __init__(
        self,
        num_subwindows: int = DEFAULT_NUM_SUBWINDOWS,
        min_range: float = MIN_RANGE_THRESHOLD
    ):
        """
        Initialize IRR Analyzer.
        
        Args:
            num_subwindows: Number of sub-windows for hierarchical analysis
            min_range: Minimum range threshold (returns None if range is below this)
        """
        self.num_subwindows = num_subwindows
        self.min_range = min_range
    
    def compute_irr(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float
    ) -> Optional[float]:
        """
        Compute IRR for a single candle/bar.
        
        IRR = 1 - |Close - Open| / (High - Low)
        
        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            
        Returns:
            IRR value (0-1) or None if range is too small
        """
        bar_range = high - low
        
        if bar_range < self.min_range:
            return None
        
        body = abs(close - open_price)
        return 1 - (body / bar_range)
    
    def compute_full_window_irr(
        self,
        df: pd.DataFrame,
        window_start: int,
        window_end: int
    ) -> Tuple[Optional[float], float, float, float, float]:
        """
        Compute IRR for entire window treated as a single candle.
        
        Uses:
        - Open: First bar's open
        - High: Maximum high across all bars
        - Low: Minimum low across all bars  
        - Close: Last bar's close
        
        Args:
            df: DataFrame with OHLC data
            window_start: Start index (inclusive)
            window_end: End index (inclusive)
            
        Returns:
            Tuple of (irr, open, high, low, close)
        """
        window_df = df.iloc[window_start:window_end + 1]
        
        open_price = window_df['open'].iloc[0]
        high = window_df['high'].max()
        low = window_df['low'].min()
        close = window_df['close'].iloc[-1]
        
        irr = self.compute_irr(open_price, high, low, close)
        
        return irr, open_price, high, low, close
    
    def compute_sub_window_irrs(
        self,
        df: pd.DataFrame,
        window_start: int,
        window_end: int,
        session_label: str = "window"
    ) -> Tuple[List[Optional[float]], List[SubWindowDetail]]:
        """
        Divide window into sub-windows and compute IRR for each.
        
        Each sub-window is treated as a single candle (aggregated OHLC).
        
        Args:
            df: DataFrame with OHLC data
            window_start: Start index of main window
            window_end: End index of main window (inclusive)
            session_label: Label prefix for time labels
            
        Returns:
            Tuple of (list of IRR values, list of SubWindowDetail objects)
        """
        window_size = window_end - window_start + 1
        sub_window_size = window_size // self.num_subwindows
        
        if sub_window_size < 1:
            return [None] * self.num_subwindows, []
        
        irr_values = []
        details = []
        
        for i in range(self.num_subwindows):
            sub_start = window_start + (i * sub_window_size)
            
            # Last sub-window takes any remaining bars
            if i == self.num_subwindows - 1:
                sub_end = window_end
            else:
                sub_end = sub_start + sub_window_size - 1
            
            sub_df = df.iloc[sub_start:sub_end + 1]
            
            if len(sub_df) == 0:
                irr_values.append(None)
                continue
            
            open_price = sub_df['open'].iloc[0]
            high = sub_df['high'].max()
            low = sub_df['low'].min()
            close = sub_df['close'].iloc[-1]
            bar_range = high - low
            body = abs(close - open_price)
            
            irr = self.compute_irr(open_price, high, low, close)
            irr_values.append(irr)
            
            detail = SubWindowDetail(
                time_label=f"{session_label}_{i + 1}",
                irr=irr,
                open=open_price,
                high=high,
                low=low,
                close=close,
                bar_range=bar_range,
                body=body
            )
            details.append(detail)
        
        return irr_values, details
    
    def find_reversal_info(
        self,
        sub_window_details: List[SubWindowDetail],
        sub_window_irrs: List[Optional[float]]
    ) -> Optional[ReversalInfo]:
        """
        Find which sub-windows made the session high and low.
        
        Useful for reversal detection - knowing the IRR of the window
        that made the extreme can indicate whether the move was exhaustive.
        
        Args:
            sub_window_details: List of sub-window details
            sub_window_irrs: List of sub-window IRR values
            
        Returns:
            ReversalInfo object or None if no details available
        """
        if not sub_window_details:
            return None
        
        # Find which window made the high
        high_idx = 0
        high_price = sub_window_details[0].high
        for i, detail in enumerate(sub_window_details):
            if detail.high > high_price:
                high_price = detail.high
                high_idx = i
        
        # Find which window made the low
        low_idx = 0
        low_price = sub_window_details[0].low
        for i, detail in enumerate(sub_window_details):
            if detail.low < low_price:
                low_price = detail.low
                low_idx = i
        
        return ReversalInfo(
            high_window_idx=high_idx,
            low_window_idx=low_idx,
            high_window_irr=sub_window_irrs[high_idx] if high_idx < len(sub_window_irrs) else None,
            low_window_irr=sub_window_irrs[low_idx] if low_idx < len(sub_window_irrs) else None,
            high_price=high_price,
            low_price=low_price
        )
    
    @staticmethod
    def classify_regime(irr_value: Optional[float]) -> str:
        """
        Classify market regime based on IRR value.
        
        Args:
            irr_value: IRR value to classify
            
        Returns:
            Regime string: 'high_reversion', 'directional', 'mixed', or 'unknown'
        """
        if irr_value is None:
            return 'unknown'
        elif irr_value > HIGH_IRR_THRESHOLD:
            return 'high_reversion'  # Consolidation, mean reversion likely
        elif irr_value < LOW_IRR_THRESHOLD:
            return 'directional'  # Trending, directional conviction
        else:
            return 'mixed'
    
    @staticmethod
    def calculate_directional_strength(average_irr: Optional[float]) -> Optional[float]:
        """
        Calculate directional strength from average IRR.
        
        Directional strength = 1 - average_irr
        - High value (near 1): Strong directional conviction
        - Low value (near 0): Weak directional conviction (consolidation)
        
        Args:
            average_irr: Average IRR value
            
        Returns:
            Directional strength (0-1) or None
        """
        if average_irr is None:
            return None
        return 1 - average_irr
    
    @staticmethod
    def interpret_result(
        full_window_irr: Optional[float],
        median_sub_irr: Optional[float],
        regime: str
    ) -> str:
        """
        Generate human-readable interpretation of IRR analysis.
        
        Args:
            full_window_irr: IRR for entire window
            median_sub_irr: Median of sub-window IRRs
            regime: Classified regime
            
        Returns:
            Interpretation string
        """
        if full_window_irr is None or median_sub_irr is None:
            return "INSUFFICIENT_DATA"
        
        # Check for consistency between full window and sub-windows
        full_regime = IRRAnalyzer.classify_regime(full_window_irr)
        sub_regime = regime
        
        if full_regime == sub_regime:
            if regime == 'directional':
                return "STRONG_TREND"
            elif regime == 'high_reversion':
                return "CONSOLIDATION"
            else:
                return "MIXED_TRANSITIONAL"
        else:
            # Divergence between full window and sub-window analysis
            if full_regime == 'directional' and sub_regime == 'high_reversion':
                return "TREND_EXHAUSTION"  # Overall directional but sub-windows showing indecision
            elif full_regime == 'high_reversion' and sub_regime == 'directional':
                return "BREAKOUT_DEVELOPING"  # Overall choppy but sub-windows showing direction
            else:
                return "MIXED_SIGNALS"
    
    def analyze(
        self,
        df: pd.DataFrame,
        window_start: int,
        window_end: int,
        session_label: str = "window"
    ) -> IRRResult:
        """
        Perform complete IRR analysis on a window.
        
        This is the main entry point for analysis. It:
        1. Computes full window IRR (entire window as one candle)
        2. Computes sub-window IRRs
        3. Calculates statistics (median, average)
        4. Identifies reversal information
        5. Classifies regime and generates interpretation
        
        Args:
            df: DataFrame with OHLC data (columns: open, high, low, close)
            window_start: Start index of analysis window
            window_end: End index of analysis window (inclusive)
            session_label: Label for time labels in sub-window details
            
        Returns:
            IRRResult with complete analysis
            
        Raises:
            ValueError: If invalid indices or insufficient data
        """
        # Validate inputs
        if window_start < 0 or window_end >= len(df):
            raise ValueError(f"Invalid window indices: start={window_start}, end={window_end}, df_len={len(df)}")
        
        if window_end <= window_start:
            raise ValueError(f"window_end ({window_end}) must be greater than window_start ({window_start})")
        
        window_size = window_end - window_start + 1
        if window_size < self.num_subwindows:
            raise ValueError(f"Window size ({window_size}) too small for {self.num_subwindows} sub-windows")
        
        # Step 1: Compute full window IRR
        full_window_irr, _, _, _, _ = self.compute_full_window_irr(df, window_start, window_end)
        
        # Step 2: Compute sub-window IRRs
        sub_window_irrs, sub_window_details = self.compute_sub_window_irrs(
            df, window_start, window_end, session_label
        )
        
        # Step 3: Calculate statistics (filter out None values)
        valid_irrs = [irr for irr in sub_window_irrs if irr is not None]
        
        median_sub_irr = median(valid_irrs) if valid_irrs else None
        average_sub_irr = mean(valid_irrs) if valid_irrs else None
        
        # Step 4: Find reversal information
        reversal_info = self.find_reversal_info(sub_window_details, sub_window_irrs)
        
        # Step 5: Classify regime (using median of sub-windows for robustness)
        regime = self.classify_regime(median_sub_irr)
        
        # Step 6: Calculate directional strength
        directional_strength = self.calculate_directional_strength(average_sub_irr)
        
        # Step 7: Generate interpretation
        interpretation = self.interpret_result(full_window_irr, median_sub_irr, regime)
        
        return IRRResult(
            full_window_irr=full_window_irr,
            sub_window_irrs=sub_window_irrs,
            median_sub_irr=median_sub_irr,
            average_sub_irr=average_sub_irr,
            sub_window_details=sub_window_details,
            regime=regime,
            directional_strength=directional_strength,
            reversal_info=reversal_info,
            interpretation=interpretation
        )


def analyze_session(
    df: pd.DataFrame,
    session_name: str = 'default',
    window_size: Optional[int] = None,
    num_subwindows: int = DEFAULT_NUM_SUBWINDOWS
) -> Dict:
    """
    Analyze a trading session using IRR analysis.
    
    Convenience function for analyzing complete session windows.
    
    Args:
        df: DataFrame with OHLC data for the session
        session_name: Name of session for configuration (e.g., '3pm', 'london')
        window_size: Override automatic window sizing
        num_subwindows: Number of sub-windows for analysis
        
    Returns:
        Dictionary with analysis results
    """
    session_configs = {
        '3pm': {'window': 50},  # 3pm-3:50pm = 50 one-minute bars
        '3pm_close': {'window': 10},  # 3:50-4pm = 10 one-minute bars
        'london': {'window': 60},
        'default': {'window': 50}
    }
    
    config = session_configs.get(session_name, session_configs['default'])
    
    if window_size is None:
        window_size = config['window']
    
    if len(df) < window_size:
        return {
            'session': session_name,
            'error': f'Insufficient data: need {window_size} bars, got {len(df)}',
            'full_window_irr': None
        }
    
    window_end = len(df) - 1
    window_start = window_end - window_size + 1
    
    analyzer = IRRAnalyzer(num_subwindows=num_subwindows)
    
    try:
        result = analyzer.analyze(df, window_start, window_end, session_label=session_name)
        
        return {
            'session': session_name,
            **result.to_dict()
        }
        
    except Exception as e:
        return {
            'session': session_name,
            'error': str(e),
            'full_window_irr': None
        }


def batch_analyze_windows(
    df: pd.DataFrame,
    window_size: int = 50,
    step_size: int = 1,
    num_subwindows: int = DEFAULT_NUM_SUBWINDOWS
) -> pd.DataFrame:
    """
    Analyze multiple rolling windows for backtesting.
    
    Args:
        df: DataFrame with OHLC data
        window_size: Size of analysis window
        step_size: Step between windows
        num_subwindows: Number of sub-windows per analysis
        
    Returns:
        DataFrame with IRR analysis for each window
    """
    results = []
    analyzer = IRRAnalyzer(num_subwindows=num_subwindows)
    
    start_idx = window_size - 1
    
    for end_idx in range(start_idx, len(df), step_size):
        window_start = end_idx - window_size + 1
        
        try:
            result = analyzer.analyze(df, window_start, end_idx)
            
            row = {
                'window_end_idx': end_idx,
                'timestamp': df.iloc[end_idx].get('timestamp', end_idx),
                'full_window_irr': result.full_window_irr,
                'median_sub_irr': result.median_sub_irr,
                'average_sub_irr': result.average_sub_irr,
                'directional_strength': result.directional_strength,
                'regime': result.regime,
                'interpretation': result.interpretation,
                'high_window_idx': result.reversal_info.high_window_idx if result.reversal_info else None,
                'low_window_idx': result.reversal_info.low_window_idx if result.reversal_info else None,
                'high_window_irr': result.reversal_info.high_window_irr if result.reversal_info else None,
                'low_window_irr': result.reversal_info.low_window_irr if result.reversal_info else None
            }
            
            # Add individual sub-window IRRs as columns
            for i, irr in enumerate(result.sub_window_irrs):
                row[f'sub_irr_{i + 1}'] = irr
            
            results.append(row)
            
        except Exception as e:
            results.append({
                'window_end_idx': end_idx,
                'timestamp': df.iloc[end_idx].get('timestamp', end_idx),
                'full_window_irr': np.nan,
                'error': str(e)
            })
    
    return pd.DataFrame(results)
