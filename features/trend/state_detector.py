"""
Market State Detector

Unified system for detecting market regime (trending, consolidating, choppy) by
combining signals from 7 complementary trend identification modules.

Architecture:
- Base classes for indicator interface
- Individual indicator wrappers
- StateDetector main class with weighted signal combination
- Dynamic Hurst-based weight adjustment
- Session-aware configuration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Set
from datetime import time, datetime
import warnings
import importlib
import importlib.util
import sys
from pathlib import Path


# ============================================================================
# MODULE LOADING
# ============================================================================

_MODULE_DIR = Path(__file__).parent
_IMPORT_CACHE: Dict[str, object] = {}

# Ensure trend directory is on sys.path so package imports (ADX) work
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))


def _load_module(module_name: str, file_path: Path):
    """Lazily import a module from file path, with caching (for dirs with spaces)."""
    key = str(file_path)
    if key not in _IMPORT_CACHE:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _IMPORT_CACHE[key] = mod
    return _IMPORT_CACHE[key]


def _load_package(dotted_name: str):
    """Import a module using dotted package notation (for proper Python packages)."""
    if dotted_name not in _IMPORT_CACHE:
        _IMPORT_CACHE[dotted_name] = importlib.import_module(dotted_name)
    return _IMPORT_CACHE[dotted_name]


# Map state_detector session names -> ATR Range session keys
_ATR_SESSION_MAP = {
    '1pm-3pm': '1pm-3pm',
    '3pm-3:50pm': '3pm-3:50pm',
    '3:50pm-4pm': '3:50-4pm',
    'auto': '1pm-3pm',
}


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

MarketState = Literal["STRONG_TREND", "WEAK_TREND", "CONSOLIDATION", "CHOPPY", "UNCERTAIN"]
TrendDirection = Literal["UP", "DOWN", "NEUTRAL"]
SessionName = Literal["1pm-3pm", "3pm-3:50pm", "3:50pm-4pm", "auto"]


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class IndicatorResult:
    """Result from a single trend indicator."""
    signal: float  # Normalized signal (0-1 or -1 to 1)
    raw_value: Optional[float] = None  # Raw indicator value
    metadata: Dict = field(default_factory=dict)  # Additional info
    error: Optional[str] = None  # Error message if calculation failed


@dataclass
class StateResult:
    """Complete market state detection result."""
    state: MarketState
    direction: TrendDirection
    confidence: float  # 0-1, how confident in the classification
    signals: Dict[str, float]  # Individual indicator signals
    weights: Dict[str, float]  # Actual weights used (after reweighting)
    warnings: List[str] = field(default_factory=list)  # Warnings from calculation
    metadata: Dict = field(default_factory=dict)  # Additional context


# ============================================================================
# BASE CLASSES
# ============================================================================

class TrendIndicator(ABC):
    """
    Base class for all trend indicators.

    All indicators should implement this interface for use in StateDetector.
    """

    @abstractmethod
    def __init__(self, config: Optional[Dict] = None):
        """Initialize indicator with optional configuration."""
        pass

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        """
        Calculate indicator from OHLC DataFrame.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'timestamp' columns

        Returns:
            IndicatorResult with signal and metadata
        """
        pass

    @abstractmethod
    def get_signal(self) -> float:
        """
        Get normalized signal from last calculation.

        Returns:
            Signal value (0-1 for unidirectional, -1 to 1 for directional)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return indicator name."""
        pass


# ============================================================================
# SESSION DETECTION
# ============================================================================

def detect_session(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> SessionName:
    """
    Auto-detect session from DataFrame timestamps.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        Session name or 'auto' if cannot determine
    """
    if timestamp_col not in df.columns:
        return "auto"

    # Get time range
    times = pd.to_datetime(df[timestamp_col]).dt.time

    start_time = times.min()
    end_time = times.max()

    # Check session windows
    if start_time >= time(13, 0) and end_time < time(15, 0):
        return "1pm-3pm"
    elif start_time >= time(15, 0) and end_time < time(15, 50):
        return "3pm-3:50pm"
    elif start_time >= time(15, 50) and end_time <= time(16, 0):
        return "3:50pm-4pm"

    return "auto"


def _normalize_resample_rule(rule: str) -> str:
    """Normalize short bar-size labels like '5m' to pandas-compatible rules."""
    if rule.endswith('min'):
        return rule
    if rule.endswith('m'):
        return f"{rule[:-1]}min"
    return rule


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLC data to a coarser bar size when timestamps are available.

    If timestamps are unavailable or the requested rule is already 1-minute,
    return the original bars.
    """
    normalized_rule = _normalize_resample_rule(rule)
    if normalized_rule in {'1min', '1T'} or 'timestamp' not in df.columns:
        return df.copy()

    bars = df.copy()
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    return (
        bars.sort_values('timestamp')
        .set_index('timestamp')
        .resample(normalized_rule)
        .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        .dropna()
        .reset_index()
    )


def _window_end_from_resampled_bars(df: pd.DataFrame, rule: str) -> time:
    """Return an exclusive end time that includes the final resampled bar."""
    if df.empty or 'timestamp' not in df.columns:
        return time(23, 59, 59)

    last_timestamp = pd.to_datetime(df['timestamp']).max()
    end_timestamp = last_timestamp + pd.to_timedelta(_normalize_resample_rule(rule))
    return end_timestamp.time()


# ============================================================================
# DYNAMIC WEIGHTING CONSTANTS
# ============================================================================

# Direction each indicator measures relative to trending:
#  +1 = trend-measuring (higher signal when trending)
#  -1 = consolidation-measuring (higher signal when NOT consolidating, inverted)
#   0 = neutral / meta-signal
INDICATOR_DIRECTION = {
    'adx': 1,
    'mss': 1,
    'dra': -1,
    'atr_range': -1,
    'irr': -1,
    'spd': 0,
    'lag': 0,
}

HURST_SENSITIVITY = 4.0   # kappa: controls steepness of tanh curve
MAX_WEIGHT_SHIFT = 0.5    # alpha: maximum relative weight adjustment (50%)


def compute_dynamic_weights(
    base_weights: Dict[str, float],
    hurst_raw: float,
    kappa: float = HURST_SENSITIVITY,
    alpha: float = MAX_WEIGHT_SHIFT,
) -> Tuple[Dict[str, float], Dict]:
    """
    Adjust indicator weights based on raw Hurst exponent.

    When Hurst > 0.5 (trending regime), upweight trend-measuring indicators
    and downweight consolidation-measuring ones. Vice versa when Hurst < 0.5.

    Args:
        base_weights: Default weight dict (indicator_name -> weight)
        hurst_raw: Raw Hurst exponent (0-1, 0.5 = neutral)
        kappa: Sensitivity parameter for tanh curve
        alpha: Maximum relative shift magnitude

    Returns:
        Tuple of (adjusted_weights, debug_info)
        adjusted_weights sum to 1.0, no weight is negative
    """
    h_dev = hurst_raw - 0.5
    adjustment = alpha * np.tanh(kappa * h_dev)

    adjusted = {}
    multipliers = {}
    for name, w in base_weights.items():
        direction = INDICATOR_DIRECTION.get(name, 0)
        multiplier = 1.0 + direction * adjustment
        # Clamp multiplier to [0.5, 1.5] — no weight goes negative or dominates
        multiplier = max(0.5, min(1.5, multiplier))
        adjusted[name] = w * multiplier
        multipliers[name] = round(multiplier, 4)

    # Renormalize to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 6) for k, v in adjusted.items()}

    debug_info = {
        'hurst_raw': round(hurst_raw, 4),
        'h_deviation': round(h_dev, 4),
        'adjustment': round(adjustment, 4),
        'multipliers': multipliers,
    }

    return adjusted, debug_info


# ============================================================================
# INDICATOR WRAPPERS
# ============================================================================

class ADXIndicator(TrendIndicator):
    """Wrapper for ADX trend quality module."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._last_result: Optional[IndicatorResult] = None

    @property
    def name(self) -> str:
        return "adx"

    def calculate(self, df: pd.DataFrame, session: SessionName = "auto") -> IndicatorResult:
        """Calculate ADX trend quality. Signal: quality_score (0-1)."""
        try:
            if session == "auto":
                session = "1pm-3pm"  # ADX requires explicit session
            mod = _load_package('ADX.trend_quality')
            config = mod.WINDOW_CONFIGS[session]
            bars = _resample_ohlc(df, config['bar_size'])
            result = mod.calculate_trend_quality(bars, session)
            self._last_result = IndicatorResult(
                signal=result['quality_score'],
                raw_value=result['components']['strength_raw'],
                metadata={
                    **result,
                    'input_bars': len(df),
                    'resampled_bars': len(bars),
                    'bar_size': config['bar_size'],
                },
            )
        except Exception as e:
            self._last_result = IndicatorResult(signal=0.5, error=str(e))
        return self._last_result

    def get_signal(self) -> float:
        if self._last_result is None:
            return 0.0
        return self._last_result.signal


class ATRRangeIndicator(TrendIndicator):
    """Wrapper for ATR/Range ratio module."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._last_result: Optional[IndicatorResult] = None

    @property
    def name(self) -> str:
        return "atr_range"

    def calculate(self, df: pd.DataFrame, session: SessionName = "auto") -> IndicatorResult:
        """
        Calculate ATR/Range ratio. Signal: 1 - raw_ratio (inverted, higher = trending).

        Resamples to 5-min bars before calculating — the ATR module's period
        configs (atr_period=10 for 1pm-3pm) were calibrated for 5-min data.
        Feeding raw 1-min bars produces a structurally tiny ratio (single
        1-min TR vs 2-hour total range) with near-zero variance.
        """
        try:
            mod = _load_module('atr', _MODULE_DIR / 'ATR Range' / 'atr.py')
            atr_session = _ATR_SESSION_MAP.get(session, '1pm-3pm')
            bar_size = mod.SESSIONS[atr_session]['bar_size']
            bars = _resample_ohlc(df, bar_size)

            result = mod.analyze_session(bars, atr_session)
            raw_ratio = result['raw_ratio']
            if raw_ratio is None:
                self._last_result = IndicatorResult(
                    signal=0.5, error="No range detected", metadata=result,
                )
            else:
                signal = float(np.clip(1.0 - raw_ratio, 0.0, 1.0))
                self._last_result = IndicatorResult(
                    signal=signal,
                    raw_value=raw_ratio,
                    metadata={
                        **result,
                        'input_bars': len(df),
                        'resampled_bars': len(bars),
                        'bar_size': bar_size,
                    },
                )
        except Exception as e:
            self._last_result = IndicatorResult(signal=0.5, error=str(e))
        return self._last_result

    def get_signal(self) -> float:
        if self._last_result is None:
            return 0.0
        return self._last_result.signal


class DRAIndicator(TrendIndicator):
    """Wrapper for Dynamic Range Analysis module."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._last_result: Optional[IndicatorResult] = None

    @property
    def name(self) -> str:
        return "dra"

    def calculate(
        self,
        df: pd.DataFrame,
        reference_bars: Optional[pd.DataFrame] = None,
    ) -> IndicatorResult:
        """
        Calculate DRA overlap. Signal: fraction of post-reference bars that
        have broken out of the initial range (higher = more trending).

        The original rolling-average approach collapses to 1.0 on any session
        longer than ~30 bars because price almost always leaves a 15-minute
        opening range within the first half-hour.  Using the *fraction of
        bars with zero overlap* preserves the temporal information — a session
        that spends 55% of its time inside the range is very different from
        one that escapes in the first 7 minutes.

        Args:
            df: Full session bars
            reference_bars: Reference range bars (defaults to first 15 of df)
        """
        try:
            mod = _load_module('dra', _MODULE_DIR / 'DRA' / 'dra.py')
            dra = mod.DRA(window=self.config.get('window', 10))

            if reference_bars is None:
                reference_bars = df.iloc[:15]

            dra.set_initial_range(reference_bars)

            # Update with remaining bars after the reference window
            n_ref = len(reference_bars)
            remaining = df.iloc[n_ref:]
            for _, bar in remaining.iterrows():
                dra.update(bar)

            # Signal: fraction of bars that broke out of the reference range
            if dra.overlaps:
                frac_inside = sum(1 for o in dra.overlaps if o > 0) / len(dra.overlaps)
            else:
                frac_inside = 0.5

            signal = float(np.clip(1.0 - frac_inside, 0.0, 1.0))
            self._last_result = IndicatorResult(
                signal=signal,
                raw_value=frac_inside,
                metadata={
                    'n_updates': len(remaining),
                    'bars_inside': sum(1 for o in dra.overlaps if o > 0),
                    'bars_total': len(dra.overlaps),
                },
            )
        except Exception as e:
            self._last_result = IndicatorResult(signal=0.5, error=str(e))
        return self._last_result

    def get_signal(self) -> float:
        if self._last_result is None:
            return 0.0
        return self._last_result.signal


class IRRIndicator(TrendIndicator):
    """Wrapper for Intraperiod Reversion Ratio module."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._last_result: Optional[IndicatorResult] = None

    @property
    def name(self) -> str:
        return "irr"

    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate IRR. Signal: directional_strength (1 - avg_irr, 0-1)."""
        try:
            mod = _load_module('irr', _MODULE_DIR / 'IRR' / 'irr.py')
            analyzer = mod.IRRAnalyzer(
                num_subwindows=self.config.get('num_subwindows', 5),
            )
            result = analyzer.analyze(df, 0, len(df) - 1, "session")
            self._last_result = IndicatorResult(
                signal=result.directional_strength,
                raw_value=result.median_sub_irr,
                metadata={
                    'regime': result.regime,
                    'full_window_irr': result.full_window_irr,
                    'interpretation': result.interpretation,
                },
            )
        except Exception as e:
            self._last_result = IndicatorResult(signal=0.5, error=str(e))
        return self._last_result

    def get_signal(self) -> float:
        if self._last_result is None:
            return 0.0
        return self._last_result.signal


class LagIndicator(TrendIndicator):
    """Wrapper for Lag Autocorrelation / Hurst module."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._last_result: Optional[IndicatorResult] = None

    @property
    def name(self) -> str:
        return "lag"

    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        """
        Calculate Hurst-based signal.

        Signal: hurst directly (0-1, 0.5 = neutral random walk).
        Raw value: raw Hurst (consumed by dynamic weighting).
        """
        try:
            mod = _load_module('lag', _MODULE_DIR / 'Lag autocorr' / 'lag.py')
            prices = df['close'].values
            tf = mod.IntradayTemporalFeatures(prices)

            # Try R/S method first (more robust), fall back to simple variance
            hurst = tf.hurst_exponent_rs()
            if np.isnan(hurst):
                hurst = tf.hurst_exponent_simple()

            if np.isnan(hurst):
                self._last_result = IndicatorResult(
                    signal=0.5, error="Insufficient data for Hurst calculation",
                )
            else:
                acf1 = tf.autocorrelation(1)
                regime = tf.get_regime_signal()
                self._last_result = IndicatorResult(
                    signal=float(hurst),
                    raw_value=float(hurst),  # Critical: dynamic weighting reads this
                    metadata={
                        'acf_lag1': float(acf1) if not np.isnan(acf1) else None,
                        'regime': regime,
                    },
                )
        except Exception as e:
            self._last_result = IndicatorResult(signal=0.5, error=str(e))
        return self._last_result

    def get_signal(self) -> float:
        if self._last_result is None:
            return 0.0
        return self._last_result.signal


class MSSIndicator(TrendIndicator):
    """Wrapper for Multi-Scale Slope module."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._last_result: Optional[IndicatorResult] = None

    @property
    def name(self) -> str:
        return "mss"

    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate MSS trending score. Signal: trending_score (0-1)."""
        try:
            mod = _load_module('mss', _MODULE_DIR / 'MSS tan' / 'mss.py')
            mss = mod.MultiScaleSlope(
                atr_period=self.config.get('atr_period', 14),
                num_subwindows=self.config.get('num_subwindows', 4),
            )
            result = mss.calculate_trending_score(
                df, window_start=0, window_end=len(df) - 1,
            )
            self._last_result = IndicatorResult(
                signal=result.trending_score,
                raw_value=result.main_slope,
                metadata={
                    'interpretation': result.interpretation,
                    'components': result.components,
                },
            )
        except Exception as e:
            self._last_result = IndicatorResult(signal=0.5, error=str(e))
        return self._last_result

    def get_signal(self) -> float:
        if self._last_result is None:
            return 0.0
        return self._last_result.signal


class SPDIndicator(TrendIndicator):
    """Wrapper for Swing Point Density module."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._last_result: Optional[IndicatorResult] = None

    @property
    def name(self) -> str:
        return "spd"

    def calculate(self, df: pd.DataFrame, session: SessionName = "auto") -> IndicatorResult:
        """
        Calculate swing point density.
        Signal: 1 - count/max_expected (inverted, fewer swings = more trending).

        Resamples to 5-min bars before detecting swings.  On raw 1-min data,
        noise produces 40-50+ swings per session regardless of regime (the
        3-bar swing detection fires on every minor tick).  5-min bars smooth
        this out and align with the module's classification thresholds (< 5
        trending, > 8 choppy) which were calibrated for larger bars.
        """
        try:
            mod = _load_module('spd', _MODULE_DIR / 'Swing Point Density' / 'spd.py')
            resample_rule = self.config.get('bar_size', '5min')

            bars_df = _resample_ohlc(df, resample_rule)
            if 'timestamp' in bars_df.columns and not bars_df.empty:
                times = pd.to_datetime(bars_df['timestamp']).dt.time
                start_time = times.min()
                end_time = _window_end_from_resampled_bars(bars_df, resample_rule)
            else:
                start_time = time(0, 0)
                end_time = time(23, 59, 59)

            bars = bars_df.to_dict('records')
            result = mod.get_swing_density(bars, start_time, end_time)
            signal = {
                'trending': 1.0,
                'mixed': 0.5,
                'chop': 0.0,
                'insufficient_data': 0.5,
            }.get(result['classification'], 0.5)
            self._last_result = IndicatorResult(
                signal=signal,
                raw_value=float(result['count']),
                metadata={
                    **result,
                    'input_bars': len(df),
                    'resampled_bars': len(bars_df),
                    'bar_size': resample_rule,
                },
            )
        except Exception as e:
            self._last_result = IndicatorResult(signal=0.5, error=str(e))
        return self._last_result

    def get_signal(self) -> float:
        if self._last_result is None:
            return 0.0
        return self._last_result.signal


# ============================================================================
# STATE CLASSIFICATION
# ============================================================================

def classify_state(
    signals: Dict[str, float],
    weights: Dict[str, float],
    confidence_threshold: float = 0.3
) -> Tuple[MarketState, float]:
    """
    Classify market state from weighted signals.

    Args:
        signals: Dict of indicator_name -> signal_value (0-1, higher = more trending)
        weights: Dict of indicator_name -> weight (should sum to 1.0)
        confidence_threshold: Minimum confidence to return a state (else UNCERTAIN)

    Returns:
        Tuple of (state, confidence)
    """
    # Calculate weighted average signal
    weighted_sum = sum(signals.get(name, 0.0) * weights.get(name, 0.0)
                      for name in weights.keys())

    # Normalize weights (in case some signals were missing)
    total_weight = sum(weights.values())
    if total_weight == 0:
        return ("UNCERTAIN", 0.0)

    composite_signal = weighted_sum / total_weight

    # Calculate confidence as distance from 0.5 (neutral)
    # Max confidence at extremes (0 or 1), min at 0.5
    confidence = abs(composite_signal - 0.5) * 2  # Maps 0.5->0, 0/1->1

    # Classify state based on composite signal
    if composite_signal >= 0.7:
        state = "STRONG_TREND"
    elif composite_signal >= 0.55:
        state = "WEAK_TREND"
    elif composite_signal <= 0.3:
        state = "CHOPPY"
    elif composite_signal <= 0.45:
        state = "CONSOLIDATION"
    else:
        state = "WEAK_TREND"  # 0.45-0.55 range

    # Check confidence threshold
    if confidence < confidence_threshold:
        state = "UNCERTAIN"

    return (state, confidence)


def determine_direction(df: pd.DataFrame) -> TrendDirection:
    """
    Determine trend direction from price movement.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Trend direction
    """
    if len(df) < 2:
        return "NEUTRAL"

    closes = df['close'].values
    price_change = (closes[-1] - closes[0]) / closes[0]

    # Threshold for significant movement (0.1%)
    if price_change > 0.001:
        return "UP"
    elif price_change < -0.001:
        return "DOWN"
    else:
        return "NEUTRAL"


# ============================================================================
# MAIN STATE DETECTOR CLASS
# ============================================================================

class StateDetector:
    """
    Unified market state detection using weighted indicator ensemble.

    Combines signals from 7 trend identification modules to classify market regime.
    Optionally uses the Hurst exponent as a meta-signal to dynamically adjust
    indicator weights toward the detected regime.
    """

    # Default weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'adx': 0.18,
        'atr_range': 0.10,
        'dra': 0.12,
        'irr': 0.12,
        'lag': 0.12,
        'mss': 0.24,
        'spd': 0.12,
    }

    def __init__(
        self,
        config: Optional[Dict] = None,
        enabled_indicators: Optional[Set[str]] = None,
        confidence_threshold: float = 0.3,
        dynamic_weights: bool = True,
    ):
        """
        Initialize StateDetector.

        Args:
            config: Optional configuration dict for individual indicators
            enabled_indicators: Set of indicator names to use (None = all)
            confidence_threshold: Minimum confidence to return state (default: 0.3)
            dynamic_weights: Enable Hurst-based dynamic weight adjustment (default: True)
        """
        self.config = config or {}
        self.confidence_threshold = confidence_threshold
        self.dynamic_weights = dynamic_weights

        # Initialize all indicators
        self.indicators: Dict[str, TrendIndicator] = {
            'adx': ADXIndicator(self.config.get('adx')),
            'atr_range': ATRRangeIndicator(self.config.get('atr_range')),
            'dra': DRAIndicator(self.config.get('dra')),
            'irr': IRRIndicator(self.config.get('irr')),
            'lag': LagIndicator(self.config.get('lag')),
            'mss': MSSIndicator(self.config.get('mss')),
            'spd': SPDIndicator(self.config.get('spd')),
        }

        # Filter to enabled indicators
        if enabled_indicators is None:
            self.enabled_indicators = set(self.indicators.keys())
        else:
            self.enabled_indicators = enabled_indicators & set(self.indicators.keys())

        # Get weights (default or from config)
        self.weights = self.config.get('weights', self.DEFAULT_WEIGHTS.copy())

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            warnings.warn(f"Weights sum to {total_weight:.3f}, not 1.0. Normalizing.")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def detect(
        self,
        df: pd.DataFrame,
        session: SessionName = "auto",
        reference_bars: Optional[pd.DataFrame] = None,
        dynamic_weights: Optional[bool] = None,
    ) -> StateResult:
        """
        Detect market state from OHLC data.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'timestamp' columns
            session: Session name or 'auto' for auto-detection
            reference_bars: Optional reference bars for DRA (defaults to first 15)
            dynamic_weights: Override instance-level dynamic_weights setting

        Returns:
            StateResult with state classification, direction, confidence, and signals
        """
        use_dynamic = dynamic_weights if dynamic_weights is not None else self.dynamic_weights

        # Validate input
        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Auto-detect session if needed
        if session == "auto":
            session = detect_session(df)

        # Prepare reference bars for DRA (default to first 15)
        if reference_bars is None and 'dra' in self.enabled_indicators:
            if len(df) >= 15:
                reference_bars = df.iloc[:15].copy()
            else:
                warnings.warn(f"Insufficient bars for DRA reference (need 15, got {len(df)})")

        # ---- Phase 1: Run all enabled indicators ----
        signals: Dict[str, float] = {}
        indicator_results: Dict[str, IndicatorResult] = {}
        result_warnings: List[str] = []
        failed_indicators: Set[str] = set()

        for name, indicator in self.indicators.items():
            if name not in self.enabled_indicators:
                continue

            try:
                # Special handling for DRA (needs reference_bars)
                if name == 'dra':
                    result = indicator.calculate(df, reference_bars=reference_bars)
                # Special handling for session-aware indicators
                elif name in ['adx', 'atr_range', 'spd']:
                    result = indicator.calculate(df, session=session)
                else:
                    result = indicator.calculate(df)

                indicator_results[name] = result

                if result.error:
                    result_warnings.append(f"{name}: {result.error}")
                    failed_indicators.add(name)
                else:
                    signals[name] = result.signal

            except Exception as e:
                result_warnings.append(f"{name}: {str(e)}")
                failed_indicators.add(name)

        if len(signals) == 0:
            return StateResult(
                state="UNCERTAIN",
                direction="NEUTRAL",
                confidence=0.0,
                signals={},
                weights={},
                warnings=result_warnings + ["All indicators failed"],
            )

        # ---- Phase 2: Dynamic weight adjustment ----
        # Start with static weights, excluding failed indicators
        inactive_indicators = failed_indicators | (set(self.weights.keys()) - self.enabled_indicators)
        active_weights = self._reweight(self.weights, inactive_indicators)
        dynamic_info = None

        if use_dynamic and 'lag' in indicator_results and indicator_results['lag'].error is None:
            hurst_raw = indicator_results['lag'].raw_value
            if hurst_raw is not None:
                active_weights, dynamic_info = compute_dynamic_weights(
                    active_weights, hurst_raw,
                )
        elif use_dynamic:
            result_warnings.append(
                "dynamic_weights: Hurst unavailable, using static weights"
            )

        # ---- Phase 3: Classify state ----
        state, confidence = classify_state(signals, active_weights, self.confidence_threshold)

        # Determine direction
        direction = determine_direction(df)

        metadata = {
            'session': session,
            'failed_indicators': list(failed_indicators),
            'n_bars': len(df),
        }
        if dynamic_info is not None:
            metadata['dynamic_weighting'] = dynamic_info

        return StateResult(
            state=state,
            direction=direction,
            confidence=confidence,
            signals=signals,
            weights=active_weights,
            warnings=result_warnings,
            metadata=metadata,
        )

    def _reweight(
        self,
        original_weights: Dict[str, float],
        failed_indicators: Set[str]
    ) -> Dict[str, float]:
        """
        Reweight remaining indicators after failures.

        Args:
            original_weights: Original weight distribution
            failed_indicators: Set of indicator names that failed

        Returns:
            New weights that sum to 1.0, excluding failed indicators
        """
        if not failed_indicators:
            return original_weights.copy()

        # Remove failed indicators
        active_weights = {
            name: weight
            for name, weight in original_weights.items()
            if name not in failed_indicators
        }

        # Renormalize
        total = sum(active_weights.values())
        if total == 0:
            return {}

        return {name: weight / total for name, weight in active_weights.items()}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_state(
    df: pd.DataFrame,
    session: SessionName = "auto",
    reference_bars: Optional[pd.DataFrame] = None,
    enabled_indicators: Optional[Set[str]] = None,
    confidence_threshold: float = 0.3,
    dynamic_weights: bool = True,
) -> StateResult:
    """
    Convenience function for one-off state detection.

    Args:
        df: OHLC DataFrame
        session: Session name or 'auto'
        reference_bars: Optional DRA reference bars
        enabled_indicators: Optional set of indicators to use
        confidence_threshold: Minimum confidence threshold
        dynamic_weights: Enable Hurst-based dynamic weight adjustment

    Returns:
        StateResult
    """
    detector = StateDetector(
        enabled_indicators=enabled_indicators,
        confidence_threshold=confidence_threshold,
        dynamic_weights=dynamic_weights,
    )
    return detector.detect(df, session=session, reference_bars=reference_bars)
