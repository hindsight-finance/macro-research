# Trend features package

from .efficiency_ratio import EfficiencyRatioResult, analyze_efficiency_ratio, calculate_efficiency_ratio
from .state_detector import StateDetector, detect_state
from .variance_ratio import (
    VarianceRatioResult,
    analyze_variance_ratio,
    calculate_variance_ratio,
    calculate_variance_ratio_from_returns,
)

__all__ = [
    "EfficiencyRatioResult",
    "StateDetector",
    "VarianceRatioResult",
    "analyze_efficiency_ratio",
    "analyze_variance_ratio",
    "calculate_efficiency_ratio",
    "detect_state",
    "calculate_variance_ratio",
    "calculate_variance_ratio_from_returns",
]



