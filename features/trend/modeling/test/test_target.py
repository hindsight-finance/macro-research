import numpy as np

from features.trend.modeling.target import build_descriptive_target


def test_build_descriptive_target_returns_bounded_components():
    result = build_descriptive_target(
        open_=np.array([100.0, 101.0, 102.0]),
        high=np.array([101.0, 102.0, 103.0]),
        low=np.array([99.5, 100.5, 101.5]),
        close=np.array([100.5, 101.5, 102.5]),
    )

    assert 0.0 <= result["descriptive_target"] <= 1.0
    assert {"target_strength", "target_consistency", "target_smoothness", "target_retention", "descriptive_target", "target_status"} <= set(result)
