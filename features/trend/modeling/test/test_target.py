import pytest
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


def test_descriptive_target_scores_clean_trend_above_round_trip_chop():
    trend = build_descriptive_target(
        open_=np.array([100.0, 100.0, 100.0, 100.0]),
        high=np.array([101.0, 102.0, 103.0, 104.0]),
        low=np.array([99.5, 100.0, 101.0, 102.0]),
        close=np.array([100.0, 101.0, 102.0, 103.0]),
    )
    chop = build_descriptive_target(
        open_=np.array([100.0, 100.0, 100.0, 100.0]),
        high=np.array([101.0, 101.5, 101.5, 101.0]),
        low=np.array([99.0, 98.5, 98.5, 99.0]),
        close=np.array([100.0, 101.0, 99.8, 100.0]),
    )

    assert trend["descriptive_target"] > chop["descriptive_target"]


def test_descriptive_target_handles_zero_return_window():
    flat = build_descriptive_target(
        open_=np.array([100.0, 100.0, 100.0]),
        high=np.array([100.0, 100.0, 100.0]),
        low=np.array([100.0, 100.0, 100.0]),
        close=np.array([100.0, 100.0, 100.0]),
    )

    assert flat["target_status"] == "ok"
    assert flat["descriptive_target"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "close",
    [
        np.array([100.0, 0.0, 101.0]),
        np.array([100.0, -1.0, 101.0]),
    ],
)
def test_descriptive_target_rejects_non_positive_close(close):
    with pytest.raises(ValueError, match="close must contain only positive values"):
        build_descriptive_target(
            open_=np.array([100.0, 100.0, 100.0]),
            high=np.array([101.0, 101.0, 101.0]),
            low=np.array([99.0, 99.0, 99.0]),
            close=close,
        )


@pytest.mark.parametrize(
    "open_, high, low, close",
    [
        (
            np.array([100.0, np.nan, 100.0]),
            np.array([101.0, 101.0, 101.0]),
            np.array([99.0, 99.0, 99.0]),
            np.array([100.0, 101.0, 102.0]),
        ),
        (
            np.array([100.0, 100.0, 100.0]),
            np.array([101.0, np.inf, 101.0]),
            np.array([99.0, 99.0, 99.0]),
            np.array([100.0, 101.0, 102.0]),
        ),
    ],
)
def test_descriptive_target_rejects_non_finite_ohlc(open_, high, low, close):
    with pytest.raises(ValueError, match="must contain only finite values"):
        build_descriptive_target(open_=open_, high=high, low=low, close=close)
