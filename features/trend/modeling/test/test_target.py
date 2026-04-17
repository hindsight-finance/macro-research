import pytest
import numpy as np

from features.trend.modeling.target import (
    build_chop_target,
    build_containment_expansion_features,
    build_containment_features,
    build_containment_target,
    build_descriptive_target,
)


def _make_ohlc(close_values: list[float], wick: float = 0.1) -> tuple[np.ndarray, ...]:
    close = np.asarray(close_values, dtype=float)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    return open_, high, low, close


def _make_ohlcv(
    close_values: list[float],
    wick: float = 0.1,
    volume: float = 100.0,
) -> tuple[np.ndarray, ...]:
    open_, high, low, close = _make_ohlc(close_values=close_values, wick=wick)
    volume_arr = np.full(close.shape, volume, dtype=float)
    return open_, high, low, close, volume_arr


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


def test_build_containment_target_returns_bounded_components():
    open_, high, low, close = _make_ohlc([100.0, 101.6, 100.9, 99.3, 100.0, 101.1, 100.2])

    result = build_containment_target(open_=open_, high=high, low=low, close=close)

    assert 0.0 <= result["containment_target"] <= 1.0
    assert {
        "containment_displacement",
        "containment_edge_balance",
        "containment_inside_share",
        "containment_path_efficiency",
        "containment_target",
        "containment_status",
    } <= set(result)


def test_build_chop_target_returns_bounded_components():
    open_, high, low, close = _make_ohlc([100.0, 101.7, 99.0, 101.5, 98.8, 101.4, 99.9])

    result = build_chop_target(open_=open_, high=high, low=low, close=close)

    assert 0.0 <= result["chop_score"] <= 1.0
    assert {
        "chop_flip_rate",
        "chop_path_waste",
        "chop_outside_share",
        "chop_instability",
        "chop_score",
        "chop_status",
    } <= set(result)


def test_chop_target_scores_ugly_chop_above_rotation_and_trend():
    trend_open, trend_high, trend_low, trend_close = _make_ohlc([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    rotating_open, rotating_high, rotating_low, rotating_close = _make_ohlc([100.0, 101.6, 100.9, 99.3, 100.0, 101.1, 100.2])
    chop_open, chop_high, chop_low, chop_close = _make_ohlc([100.0, 101.7, 99.0, 101.5, 98.8, 101.4, 99.9])

    trend = build_chop_target(
        open_=trend_open,
        high=trend_high,
        low=trend_low,
        close=trend_close,
    )
    rotating = build_chop_target(
        open_=rotating_open,
        high=rotating_high,
        low=rotating_low,
        close=rotating_close,
    )
    chop = build_chop_target(
        open_=chop_open,
        high=chop_high,
        low=chop_low,
        close=chop_close,
    )

    assert chop["chop_score"] > rotating["chop_score"]
    assert chop["chop_score"] > trend["chop_score"]


def test_containment_target_scores_clean_rotating_range_above_trend_and_noisy_chop():
    trend_open, trend_high, trend_low, trend_close = _make_ohlc([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    rotating_open, rotating_high, rotating_low, rotating_close = _make_ohlc([100.0, 101.6, 100.9, 99.3, 100.0, 101.1, 100.2])
    chop_open, chop_high, chop_low, chop_close = _make_ohlc([100.0, 101.7, 99.0, 101.5, 98.8, 101.4, 99.9])

    trend = build_containment_target(
        open_=trend_open,
        high=trend_high,
        low=trend_low,
        close=trend_close,
    )
    rotating = build_containment_target(
        open_=rotating_open,
        high=rotating_high,
        low=rotating_low,
        close=rotating_close,
    )
    chop = build_containment_target(
        open_=chop_open,
        high=chop_high,
        low=chop_low,
        close=chop_close,
    )

    assert rotating["containment_target"] > trend["containment_target"]
    assert rotating["containment_target"] > chop["containment_target"]


@pytest.mark.parametrize(
    "close",
    [
        np.array([100.0, 0.0, 101.0]),
        np.array([100.0, -1.0, 101.0]),
    ],
)
def test_containment_target_rejects_non_positive_close(close):
    with pytest.raises(ValueError, match="close must contain only positive values"):
        build_containment_target(
            open_=np.array([100.0, 100.0, 100.0]),
            high=np.array([101.0, 101.0, 101.0]),
            low=np.array([99.0, 99.0, 99.0]),
            close=close,
        )


@pytest.mark.parametrize(
    "close",
    [
        np.array([100.0, 0.0, 101.0]),
        np.array([100.0, -1.0, 101.0]),
    ],
)
def test_chop_target_rejects_non_positive_close(close):
    with pytest.raises(ValueError, match="close must contain only positive values"):
        build_chop_target(
            open_=np.array([100.0, 100.0, 100.0]),
            high=np.array([101.0, 101.0, 101.0]),
            low=np.array([99.0, 99.0, 99.0]),
            close=close,
        )


def test_build_containment_features_returns_bounded_components():
    open_, high, low, close = _make_ohlc([100.0, 101.6, 100.9, 99.3, 100.0, 101.1, 100.2])

    result = build_containment_features(open_=open_, high=high, low=low, close=close)

    assert {
        "containment_overshoot_ratio",
        "containment_range_stability",
        "containment_mid_cross_count",
        "containment_swing_symmetry",
    } <= set(result)
    assert 0.0 <= result["containment_overshoot_ratio"] <= 1.0
    assert 0.0 <= result["containment_range_stability"] <= 1.0
    assert 0.0 <= result["containment_mid_cross_count"] <= 1.0
    assert 0.0 <= result["containment_swing_symmetry"] <= 1.0


def test_containment_features_score_clean_rotation_above_trend_and_noisy_chop():
    trend_open, trend_high, trend_low, trend_close = _make_ohlc([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    rotating_open, rotating_high, rotating_low, rotating_close = _make_ohlc([100.0, 101.6, 100.9, 99.3, 100.0, 101.1, 100.2])
    chop_open, chop_high, chop_low, chop_close = _make_ohlc([100.0, 101.8, 99.2, 100.2, 98.7, 99.4, 99.1])

    trend = build_containment_features(
        open_=trend_open,
        high=trend_high,
        low=trend_low,
        close=trend_close,
    )
    rotating = build_containment_features(
        open_=rotating_open,
        high=rotating_high,
        low=rotating_low,
        close=rotating_close,
    )
    chop = build_containment_features(
        open_=chop_open,
        high=chop_high,
        low=chop_low,
        close=chop_close,
    )

    assert rotating["containment_overshoot_ratio"] < chop["containment_overshoot_ratio"]
    assert rotating["containment_range_stability"] > chop["containment_range_stability"]
    assert rotating["containment_swing_symmetry"] > chop["containment_swing_symmetry"]
    assert rotating["containment_swing_symmetry"] > trend["containment_swing_symmetry"]
    assert rotating["containment_mid_cross_count"] > trend["containment_mid_cross_count"]


def test_containment_features_reject_non_positive_close():
    with pytest.raises(ValueError, match="close must contain only positive values"):
        build_containment_features(
            open_=np.array([100.0, 100.0, 100.0]),
            high=np.array([101.0, 101.0, 101.0]),
            low=np.array([99.0, 99.0, 99.0]),
            close=np.array([100.0, 0.0, 101.0]),
        )


def test_build_containment_expansion_features_returns_expected_columns():
    open_, high, low, close, volume = _make_ohlcv(
        [
            100.0, 100.2, 100.5, 100.1, 99.9,
            100.3, 100.6, 100.2, 99.8, 100.1,
            100.4, 100.7, 100.2, 99.9, 100.0,
            100.2, 100.4, 100.1, 99.9, 100.0,
        ],
        wick=0.25,
        volume=120.0,
    )

    result = build_containment_expansion_features(
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        session_name="1pm-3pm",
    )

    assert {
        "containment_ib_extension_ratio",
        "containment_ib_asymmetry",
        "containment_bandwidth_squeeze",
        "containment_vwap_acceptance",
        "containment_excess_rejection",
    } <= set(result)
    assert result["containment_ib_extension_ratio"] >= 0.0
    assert 0.0 <= result["containment_ib_asymmetry"] <= 1.0
    assert 0.0 <= result["containment_bandwidth_squeeze"] <= 1.0
    assert 0.0 <= result["containment_vwap_acceptance"] <= 1.0
    assert 0.0 <= result["containment_excess_rejection"] <= 1.0


def test_containment_expansion_features_score_clean_rotation_above_trend():
    trend_open, trend_high, trend_low, trend_close, trend_volume = _make_ohlcv(
        [100.0 + 0.5 * idx for idx in range(20)],
        wick=0.05,
        volume=120.0,
    )
    rotating_open, rotating_high, rotating_low, rotating_close, rotating_volume = _make_ohlcv(
        [
            100.0, 100.2, 100.5, 100.1, 99.9,
            100.3, 100.6, 100.2, 99.8, 100.1,
            100.4, 100.7, 100.2, 99.9, 100.0,
            100.2, 100.4, 100.1, 99.9, 100.0,
        ],
        wick=0.25,
        volume=120.0,
    )

    trend = build_containment_expansion_features(
        open_=trend_open,
        high=trend_high,
        low=trend_low,
        close=trend_close,
        volume=trend_volume,
        session_name="1pm-3pm",
    )
    rotating = build_containment_expansion_features(
        open_=rotating_open,
        high=rotating_high,
        low=rotating_low,
        close=rotating_close,
        volume=rotating_volume,
        session_name="1pm-3pm",
    )

    assert rotating["containment_ib_extension_ratio"] < trend["containment_ib_extension_ratio"]
    assert rotating["containment_ib_asymmetry"] < trend["containment_ib_asymmetry"]
    assert rotating["containment_bandwidth_squeeze"] > trend["containment_bandwidth_squeeze"]
    assert rotating["containment_vwap_acceptance"] > trend["containment_vwap_acceptance"]


def test_containment_expansion_features_reject_negative_volume():
    with pytest.raises(ValueError, match="volume must contain only finite non-negative values"):
        build_containment_expansion_features(
            open_=np.array([100.0, 100.0, 100.0]),
            high=np.array([101.0, 101.0, 101.0]),
            low=np.array([99.0, 99.0, 99.0]),
            close=np.array([100.0, 100.5, 100.2]),
            volume=np.array([100.0, -1.0, 100.0]),
            session_name="1pm-3pm",
        )
