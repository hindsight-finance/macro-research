import pandas as pd
import pytest

from features.trend.efficiency_ratio.er import analyze_efficiency_ratio, calculate_efficiency_ratio


def test_efficiency_ratio_is_one_for_straight_line_path():
    result = calculate_efficiency_ratio([100.0, 101.0, 102.0, 103.0])

    assert result.efficiency_ratio == pytest.approx(1.0)
    assert result.net_change == pytest.approx(3.0)
    assert result.path_length == pytest.approx(3.0)
    assert result.direction == "UP"


def test_efficiency_ratio_is_zero_for_round_trip_path():
    result = calculate_efficiency_ratio([100.0, 102.0, 100.0, 102.0, 100.0])

    assert result.efficiency_ratio == pytest.approx(0.0)
    assert result.net_change == pytest.approx(0.0)
    assert result.path_length == pytest.approx(8.0)
    assert result.direction == "NEUTRAL"


def test_analyze_efficiency_ratio_uses_close_prices_from_dataframe():
    df = pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0, 101.5],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.5, 100.0, 100.5, 101.0],
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )

    result = analyze_efficiency_ratio(df)

    assert result.efficiency_ratio == pytest.approx(1.0)
    assert result.direction == "UP"
    assert result.n_prices == 4


def test_efficiency_ratio_requires_at_least_two_prices():
    with pytest.raises(ValueError, match="at least two prices"):
        calculate_efficiency_ratio([100.0])
