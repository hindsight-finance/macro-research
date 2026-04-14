import math

import numpy as np
import pandas as pd

from features.trend.variance_ratio.vr import (
    analyze_variance_ratio,
    calculate_variance_ratio,
    calculate_variance_ratio_from_returns,
)


def _prices_from_log_returns(log_returns):
    cumulative = np.concatenate([[0.0], np.cumsum(log_returns)])
    return 100.0 * np.exp(cumulative)


def test_variance_ratio_is_above_one_for_persistent_return_blocks():
    log_returns = np.array([0.01, 0.01, -0.01, -0.01, 0.01, 0.01, -0.01, -0.01])

    result = calculate_variance_ratio_from_returns(log_returns, lag=2)

    assert result.variance_ratio > 1.0
    assert result.log_variance_ratio > 0.0
    assert result.interpretation == "persistent"


def test_variance_ratio_is_below_one_for_alternating_returns():
    log_returns = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01])

    result = calculate_variance_ratio_from_returns(log_returns, lag=2)

    assert result.variance_ratio < 1.0
    assert result.log_variance_ratio < 0.0
    assert result.interpretation == "mean_reverting"


def test_analyze_variance_ratio_uses_close_prices_from_dataframe():
    log_returns = np.array([0.01, 0.01, -0.01, -0.01, 0.01, 0.01, -0.01, -0.01])
    close = _prices_from_log_returns(log_returns)
    df = pd.DataFrame({"close": close})

    result = analyze_variance_ratio(df, lag=2)

    assert result.lag == 2
    assert result.n_returns == len(log_returns)
    assert result.log_variance_ratio > 0.0


def test_flat_prices_map_to_neutral_log_variance_ratio():
    result = calculate_variance_ratio([100.0, 100.0, 100.0, 100.0, 100.0], lag=2)

    assert result.variance_ratio == 1.0
    assert math.isclose(result.log_variance_ratio, 0.0)
    assert result.interpretation == "neutral"
