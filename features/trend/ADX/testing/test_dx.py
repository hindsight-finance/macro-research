from pathlib import Path

import numpy as np
import pandas as pd

from features.trend.ADX.adx_calc import calculate_dx_full_from_df
from features.trend.ADX.di_crossovers import calculate_crossover_penalty, count_di_crossovers
from features.trend.ADX.di_persistence import (
    calculate_di_persistence,
    calculate_di_persistence_avg,
    calculate_margin_weighted_persistence,
    calculate_recency_weighted_persistence,
    calculate_time_in_control_persistence,
)


def _load_window(start_time: str, end_time: str) -> pd.DataFrame:
    data_path = Path(__file__).parent / "NQ.csv"
    df = pd.read_csv(data_path, parse_dates=["DateTime_ET"])
    times = df["DateTime_ET"].dt.strftime("%H:%M")
    mask = (times >= start_time) & (times < end_time)
    return df.loc[mask].reset_index(drop=True)


def test_calculate_dx_full_from_df_excludes_adx_column():
    bars = _load_window("15:45", "16:00")

    result = calculate_dx_full_from_df(
        bars,
        period=5,
        high_col="High",
        low_col="Low",
        close_col="Close",
    )

    assert len(result) == len(bars)
    assert {"TR", "+DM", "-DM", "ATR", "+DI", "-DI", "DX"} <= set(result.columns)
    assert "ADX" not in result.columns


def test_dx_helpers_return_bounded_scores_on_valid_di_window():
    bars = _load_window("15:45", "16:00")
    result = calculate_dx_full_from_df(
        bars,
        period=5,
        high_col="High",
        low_col="Low",
        close_col="Close",
    )

    plus_di = result["+DI"].to_numpy()
    minus_di = result["-DI"].to_numpy()
    valid_mask = ~(np.isnan(plus_di) | np.isnan(minus_di))
    plus_di = plus_di[valid_mask]
    minus_di = minus_di[valid_mask]

    assert len(plus_di) > 0
    assert 0.0 <= calculate_di_persistence(plus_di, minus_di) <= 1.0
    assert 0.0 <= calculate_di_persistence_avg(plus_di, minus_di) <= 1.0
    assert 0.0 <= calculate_margin_weighted_persistence(plus_di, minus_di) <= 1.0
    assert 0.0 <= calculate_time_in_control_persistence(plus_di, minus_di) <= 1.0
    assert 0.0 <= calculate_recency_weighted_persistence(plus_di, minus_di) <= 1.0
    assert 0.0 <= calculate_crossover_penalty(plus_di, minus_di) <= 1.0
    assert count_di_crossovers(plus_di, minus_di) >= 0


def test_new_persistence_variants_score_clean_control_above_alternation():
    plus_clean = np.array([30, 31, 33, 34, 35, 36], dtype=float)
    minus_clean = np.array([10, 10, 11, 11, 12, 12], dtype=float)
    plus_chop = np.array([30, 10, 30, 10, 30, 10], dtype=float)
    minus_chop = np.array([10, 30, 10, 30, 10, 30], dtype=float)

    assert calculate_margin_weighted_persistence(plus_clean, minus_clean) > calculate_margin_weighted_persistence(plus_chop, minus_chop)
    assert calculate_time_in_control_persistence(plus_clean, minus_clean) > calculate_time_in_control_persistence(plus_chop, minus_chop)
    assert calculate_recency_weighted_persistence(plus_clean, minus_clean) > calculate_recency_weighted_persistence(plus_chop, minus_chop)


def test_margin_weighted_persistence_rewards_stronger_di_margin():
    plus_strong = np.array([40, 42, 44, 46], dtype=float)
    minus_strong = np.array([10, 11, 12, 13], dtype=float)
    plus_weak = np.array([21, 21, 22, 22], dtype=float)
    minus_weak = np.array([20, 20, 21, 21], dtype=float)

    assert calculate_margin_weighted_persistence(plus_strong, minus_strong) > calculate_margin_weighted_persistence(plus_weak, minus_weak)


def test_calculate_di_persistence_uses_margin_weighted_definition():
    plus_di = np.array([30, 31, 33, 15, 16, 36], dtype=float)
    minus_di = np.array([10, 10, 11, 25, 28, 12], dtype=float)

    assert calculate_di_persistence(plus_di, minus_di) == calculate_margin_weighted_persistence(plus_di, minus_di)
