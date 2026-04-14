from pathlib import Path

import numpy as np
import pandas as pd

from features.trend.ADX.adx_calc import calculate_dx_full_from_df
from features.trend.ADX.di_crossovers import calculate_crossover_penalty, count_di_crossovers
from features.trend.ADX.di_persistence import calculate_di_persistence, calculate_di_persistence_avg


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
    assert 0.0 <= calculate_crossover_penalty(plus_di, minus_di) <= 1.0
    assert count_di_crossovers(plus_di, minus_di) >= 0
