from pathlib import Path

import pandas as pd

from features.trend.ADX.adx_calc import calculate_adx_full_from_df


def _load_window(start_time: str, end_time: str) -> pd.DataFrame:
    data_path = Path(__file__).parent / "NQ.csv"
    df = pd.read_csv(data_path, parse_dates=["DateTime_ET"])
    times = df["DateTime_ET"].dt.strftime("%H:%M")
    mask = (times >= start_time) & (times < end_time)
    return df.loc[mask].reset_index(drop=True)


def test_calculate_adx_full_from_df_returns_expected_columns():
    bars = _load_window("15:00", "15:50")

    result = calculate_adx_full_from_df(
        bars,
        period=12,
        high_col="High",
        low_col="Low",
        close_col="Close",
    )

    assert len(result) == len(bars)
    assert {"TR", "+DM", "-DM", "ATR", "+DI", "-DI", "DX", "ADX"} <= set(result.columns)
    assert result["ADX"].notna().any()
