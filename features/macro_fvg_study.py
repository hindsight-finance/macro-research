from pathlib import Path

import pandas as pd


INPUT_PATH = Path("outputs/nq_1m.parquet")
EVENTS_OUTPUT_PATH = Path("outputs/nq_macro_fvg_events.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_fvg_summary.parquet")
FIGURES_DIR = Path("outputs/figs/fvg")

MACRO_WINDOW = "MACRO"
NO_NEW_ASSIGNMENTS_AT = "15:59:00"
STAGE_1_END = "15:54:00"
STAGE_2_END = "15:58:00"


def assign_stage(ts: pd.Timestamp) -> str:
    hhmmss = ts.strftime("%H:%M:%S")
    if "15:50:00" <= hhmmss <= STAGE_1_END:
        return "stage_1"
    if "15:55:00" <= hhmmss <= STAGE_2_END:
        return "stage_2"
    return "outside"


def detect_macro_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    required = {"DateTime_ET", "Open", "High", "Low", "Close", "window"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["DateTime_ET"] = pd.to_datetime(work["DateTime_ET"])
    work = work.sort_values("DateTime_ET").reset_index(drop=True)

    events = []

    for idx in range(2, len(work)):
        bar1 = work.iloc[idx - 2]
        bar2 = work.iloc[idx - 1]
        bar3 = work.iloc[idx]

        assigned_at = pd.Timestamp(bar2["DateTime_ET"])
        if bar2["window"] != MACRO_WINDOW:
            continue
        if assigned_at.strftime("%H:%M:%S") == NO_NEW_ASSIGNMENTS_AT:
            continue

        is_bullish = float(bar3["Low"]) > float(bar1["High"])
        is_bearish = float(bar3["High"]) < float(bar1["Low"])
        if not is_bullish and not is_bearish:
            continue

        if is_bullish:
            gap_bottom = float(bar1["High"])
            gap_top = float(bar3["Low"])
            fvg_side = "bullish"
        else:
            gap_bottom = float(bar3["High"])
            gap_top = float(bar1["Low"])
            fvg_side = "bearish"

        events.append(
            {
                "date": assigned_at.normalize(),
                "fvg_side": fvg_side,
                "assigned_at": assigned_at,
                "confirmed_at": pd.Timestamp(bar3["DateTime_ET"]) + pd.Timedelta(minutes=1),
                "assigned_stage": assign_stage(assigned_at),
                "gap_bottom": gap_bottom,
                "gap_top": gap_top,
                "gap_size": gap_top - gap_bottom,
            }
        )

    return pd.DataFrame(events)
