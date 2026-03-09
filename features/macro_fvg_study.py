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
FINAL_SCAN_TIME = "15:59:00"


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
                "is_confirmable_by_1559": (
                    pd.Timestamp(bar3["DateTime_ET"]) + pd.Timedelta(minutes=1)
                ).strftime("%H:%M:%S")
                <= FINAL_SCAN_TIME,
            }
        )

    return pd.DataFrame(events)


def _bar_retraces_gap(bar: pd.Series, gap_bottom: float, gap_top: float) -> bool:
    return float(bar["High"]) >= gap_bottom and float(bar["Low"]) <= gap_top


def _bar_invalidates_gap(bar: pd.Series, fvg_side: str, gap_bottom: float, gap_top: float) -> bool:
    close = float(bar["Close"])
    if fvg_side == "bullish":
        return close < gap_bottom
    return close > gap_top


def scan_fvg_outcomes_until_1559_close(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    work_bars = bars.copy()
    work_bars["DateTime_ET"] = pd.to_datetime(work_bars["DateTime_ET"])
    work_bars = work_bars.sort_values("DateTime_ET").reset_index(drop=True)

    scanned_rows = []
    for _, event in events.iterrows():
        event_dict = event.to_dict()
        if not event_dict["is_confirmable_by_1559"]:
            event_dict.update(
                {
                    "first_retrace_at": pd.NaT,
                    "first_invalidation_at": pd.NaT,
                    "retraced_by_1559": False,
                    "invalidated_by_1559": False,
                    "held_to_1559_close": False,
                    "untouched_to_1559_close": False,
                    "last_observed_at": pd.NaT,
                }
            )
            scanned_rows.append(event_dict)
            continue

        scan_end = pd.Timestamp(event_dict["assigned_at"]).normalize() + pd.Timedelta(
            hours=15, minutes=59
        )
        scan_df = work_bars[
            (work_bars["DateTime_ET"] >= event_dict["confirmed_at"])
            & (work_bars["DateTime_ET"] <= scan_end)
        ]

        first_retrace_at = pd.NaT
        first_invalidation_at = pd.NaT

        for _, bar in scan_df.iterrows():
            if pd.isna(first_retrace_at) and _bar_retraces_gap(
                bar, event_dict["gap_bottom"], event_dict["gap_top"]
            ):
                first_retrace_at = bar["DateTime_ET"]

            if pd.isna(first_invalidation_at) and _bar_invalidates_gap(
                bar,
                event_dict["fvg_side"],
                event_dict["gap_bottom"],
                event_dict["gap_top"],
            ):
                first_invalidation_at = bar["DateTime_ET"]

            if pd.notna(first_retrace_at) and pd.notna(first_invalidation_at):
                break

        event_dict.update(
            {
                "first_retrace_at": first_retrace_at,
                "first_invalidation_at": first_invalidation_at,
                "retraced_by_1559": bool(pd.notna(first_retrace_at)),
                "invalidated_by_1559": bool(pd.notna(first_invalidation_at)),
                "held_to_1559_close": bool(pd.isna(first_invalidation_at)),
                "untouched_to_1559_close": bool(pd.isna(first_retrace_at)),
                "last_observed_at": scan_end,
            }
        )
        scanned_rows.append(event_dict)

    return pd.DataFrame(scanned_rows)
