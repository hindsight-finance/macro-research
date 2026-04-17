from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from utils.minute_bars import (
        build_market_time_columns,
        derive_session_window,
        load_minute_bars,
        normalize_minute_bars,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.minute_bars import (
        build_market_time_columns,
        derive_session_window,
        load_minute_bars,
        normalize_minute_bars,
    )

INPUT_PATH = Path("outputs/nq_minute_base.parquet")
EVENTS_OUTPUT_PATH = Path("outputs/nq_macro_fvg_events.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_fvg_summary.parquet")
FIGURES_DIR = Path("outputs/figs/fvg")

MACRO_WINDOW = "MACRO"
NO_NEW_ASSIGNMENTS_AT = "15:59:00"
STAGE_1_END = "15:54:00"
STAGE_2_END = "15:58:00"
FINAL_SCAN_TIME = "15:59:00"
ALIGNMENT_BUCKET_ORDER = [
    "3_aligned",
    "2_aligned_1_opposite",
    "1_aligned_2_opposite",
    "contains_neutral",
]
MINUTE_BLOCK_ORDER = ["15:50-15:52", "15:53-15:57", "15:58_unconfirmable"]
GAP_SIZE_BUCKET_ORDER = ["<2.25", ">=2.25"]
SUMMARY_COLUMNS = [
    "summary_scope",
    "fvg_side",
    "alignment_bucket",
    "stacked_continuation_fvg",
    "minute_block",
    "gap_size_bucket_225",
    "n_total",
    "n_confirmable",
    "n_retraced",
    "n_successful",
    "n_triggered",
    "entry_trigger_rate",
    "hold_rate",
    "retrace_rate",
    "untouched_rate",
    "invalidation_rate",
    "success_after_retrace_rate",
    "successful_share_of_confirmable",
    "mfe_pct_mean",
    "mfe_pct_median",
    "mfe_pct_p75",
    "mfe_pct_p90",
    "mae_pct_mean",
    "mae_pct_median",
    "mae_pct_p75",
    "mae_pct_p90",
    "assigned_minute_hhmm",
    "assigned_minute_index",
    "bar2_volume_bucket",
]


def _prepare_macro_bars(df: pd.DataFrame) -> pd.DataFrame:
    work = derive_session_window(build_market_time_columns(normalize_minute_bars(df)))
    work["DateTime_ET"] = work["datetime_et"]
    return work.sort_values("DateTime_ET").reset_index(drop=True)


def assign_stage(ts: pd.Timestamp) -> str:
    hhmmss = ts.strftime("%H:%M:%S")
    if "15:50:00" <= hhmmss <= STAGE_1_END:
        return "stage_1"
    if "15:55:00" <= hhmmss <= STAGE_2_END:
        return "stage_2"
    return "outside"


def classify_candle_direction(open_price: float, close_price: float) -> str:
    if close_price > open_price:
        return "bullish"
    if close_price < open_price:
        return "bearish"
    return "neutral"


def assign_alignment_bucket(
    fvg_side: str,
    directions: list[str],
) -> tuple[int, int, int, str]:
    aligned_label = "bullish" if fvg_side == "bullish" else "bearish"
    aligned_count = sum(direction == aligned_label for direction in directions)
    neutral_count = sum(direction == "neutral" for direction in directions)
    opposite_count = len(directions) - aligned_count - neutral_count

    if neutral_count > 0:
        bucket = "contains_neutral"
    elif aligned_count == 3:
        bucket = "3_aligned"
    elif aligned_count == 2:
        bucket = "2_aligned_1_opposite"
    elif aligned_count == 1:
        bucket = "1_aligned_2_opposite"
    else:
        raise ValueError("Unexpected zero-aligned non-neutral FVG pattern")

    return aligned_count, opposite_count, neutral_count, bucket


def assign_entry_price(event: pd.Series | dict) -> float:
    if event["fvg_side"] == "bullish":
        return float(event["bar3_high"])
    return float(event["bar3_low"])


def _bar_triggers_entry(bar: pd.Series, fvg_side: str, entry_price: float) -> bool:
    if fvg_side == "bullish":
        return float(bar["High"]) >= entry_price
    return float(bar["Low"]) <= entry_price


def _calculate_excursions_from_entry(
    scan_df: pd.DataFrame,
    fvg_side: str,
    entry_price: float,
) -> tuple[float, float]:
    if scan_df.empty:
        return float("nan"), float("nan")

    max_high = float(scan_df["High"].max())
    min_low = float(scan_df["Low"].min())
    if fvg_side == "bullish":
        mfe_pct = (max_high - entry_price) / entry_price
        mae_pct = (entry_price - min_low) / entry_price
    else:
        mfe_pct = (entry_price - min_low) / entry_price
        mae_pct = (max_high - entry_price) / entry_price

    return float(mfe_pct), float(mae_pct)


def mark_stacked_continuation_fvgs(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["stacked_continuation_fvg"] = False
    work["stack_predecessor_assigned_at"] = pd.NaT
    if work.empty:
        return work

    for idx in range(1, len(work)):
        current = work.iloc[idx]
        previous = work.iloc[idx - 1]
        if (
            current["date"] == previous["date"]
            and current["fvg_side"] == previous["fvg_side"]
            and current["assigned_at"] == previous["bar3_time"]
        ):
            work.at[work.index[idx], "stacked_continuation_fvg"] = True
            work.at[work.index[idx], "stack_predecessor_assigned_at"] = previous["assigned_at"]
    return work


def detect_macro_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = _prepare_macro_bars(df)
    work["bar1_date"] = work["DateTime_ET"].shift(1).dt.normalize()
    work["bar3_date"] = work["DateTime_ET"].shift(-1).dt.normalize()
    work["bar1_open"] = work["Open"].shift(1)
    work["bar1_close"] = work["Close"].shift(1)
    work["bar1_high"] = work["High"].shift(1)
    work["bar1_low"] = work["Low"].shift(1)
    work["bar3_time"] = work["DateTime_ET"].shift(-1)
    work["bar3_open"] = work["Open"].shift(-1)
    work["bar3_close"] = work["Close"].shift(-1)
    work["bar3_high"] = work["High"].shift(-1)
    work["bar3_low"] = work["Low"].shift(-1)

    same_day = (
        work["DateTime_ET"].dt.normalize().eq(work["bar1_date"])
        & work["DateTime_ET"].dt.normalize().eq(work["bar3_date"])
    )
    not_last_assignment = work["DateTime_ET"].dt.strftime("%H:%M:%S") != NO_NEW_ASSIGNMENTS_AT
    bullish_mask = work["bar3_low"] > work["bar1_high"]
    bearish_mask = work["bar3_high"] < work["bar1_low"]

    event_rows = work[
        (work["window"] == MACRO_WINDOW)
        & same_day
        & not_last_assignment
        & (bullish_mask | bearish_mask)
    ].copy()
    if event_rows.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "fvg_side",
                "assigned_at",
                "confirmed_at",
                "assigned_stage",
                "assigned_minute_hhmm",
                "assigned_minute_index",
                "gap_bottom",
                "gap_top",
                "gap_size",
                "bar2_volume",
                "bar3_high",
                "bar3_low",
                "is_confirmable_by_1559",
                "bar1_direction",
                "bar2_direction",
                "bar3_direction",
                "aligned_count",
                "opposite_count",
                "neutral_count",
                "alignment_bucket",
                "minute_block",
                "gap_size_bucket_225",
                "stacked_continuation_fvg",
                "stack_predecessor_assigned_at",
            ]
        )

    event_rows["date"] = event_rows["DateTime_ET"].dt.normalize()
    event_rows["assigned_at"] = event_rows["DateTime_ET"]
    event_rows["confirmed_at"] = event_rows["bar3_time"] + pd.Timedelta(minutes=1)
    event_rows["assigned_stage"] = event_rows["assigned_at"].map(assign_stage)
    event_rows["assigned_minute_hhmm"] = event_rows["assigned_at"].dt.strftime("%H:%M")
    event_rows["assigned_minute_index"] = (
        event_rows["assigned_at"].dt.hour * 60
        + event_rows["assigned_at"].dt.minute
        - (15 * 60 + 50)
    )
    event_rows["fvg_side"] = np.where(bullish_mask.loc[event_rows.index], "bullish", "bearish")
    event_rows["gap_bottom"] = np.where(
        bullish_mask.loc[event_rows.index],
        event_rows["bar1_high"],
        event_rows["bar3_high"],
    )
    event_rows["gap_top"] = np.where(
        bullish_mask.loc[event_rows.index],
        event_rows["bar3_low"],
        event_rows["bar1_low"],
    )
    event_rows["gap_size"] = event_rows["gap_top"] - event_rows["gap_bottom"]
    event_rows["bar2_volume"] = event_rows["Volume"]
    event_rows["is_confirmable_by_1559"] = (
        event_rows["confirmed_at"].dt.strftime("%H:%M:%S") <= FINAL_SCAN_TIME
    )
    event_rows["minute_block"] = np.where(
        event_rows["assigned_minute_index"] <= 2,
        "15:50-15:52",
        np.where(
            event_rows["assigned_minute_index"] <= 7,
            "15:53-15:57",
            "15:58_unconfirmable",
        ),
    )
    event_rows["gap_size_bucket_225"] = np.where(
        event_rows["gap_size"] < 2.25,
        "<2.25",
        ">=2.25",
    )
    event_rows["bar1_direction"] = [
        classify_candle_direction(open_price, close_price)
        for open_price, close_price in zip(event_rows["bar1_open"], event_rows["bar1_close"])
    ]
    event_rows["bar2_direction"] = [
        classify_candle_direction(open_price, close_price)
        for open_price, close_price in zip(event_rows["Open"], event_rows["Close"])
    ]
    event_rows["bar3_direction"] = [
        classify_candle_direction(open_price, close_price)
        for open_price, close_price in zip(event_rows["bar3_open"], event_rows["bar3_close"])
    ]
    alignment = [
        assign_alignment_bucket(fvg_side, [bar1_direction, bar2_direction, bar3_direction])
        for fvg_side, bar1_direction, bar2_direction, bar3_direction in zip(
            event_rows["fvg_side"],
            event_rows["bar1_direction"],
            event_rows["bar2_direction"],
            event_rows["bar3_direction"],
        )
    ]
    event_rows[
        ["aligned_count", "opposite_count", "neutral_count", "alignment_bucket"]
    ] = pd.DataFrame(
        alignment,
        index=event_rows.index,
        columns=["aligned_count", "opposite_count", "neutral_count", "alignment_bucket"],
    )
    event_rows = mark_stacked_continuation_fvgs(event_rows)

    return event_rows[
        [
            "date",
            "fvg_side",
            "assigned_at",
            "confirmed_at",
            "assigned_stage",
            "assigned_minute_hhmm",
            "assigned_minute_index",
            "gap_bottom",
            "gap_top",
            "gap_size",
            "bar2_volume",
            "bar3_high",
            "bar3_low",
            "is_confirmable_by_1559",
            "bar1_direction",
            "bar2_direction",
            "bar3_direction",
            "aligned_count",
            "opposite_count",
            "neutral_count",
            "alignment_bucket",
            "minute_block",
            "gap_size_bucket_225",
            "stacked_continuation_fvg",
            "stack_predecessor_assigned_at",
        ]
    ].reset_index(drop=True)


def _bar_retraces_gap(bar: pd.Series, gap_bottom: float, gap_top: float) -> bool:
    return float(bar["High"]) >= gap_bottom and float(bar["Low"]) <= gap_top


def _bar_invalidates_gap(bar: pd.Series, fvg_side: str, gap_bottom: float, gap_top: float) -> bool:
    close = float(bar["Close"])
    if fvg_side == "bullish":
        return close < gap_bottom
    return close > gap_top


def _success_reference_price(event: pd.Series | dict) -> float:
    if event["fvg_side"] == "bullish":
        return float(event["first_retrace_candle_high"])
    return float(event["first_retrace_candle_low"])


def _bar_breaks_retrace_reference(bar: pd.Series, fvg_side: str, reference_price: float) -> bool:
    if fvg_side == "bullish":
        return float(bar["High"]) > reference_price
    return float(bar["Low"]) < reference_price


def scan_fvg_outcomes_until_1559_close(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    work_bars = _prepare_macro_bars(bars)
    work_bars = work_bars[work_bars["window"] == MACRO_WINDOW].copy()
    work_bars["date"] = work_bars["DateTime_ET"].dt.normalize()
    bars_by_date = {
        date: group.reset_index(drop=True)
        for date, group in work_bars.groupby("date", sort=False)
    }

    scanned_rows = []
    for _, event in events.iterrows():
        event_dict = event.to_dict()
        entry_price = assign_entry_price(event_dict)
        if not event_dict["is_confirmable_by_1559"]:
            event_dict.update(
                {
                    "entry_price": entry_price,
                    "entry_triggered_by_1559": False,
                    "first_entry_trigger_at": pd.NaT,
                    "entry_trigger_minute_hhmm": pd.NA,
                    "entry_trigger_minute_index": pd.NA,
                    "mfe_pct_to_1559": float("nan"),
                    "mae_pct_to_1559": float("nan"),
                    "first_retrace_at": pd.NaT,
                    "first_retrace_candle_at": pd.NaT,
                    "first_retrace_candle_open": float("nan"),
                    "first_retrace_candle_high": float("nan"),
                    "first_retrace_candle_low": float("nan"),
                    "first_retrace_candle_close": float("nan"),
                    "success_reference_price": float("nan"),
                    "successful_by_1559": False,
                    "success_break_at": pd.NaT,
                    "first_invalidation_at": pd.NaT,
                    "retraced_by_1559": False,
                    "invalidated_by_1559": False,
                    "held_to_1559_close": False,
                    "untouched_to_1559_close": False,
                    "retraced_in_stage_2": False,
                    "invalidated_in_stage_2": False,
                    "held_through_stage_2": False,
                    "untouched_through_stage_2": False,
                    "last_observed_at": pd.NaT,
                }
            )
            scanned_rows.append(event_dict)
            continue

        session_date = pd.Timestamp(event_dict["assigned_at"]).normalize()
        day_bars = bars_by_date.get(session_date)
        if day_bars is None:
            event_dict.update(
                {
                    "entry_price": entry_price,
                    "entry_triggered_by_1559": False,
                    "first_entry_trigger_at": pd.NaT,
                    "entry_trigger_minute_hhmm": pd.NA,
                    "entry_trigger_minute_index": pd.NA,
                    "mfe_pct_to_1559": float("nan"),
                    "mae_pct_to_1559": float("nan"),
                    "first_retrace_at": pd.NaT,
                    "first_retrace_candle_at": pd.NaT,
                    "first_retrace_candle_open": float("nan"),
                    "first_retrace_candle_high": float("nan"),
                    "first_retrace_candle_low": float("nan"),
                    "first_retrace_candle_close": float("nan"),
                    "success_reference_price": float("nan"),
                    "successful_by_1559": False,
                    "success_break_at": pd.NaT,
                    "first_invalidation_at": pd.NaT,
                    "retraced_by_1559": False,
                    "invalidated_by_1559": False,
                    "held_to_1559_close": True,
                    "untouched_to_1559_close": True,
                    "retraced_in_stage_2": False,
                    "invalidated_in_stage_2": False,
                    "held_through_stage_2": True,
                    "untouched_through_stage_2": True,
                    "last_observed_at": session_date + pd.Timedelta(hours=15, minutes=59),
                }
            )
            scanned_rows.append(event_dict)
            continue

        scan_end = session_date + pd.Timedelta(
            hours=15, minutes=59
        )
        stage_2_start = session_date + pd.Timedelta(hours=15, minutes=55)
        scan_df = day_bars[
            (day_bars["DateTime_ET"] >= event_dict["confirmed_at"])
            & (day_bars["DateTime_ET"] <= scan_end)
        ]
        stage_2_scan_start = max(pd.Timestamp(event_dict["confirmed_at"]), stage_2_start)
        stage_2_scan_df = day_bars[
            (day_bars["DateTime_ET"] >= stage_2_scan_start)
            & (day_bars["DateTime_ET"] <= scan_end)
        ]

        first_retrace_at = pd.NaT
        first_retrace_candle = None
        first_invalidation_at = pd.NaT
        first_stage_2_retrace_at = pd.NaT
        first_stage_2_invalidation_at = pd.NaT
        first_entry_trigger_at = pd.NaT

        for _, bar in scan_df.iterrows():
            if pd.isna(first_entry_trigger_at) and _bar_triggers_entry(
                bar,
                event_dict["fvg_side"],
                entry_price,
            ):
                first_entry_trigger_at = bar["DateTime_ET"]

            if pd.isna(first_retrace_at) and _bar_retraces_gap(
                bar, event_dict["gap_bottom"], event_dict["gap_top"]
            ):
                first_retrace_at = bar["DateTime_ET"]
                first_retrace_candle = bar

            if pd.isna(first_invalidation_at) and _bar_invalidates_gap(
                bar,
                event_dict["fvg_side"],
                event_dict["gap_bottom"],
                event_dict["gap_top"],
            ):
                first_invalidation_at = bar["DateTime_ET"]

            if pd.notna(first_retrace_at) and pd.notna(first_invalidation_at):
                break

        for _, bar in stage_2_scan_df.iterrows():
            if pd.isna(first_stage_2_retrace_at) and _bar_retraces_gap(
                bar, event_dict["gap_bottom"], event_dict["gap_top"]
            ):
                first_stage_2_retrace_at = bar["DateTime_ET"]

            if pd.isna(first_stage_2_invalidation_at) and _bar_invalidates_gap(
                bar,
                event_dict["fvg_side"],
                event_dict["gap_bottom"],
                event_dict["gap_top"],
            ):
                first_stage_2_invalidation_at = bar["DateTime_ET"]

            if pd.notna(first_stage_2_retrace_at) and pd.notna(first_stage_2_invalidation_at):
                break

        if pd.isna(first_entry_trigger_at):
            entry_trigger_minute_hhmm = pd.NA
            entry_trigger_minute_index = pd.NA
            mfe_pct_to_1559 = float("nan")
            mae_pct_to_1559 = float("nan")
        else:
            entry_trigger_minute_hhmm = pd.Timestamp(first_entry_trigger_at).strftime("%H:%M")
            entry_trigger_minute_index = (
                pd.Timestamp(first_entry_trigger_at).hour * 60
                + pd.Timestamp(first_entry_trigger_at).minute
                - (15 * 60 + 50)
            )
            post_trigger_df = day_bars[
                (day_bars["DateTime_ET"] > first_entry_trigger_at)
                & (day_bars["DateTime_ET"] <= scan_end)
            ]
            mfe_pct_to_1559, mae_pct_to_1559 = _calculate_excursions_from_entry(
                post_trigger_df,
                event_dict["fvg_side"],
                entry_price,
            )

        if first_retrace_candle is None:
            success_reference_price = float("nan")
            successful_by_1559 = False
            success_break_at = pd.NaT
        else:
            success_reference_price = _success_reference_price(
                {
                    "fvg_side": event_dict["fvg_side"],
                    "first_retrace_candle_high": first_retrace_candle["High"],
                    "first_retrace_candle_low": first_retrace_candle["Low"],
                }
            )
            success_break_at = pd.NaT
            success_scan_df = day_bars[
                (day_bars["DateTime_ET"] > first_retrace_at)
                & (day_bars["DateTime_ET"] <= scan_end)
            ]
            for _, bar in success_scan_df.iterrows():
                if _bar_breaks_retrace_reference(
                    bar,
                    event_dict["fvg_side"],
                    success_reference_price,
                ):
                    success_break_at = bar["DateTime_ET"]
                    break
            successful_by_1559 = bool(pd.notna(success_break_at))

        event_dict.update(
            {
                "entry_price": entry_price,
                "entry_triggered_by_1559": bool(pd.notna(first_entry_trigger_at)),
                "first_entry_trigger_at": first_entry_trigger_at,
                "entry_trigger_minute_hhmm": entry_trigger_minute_hhmm,
                "entry_trigger_minute_index": entry_trigger_minute_index,
                "mfe_pct_to_1559": mfe_pct_to_1559,
                "mae_pct_to_1559": mae_pct_to_1559,
                "first_retrace_at": first_retrace_at,
                "first_retrace_candle_at": first_retrace_at,
                "first_retrace_candle_open": (
                    float(first_retrace_candle["Open"])
                    if first_retrace_candle is not None
                    else float("nan")
                ),
                "first_retrace_candle_high": (
                    float(first_retrace_candle["High"])
                    if first_retrace_candle is not None
                    else float("nan")
                ),
                "first_retrace_candle_low": (
                    float(first_retrace_candle["Low"])
                    if first_retrace_candle is not None
                    else float("nan")
                ),
                "first_retrace_candle_close": (
                    float(first_retrace_candle["Close"])
                    if first_retrace_candle is not None
                    else float("nan")
                ),
                "success_reference_price": success_reference_price,
                "successful_by_1559": successful_by_1559,
                "success_break_at": success_break_at,
                "first_invalidation_at": first_invalidation_at,
                "retraced_by_1559": bool(pd.notna(first_retrace_at)),
                "invalidated_by_1559": bool(pd.notna(first_invalidation_at)),
                "held_to_1559_close": bool(pd.isna(first_invalidation_at)),
                "untouched_to_1559_close": bool(pd.isna(first_retrace_at)),
                "retraced_in_stage_2": bool(pd.notna(first_stage_2_retrace_at)),
                "invalidated_in_stage_2": bool(pd.notna(first_stage_2_invalidation_at)),
                "held_through_stage_2": bool(pd.isna(first_stage_2_invalidation_at)),
                "untouched_through_stage_2": bool(pd.isna(first_stage_2_retrace_at)),
                "last_observed_at": scan_end,
            }
        )
        scanned_rows.append(event_dict)

    return pd.DataFrame(scanned_rows)


def _empty_summary_table() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def _build_scope_summary(
    events: pd.DataFrame,
    scope_name: str,
    retrace_col: str,
    invalidate_col: str,
    held_col: str,
    untouched_col: str,
) -> pd.DataFrame:
    if events.empty:
        return _empty_summary_table()

    rows = []
    for fvg_side, group in events.groupby("fvg_side"):
        n_total = len(group)
        n_confirmable = int(group["is_confirmable_by_1559"].sum())
        if n_confirmable == 0:
            hold_rate = float("nan")
            retrace_rate = float("nan")
            untouched_rate = float("nan")
            invalidation_rate = float("nan")
        else:
            hold_rate = float(group[held_col].sum() / n_confirmable)
            retrace_rate = float(group[retrace_col].sum() / n_confirmable)
            untouched_rate = float(group[untouched_col].sum() / n_confirmable)
            invalidation_rate = float(group[invalidate_col].sum() / n_confirmable)

        rows.append(
            {
                "summary_scope": scope_name,
                "fvg_side": fvg_side,
                "n_total": int(n_total),
                "n_confirmable": n_confirmable,
                "hold_rate": hold_rate,
                "retrace_rate": retrace_rate,
                "untouched_rate": untouched_rate,
                "invalidation_rate": invalidation_rate,
            }
        )

    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)


def _group_outcome_rates(events: pd.DataFrame, group_cols: list[str], scope_name: str) -> pd.DataFrame:
    if events.empty:
        return _empty_summary_table()

    rows = []
    for group_key, group in events.groupby(group_cols, dropna=False, sort=False):
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_cols, group_values))
        n_total = len(group)
        n_confirmable = int(group["is_confirmable_by_1559"].sum())
        if n_confirmable == 0:
            hold_rate = float("nan")
            retrace_rate = float("nan")
            untouched_rate = float("nan")
            invalidation_rate = float("nan")
        else:
            hold_rate = float(group["held_to_1559_close"].sum() / n_confirmable)
            retrace_rate = float(group["retraced_by_1559"].sum() / n_confirmable)
            untouched_rate = float(group["untouched_to_1559_close"].sum() / n_confirmable)
            invalidation_rate = float(group["invalidated_by_1559"].sum() / n_confirmable)

        row.update(
            {
                "summary_scope": scope_name,
                "fvg_side": np.nan,
                "n_total": int(n_total),
                "n_confirmable": n_confirmable,
                "hold_rate": hold_rate,
                "retrace_rate": retrace_rate,
                "untouched_rate": untouched_rate,
                "invalidation_rate": invalidation_rate,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)


def _percentile_or_nan(series: pd.Series, q: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def _group_entry_excursion_stats(
    events: pd.DataFrame,
    group_cols: list[str],
    scope_name: str,
) -> pd.DataFrame:
    if events.empty:
        return _empty_summary_table()

    grouped = [((None,), events)] if not group_cols else events.groupby(group_cols, dropna=False, sort=False)
    rows = []
    for group_key, group in grouped:
        row = {}
        if group_cols:
            group_values = group_key if isinstance(group_key, tuple) else (group_key,)
            row.update(dict(zip(group_cols, group_values)))

        n_total = len(group)
        n_confirmable = int(group["is_confirmable_by_1559"].sum())
        triggered = group[group["entry_triggered_by_1559"].fillna(False)]
        n_triggered = len(triggered)
        if n_confirmable == 0:
            entry_trigger_rate = float("nan")
        else:
            entry_trigger_rate = float(n_triggered / n_confirmable)

        mfe = triggered["mfe_pct_to_1559"].dropna()
        mae = triggered["mae_pct_to_1559"].dropna()
        row.update(
            {
                "summary_scope": scope_name,
                "fvg_side": np.nan,
                "n_total": int(n_total),
                "n_confirmable": n_confirmable,
                "n_triggered": int(n_triggered),
                "entry_trigger_rate": entry_trigger_rate,
                "mfe_pct_mean": float(mfe.mean()) if not mfe.empty else float("nan"),
                "mfe_pct_median": float(mfe.median()) if not mfe.empty else float("nan"),
                "mfe_pct_p75": _percentile_or_nan(mfe, 0.75),
                "mfe_pct_p90": _percentile_or_nan(mfe, 0.90),
                "mae_pct_mean": float(mae.mean()) if not mae.empty else float("nan"),
                "mae_pct_median": float(mae.median()) if not mae.empty else float("nan"),
                "mae_pct_p75": _percentile_or_nan(mae, 0.75),
                "mae_pct_p90": _percentile_or_nan(mae, 0.90),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)


def _group_success_context_stats(
    events: pd.DataFrame,
    group_cols: list[str],
    scope_name: str,
) -> pd.DataFrame:
    if events.empty:
        return _empty_summary_table()

    grouped = [((None,), events)] if not group_cols else events.groupby(group_cols, dropna=False, sort=False)
    rows = []
    for group_key, group in grouped:
        row = {}
        if group_cols:
            group_values = group_key if isinstance(group_key, tuple) else (group_key,)
            row.update(dict(zip(group_cols, group_values)))

        n_total = len(group)
        n_confirmable = int(group["is_confirmable_by_1559"].fillna(False).sum())
        n_retraced = int(group["retraced_by_1559"].fillna(False).sum())
        n_successful = int(group["successful_by_1559"].fillna(False).sum())
        successful = group[group["successful_by_1559"].fillna(False)]
        mfe = successful["mfe_pct_to_1559"].dropna()
        mae = successful["mae_pct_to_1559"].dropna()

        if n_confirmable == 0:
            retrace_rate = float("nan")
            successful_share_of_confirmable = float("nan")
        else:
            retrace_rate = float(n_retraced / n_confirmable)
            successful_share_of_confirmable = float(n_successful / n_confirmable)

        if n_retraced == 0:
            success_after_retrace_rate = float("nan")
        else:
            success_after_retrace_rate = float(n_successful / n_retraced)

        row.update(
            {
                "summary_scope": scope_name,
                "fvg_side": np.nan,
                "n_total": int(n_total),
                "n_confirmable": n_confirmable,
                "n_retraced": n_retraced,
                "n_successful": n_successful,
                "retrace_rate": retrace_rate,
                "success_after_retrace_rate": success_after_retrace_rate,
                "successful_share_of_confirmable": successful_share_of_confirmable,
                "mfe_pct_mean": float(mfe.mean()) if not mfe.empty else float("nan"),
                "mfe_pct_median": float(mfe.median()) if not mfe.empty else float("nan"),
                "mfe_pct_p75": _percentile_or_nan(mfe, 0.75),
                "mae_pct_mean": float(mae.mean()) if not mae.empty else float("nan"),
                "mae_pct_median": float(mae.median()) if not mae.empty else float("nan"),
                "mae_pct_p75": _percentile_or_nan(mae, 0.75),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)


def build_creation_minute_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_outcome_rates(
        events,
        ["assigned_minute_index", "assigned_minute_hhmm"],
        "creation_minute",
    )


def _add_bar2_volume_bucket(events: pd.DataFrame, bucket_count: int = 4) -> pd.DataFrame:
    work = events.copy()
    if work.empty:
        work["bar2_volume_bucket"] = pd.Series(dtype="object")
        return work

    unique_count = work["bar2_volume"].nunique(dropna=False)
    if unique_count <= 1:
        work["bar2_volume_bucket"] = "all"
    else:
        work["bar2_volume_bucket"] = pd.qcut(
            work["bar2_volume"],
            q=min(bucket_count, unique_count),
            duplicates="drop",
        ).astype(str)
    return work


def build_bar2_volume_summary(events: pd.DataFrame, bucket_count: int = 4) -> pd.DataFrame:
    if events.empty:
        return _empty_summary_table()

    work = _add_bar2_volume_bucket(events, bucket_count=bucket_count)
    return _group_outcome_rates(work, ["bar2_volume_bucket"], "bar2_volume_bucket")


def build_alignment_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_outcome_rates(events, ["alignment_bucket"], "alignment_bucket")


def build_alignment_bucket_minute_block_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_outcome_rates(
        events,
        ["minute_block", "alignment_bucket"],
        "alignment_bucket_minute_block",
    )


def build_alignment_bucket_gap_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_outcome_rates(
        events,
        ["gap_size_bucket_225", "alignment_bucket"],
        "alignment_bucket_gap_bucket",
    )


def build_entry_excursion_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_entry_excursion_stats(events, [], "entry_excursion_overall")


def build_entry_excursion_alignment_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_entry_excursion_stats(
        events,
        ["alignment_bucket"],
        "entry_excursion_alignment_bucket",
    )


def build_entry_excursion_minute_block_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_entry_excursion_stats(
        events,
        ["minute_block"],
        "entry_excursion_minute_block",
    )


def build_entry_excursion_gap_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_entry_excursion_stats(
        events,
        ["gap_size_bucket_225"],
        "entry_excursion_gap_bucket",
    )


def build_entry_excursion_alignment_bucket_minute_block_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    return _group_entry_excursion_stats(
        events,
        ["alignment_bucket", "minute_block"],
        "entry_excursion_alignment_bucket_minute_block",
    )


def build_success_context_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_success_context_stats(events, [], "success_context_overall")


def build_success_context_alignment_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_success_context_stats(
        events,
        ["alignment_bucket"],
        "success_context_alignment_bucket",
    )


def build_success_context_stacked_flag_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_success_context_stats(
        events,
        ["stacked_continuation_fvg"],
        "success_context_stacked_flag",
    )


def build_success_context_alignment_bucket_stacked_flag_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    return _group_success_context_stats(
        events,
        ["alignment_bucket", "stacked_continuation_fvg"],
        "success_context_alignment_bucket_stacked_flag",
    )


def build_stage_summary_tables(events: pd.DataFrame) -> pd.DataFrame:
    stage_1 = events[events["assigned_stage"] == "stage_1"]
    stage_2 = events[events["assigned_stage"] == "stage_2"]
    frames = [
        _build_scope_summary(
            stage_1,
            "stage_1",
            "retraced_by_1559",
            "invalidated_by_1559",
            "held_to_1559_close",
            "untouched_to_1559_close",
        ),
        _build_scope_summary(
            stage_2,
            "stage_2",
            "retraced_by_1559",
            "invalidated_by_1559",
            "held_to_1559_close",
            "untouched_to_1559_close",
        ),
        _build_scope_summary(
            stage_1,
            "stage_1_to_stage_2",
            "retraced_in_stage_2",
            "invalidated_in_stage_2",
            "held_through_stage_2",
            "untouched_through_stage_2",
        ),
    ]
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return _empty_summary_table()
    return pd.concat(non_empty_frames, ignore_index=True).reindex(columns=SUMMARY_COLUMNS)


def build_summary_tables(events: pd.DataFrame) -> pd.DataFrame:
    frames = [
        build_stage_summary_tables(events),
        build_creation_minute_summary(events),
        build_bar2_volume_summary(events),
        build_alignment_bucket_summary(events),
        build_alignment_bucket_minute_block_summary(events),
        build_alignment_bucket_gap_bucket_summary(events),
        build_entry_excursion_summary(events),
        build_entry_excursion_alignment_bucket_summary(events),
        build_entry_excursion_minute_block_summary(events),
        build_entry_excursion_gap_bucket_summary(events),
        build_entry_excursion_alignment_bucket_minute_block_summary(events),
        build_success_context_summary(events),
        build_success_context_alignment_bucket_summary(events),
        build_success_context_stacked_flag_summary(events),
        build_success_context_alignment_bucket_stacked_flag_summary(events),
    ]
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return _empty_summary_table()
    return pd.concat(non_empty_frames, ignore_index=True).reindex(columns=SUMMARY_COLUMNS)


def _event_outcome_bucket(event: pd.Series) -> str:
    if not event["is_confirmable_by_1559"]:
        return "unconfirmable"
    if event["invalidated_by_1559"]:
        return "invalidated"
    if event["retraced_by_1559"]:
        return "retraced_held"
    return "untouched_held"


def _save_placeholder_figure(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_creation_minute_outcomes(events: pd.DataFrame, figures_dir: Path) -> None:
    minute_summary = build_creation_minute_summary(events)
    if minute_summary.empty:
        _save_placeholder_figure(
            figures_dir / "creation_minute_outcome_bars.png",
            "Creation Minute Outcome Bars",
        )
        return

    plot_frame = (
        minute_summary.sort_values(["assigned_minute_index", "assigned_minute_hhmm"])
        .set_index("assigned_minute_hhmm")[["hold_rate", "retrace_rate", "invalidation_rate"]]
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#31a354", "#fd8d3c", "#de2d26"])
    ax.set_title("Creation Minute Outcomes")
    ax.set_xlabel("Creation Minute")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "creation_minute_outcome_bars.png")
    plt.close(fig)


def plot_bar2_volume_bucket_outcomes(events: pd.DataFrame, figures_dir: Path) -> None:
    volume_summary = build_bar2_volume_summary(events)
    if volume_summary.empty:
        _save_placeholder_figure(
            figures_dir / "bar2_volume_bucket_outcomes.png",
            "Bar-2 Volume Bucket Outcomes",
        )
        return

    plot_frame = volume_summary.set_index("bar2_volume_bucket")[
        ["hold_rate", "retrace_rate", "invalidation_rate"]
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#31a354", "#fd8d3c", "#de2d26"])
    ax.set_title("Bar-2 Volume Bucket Outcomes")
    ax.set_xlabel("Bar-2 Volume Bucket")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "bar2_volume_bucket_outcomes.png")
    plt.close(fig)


def plot_creation_minute_avg_bar2_volume(events: pd.DataFrame, figures_dir: Path) -> None:
    if events.empty:
        _save_placeholder_figure(
            figures_dir / "creation_minute_avg_bar2_volume.png",
            "Creation Minute Average Bar-2 Volume",
        )
        return

    volume_frame = (
        events.groupby(["assigned_minute_index", "assigned_minute_hhmm"], sort=False)["bar2_volume"]
        .mean()
        .reset_index()
        .sort_values(["assigned_minute_index", "assigned_minute_hhmm"])
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(volume_frame["assigned_minute_hhmm"], volume_frame["bar2_volume"], color="#3182bd")
    ax.set_title("Creation Minute Average Bar-2 Volume")
    ax.set_xlabel("Creation Minute")
    ax.set_ylabel("Average Volume")
    fig.tight_layout()
    fig.savefig(figures_dir / "creation_minute_avg_bar2_volume.png")
    plt.close(fig)


def plot_creation_minute_volume_heatmap(events: pd.DataFrame, figures_dir: Path) -> None:
    if events.empty:
        _save_placeholder_figure(
            figures_dir / "creation_minute_volume_heatmap.png",
            "Creation Minute Volume Heatmap",
        )
        return

    work = _add_bar2_volume_bucket(events)
    minute_order = (
        work[["assigned_minute_index", "assigned_minute_hhmm"]]
        .drop_duplicates()
        .sort_values(["assigned_minute_index", "assigned_minute_hhmm"])["assigned_minute_hhmm"]
    )
    heatmap_frame = (
        work.groupby(["assigned_minute_hhmm", "bar2_volume_bucket"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=minute_order, fill_value=0)
    )
    if heatmap_frame.empty:
        _save_placeholder_figure(
            figures_dir / "creation_minute_volume_heatmap.png",
            "Creation Minute Volume Heatmap",
        )
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    image = ax.imshow(heatmap_frame.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_title("Creation Minute Volume Heatmap")
    ax.set_xlabel("Bar-2 Volume Bucket")
    ax.set_ylabel("Creation Minute")
    ax.set_xticks(range(len(heatmap_frame.columns)))
    ax.set_xticklabels(heatmap_frame.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heatmap_frame.index)))
    ax.set_yticklabels(heatmap_frame.index)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(figures_dir / "creation_minute_volume_heatmap.png")
    plt.close(fig)


def plot_alignment_bucket_outcomes(summary: pd.DataFrame, figures_dir: Path) -> None:
    alignment_summary = summary[summary["summary_scope"] == "alignment_bucket"].copy()
    if alignment_summary.empty:
        _save_placeholder_figure(
            figures_dir / "alignment_bucket_outcomes.png",
            "Alignment Bucket Outcomes",
        )
        return

    alignment_summary["alignment_bucket"] = pd.Categorical(
        alignment_summary["alignment_bucket"],
        categories=ALIGNMENT_BUCKET_ORDER,
        ordered=True,
    )
    plot_frame = (
        alignment_summary.sort_values("alignment_bucket")
        .dropna(subset=["alignment_bucket"])
        .set_index("alignment_bucket")[["hold_rate", "retrace_rate", "invalidation_rate"]]
    )
    if plot_frame.empty:
        _save_placeholder_figure(
            figures_dir / "alignment_bucket_outcomes.png",
            "Alignment Bucket Outcomes",
        )
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#31a354", "#fd8d3c", "#de2d26"])
    ax.set_title("Alignment Bucket Outcomes")
    ax.set_xlabel("Alignment Bucket")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "alignment_bucket_outcomes.png")
    plt.close(fig)


def plot_alignment_bucket_by_minute_block(summary: pd.DataFrame, figures_dir: Path) -> None:
    minute_summary = summary[summary["summary_scope"] == "alignment_bucket_minute_block"].copy()
    if minute_summary.empty:
        _save_placeholder_figure(
            figures_dir / "alignment_bucket_by_minute_block.png",
            "Alignment Bucket by Minute Block",
        )
        return

    minute_summary["alignment_bucket"] = pd.Categorical(
        minute_summary["alignment_bucket"],
        categories=ALIGNMENT_BUCKET_ORDER,
        ordered=True,
    )
    plot_frame = (
        minute_summary.pivot(
            index="alignment_bucket",
            columns="minute_block",
            values="hold_rate",
        )
        .reindex(index=ALIGNMENT_BUCKET_ORDER, columns=MINUTE_BLOCK_ORDER)
        .dropna(how="all")
    )
    if plot_frame.empty:
        _save_placeholder_figure(
            figures_dir / "alignment_bucket_by_minute_block.png",
            "Alignment Bucket by Minute Block",
        )
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#3182bd", "#6baed6", "#9ecae1"])
    ax.set_title("Alignment Hold Rate by Minute Block")
    ax.set_xlabel("Alignment Bucket")
    ax.set_ylabel("Hold Rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Minute Block", loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "alignment_bucket_by_minute_block.png")
    plt.close(fig)


def plot_alignment_bucket_by_gap_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    gap_summary = summary[summary["summary_scope"] == "alignment_bucket_gap_bucket"].copy()
    if gap_summary.empty:
        _save_placeholder_figure(
            figures_dir / "alignment_bucket_by_gap_bucket.png",
            "Alignment Bucket by Gap Bucket",
        )
        return

    gap_summary["alignment_bucket"] = pd.Categorical(
        gap_summary["alignment_bucket"],
        categories=ALIGNMENT_BUCKET_ORDER,
        ordered=True,
    )
    plot_frame = (
        gap_summary.pivot(
            index="alignment_bucket",
            columns="gap_size_bucket_225",
            values="hold_rate",
        )
        .reindex(index=ALIGNMENT_BUCKET_ORDER, columns=GAP_SIZE_BUCKET_ORDER)
        .dropna(how="all")
    )
    if plot_frame.empty:
        _save_placeholder_figure(
            figures_dir / "alignment_bucket_by_gap_bucket.png",
            "Alignment Bucket by Gap Bucket",
        )
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#756bb1", "#bcbddc"])
    ax.set_title("Alignment Hold Rate by Gap Bucket")
    ax.set_xlabel("Alignment Bucket")
    ax.set_ylabel("Hold Rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Gap Bucket", loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "alignment_bucket_by_gap_bucket.png")
    plt.close(fig)


def plot_alignment_bucket_counts(events: pd.DataFrame, figures_dir: Path) -> None:
    if events.empty:
        _save_placeholder_figure(
            figures_dir / "alignment_bucket_counts.png",
            "Alignment Bucket Counts",
        )
        return

    count_frame = events["alignment_bucket"].value_counts().reindex(ALIGNMENT_BUCKET_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(count_frame.index, count_frame.values, color="#636363")
    ax.set_title("Alignment Bucket Counts")
    ax.set_xlabel("Alignment Bucket")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "alignment_bucket_counts.png")
    plt.close(fig)


def plot_entry_trigger_rate_by_alignment_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    excursion_summary = summary[
        summary["summary_scope"] == "entry_excursion_alignment_bucket"
    ].copy()
    if excursion_summary.empty:
        _save_placeholder_figure(
            figures_dir / "entry_trigger_rate_by_alignment_bucket.png",
            "Entry Trigger Rate by Alignment Bucket",
        )
        return

    excursion_summary["alignment_bucket"] = pd.Categorical(
        excursion_summary["alignment_bucket"],
        categories=ALIGNMENT_BUCKET_ORDER,
        ordered=True,
    )
    plot_frame = excursion_summary.sort_values("alignment_bucket").dropna(
        subset=["alignment_bucket"]
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(plot_frame["alignment_bucket"], plot_frame["entry_trigger_rate"], color="#3182bd")
    ax.set_title("Entry Trigger Rate by Alignment Bucket")
    ax.set_xlabel("Alignment Bucket")
    ax.set_ylabel("Trigger Rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "entry_trigger_rate_by_alignment_bucket.png")
    plt.close(fig)


def plot_mfe_mae_pct_by_alignment_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    excursion_summary = summary[
        summary["summary_scope"] == "entry_excursion_alignment_bucket"
    ].copy()
    if excursion_summary.empty:
        _save_placeholder_figure(
            figures_dir / "mfe_mae_pct_by_alignment_bucket.png",
            "MFE and MAE Percent by Alignment Bucket",
        )
        return

    excursion_summary["alignment_bucket"] = pd.Categorical(
        excursion_summary["alignment_bucket"],
        categories=ALIGNMENT_BUCKET_ORDER,
        ordered=True,
    )
    plot_frame = (
        excursion_summary.sort_values("alignment_bucket")
        .dropna(subset=["alignment_bucket"])
        .set_index("alignment_bucket")[["mfe_pct_mean", "mae_pct_mean"]]
        * 100.0
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#31a354", "#de2d26"])
    ax.set_title("MFE and MAE Percent by Alignment Bucket")
    ax.set_xlabel("Alignment Bucket")
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "mfe_mae_pct_by_alignment_bucket.png")
    plt.close(fig)


def plot_mfe_pct_by_minute_block(summary: pd.DataFrame, figures_dir: Path) -> None:
    minute_summary = summary[summary["summary_scope"] == "entry_excursion_minute_block"].copy()
    if minute_summary.empty:
        _save_placeholder_figure(
            figures_dir / "mfe_pct_by_minute_block.png",
            "MFE Percent by Minute Block",
        )
        return

    minute_summary["minute_block"] = pd.Categorical(
        minute_summary["minute_block"],
        categories=MINUTE_BLOCK_ORDER,
        ordered=True,
    )
    plot_frame = minute_summary.sort_values("minute_block").dropna(subset=["minute_block"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(plot_frame["minute_block"], plot_frame["mfe_pct_mean"] * 100.0, color="#756bb1")
    ax.set_title("MFE Percent by Minute Block")
    ax.set_xlabel("Minute Block")
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "mfe_pct_by_minute_block.png")
    plt.close(fig)


def plot_mfe_pct_by_gap_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    gap_summary = summary[summary["summary_scope"] == "entry_excursion_gap_bucket"].copy()
    if gap_summary.empty:
        _save_placeholder_figure(
            figures_dir / "mfe_pct_by_gap_bucket.png",
            "MFE Percent by Gap Bucket",
        )
        return

    gap_summary["gap_size_bucket_225"] = pd.Categorical(
        gap_summary["gap_size_bucket_225"],
        categories=GAP_SIZE_BUCKET_ORDER,
        ordered=True,
    )
    plot_frame = gap_summary.sort_values("gap_size_bucket_225").dropna(
        subset=["gap_size_bucket_225"]
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(plot_frame["gap_size_bucket_225"], plot_frame["mfe_pct_mean"] * 100.0, color="#fd8d3c")
    ax.set_title("MFE Percent by Gap Bucket")
    ax.set_xlabel("Gap Bucket")
    ax.set_ylabel("Percent")
    fig.tight_layout()
    fig.savefig(figures_dir / "mfe_pct_by_gap_bucket.png")
    plt.close(fig)


def plot_successful_fvg_mae_by_alignment_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    success_summary = summary[
        summary["summary_scope"] == "success_context_alignment_bucket"
    ].copy()
    if success_summary.empty:
        _save_placeholder_figure(
            figures_dir / "successful_fvg_mae_by_alignment_bucket.png",
            "Successful FVG MAE by Alignment Bucket",
        )
        return

    success_summary["alignment_bucket"] = pd.Categorical(
        success_summary["alignment_bucket"],
        categories=ALIGNMENT_BUCKET_ORDER,
        ordered=True,
    )
    ordered_summary = (
        success_summary.sort_values("alignment_bucket")
        .dropna(subset=["alignment_bucket"])
    )
    plot_frame = ordered_summary.set_index("alignment_bucket")[
        ["mae_pct_mean", "mae_pct_median", "mae_pct_p75"]
    ] * 100.0
    if plot_frame.empty:
        _save_placeholder_figure(
            figures_dir / "successful_fvg_mae_by_alignment_bucket.png",
            "Successful FVG MAE by Alignment Bucket",
        )
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#de2d26", "#fd8d3c", "#3182bd"])
    ax.set_title("Successful FVG MAE by Alignment Bucket")
    ax.set_xlabel("Alignment Bucket")
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", rotation=20)
    success_rate_pct = ordered_summary.set_index("alignment_bucket")[
        "successful_share_of_confirmable"
    ] * 100.0
    ax2 = ax.twinx()
    ax2.plot(
        range(len(plot_frame.index)),
        success_rate_pct.reindex(plot_frame.index).to_numpy(),
        color="#252525",
        marker="o",
        linewidth=2,
        label="Success Rate",
    )
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 100)

    mae_handles, mae_labels = ax.get_legend_handles_labels()
    success_handles, success_labels = ax2.get_legend_handles_labels()
    ax.legend(mae_handles + success_handles, mae_labels + success_labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "successful_fvg_mae_by_alignment_bucket.png")
    plt.close(fig)


def plot_successful_fvg_mae_by_stacked_flag(summary: pd.DataFrame, figures_dir: Path) -> None:
    success_summary = summary[
        summary["summary_scope"] == "success_context_stacked_flag"
    ].copy()
    if success_summary.empty:
        _save_placeholder_figure(
            figures_dir / "successful_fvg_mae_by_stacked_flag.png",
            "Successful FVG MAE by Stacked Flag",
        )
        return

    success_summary["stack_label"] = np.where(
        success_summary["stacked_continuation_fvg"].fillna(False),
        "stacked_continuation",
        "base_fvg",
    )
    plot_frame = (
        success_summary.set_index("stack_label")[["mae_pct_mean", "mae_pct_median", "mae_pct_p75"]]
        * 100.0
    )
    if plot_frame.empty:
        _save_placeholder_figure(
            figures_dir / "successful_fvg_mae_by_stacked_flag.png",
            "Successful FVG MAE by Stacked Flag",
        )
        return

    desired_order = [label for label in ["base_fvg", "stacked_continuation"] if label in plot_frame.index]
    plot_frame = plot_frame.reindex(desired_order)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#de2d26", "#fd8d3c", "#3182bd"])
    ax.set_title("Successful FVG MAE by Stacked Flag")
    ax.set_xlabel("FVG Type")
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(["MAE Mean", "MAE Median", "MAE P75"], loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "successful_fvg_mae_by_stacked_flag.png")
    plt.close(fig)


def plot_successful_fvg_mfe_by_alignment_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    success_summary = summary[
        summary["summary_scope"] == "success_context_alignment_bucket"
    ].copy()
    if success_summary.empty:
        _save_placeholder_figure(
            figures_dir / "successful_fvg_mfe_by_alignment_bucket.png",
            "Successful FVG MFE by Alignment Bucket",
        )
        return

    success_summary["alignment_bucket"] = pd.Categorical(
        success_summary["alignment_bucket"],
        categories=ALIGNMENT_BUCKET_ORDER,
        ordered=True,
    )
    plot_frame = (
        success_summary.sort_values("alignment_bucket")
        .dropna(subset=["alignment_bucket"])
        .set_index("alignment_bucket")[["mfe_pct_mean", "mfe_pct_median", "mfe_pct_p75"]]
        * 100.0
    )
    if plot_frame.empty:
        _save_placeholder_figure(
            figures_dir / "successful_fvg_mfe_by_alignment_bucket.png",
            "Successful FVG MFE by Alignment Bucket",
        )
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#31a354", "#74c476", "#3182bd"])
    ax.set_title("Successful FVG MFE by Alignment Bucket")
    ax.set_xlabel("Alignment Bucket")
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(["MFE Mean", "MFE Median", "MFE P75"], loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "successful_fvg_mfe_by_alignment_bucket.png")
    plt.close(fig)


def plot_successful_fvg_mfe_by_stacked_flag(summary: pd.DataFrame, figures_dir: Path) -> None:
    success_summary = summary[
        summary["summary_scope"] == "success_context_stacked_flag"
    ].copy()
    if success_summary.empty:
        _save_placeholder_figure(
            figures_dir / "successful_fvg_mfe_by_stacked_flag.png",
            "Successful FVG MFE by Stacked Flag",
        )
        return

    success_summary["stack_label"] = np.where(
        success_summary["stacked_continuation_fvg"].fillna(False),
        "stacked_continuation",
        "base_fvg",
    )
    plot_frame = (
        success_summary.set_index("stack_label")[["mfe_pct_mean", "mfe_pct_median", "mfe_pct_p75"]]
        * 100.0
    )
    if plot_frame.empty:
        _save_placeholder_figure(
            figures_dir / "successful_fvg_mfe_by_stacked_flag.png",
            "Successful FVG MFE by Stacked Flag",
        )
        return

    desired_order = [label for label in ["base_fvg", "stacked_continuation"] if label in plot_frame.index]
    plot_frame = plot_frame.reindex(desired_order)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_frame.plot(kind="bar", ax=ax, color=["#31a354", "#74c476", "#3182bd"])
    ax.set_title("Successful FVG MFE by Stacked Flag")
    ax.set_xlabel("FVG Type")
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(["MFE Mean", "MFE Median", "MFE P75"], loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "successful_fvg_mfe_by_stacked_flag.png")
    plt.close(fig)


def plot_fvg_summary_figures(events: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    if events.empty:
        for filename, title in [
            ("hold_vs_invalidate_by_side.png", "Hold vs Invalidate by Side"),
            ("stage1_to_stage2_outcomes.png", "Stage 1 to Stage 2 Outcomes"),
            ("creation_minute_outcome_heatmap.png", "Creation Minute Outcome Heatmap"),
            ("gap_size_vs_outcome.png", "Gap Size vs Outcome"),
            ("creation_minute_outcome_bars.png", "Creation Minute Outcome Bars"),
            ("bar2_volume_bucket_outcomes.png", "Bar-2 Volume Bucket Outcomes"),
            ("creation_minute_avg_bar2_volume.png", "Creation Minute Average Bar-2 Volume"),
            ("creation_minute_volume_heatmap.png", "Creation Minute Volume Heatmap"),
            ("alignment_bucket_outcomes.png", "Alignment Bucket Outcomes"),
            ("alignment_bucket_by_minute_block.png", "Alignment Bucket by Minute Block"),
            ("alignment_bucket_by_gap_bucket.png", "Alignment Bucket by Gap Bucket"),
            ("alignment_bucket_counts.png", "Alignment Bucket Counts"),
            ("entry_trigger_rate_by_alignment_bucket.png", "Entry Trigger Rate by Alignment Bucket"),
            ("mfe_mae_pct_by_alignment_bucket.png", "MFE and MAE Percent by Alignment Bucket"),
            ("mfe_pct_by_minute_block.png", "MFE Percent by Minute Block"),
            ("mfe_pct_by_gap_bucket.png", "MFE Percent by Gap Bucket"),
            ("successful_fvg_mae_by_alignment_bucket.png", "Successful FVG MAE by Alignment Bucket"),
            ("successful_fvg_mae_by_stacked_flag.png", "Successful FVG MAE by Stacked Flag"),
            ("successful_fvg_mfe_by_alignment_bucket.png", "Successful FVG MFE by Alignment Bucket"),
            ("successful_fvg_mfe_by_stacked_flag.png", "Successful FVG MFE by Stacked Flag"),
        ]:
            _save_placeholder_figure(figures_dir / filename, title)
        return

    plot_events = events.copy()
    plot_events["outcome_bucket"] = plot_events.apply(_event_outcome_bucket, axis=1)

    # Hold vs invalidate by side
    hold_frame = (
        plot_events.assign(
            hold_bucket=np.where(
                plot_events["invalidated_by_1559"],
                "invalidated",
                np.where(plot_events["retraced_by_1559"], "retraced_held", "untouched_held"),
            )
        )
        .groupby(["fvg_side", "hold_bucket"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["untouched_held", "retraced_held", "invalidated"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    hold_frame.plot(kind="bar", stacked=True, ax=ax, color=["#6baed6", "#fd8d3c", "#de2d26"])
    ax.set_title("Hold vs Invalidate by Side")
    ax.set_xlabel("FVG Side")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "hold_vs_invalidate_by_side.png")
    plt.close(fig)

    # Stage 1 to stage 2 outcomes
    transition_frame = (
        plot_events[plot_events["assigned_stage"] == "stage_1"]
        .assign(
            stage_2_bucket=np.where(
                plot_events.loc[plot_events["assigned_stage"] == "stage_1", "invalidated_in_stage_2"],
                "invalidated_in_stage_2",
                np.where(
                    plot_events.loc[plot_events["assigned_stage"] == "stage_1", "retraced_in_stage_2"],
                    "retraced_in_stage_2",
                    "untouched_through_stage_2",
                ),
            )
        )
        .groupby(["fvg_side", "stage_2_bucket"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["retraced_in_stage_2", "invalidated_in_stage_2", "untouched_through_stage_2"],
            fill_value=0,
        )
    )
    if transition_frame.empty:
        _save_placeholder_figure(
            figures_dir / "stage1_to_stage2_outcomes.png",
            "Stage 1 to Stage 2 Outcomes",
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        transition_frame.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=["#31a354", "#de2d26", "#9ecae1"],
        )
        ax.set_title("Stage 1 to Stage 2 Outcomes")
        ax.set_xlabel("FVG Side")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(figures_dir / "stage1_to_stage2_outcomes.png")
        plt.close(fig)

    # Creation minute heatmap
    heatmap_frame = (
        plot_events.assign(creation_minute=plot_events["assigned_at"].dt.strftime("%H:%M"))
        .groupby(["creation_minute", "outcome_bucket"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["untouched_held", "retraced_held", "invalidated", "unconfirmable"],
            fill_value=0,
        )
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(heatmap_frame.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_title("Creation Minute Outcome Heatmap")
    ax.set_xticks(range(len(heatmap_frame.columns)))
    ax.set_xticklabels(heatmap_frame.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heatmap_frame.index)))
    ax.set_yticklabels(heatmap_frame.index)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(figures_dir / "creation_minute_outcome_heatmap.png")
    plt.close(fig)

    # Gap size vs outcome
    bucket_count = min(3, max(1, plot_events["gap_size"].nunique()))
    if bucket_count == 1:
        plot_events["gap_bucket"] = "all"
    else:
        plot_events["gap_bucket"] = pd.qcut(
            plot_events["gap_size"],
            q=bucket_count,
            duplicates="drop",
        ).astype(str)
    gap_frame = (
        plot_events.groupby(["gap_bucket", "outcome_bucket"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["untouched_held", "retraced_held", "invalidated", "unconfirmable"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    gap_frame.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Gap Size vs Outcome")
    ax.set_xlabel("Gap Size Bucket")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "gap_size_vs_outcome.png")
    plt.close(fig)

    plot_creation_minute_outcomes(events, figures_dir)
    plot_bar2_volume_bucket_outcomes(events, figures_dir)
    plot_creation_minute_avg_bar2_volume(events, figures_dir)
    plot_creation_minute_volume_heatmap(events, figures_dir)
    plot_alignment_bucket_outcomes(summary, figures_dir)
    plot_alignment_bucket_by_minute_block(summary, figures_dir)
    plot_alignment_bucket_by_gap_bucket(summary, figures_dir)
    plot_alignment_bucket_counts(events, figures_dir)
    plot_entry_trigger_rate_by_alignment_bucket(summary, figures_dir)
    plot_mfe_mae_pct_by_alignment_bucket(summary, figures_dir)
    plot_mfe_pct_by_minute_block(summary, figures_dir)
    plot_mfe_pct_by_gap_bucket(summary, figures_dir)
    plot_successful_fvg_mae_by_alignment_bucket(summary, figures_dir)
    plot_successful_fvg_mae_by_stacked_flag(summary, figures_dir)
    plot_successful_fvg_mfe_by_alignment_bucket(summary, figures_dir)
    plot_successful_fvg_mfe_by_stacked_flag(summary, figures_dir)


def run_macro_fvg_study(
    input_path: Path = INPUT_PATH,
    events_output_path: Path = EVENTS_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = load_minute_bars(input_path)
    events = detect_macro_fvgs(bars)
    events = scan_fvg_outcomes_until_1559_close(events, bars)
    summary = build_summary_tables(events)

    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_output_path, index=False)
    summary.to_parquet(summary_output_path, index=False)
    plot_fvg_summary_figures(events, summary, figures_dir)
    return events, summary


if __name__ == "__main__":
    run_macro_fvg_study()
