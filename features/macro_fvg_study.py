from pathlib import Path
import sys
from datetime import datetime, timedelta
from math import isnan

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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

try:
    from features.macro_fvg_delta_dominance import (
        DELTA_DOMINANCE_COLUMNS,
        try_enrich_fvg_events_with_delta_dominance,
    )
    from volume_delta import OUTPUT_MACRO_5S_PATH
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from features.macro_fvg_delta_dominance import (
        DELTA_DOMINANCE_COLUMNS,
        try_enrich_fvg_events_with_delta_dominance,
    )
    from volume_delta import OUTPUT_MACRO_5S_PATH

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
    "aligned_delta_imbalance_quantile",
    "abs_delta_imbalance_quantile",
]

EVENT_COLUMNS = [
    "date", "fvg_side", "assigned_at", "confirmed_at", "assigned_stage",
    "assigned_minute_hhmm", "assigned_minute_index", "gap_bottom", "gap_top",
    "gap_size", "bar2_volume", "bar3_high", "bar3_low", "is_confirmable_by_1559",
    "bar1_direction", "bar2_direction", "bar3_direction", "aligned_count",
    "opposite_count", "neutral_count", "alignment_bucket", "minute_block",
    "gap_size_bucket_225", "stacked_continuation_fvg", "stack_predecessor_assigned_at",
    *DELTA_DOMINANCE_COLUMNS,
]

OUTCOME_COLUMNS = [
    "entry_price", "entry_triggered_by_1559", "first_entry_trigger_at",
    "entry_trigger_minute_hhmm", "entry_trigger_minute_index", "mfe_pct_to_1559",
    "mae_pct_to_1559", "first_retrace_at", "first_retrace_candle_at",
    "first_retrace_candle_open", "first_retrace_candle_high", "first_retrace_candle_low",
    "first_retrace_candle_close", "success_reference_price", "successful_by_1559",
    "success_break_at", "first_invalidation_at", "retraced_by_1559",
    "invalidated_by_1559", "held_to_1559_close", "untouched_to_1559_close",
    "retraced_in_stage_2", "invalidated_in_stage_2", "held_through_stage_2",
    "untouched_through_stage_2", "last_observed_at",
]

def _nan() -> float:
    return float("nan")


def _is_null(value) -> bool:
    return value is None or (isinstance(value, float) and isnan(value))


def _hhmm(dt: datetime | None) -> str | None:
    return dt.strftime("%H:%M") if dt is not None else None


def _minute_index(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    return dt.hour * 60 + dt.minute - (15 * 60 + 50)


def _summary_frame(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({column: [] for column in SUMMARY_COLUMNS})
    return pl.DataFrame(rows).select([pl.col(c) if c in rows[0] else pl.lit(None).alias(c) for c in SUMMARY_COLUMNS])


def _frame_from_rows(rows: list[dict], columns: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({column: [] for column in columns})
    return pl.DataFrame(rows).select([pl.col(c) if c in rows[0] else pl.lit(None).alias(c) for c in columns])


def _prepare_macro_bars(df: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(df, pl.DataFrame):
        raise TypeError("macro FVG study expects Polars DataFrame inputs")
    work = derive_session_window(build_market_time_columns(normalize_minute_bars(df)))
    return work.with_columns(DateTime_ET=pl.col("datetime_et")).sort("DateTime_ET")


def assign_stage(ts: datetime) -> str:
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


def assign_alignment_bucket(fvg_side: str, directions: list[str]) -> tuple[int, int, int, str]:
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


def assign_entry_price(event: dict) -> float:
    return float(event["bar3_high"] if event["fvg_side"] == "bullish" else event["bar3_low"])


def _bar_triggers_entry(bar: dict, fvg_side: str, entry_price: float) -> bool:
    return float(bar["High"]) >= entry_price if fvg_side == "bullish" else float(bar["Low"]) <= entry_price


def _bar_retraces_gap(bar: dict, gap_bottom: float, gap_top: float) -> bool:
    return float(bar["High"]) >= gap_bottom and float(bar["Low"]) <= gap_top


def _bar_invalidates_gap(bar: dict, fvg_side: str, gap_bottom: float, gap_top: float) -> bool:
    close = float(bar["Close"])
    return close < gap_bottom if fvg_side == "bullish" else close > gap_top


def _success_reference_price(event: dict) -> float:
    return float(event["first_retrace_candle_high"] if event["fvg_side"] == "bullish" else event["first_retrace_candle_low"])


def _bar_breaks_retrace_reference(bar: dict, fvg_side: str, reference_price: float) -> bool:
    return float(bar["High"]) > reference_price if fvg_side == "bullish" else float(bar["Low"]) < reference_price


def _calculate_excursions_from_entry(scan_rows: list[dict], fvg_side: str, entry_price: float) -> tuple[float, float]:
    if not scan_rows:
        return _nan(), _nan()
    max_high = max(float(row["High"]) for row in scan_rows)
    min_low = min(float(row["Low"]) for row in scan_rows)
    if fvg_side == "bullish":
        return float((max_high - entry_price) / entry_price), float((entry_price - min_low) / entry_price)
    return float((entry_price - min_low) / entry_price), float((max_high - entry_price) / entry_price)

def mark_stacked_continuation_fvgs(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty():
        return events.with_columns(
            pl.lit(False).alias("stacked_continuation_fvg"),
            pl.lit(None).alias("stack_predecessor_assigned_at"),
        ) if "stacked_continuation_fvg" not in events.columns else events

    rows = events.to_dicts()
    for i, row in enumerate(rows):
        row["stacked_continuation_fvg"] = False
        row["stack_predecessor_assigned_at"] = None
        if i == 0:
            continue
        previous = rows[i - 1]
        if (
            row.get("date") == previous.get("date")
            and row.get("fvg_side") == previous.get("fvg_side")
            and row.get("assigned_at") == previous.get("bar3_time", previous.get("assigned_at"))
        ):
            row["stacked_continuation_fvg"] = True
            row["stack_predecessor_assigned_at"] = previous.get("assigned_at")
    return _frame_from_rows(rows, list(dict.fromkeys([*events.columns, "stacked_continuation_fvg", "stack_predecessor_assigned_at"])))


def detect_macro_fvgs(df: pl.DataFrame) -> pl.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = _prepare_macro_bars(df).with_columns(
        bar1_date=pl.col("DateTime_ET").shift(1).dt.date(),
        bar3_date=pl.col("DateTime_ET").shift(-1).dt.date(),
        bar1_open=pl.col("Open").shift(1),
        bar1_close=pl.col("Close").shift(1),
        bar1_high=pl.col("High").shift(1),
        bar1_low=pl.col("Low").shift(1),
        bar3_time=pl.col("DateTime_ET").shift(-1),
        bar3_open=pl.col("Open").shift(-1),
        bar3_close=pl.col("Close").shift(-1),
        bar3_high=pl.col("High").shift(-1),
        bar3_low=pl.col("Low").shift(-1),
    ).with_columns(
        current_date=pl.col("DateTime_ET").dt.date(),
        hhmmss=pl.col("DateTime_ET").dt.strftime("%H:%M:%S"),
        bullish_mask=pl.col("bar3_low") > pl.col("bar1_high"),
        bearish_mask=pl.col("bar3_high") < pl.col("bar1_low"),
    )

    candidates = work.filter(
        (pl.col("window") == MACRO_WINDOW)
        & (pl.col("current_date") == pl.col("bar1_date"))
        & (pl.col("current_date") == pl.col("bar3_date"))
        & (pl.col("hhmmss") != NO_NEW_ASSIGNMENTS_AT)
        & (pl.col("bullish_mask") | pl.col("bearish_mask"))
    )
    if candidates.is_empty():
        return _frame_from_rows([], EVENT_COLUMNS)

    rows = []
    for row in candidates.to_dicts():
        fvg_side = "bullish" if row["bullish_mask"] else "bearish"
        assigned_at = row["DateTime_ET"]
        confirmed_at = row["bar3_time"] + timedelta(minutes=1)
        gap_bottom = float(row["bar1_high"] if fvg_side == "bullish" else row["bar3_high"])
        gap_top = float(row["bar3_low"] if fvg_side == "bullish" else row["bar1_low"])
        directions = [
            classify_candle_direction(row["bar1_open"], row["bar1_close"]),
            classify_candle_direction(row["Open"], row["Close"]),
            classify_candle_direction(row["bar3_open"], row["bar3_close"]),
        ]
        aligned_count, opposite_count, neutral_count, alignment_bucket = assign_alignment_bucket(fvg_side, directions)
        assigned_idx = _minute_index(assigned_at)
        rows.append({
            "date": assigned_at.date(),
            "fvg_side": fvg_side,
            "assigned_at": assigned_at,
            "confirmed_at": confirmed_at,
            "assigned_stage": assign_stage(assigned_at),
            "assigned_minute_hhmm": _hhmm(assigned_at),
            "assigned_minute_index": assigned_idx,
            "gap_bottom": gap_bottom,
            "gap_top": gap_top,
            "gap_size": gap_top - gap_bottom,
            "bar2_volume": row["Volume"],
            "bar3_high": row["bar3_high"],
            "bar3_low": row["bar3_low"],
            "is_confirmable_by_1559": confirmed_at.strftime("%H:%M:%S") <= FINAL_SCAN_TIME,
            "bar1_direction": directions[0],
            "bar2_direction": directions[1],
            "bar3_direction": directions[2],
            "aligned_count": aligned_count,
            "opposite_count": opposite_count,
            "neutral_count": neutral_count,
            "alignment_bucket": alignment_bucket,
            "minute_block": "15:50-15:52" if assigned_idx <= 2 else ("15:53-15:57" if assigned_idx <= 7 else "15:58_unconfirmable"),
            "gap_size_bucket_225": "<2.25" if (gap_top - gap_bottom) < 2.25 else ">=2.25",
            "bar3_time": row["bar3_time"],
        })
    if not rows:
        return _frame_from_rows([], EVENT_COLUMNS)
    events = mark_stacked_continuation_fvgs(pl.DataFrame(rows))
    return events.with_columns(
        [pl.lit(None).alias(column) for column in EVENT_COLUMNS if column not in events.columns]
    ).select(EVENT_COLUMNS)

def _blank_outcomes(entry_price: float, held: bool, untouched: bool, last_observed_at: datetime | None) -> dict:
    return {
        "entry_price": entry_price,
        "entry_triggered_by_1559": False,
        "first_entry_trigger_at": None,
        "entry_trigger_minute_hhmm": None,
        "entry_trigger_minute_index": None,
        "mfe_pct_to_1559": _nan(),
        "mae_pct_to_1559": _nan(),
        "first_retrace_at": None,
        "first_retrace_candle_at": None,
        "first_retrace_candle_open": _nan(),
        "first_retrace_candle_high": _nan(),
        "first_retrace_candle_low": _nan(),
        "first_retrace_candle_close": _nan(),
        "success_reference_price": _nan(),
        "successful_by_1559": False,
        "success_break_at": None,
        "first_invalidation_at": None,
        "retraced_by_1559": False,
        "invalidated_by_1559": False,
        "held_to_1559_close": held,
        "untouched_to_1559_close": untouched,
        "retraced_in_stage_2": False,
        "invalidated_in_stage_2": False,
        "held_through_stage_2": held,
        "untouched_through_stage_2": untouched,
        "last_observed_at": last_observed_at,
    }


def scan_fvg_outcomes_until_1559_close(events: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty():
        return events.clone()
    work_bars = _prepare_macro_bars(bars).filter(pl.col("window") == MACRO_WINDOW).with_columns(date=pl.col("DateTime_ET").dt.date())
    bars_by_date: dict[object, list[dict]] = {}
    for row in work_bars.to_dicts():
        bars_by_date.setdefault(row["date"], []).append(row)

    scanned_rows = []
    for event in events.to_dicts():
        entry_price = assign_entry_price(event)
        if not event["is_confirmable_by_1559"]:
            event.update(_blank_outcomes(entry_price, held=False, untouched=False, last_observed_at=None))
            scanned_rows.append(event)
            continue

        session_date = event["assigned_at"].date()
        scan_end = datetime.combine(session_date, datetime.min.time()) + timedelta(hours=15, minutes=59)
        day_bars = bars_by_date.get(session_date, [])
        if not day_bars:
            event.update(_blank_outcomes(entry_price, held=True, untouched=True, last_observed_at=scan_end))
            scanned_rows.append(event)
            continue

        scan_rows = [bar for bar in day_bars if event["confirmed_at"] <= bar["DateTime_ET"] <= scan_end]
        stage_2_start = datetime.combine(session_date, datetime.min.time()) + timedelta(hours=15, minutes=55)
        stage_2_scan_start = max(event["confirmed_at"], stage_2_start)
        stage_2_rows = [bar for bar in day_bars if stage_2_scan_start <= bar["DateTime_ET"] <= scan_end]

        first_entry_trigger_at = None
        first_retrace_at = None
        first_retrace_candle = None
        first_invalidation_at = None
        first_stage_2_retrace_at = None
        first_stage_2_invalidation_at = None

        for bar in scan_rows:
            if first_entry_trigger_at is None and _bar_triggers_entry(bar, event["fvg_side"], entry_price):
                first_entry_trigger_at = bar["DateTime_ET"]
            if first_retrace_at is None and _bar_retraces_gap(bar, event["gap_bottom"], event["gap_top"]):
                first_retrace_at = bar["DateTime_ET"]
                first_retrace_candle = bar
            if first_invalidation_at is None and _bar_invalidates_gap(bar, event["fvg_side"], event["gap_bottom"], event["gap_top"]):
                first_invalidation_at = bar["DateTime_ET"]
            if first_retrace_at is not None and first_invalidation_at is not None:
                break

        for bar in stage_2_rows:
            if first_stage_2_retrace_at is None and _bar_retraces_gap(bar, event["gap_bottom"], event["gap_top"]):
                first_stage_2_retrace_at = bar["DateTime_ET"]
            if first_stage_2_invalidation_at is None and _bar_invalidates_gap(bar, event["fvg_side"], event["gap_bottom"], event["gap_top"]):
                first_stage_2_invalidation_at = bar["DateTime_ET"]
            if first_stage_2_retrace_at is not None and first_stage_2_invalidation_at is not None:
                break

        if first_entry_trigger_at is None:
            mfe_pct_to_1559, mae_pct_to_1559 = _nan(), _nan()
        else:
            post_trigger_rows = [bar for bar in day_bars if first_entry_trigger_at < bar["DateTime_ET"] <= scan_end]
            mfe_pct_to_1559, mae_pct_to_1559 = _calculate_excursions_from_entry(post_trigger_rows, event["fvg_side"], entry_price)

        if first_retrace_candle is None:
            success_reference_price = _nan()
            success_break_at = None
        else:
            success_reference_price = _success_reference_price({
                "fvg_side": event["fvg_side"],
                "first_retrace_candle_high": first_retrace_candle["High"],
                "first_retrace_candle_low": first_retrace_candle["Low"],
            })
            success_break_at = None
            for bar in [b for b in day_bars if first_retrace_at < b["DateTime_ET"] <= scan_end]:
                if _bar_breaks_retrace_reference(bar, event["fvg_side"], success_reference_price):
                    success_break_at = bar["DateTime_ET"]
                    break

        event.update({
            "entry_price": entry_price,
            "entry_triggered_by_1559": first_entry_trigger_at is not None,
            "first_entry_trigger_at": first_entry_trigger_at,
            "entry_trigger_minute_hhmm": _hhmm(first_entry_trigger_at),
            "entry_trigger_minute_index": _minute_index(first_entry_trigger_at),
            "mfe_pct_to_1559": mfe_pct_to_1559,
            "mae_pct_to_1559": mae_pct_to_1559,
            "first_retrace_at": first_retrace_at,
            "first_retrace_candle_at": first_retrace_at,
            "first_retrace_candle_open": float(first_retrace_candle["Open"]) if first_retrace_candle is not None else _nan(),
            "first_retrace_candle_high": float(first_retrace_candle["High"]) if first_retrace_candle is not None else _nan(),
            "first_retrace_candle_low": float(first_retrace_candle["Low"]) if first_retrace_candle is not None else _nan(),
            "first_retrace_candle_close": float(first_retrace_candle["Close"]) if first_retrace_candle is not None else _nan(),
            "success_reference_price": success_reference_price,
            "successful_by_1559": success_break_at is not None,
            "success_break_at": success_break_at,
            "first_invalidation_at": first_invalidation_at,
            "retraced_by_1559": first_retrace_at is not None,
            "invalidated_by_1559": first_invalidation_at is not None,
            "held_to_1559_close": first_invalidation_at is None,
            "untouched_to_1559_close": first_retrace_at is None,
            "retraced_in_stage_2": first_stage_2_retrace_at is not None,
            "invalidated_in_stage_2": first_stage_2_invalidation_at is not None,
            "held_through_stage_2": first_stage_2_invalidation_at is None,
            "untouched_through_stage_2": first_stage_2_retrace_at is None,
            "last_observed_at": scan_end,
        })
        scanned_rows.append(event)
    return pl.DataFrame(scanned_rows)

def _bool_sum(rows: list[dict], col: str) -> int:
    return sum(1 for row in rows if bool(row.get(col) or False))


def _clean_numbers(rows: list[dict], col: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(col)
        if value is None:
            continue
        value = float(value)
        if not isnan(value):
            out.append(value)
    return out


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else _nan()


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else _nan()


def _percentile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else _nan()


def _group_rows(rows: list[dict], group_cols: list[str]) -> list[tuple[tuple, list[dict]]]:
    if not group_cols:
        return [((), rows)]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(col) for col in group_cols)
        groups.setdefault(key, []).append(row)
    return list(groups.items())


def _build_scope_summary(events: pl.DataFrame, scope_name: str, retrace_col: str, invalidate_col: str, held_col: str, untouched_col: str) -> pl.DataFrame:
    rows = events.to_dicts()
    out = []
    for (fvg_side,), group in _group_rows(rows, ["fvg_side"]):
        n_total = len(group)
        n_confirmable = _bool_sum(group, "is_confirmable_by_1559")
        out.append({
            "summary_scope": scope_name,
            "fvg_side": fvg_side,
            "n_total": n_total,
            "n_confirmable": n_confirmable,
            "hold_rate": _bool_sum(group, held_col) / n_confirmable if n_confirmable else _nan(),
            "retrace_rate": _bool_sum(group, retrace_col) / n_confirmable if n_confirmable else _nan(),
            "untouched_rate": _bool_sum(group, untouched_col) / n_confirmable if n_confirmable else _nan(),
            "invalidation_rate": _bool_sum(group, invalidate_col) / n_confirmable if n_confirmable else _nan(),
        })
    return _summary_frame(out)


def _group_outcome_rates(events: pl.DataFrame, group_cols: list[str], scope_name: str) -> pl.DataFrame:
    out = []
    for key, group in _group_rows(events.to_dicts(), group_cols):
        n_total = len(group)
        n_confirmable = _bool_sum(group, "is_confirmable_by_1559")
        row = dict(zip(group_cols, key))
        row.update({
            "summary_scope": scope_name,
            "fvg_side": None,
            "n_total": n_total,
            "n_confirmable": n_confirmable,
            "hold_rate": _bool_sum(group, "held_to_1559_close") / n_confirmable if n_confirmable else _nan(),
            "retrace_rate": _bool_sum(group, "retraced_by_1559") / n_confirmable if n_confirmable else _nan(),
            "untouched_rate": _bool_sum(group, "untouched_to_1559_close") / n_confirmable if n_confirmable else _nan(),
            "invalidation_rate": _bool_sum(group, "invalidated_by_1559") / n_confirmable if n_confirmable else _nan(),
        })
        out.append(row)
    return _summary_frame(out)


def _group_entry_excursion_stats(events: pl.DataFrame, group_cols: list[str], scope_name: str) -> pl.DataFrame:
    out = []
    for key, group in _group_rows(events.to_dicts(), group_cols):
        triggered = [row for row in group if bool(row.get("entry_triggered_by_1559") or False)]
        n_confirmable = _bool_sum(group, "is_confirmable_by_1559")
        mfe = _clean_numbers(triggered, "mfe_pct_to_1559")
        mae = _clean_numbers(triggered, "mae_pct_to_1559")
        row = dict(zip(group_cols, key))
        row.update({
            "summary_scope": scope_name,
            "fvg_side": None,
            "n_total": len(group),
            "n_confirmable": n_confirmable,
            "n_triggered": len(triggered),
            "entry_trigger_rate": len(triggered) / n_confirmable if n_confirmable else _nan(),
            "mfe_pct_mean": _mean(mfe), "mfe_pct_median": _median(mfe), "mfe_pct_p75": _percentile(mfe, 0.75), "mfe_pct_p90": _percentile(mfe, 0.90),
            "mae_pct_mean": _mean(mae), "mae_pct_median": _median(mae), "mae_pct_p75": _percentile(mae, 0.75), "mae_pct_p90": _percentile(mae, 0.90),
        })
        out.append(row)
    return _summary_frame(out)


def _group_success_context_stats(events: pl.DataFrame, group_cols: list[str], scope_name: str) -> pl.DataFrame:
    out = []
    for key, group in _group_rows(events.to_dicts(), group_cols):
        successful = [row for row in group if bool(row.get("successful_by_1559") or False)]
        n_confirmable = _bool_sum(group, "is_confirmable_by_1559")
        n_retraced = _bool_sum(group, "retraced_by_1559")
        n_successful = len(successful)
        mfe = _clean_numbers(successful, "mfe_pct_to_1559")
        mae = _clean_numbers(successful, "mae_pct_to_1559")
        row = dict(zip(group_cols, key))
        row.update({
            "summary_scope": scope_name,
            "fvg_side": row.get("fvg_side"),
            "n_total": len(group),
            "n_confirmable": n_confirmable,
            "n_retraced": n_retraced,
            "n_successful": n_successful,
            "retrace_rate": n_retraced / n_confirmable if n_confirmable else _nan(),
            "success_after_retrace_rate": n_successful / n_retraced if n_retraced else _nan(),
            "successful_share_of_confirmable": n_successful / n_confirmable if n_confirmable else _nan(),
            "mfe_pct_mean": _mean(mfe), "mfe_pct_median": _median(mfe), "mfe_pct_p75": _percentile(mfe, 0.75),
            "mae_pct_mean": _mean(mae), "mae_pct_median": _median(mae), "mae_pct_p75": _percentile(mae, 0.75),
        })
        out.append(row)
    return _summary_frame(out)


def _with_bar2_volume_bucket(events: pl.DataFrame, bucket_count: int = 4) -> pl.DataFrame:
    if events.is_empty():
        return events.with_columns(pl.lit(None).alias("bar2_volume_bucket"))
    unique_count = events.select(pl.col("bar2_volume").n_unique()).item()
    if unique_count <= 1:
        return events.with_columns(pl.lit("all").alias("bar2_volume_bucket"))
    q = min(bucket_count, unique_count)
    ranked = events.with_columns(pl.col("bar2_volume").rank(method="ordinal").alias("_rank"))
    n = events.height
    return ranked.with_columns(
        (((pl.col("_rank") - 1) * q / n).floor().cast(pl.Int64)).alias("_bucket")
    ).with_columns(
        pl.format("q{}", pl.col("_bucket") + 1).alias("bar2_volume_bucket")
    ).drop(["_rank", "_bucket"])

def build_creation_minute_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_outcome_rates(events, ["assigned_minute_index", "assigned_minute_hhmm"], "creation_minute")


def build_bar2_volume_summary(events: pl.DataFrame, bucket_count: int = 4) -> pl.DataFrame:
    return _group_outcome_rates(_with_bar2_volume_bucket(events, bucket_count), ["bar2_volume_bucket"], "bar2_volume_bucket")


def build_alignment_bucket_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_outcome_rates(events, ["alignment_bucket"], "alignment_bucket")


def build_alignment_bucket_minute_block_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_outcome_rates(events, ["minute_block", "alignment_bucket"], "alignment_bucket_minute_block")


def build_alignment_bucket_gap_bucket_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_outcome_rates(events, ["gap_size_bucket_225", "alignment_bucket"], "alignment_bucket_gap_bucket")


def build_entry_excursion_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_entry_excursion_stats(events, [], "entry_excursion_overall")


def build_entry_excursion_alignment_bucket_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_entry_excursion_stats(events, ["alignment_bucket"], "entry_excursion_alignment_bucket")


def build_entry_excursion_minute_block_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_entry_excursion_stats(events, ["minute_block"], "entry_excursion_minute_block")


def build_entry_excursion_gap_bucket_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_entry_excursion_stats(events, ["gap_size_bucket_225"], "entry_excursion_gap_bucket")


def build_entry_excursion_alignment_bucket_minute_block_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_entry_excursion_stats(events, ["alignment_bucket", "minute_block"], "entry_excursion_alignment_bucket_minute_block")


def build_success_context_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_success_context_stats(events, [], "success_context_overall")


def build_success_context_alignment_bucket_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_success_context_stats(events, ["alignment_bucket"], "success_context_alignment_bucket")


def build_success_context_stacked_flag_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_success_context_stats(events, ["stacked_continuation_fvg"], "success_context_stacked_flag")


def build_success_context_alignment_bucket_stacked_flag_summary(events: pl.DataFrame) -> pl.DataFrame:
    return _group_success_context_stats(events, ["alignment_bucket", "stacked_continuation_fvg"], "success_context_alignment_bucket_stacked_flag")


def _filter_non_null(events: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in events.columns:
        return pl.DataFrame({name: [] for name in events.columns})
    return events.filter(pl.col(column).is_not_null())


def build_success_context_aligned_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "aligned_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        [column],
        "success_context_aligned_delta_imbalance_quantile",
    )


def build_success_context_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        [column],
        "success_context_abs_delta_imbalance_quantile",
    )


def build_success_context_side_aligned_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "aligned_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["fvg_side", column],
        "success_context_side_aligned_delta_imbalance_quantile",
    )


def build_success_context_side_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["fvg_side", column],
        "success_context_side_abs_delta_imbalance_quantile",
    )


def build_success_context_creation_minute_aligned_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "aligned_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["assigned_minute_index", "assigned_minute_hhmm", column],
        "success_context_creation_minute_aligned_delta_imbalance_quantile",
    )


def build_success_context_creation_minute_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["assigned_minute_index", "assigned_minute_hhmm", column],
        "success_context_creation_minute_abs_delta_imbalance_quantile",
    )


def build_success_context_minute_block_aligned_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "aligned_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["minute_block", column],
        "success_context_minute_block_aligned_delta_imbalance_quantile",
    )


def build_success_context_minute_block_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_success_context_stats(
        _filter_non_null(events, column),
        ["minute_block", column],
        "success_context_minute_block_abs_delta_imbalance_quantile",
    )


def build_stage_summary_tables(events: pl.DataFrame) -> pl.DataFrame:
    stage_1 = events.filter(pl.col("assigned_stage") == "stage_1")
    stage_2 = events.filter(pl.col("assigned_stage") == "stage_2")
    frames = [
        _build_scope_summary(stage_1, "stage_1", "retraced_by_1559", "invalidated_by_1559", "held_to_1559_close", "untouched_to_1559_close"),
        _build_scope_summary(stage_2, "stage_2", "retraced_by_1559", "invalidated_by_1559", "held_to_1559_close", "untouched_to_1559_close"),
        _build_scope_summary(stage_1, "stage_1_to_stage_2", "retraced_in_stage_2", "invalidated_in_stage_2", "held_through_stage_2", "untouched_through_stage_2"),
    ]
    return pl.concat([f for f in frames if not f.is_empty()], how="diagonal_relaxed") if any(not f.is_empty() for f in frames) else _summary_frame([])


def build_summary_tables(events: pl.DataFrame) -> pl.DataFrame:
    frames = [
        build_stage_summary_tables(events), build_creation_minute_summary(events), build_bar2_volume_summary(events),
        build_alignment_bucket_summary(events), build_alignment_bucket_minute_block_summary(events), build_alignment_bucket_gap_bucket_summary(events),
        build_entry_excursion_summary(events), build_entry_excursion_alignment_bucket_summary(events), build_entry_excursion_minute_block_summary(events),
        build_entry_excursion_gap_bucket_summary(events), build_entry_excursion_alignment_bucket_minute_block_summary(events),
        build_success_context_summary(events), build_success_context_alignment_bucket_summary(events), build_success_context_stacked_flag_summary(events),
        build_success_context_alignment_bucket_stacked_flag_summary(events),
        build_success_context_aligned_delta_imbalance_quantile_summary(events),
        build_success_context_abs_delta_imbalance_quantile_summary(events),
        build_success_context_side_aligned_delta_imbalance_quantile_summary(events),
        build_success_context_side_abs_delta_imbalance_quantile_summary(events),
        build_success_context_creation_minute_aligned_delta_imbalance_quantile_summary(events),
        build_success_context_creation_minute_abs_delta_imbalance_quantile_summary(events),
        build_success_context_minute_block_aligned_delta_imbalance_quantile_summary(events),
        build_success_context_minute_block_abs_delta_imbalance_quantile_summary(events),
    ]
    non_empty = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(non_empty, how="diagonal_relaxed").select(SUMMARY_COLUMNS) if non_empty else _summary_frame([])

def _save_placeholder_figure(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _ordered_records(df: pl.DataFrame, order_col: str, order: list[str] | None = None) -> list[dict]:
    records = df.to_dicts()
    if order is not None:
        pos = {value: idx for idx, value in enumerate(order)}
        records.sort(key=lambda r: pos.get(r.get(order_col), 999))
    return records


def _plot_grouped_rates(records: list[dict], labels_col: str, value_cols: list[str], path: Path, title: str, xlabel: str) -> None:
    if not records:
        _save_placeholder_figure(path, title)
        return
    labels = [str(r.get(labels_col)) for r in records]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(value_cols))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, col in enumerate(value_cols):
        vals = [0 if _is_null(r.get(col)) else float(r.get(col)) for r in records]
        ax.bar(x + (i - (len(value_cols)-1)/2) * width, vals, width, label=col)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Rate"); ax.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_creation_minute_outcomes(events: pl.DataFrame, figures_dir: Path) -> None:
    summary = build_creation_minute_summary(events).sort(["assigned_minute_index", "assigned_minute_hhmm"])
    _plot_grouped_rates(summary.to_dicts(), "assigned_minute_hhmm", ["hold_rate", "retrace_rate", "invalidation_rate"], figures_dir / "creation_minute_outcome_bars.png", "Creation Minute Outcomes", "Creation Minute")


def plot_bar2_volume_bucket_outcomes(events: pl.DataFrame, figures_dir: Path) -> None:
    summary = build_bar2_volume_summary(events)
    _plot_grouped_rates(summary.to_dicts(), "bar2_volume_bucket", ["hold_rate", "retrace_rate", "invalidation_rate"], figures_dir / "bar2_volume_bucket_outcomes.png", "Bar-2 Volume Bucket Outcomes", "Bar-2 Volume Bucket")


def plot_creation_minute_avg_bar2_volume(events: pl.DataFrame, figures_dir: Path) -> None:
    if events.is_empty():
        _save_placeholder_figure(figures_dir / "creation_minute_avg_bar2_volume.png", "Creation Minute Average Bar-2 Volume"); return
    frame = events.group_by(["assigned_minute_index", "assigned_minute_hhmm"]).agg(pl.col("bar2_volume").mean()).sort(["assigned_minute_index", "assigned_minute_hhmm"])
    records = frame.to_dicts()
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar([r["assigned_minute_hhmm"] for r in records], [r["bar2_volume"] for r in records], color="#3182bd")
    ax.set_title("Creation Minute Average Bar-2 Volume"); ax.set_xlabel("Creation Minute"); ax.set_ylabel("Average Volume")
    fig.tight_layout(); fig.savefig(figures_dir / "creation_minute_avg_bar2_volume.png"); plt.close(fig)


def plot_creation_minute_volume_heatmap(events: pl.DataFrame, figures_dir: Path) -> None:
    if events.is_empty():
        _save_placeholder_figure(figures_dir / "creation_minute_volume_heatmap.png", "Creation Minute Volume Heatmap"); return
    work = _with_bar2_volume_bucket(events)
    minutes = work.select(["assigned_minute_index", "assigned_minute_hhmm"]).unique().sort(["assigned_minute_index", "assigned_minute_hhmm"])["assigned_minute_hhmm"].to_list()
    buckets = sorted(work["bar2_volume_bucket"].unique().to_list())
    counts = work.group_by(["assigned_minute_hhmm", "bar2_volume_bucket"]).len().to_dicts()
    lookup = {(r["assigned_minute_hhmm"], r["bar2_volume_bucket"]): r["len"] for r in counts}
    matrix = np.array([[lookup.get((m, b), 0) for b in buckets] for m in minutes])
    fig, ax = plt.subplots(figsize=(9, 4)); image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_title("Creation Minute Volume Heatmap"); ax.set_xticks(range(len(buckets))); ax.set_xticklabels(buckets, rotation=30, ha="right"); ax.set_yticks(range(len(minutes))); ax.set_yticklabels(minutes)
    fig.colorbar(image, ax=ax, shrink=0.8); fig.tight_layout(); fig.savefig(figures_dir / "creation_minute_volume_heatmap.png"); plt.close(fig)

def _summary_scope(summary: pl.DataFrame, scope: str) -> pl.DataFrame:
    return summary.filter(pl.col("summary_scope") == scope) if not summary.is_empty() else summary


def _plot_rate_by(summary: pl.DataFrame, scope: str, label_col: str, value_col: str, filename: str, title: str, figures_dir: Path, order: list[str] | None = None, ylabel: str = "Rate") -> None:
    frame = _summary_scope(summary, scope)
    records = _ordered_records(frame, label_col, order)
    if not records:
        _save_placeholder_figure(figures_dir / filename, title); return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([str(r.get(label_col)) for r in records], [0 if _is_null(r.get(value_col)) else float(r.get(value_col)) for r in records], color="#3182bd")
    ax.set_title(title); ax.set_xlabel(label_col); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=20)
    if ylabel == "Rate": ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(figures_dir / filename); plt.close(fig)


def plot_alignment_bucket_outcomes(summary: pl.DataFrame, figures_dir: Path) -> None:
    frame = _summary_scope(summary, "alignment_bucket")
    _plot_grouped_rates(_ordered_records(frame, "alignment_bucket", ALIGNMENT_BUCKET_ORDER), "alignment_bucket", ["hold_rate", "retrace_rate", "invalidation_rate"], figures_dir / "alignment_bucket_outcomes.png", "Alignment Bucket Outcomes", "Alignment Bucket")


def plot_alignment_bucket_by_minute_block(summary: pl.DataFrame, figures_dir: Path) -> None:
    _save_or_pivot_plot(summary, "alignment_bucket_minute_block", "alignment_bucket", "minute_block", "hold_rate", ALIGNMENT_BUCKET_ORDER, MINUTE_BLOCK_ORDER, figures_dir / "alignment_bucket_by_minute_block.png", "Alignment Hold Rate by Minute Block", "Hold Rate")


def plot_alignment_bucket_by_gap_bucket(summary: pl.DataFrame, figures_dir: Path) -> None:
    _save_or_pivot_plot(summary, "alignment_bucket_gap_bucket", "alignment_bucket", "gap_size_bucket_225", "hold_rate", ALIGNMENT_BUCKET_ORDER, GAP_SIZE_BUCKET_ORDER, figures_dir / "alignment_bucket_by_gap_bucket.png", "Alignment Hold Rate by Gap Bucket", "Hold Rate")


def _save_or_pivot_plot(summary: pl.DataFrame, scope: str, x_col: str, series_col: str, value_col: str, x_order: list[str], series_order: list[str], path: Path, title: str, ylabel: str) -> None:
    records = _summary_scope(summary, scope).to_dicts()
    if not records:
        _save_placeholder_figure(path, title); return
    lookup = {(r.get(x_col), r.get(series_col)): r.get(value_col) for r in records}
    x_labels = [x for x in x_order if any(r.get(x_col) == x for r in records)]
    fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(x_labels)); width = 0.8 / max(1, len(series_order))
    for i, s in enumerate(series_order):
        vals = [0 if _is_null(lookup.get((xv, s))) else float(lookup[(xv, s)]) for xv in x_labels]
        ax.bar(x + (i - (len(series_order)-1)/2) * width, vals, width, label=s)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xticks(x); ax.set_xticklabels(x_labels, rotation=20, ha="right"); ax.set_ylim(0, 1); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_alignment_bucket_counts(events: pl.DataFrame, figures_dir: Path) -> None:
    if events.is_empty():
        _save_placeholder_figure(figures_dir / "alignment_bucket_counts.png", "Alignment Bucket Counts"); return
    counts = {r["alignment_bucket"]: r["len"] for r in events.group_by("alignment_bucket").len().to_dicts()}
    labels = [x for x in ALIGNMENT_BUCKET_ORDER if x in counts]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(labels, [counts[x] for x in labels], color="#636363")
    ax.set_title("Alignment Bucket Counts"); ax.set_xlabel("Alignment Bucket"); ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(figures_dir / "alignment_bucket_counts.png"); plt.close(fig)


def plot_entry_trigger_rate_by_alignment_bucket(summary: pl.DataFrame, figures_dir: Path) -> None:
    _plot_rate_by(summary, "entry_excursion_alignment_bucket", "alignment_bucket", "entry_trigger_rate", "entry_trigger_rate_by_alignment_bucket.png", "Entry Trigger Rate by Alignment Bucket", figures_dir, ALIGNMENT_BUCKET_ORDER)


def plot_mfe_mae_pct_by_alignment_bucket(summary: pl.DataFrame, figures_dir: Path) -> None:
    frame = _summary_scope(summary, "entry_excursion_alignment_bucket")
    records = _ordered_records(frame, "alignment_bucket", ALIGNMENT_BUCKET_ORDER)
    _plot_grouped_rates([{**r, "mfe_pct_mean": (r.get("mfe_pct_mean") or 0)*100, "mae_pct_mean": (r.get("mae_pct_mean") or 0)*100} for r in records], "alignment_bucket", ["mfe_pct_mean", "mae_pct_mean"], figures_dir / "mfe_mae_pct_by_alignment_bucket.png", "MFE and MAE Percent by Alignment Bucket", "Alignment Bucket")


def plot_mfe_pct_by_minute_block(summary: pl.DataFrame, figures_dir: Path) -> None:
    _plot_rate_by(summary, "entry_excursion_minute_block", "minute_block", "mfe_pct_mean", "mfe_pct_by_minute_block.png", "MFE Percent by Minute Block", figures_dir, MINUTE_BLOCK_ORDER, ylabel="Percent")


def plot_mfe_pct_by_gap_bucket(summary: pl.DataFrame, figures_dir: Path) -> None:
    _plot_rate_by(summary, "entry_excursion_gap_bucket", "gap_size_bucket_225", "mfe_pct_mean", "mfe_pct_by_gap_bucket.png", "MFE Percent by Gap Bucket", figures_dir, GAP_SIZE_BUCKET_ORDER, ylabel="Percent")

def _plot_success_metric(summary: pl.DataFrame, scope: str, group_col: str, metrics: list[str], filename: str, title: str, figures_dir: Path, order: list | None = None, overlay_success: bool = False) -> None:
    records = _ordered_records(_summary_scope(summary, scope), group_col, order)
    if not records:
        _save_placeholder_figure(figures_dir / filename, title); return
    labels = [str(r.get(group_col)) for r in records]
    x = np.arange(len(labels)); width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, metric in enumerate(metrics):
        vals = [0 if _is_null(r.get(metric)) else float(r.get(metric)) * 100.0 for r in records]
        ax.bar(x + (i - (len(metrics)-1)/2) * width, vals, width, label=metric)
    ax.set_title(title); ax.set_ylabel("Percent"); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    if overlay_success:
        ax2 = ax.twinx()
        ax2.plot(x, [0 if _is_null(r.get("successful_share_of_confirmable")) else float(r["successful_share_of_confirmable"]) * 100 for r in records], color="#252525", marker="o", linewidth=2, label="Success Rate")
        ax2.set_ylabel("Success Rate"); ax2.set_ylim(0, 100)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1+h2, l1+l2, loc="upper right")
    else:
        ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(figures_dir / filename); plt.close(fig)


def plot_successful_fvg_mae_by_alignment_bucket(summary: pl.DataFrame, figures_dir: Path) -> None:
    _plot_success_metric(summary, "success_context_alignment_bucket", "alignment_bucket", ["mae_pct_mean", "mae_pct_median", "mae_pct_p75"], "successful_fvg_mae_by_alignment_bucket.png", "Successful FVG MAE by Alignment Bucket", figures_dir, ALIGNMENT_BUCKET_ORDER, overlay_success=True)


def plot_successful_fvg_mae_by_stacked_flag(summary: pl.DataFrame, figures_dir: Path) -> None:
    frame = _summary_scope(summary, "success_context_stacked_flag").with_columns(pl.when(pl.col("stacked_continuation_fvg").fill_null(False)).then(pl.lit("stacked_continuation")).otherwise(pl.lit("base_fvg")).alias("stack_label")) if not summary.is_empty() else summary
    _plot_success_metric(frame, "success_context_stacked_flag", "stack_label", ["mae_pct_mean", "mae_pct_median", "mae_pct_p75"], "successful_fvg_mae_by_stacked_flag.png", "Successful FVG MAE by Stacked Flag", figures_dir, ["base_fvg", "stacked_continuation"])


def plot_successful_fvg_mfe_by_alignment_bucket(summary: pl.DataFrame, figures_dir: Path) -> None:
    _plot_success_metric(summary, "success_context_alignment_bucket", "alignment_bucket", ["mfe_pct_mean", "mfe_pct_median", "mfe_pct_p75"], "successful_fvg_mfe_by_alignment_bucket.png", "Successful FVG MFE by Alignment Bucket", figures_dir, ALIGNMENT_BUCKET_ORDER)


def plot_successful_fvg_mfe_by_stacked_flag(summary: pl.DataFrame, figures_dir: Path) -> None:
    frame = _summary_scope(summary, "success_context_stacked_flag").with_columns(pl.when(pl.col("stacked_continuation_fvg").fill_null(False)).then(pl.lit("stacked_continuation")).otherwise(pl.lit("base_fvg")).alias("stack_label")) if not summary.is_empty() else summary
    _plot_success_metric(frame, "success_context_stacked_flag", "stack_label", ["mfe_pct_mean", "mfe_pct_median", "mfe_pct_p75"], "successful_fvg_mfe_by_stacked_flag.png", "Successful FVG MFE by Stacked Flag", figures_dir, ["base_fvg", "stacked_continuation"])


def _event_outcome_bucket(event: dict) -> str:
    if not event.get("is_confirmable_by_1559"):
        return "unconfirmable"
    if event.get("invalidated_by_1559"):
        return "invalidated"
    if event.get("retraced_by_1559"):
        return "retraced_held"
    return "untouched_held"


def _stacked_count_plot(records: list[dict], x_col: str, bucket_col: str, buckets: list[str], path: Path, title: str) -> None:
    if not records:
        _save_placeholder_figure(path, title); return
    labels = sorted({str(r[x_col]) for r in records})
    lookup = {(str(r[x_col]), r[bucket_col]): r["count"] for r in records}
    bottoms = np.zeros(len(labels)); fig, ax = plt.subplots(figsize=(8, 4))
    for bucket in buckets:
        vals = np.array([lookup.get((label, bucket), 0) for label in labels])
        ax.bar(labels, vals, bottom=bottoms, label=bucket); bottoms += vals
    ax.set_title(title); ax.set_ylabel("Count"); ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_fvg_summary_figures(events: pl.DataFrame, summary: pl.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    placeholder_files = [
        ("hold_vs_invalidate_by_side.png", "Hold vs Invalidate by Side"), ("stage1_to_stage2_outcomes.png", "Stage 1 to Stage 2 Outcomes"),
        ("creation_minute_outcome_heatmap.png", "Creation Minute Outcome Heatmap"), ("gap_size_vs_outcome.png", "Gap Size vs Outcome"),
    ]
    if events.is_empty():
        for filename, title in placeholder_files:
            _save_placeholder_figure(figures_dir / filename, title)
    else:
        plot_events = events.with_columns(
            pl.struct(pl.all()).map_elements(_event_outcome_bucket, return_dtype=pl.String).alias("outcome_bucket"),
            pl.when(pl.col("invalidated_by_1559")).then(pl.lit("invalidated")).when(pl.col("retraced_by_1559")).then(pl.lit("retraced_held")).otherwise(pl.lit("untouched_held")).alias("hold_bucket"),
        )
        hold_records = plot_events.group_by(["fvg_side", "hold_bucket"]).len().rename({"len": "count"}).to_dicts()
        _stacked_count_plot(hold_records, "fvg_side", "hold_bucket", ["untouched_held", "retraced_held", "invalidated"], figures_dir / "hold_vs_invalidate_by_side.png", "Hold vs Invalidate by Side")
        stage1 = plot_events.filter(pl.col("assigned_stage") == "stage_1").with_columns(pl.when(pl.col("invalidated_in_stage_2")).then(pl.lit("invalidated_in_stage_2")).when(pl.col("retraced_in_stage_2")).then(pl.lit("retraced_in_stage_2")).otherwise(pl.lit("untouched_through_stage_2")).alias("stage_2_bucket"))
        trans_records = stage1.group_by(["fvg_side", "stage_2_bucket"]).len().rename({"len": "count"}).to_dicts() if not stage1.is_empty() else []
        _stacked_count_plot(trans_records, "fvg_side", "stage_2_bucket", ["retraced_in_stage_2", "invalidated_in_stage_2", "untouched_through_stage_2"], figures_dir / "stage1_to_stage2_outcomes.png", "Stage 1 to Stage 2 Outcomes")
        heat = plot_events.with_columns(pl.col("assigned_at").dt.strftime("%H:%M").alias("creation_minute")).group_by(["creation_minute", "outcome_bucket"]).len().rename({"len": "count"}).to_dicts()
        _stacked_count_plot(heat, "creation_minute", "outcome_bucket", ["untouched_held", "retraced_held", "invalidated", "unconfirmable"], figures_dir / "creation_minute_outcome_heatmap.png", "Creation Minute Outcome Heatmap")
        gap_events = plot_events.with_columns(pl.when(pl.col("gap_size") < pl.col("gap_size").median()).then(pl.lit("low")).otherwise(pl.lit("high")).alias("gap_bucket"))
        gap_records = gap_events.group_by(["gap_bucket", "outcome_bucket"]).len().rename({"len": "count"}).to_dicts()
        _stacked_count_plot(gap_records, "gap_bucket", "outcome_bucket", ["untouched_held", "retraced_held", "invalidated", "unconfirmable"], figures_dir / "gap_size_vs_outcome.png", "Gap Size vs Outcome")
    plot_creation_minute_outcomes(events, figures_dir); plot_bar2_volume_bucket_outcomes(events, figures_dir); plot_creation_minute_avg_bar2_volume(events, figures_dir); plot_creation_minute_volume_heatmap(events, figures_dir)
    plot_alignment_bucket_outcomes(summary, figures_dir); plot_alignment_bucket_by_minute_block(summary, figures_dir); plot_alignment_bucket_by_gap_bucket(summary, figures_dir); plot_alignment_bucket_counts(events, figures_dir)
    plot_entry_trigger_rate_by_alignment_bucket(summary, figures_dir); plot_mfe_mae_pct_by_alignment_bucket(summary, figures_dir); plot_mfe_pct_by_minute_block(summary, figures_dir); plot_mfe_pct_by_gap_bucket(summary, figures_dir)
    plot_successful_fvg_mae_by_alignment_bucket(summary, figures_dir); plot_successful_fvg_mae_by_stacked_flag(summary, figures_dir); plot_successful_fvg_mfe_by_alignment_bucket(summary, figures_dir); plot_successful_fvg_mfe_by_stacked_flag(summary, figures_dir)

def run_macro_fvg_study(
    input_path: Path = INPUT_PATH,
    events_output_path: Path = EVENTS_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    figures_dir: Path = FIGURES_DIR,
    volume_delta_5s_path: Path = OUTPUT_MACRO_5S_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    bars = load_minute_bars(input_path)
    events = detect_macro_fvgs(bars)
    events = scan_fvg_outcomes_until_1559_close(events, bars)
    events = try_enrich_fvg_events_with_delta_dominance(events, volume_delta_5s_path)
    summary = build_summary_tables(events)
    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    events.write_parquet(events_output_path)
    summary.write_parquet(summary_output_path)
    plot_fvg_summary_figures(events, summary, figures_dir)
    return events, summary


if __name__ == "__main__":
    run_macro_fvg_study()
