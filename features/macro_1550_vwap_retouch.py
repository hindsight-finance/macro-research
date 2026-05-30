#!/usr/bin/env python3
"""First-10s barrier-break direction signal + VWAP-retouch event study (15:50 macro open).

Causal: the first side of the first-10s barrier range [low_10s, high_10s] to break after
15:50:10 ET sets the directional bias; a retouch of the first-10s VWAP (frozen) or the rolling
15:50-anchored VWAP is the candidate entry. Forward outcomes are measured from the break and
from each retouch. The realised 15:50-candle and macro direction are recorded as outcomes only,
never as signals. Descriptive event study — no target/stop backtest in this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from utils import data_sources
from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR, get_tick_schema, scan_source

INPUT_PATH = data_sources.tick_data_url()
OUTPUT_PATH = Path("outputs/nq_macro_1550_vwap_retouch.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_1550_vwap_retouch_summary.parquet")

UTC_NS = pl.Datetime("ns", time_zone="UTC")
DEFAULT_BARRIER_SECONDS = 10
TOUCH_THRESHOLD_POINTS = 0.25
_REQUIRED_TICK_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks", "size"}


def _ets(hour: int, minute: int, second: int = 0) -> int:
    return hour * 3600 + minute * 60 + second


# ET-second-of-day window boundaries (time-of-day only; date-independent).
S_1550 = _ets(15, 50)
S_1551 = _ets(15, 51)
S_1555 = _ets(15, 55)
S_1600 = _ets(16, 0)
S_1601 = _ets(16, 1)

MACRO_1550_VWAP_RETOUCH_COLUMNS = [
    "date", "tick_count_macro", "has_first10", "trigger_state",
    "open_1550", "high_10s", "low_10s", "range_10s_points", "vwap_10s_frozen", "vol_share_first10",
    "break_side", "bias", "break_time_s", "break_ts_utc", "break_price",
    "retouch_frozen_occurred", "retouch_frozen_time_s", "retouch_frozen_ts_utc",
    "retouch_frozen_price", "retouch_frozen_lag_s",
    "retouch_rolling_occurred", "retouch_rolling_time_s", "retouch_rolling_ts_utc",
    "retouch_rolling_price", "retouch_rolling_lag_s", "vwap_rolling_at_retouch",
    "fwd_break_1554_points", "fwd_break_1559_points", "fwd_break_1600_points",
    "mfe_break_points", "mae_break_points",
    "fwd_retouch_frozen_1554_points", "fwd_retouch_frozen_1559_points", "fwd_retouch_frozen_1600_points",
    "mfe_retouch_frozen_points", "mae_retouch_frozen_points",
    "fwd_retouch_rolling_1554_points", "fwd_retouch_rolling_1559_points", "fwd_retouch_rolling_1600_points",
    "mfe_retouch_rolling_points", "mae_retouch_rolling_points",
    "candle_1550_dir_points", "candle_1550_dir_sign", "candle_1550_state",
    "macro_dir_points", "macro_dir_sign", "macro_trend_state",
    "bias_matches_1550_candle", "bias_matches_macro",
]

_STR_COLUMNS = {"trigger_state", "break_side", "bias", "candle_1550_state", "macro_trend_state"}
_BOOL_COLUMNS = {
    "has_first10", "retouch_frozen_occurred", "retouch_rolling_occurred",
    "bias_matches_1550_candle", "bias_matches_macro",
}
_INT_COLUMNS = {
    "tick_count_macro", "break_time_s", "retouch_frozen_time_s", "retouch_frozen_lag_s",
    "retouch_rolling_time_s", "retouch_rolling_lag_s", "candle_1550_dir_sign", "macro_dir_sign",
}
_TS_COLUMNS = {"break_ts_utc", "retouch_frozen_ts_utc", "retouch_rolling_ts_utc"}


def _schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {}
    for col in MACRO_1550_VWAP_RETOUCH_COLUMNS:
        if col == "date":
            schema[col] = pl.Date
        elif col in _STR_COLUMNS:
            schema[col] = pl.String
        elif col in _BOOL_COLUMNS:
            schema[col] = pl.Boolean
        elif col in _INT_COLUMNS:
            schema[col] = pl.Int64
        elif col in _TS_COLUMNS:
            schema[col] = UTC_NS
        else:
            schema[col] = pl.Float64
    return schema


def _blank_row(date, tick_count_macro: int, has_first10: bool) -> dict:
    row = {col: None for col in MACRO_1550_VWAP_RETOUCH_COLUMNS}
    row["date"] = date
    row["tick_count_macro"] = tick_count_macro
    row["has_first10"] = has_first10
    return row


def _scan_macro_window(path: str | Path) -> pl.LazyFrame:
    """Lazy, R2-aware scan of the 15:50:00–16:00:59 ET window across all dates.

    Reads through ``scan_source`` (synthesises ``price_ticks`` from the lake's float price and
    attaches R2 ``storage_options`` for s3:// URLs); ET is derived from UTC so DST is correct.
    """
    schema = get_tick_schema(path)
    missing = sorted(_REQUIRED_TICK_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")
    ts_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ)
    et_second = (
        ts_et.dt.hour().cast(pl.Int32) * 3600
        + ts_et.dt.minute().cast(pl.Int32) * 60
        + ts_et.dt.second().cast(pl.Int32)
    )
    return (
        scan_source(path)
        .select(
            pl.col("ts_event").cast(UTC_NS).alias("ts_event"),
            pl.col("intra_ts_rank").cast(pl.Int64),
            pl.col("price_ticks").cast(pl.Int64),
            pl.col("size").cast(pl.Int64),
        )
        .with_columns(
            date=ts_et.dt.date(),
            et_second=et_second,
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .filter((pl.col("et_second") >= S_1550) & (pl.col("et_second") < S_1601))
    )


def _signed(anchor_price, target_price, bias: str | None):
    """Forward move signed so positive = favourable to the bias."""
    if anchor_price is None or target_price is None or bias not in ("bullish", "bearish"):
        return None
    return (target_price - anchor_price) if bias == "bullish" else (anchor_price - target_price)


def _last_price_before(win: pl.DataFrame, cutoff_second: int):
    sub = win.filter(pl.col("et_second") < cutoff_second)
    return float(sub["price"][-1]) if sub.height else None


def _candle_dir(win: pl.DataFrame, start_second: int, end_second: int):
    sub = win.filter((pl.col("et_second") >= start_second) & (pl.col("et_second") < end_second))
    if sub.height == 0:
        return None, None, None
    points = float(sub["price"][-1] - sub["price"][0])
    sign = 1 if points > 0 else (-1 if points < 0 else 0)
    state = "bullish" if points > 0 else ("bearish" if points < 0 else "neutral")
    return points, sign, state


def _bias_match(bias: str | None, state: str | None):
    if bias not in ("bullish", "bearish") or state not in ("bullish", "bearish"):
        return None
    return bias == state


def _fill_forward(row, win, anchor, anchor_ts, anchor_rank, anchor_et, anchor_price, bias, p1554, p1559, p1600):
    """Forward points to each horizon (null if the anchor is at/after the horizon) + MFE/MAE."""
    def fwd(cutoff, price_at_horizon):
        if anchor_et is None or anchor_et >= cutoff:
            return None
        return _signed(anchor_price, price_at_horizon, bias)

    row[f"fwd_{anchor}_1554_points"] = fwd(S_1555, p1554)
    row[f"fwd_{anchor}_1559_points"] = fwd(S_1600, p1559)
    row[f"fwd_{anchor}_1600_points"] = fwd(S_1601, p1600)

    seg = win.filter(
        (pl.col("et_second") < S_1600)
        & (
            (pl.col("ts_event") > anchor_ts)
            | ((pl.col("ts_event") == anchor_ts) & (pl.col("intra_ts_rank") >= anchor_rank))
        )
    )
    if seg.height == 0:
        row[f"mfe_{anchor}_points"] = 0.0
        row[f"mae_{anchor}_points"] = 0.0
        return
    signed_expr = (pl.col("price") - anchor_price) if bias == "bullish" else (anchor_price - pl.col("price"))
    row[f"mfe_{anchor}_points"] = float(seg.select(signed_expr.max()).item())
    row[f"mae_{anchor}_points"] = float(seg.select(signed_expr.min()).item())


def _fill_retouch(row, win, ref, rrow, break_et, bias, p1554, p1559, p1600):
    r_et = int(rrow["et_second"])
    row[f"retouch_{ref}_occurred"] = True
    row[f"retouch_{ref}_time_s"] = r_et - S_1550
    row[f"retouch_{ref}_ts_utc"] = rrow["ts_event"]
    row[f"retouch_{ref}_price"] = float(rrow["price"])
    row[f"retouch_{ref}_lag_s"] = r_et - break_et
    _fill_forward(row, win, f"retouch_{ref}", rrow["ts_event"], int(rrow["intra_ts_rank"]),
                  r_et, float(rrow["price"]), bias, p1554, p1559, p1600)


def detect_retouch_events(
    day_ticks: pl.DataFrame,
    *,
    date,
    barrier_seconds: int = DEFAULT_BARRIER_SECONDS,
    touch: float = TOUCH_THRESHOLD_POINTS,
) -> dict:
    """One row of retouch-event features for a single macro date. Pure (no I/O)."""
    win = day_ticks.sort("ts_event", "intra_ts_rank")
    macro = win.filter((pl.col("et_second") >= S_1550) & (pl.col("et_second") < S_1600))
    tick_count_macro = macro.height

    first10 = win.filter((pl.col("et_second") >= S_1550) & (pl.col("et_second") < S_1550 + barrier_seconds))
    if first10.height == 0:
        return _blank_row(date, tick_count_macro, has_first10=False)

    high_10s = float(first10["price"].max())
    low_10s = float(first10["price"].min())
    open_1550 = float(first10["price"][0])
    f10_size = int(first10["size"].sum())
    vwap_10s_frozen = float((first10["price"] * first10["size"]).sum() / f10_size) if f10_size > 0 else None
    minute_1550_size = int(
        win.filter((pl.col("et_second") >= S_1550) & (pl.col("et_second") < S_1551))["size"].sum()
    )
    vol_share_first10 = (f10_size / minute_1550_size) if minute_1550_size > 0 else None

    # rolling 15:50-anchored VWAP across the whole read window
    win = win.with_columns(pv=pl.col("price") * pl.col("size")).with_columns(
        cum_pv=pl.col("pv").cum_sum(),
        cum_size=pl.col("size").cum_sum(),
    ).with_columns(
        rolling_vwap=pl.when(pl.col("cum_size") > 0)
        .then(pl.col("cum_pv") / pl.col("cum_size"))
        .otherwise(None)
    )

    candle_pts, candle_sign, candle_state = _candle_dir(win, S_1550, S_1551)
    macro_pts, macro_sign, macro_state = _candle_dir(win, S_1550, S_1600)

    row = _blank_row(date, tick_count_macro, has_first10=True)
    row.update(
        open_1550=open_1550, high_10s=high_10s, low_10s=low_10s,
        range_10s_points=high_10s - low_10s, vwap_10s_frozen=vwap_10s_frozen,
        vol_share_first10=vol_share_first10,
        candle_1550_dir_points=candle_pts, candle_1550_dir_sign=candle_sign, candle_1550_state=candle_state,
        macro_dir_points=macro_pts, macro_dir_sign=macro_sign, macro_trend_state=macro_state,
    )

    p1554 = _last_price_before(win, S_1555)
    p1559 = _last_price_before(win, S_1600)
    p1600 = _last_price_before(win, S_1601)

    post = win.filter((pl.col("et_second") >= S_1550 + barrier_seconds) & (pl.col("et_second") < S_1600))
    broke = post.filter((pl.col("price") > high_10s) | (pl.col("price") < low_10s)).head(1)
    if broke.height == 0:
        row["trigger_state"] = "no_trigger"
        return row

    b = broke.row(0, named=True)
    break_price = float(b["price"])
    break_ts = b["ts_event"]
    break_rank = int(b["intra_ts_rank"])
    break_et = int(b["et_second"])
    break_side = "high" if break_price > high_10s else "low"
    bias = "bullish" if break_side == "high" else "bearish"
    row.update(
        trigger_state="triggered", break_side=break_side, bias=bias,
        break_time_s=break_et - S_1550, break_ts_utc=break_ts, break_price=break_price,
        bias_matches_1550_candle=_bias_match(bias, candle_state),
        bias_matches_macro=_bias_match(bias, macro_state),
        retouch_frozen_occurred=False, retouch_rolling_occurred=False,
    )
    _fill_forward(row, win, "break", break_ts, break_rank, break_et, break_price, bias, p1554, p1559, p1600)

    post_break = win.filter(
        (pl.col("et_second") < S_1600)
        & (
            (pl.col("ts_event") > break_ts)
            | ((pl.col("ts_event") == break_ts) & (pl.col("intra_ts_rank") > break_rank))
        )
    )

    if vwap_10s_frozen is not None:
        cond = (
            (pl.col("price") <= vwap_10s_frozen + touch)
            if bias == "bullish"
            else (pl.col("price") >= vwap_10s_frozen - touch)
        )
        rf = post_break.filter(cond).head(1)
        if rf.height:
            _fill_retouch(row, win, "frozen", rf.row(0, named=True), break_et, bias, p1554, p1559, p1600)

    cond_r = (
        (pl.col("price") <= pl.col("rolling_vwap") + touch)
        if bias == "bullish"
        else (pl.col("price") >= pl.col("rolling_vwap") - touch)
    )
    rr = post_break.filter(pl.col("rolling_vwap").is_not_null() & cond_r).head(1)
    if rr.height:
        rr_row = rr.row(0, named=True)
        _fill_retouch(row, win, "rolling", rr_row, break_et, bias, p1554, p1559, p1600)
        row["vwap_rolling_at_retouch"] = float(rr_row["rolling_vwap"])

    return row
