from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import polars as pl

from utils import data_sources
from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR, get_tick_schema, scan_source

TICK_INPUT_PATH = data_sources.tick_data_url()
BARRIER_INPUT_PATH = Path("outputs/nq_macro_1550_barrier.parquet")
VWAP_INPUT_PATH = Path("outputs/nq_macro_vwap_intramacro.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context_summary.parquet")
UTC_NS = pl.Datetime("ns", time_zone="UTC")

TICK_REQUIRED_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks", "size"}
BARRIER_REQUIRED_COLUMNS = {
    "date", "macro_trend_state", "barrier_extreme", "barrier_price", "barrier_time",
    "barrier_first10", "barrier_is_macro_extreme", "barrier_holds", "edge_case",
}
VWAP_REQUIRED_COLUMNS = {
    "date",
    "macro_1550_at_1550_10s_vwap_side",
    "macro_1550_at_1550_10s_vwap_dist_points",
    "macro_1550_at_1550_10s_vwap_dist_bps",
    "macro_1550_at_1555_vwap_side",
    "macro_1550_at_1555_vwap_dist_points",
    "macro_1550_at_1555_vwap_dist_bps",
    "target_1550_1554_points", "target_1550_1554_sign", "target_1550_1554_state",
    "target_1555_1559_points", "target_1555_1559_sign", "target_1555_1559_state",
    "target_1550_1559_points", "target_1550_1559_sign", "target_1550_1559_state",
}

TARGET_PREFIXES = (
    "target_1550_10s_1554",
    "target_1550_10s_1559",
    "target_1551_1559",
    "target_1555_1559",
)

MACRO_VWAP_BARRIER_CONTEXT_COLUMNS = [
    "date",
    "macro_trend_state",
    "barrier_extreme",
    "barrier_price",
    "barrier_time",
    "barrier_first10",
    "barrier_is_macro_extreme",
    "barrier_holds",
    "edge_case",
    "vwap_10s_side",
    "vwap_10s_dist_points",
    "vwap_10s_dist_bps",
    "vwap_10s_constructive",
    "barrier_first10_and_vwap_constructive",
    "barrier_ts_utc",
    "post_barrier_tick_count_1550",
    "vwap_side_at_barrier",
    "vwap_dist_at_barrier_points",
    "vwap_side_at_1550_close",
    "vwap_dist_at_1550_close_points",
    "closed_wrong_side_1550",
    "closed_wrong_side_more_than_1tick",
    "closed_wrong_side_more_than_2pts",
    "closed_wrong_side_more_than_5pts",
    "worst_wrong_side_dist_points",
    "worst_wrong_side_dist_bps",
    "seconds_wrong_side_vwap",
    "wrong_side_share_1550",
    "vwap_1555_side",
    "vwap_1555_dist_points",
    "vwap_1555_dist_bps",
    "vwap_1555_constructive",
    "vwap_context_10s_to_1555",
    "barrier_holds_and_1555_constructive",
    "barrier_first10_and_1555_constructive",
    "target_1550_1554_points", "target_1550_1554_sign", "target_1550_1554_state",
    "target_1555_1559_points", "target_1555_1559_sign", "target_1555_1559_state",
    "target_1550_1559_points", "target_1550_1559_sign", "target_1550_1559_state",
    "target_1550_10s_1554_points", "target_1550_10s_1554_sign", "target_1550_10s_1554_state",
    "target_1550_10s_1559_points", "target_1550_10s_1559_sign", "target_1550_10s_1559_state",
    "target_1551_1559_points", "target_1551_1559_sign", "target_1551_1559_state",
]

MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS = [
    "scope", "bucket", "target_name", "sample_size", "bullish_count", "bearish_count", "neutral_count",
    "bullish_pct", "bearish_pct", "neutral_pct", "avg_target_points", "median_target_points",
    "p10_target_points", "p25_target_points", "p75_target_points", "p90_target_points",
]


def _context_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {column: pl.Float64 for column in MACRO_VWAP_BARRIER_CONTEXT_COLUMNS}
    for column in [
        "date",
    ]:
        schema[column] = pl.Date
    for column in [
        "macro_trend_state", "barrier_extreme", "vwap_10s_side", "vwap_10s_constructive",
        "vwap_side_at_barrier", "vwap_side_at_1550_close", "vwap_1555_side",
        "vwap_1555_constructive", "vwap_context_10s_to_1555", "target_1550_1554_state",
        "target_1555_1559_state", "target_1550_1559_state", "target_1550_10s_1554_state",
        "target_1550_10s_1559_state", "target_1551_1559_state",
    ]:
        schema[column] = pl.String
    for column in [
        "barrier_first10", "barrier_is_macro_extreme", "barrier_holds", "edge_case",
        "barrier_first10_and_vwap_constructive", "closed_wrong_side_1550",
        "closed_wrong_side_more_than_1tick", "closed_wrong_side_more_than_2pts",
        "closed_wrong_side_more_than_5pts", "barrier_holds_and_1555_constructive",
        "barrier_first10_and_1555_constructive",
    ]:
        schema[column] = pl.Boolean
    for column in [
        "barrier_time", "post_barrier_tick_count_1550", "seconds_wrong_side_vwap",
        "target_1550_1554_sign", "target_1555_1559_sign", "target_1550_1559_sign",
        "target_1550_10s_1554_sign", "target_1550_10s_1559_sign", "target_1551_1559_sign",
    ]:
        schema[column] = pl.Int64 if column != "seconds_wrong_side_vwap" else pl.Float64
    schema["barrier_ts_utc"] = UTC_NS
    return schema


def _empty_context_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_context_schema()).select(MACRO_VWAP_BARRIER_CONTEXT_COLUMNS)


def _empty_tick_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ts_event": UTC_NS,
            "intra_ts_rank": pl.Int64,
            "price_ticks": pl.Int64,
            "size": pl.Int64,
            "datetime_et": pl.Datetime("ns", time_zone=MARKET_TZ),
            "date": pl.Date,
            "et_second": pl.Int32,
            "price": pl.Float64,
        }
    )


def _et_second(hour: int, minute: int, second: int = 0) -> int:
    return hour * 3600 + minute * 60 + second


def _missing_columns(df: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(df.columns))


def _validate_frame(df: pl.DataFrame, required: set[str], name: str) -> None:
    missing = _missing_columns(df, required)
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = sorted(TICK_REQUIRED_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def classify_constructive_side(macro_trend_state: str | None, vwap_side: str | None) -> str:
    if macro_trend_state not in {"bullish", "bearish"} or vwap_side is None:
        return "unknown"
    if vwap_side == "touch":
        return "touch"
    if macro_trend_state == "bullish":
        return "constructive" if vwap_side == "above" else "wrong"
    return "constructive" if vwap_side == "below" else "wrong"


def _sign_state(points: float | None) -> tuple[int | None, str | None]:
    if points is None:
        return None, None
    if points > 0:
        return 1, "bullish"
    if points < 0:
        return -1, "bearish"
    return 0, "neutral"


def _side_from_raw_dist(value: float | None) -> str | None:
    if value is None:
        return None
    if abs(value) <= 0.25:
        return "touch"
    return "above" if value > 0 else "below"


def _barrier_ts_utc(date: object, barrier_time: object) -> datetime | None:
    if date is None or barrier_time is None:
        return None
    local_dt = datetime.combine(date, time(15, 50), tzinfo=ZoneInfo(MARKET_TZ)) + timedelta(seconds=int(barrier_time))
    return local_dt.astimezone(ZoneInfo("UTC"))


def _wrong_side_close_bucket(value: float | None) -> str:
    if value is None or value <= 0:
        return "no_wrong_side_close"
    if value <= 0.25:
        return "wrong_le_1tick"
    if value <= 2.0:
        return "wrong_1tick_to_2pts"
    if value <= 5.0:
        return "wrong_2_to_5pts"
    return "wrong_gt_5pts"


def _scan_macro_ticks(path: str | Path, target_dates: set[object]) -> pl.DataFrame:
    _validate_tick_schema(path)
    if not target_dates:
        return _empty_tick_frame()
    dates = sorted(target_dates)
    start_utc = datetime.combine(dates[0], time(15, 50), tzinfo=ZoneInfo(MARKET_TZ)).astimezone(ZoneInfo("UTC"))
    end_utc = datetime.combine(dates[-1], time(16, 0), tzinfo=ZoneInfo(MARKET_TZ)).astimezone(ZoneInfo("UTC"))
    ts_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ)
    return (
        scan_source(path)
        .select(
            pl.col("ts_event").cast(UTC_NS).alias("ts_event"),
            pl.col("intra_ts_rank").cast(pl.Int64),
            pl.col("price_ticks").cast(pl.Int64),
            pl.col("size").cast(pl.Int64),
        )
        .filter((pl.col("ts_event") >= pl.lit(start_utc).cast(UTC_NS)) & (pl.col("ts_event") < pl.lit(end_utc).cast(UTC_NS)))
        .with_columns(
            datetime_et=ts_et,
            date=ts_et.dt.date(),
            et_second=(ts_et.dt.hour().cast(pl.Int32) * 3600 + ts_et.dt.minute().cast(pl.Int32) * 60 + ts_et.dt.second().cast(pl.Int32)),
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .filter(pl.col("date").is_in(dates))
        .filter((pl.col("et_second") >= _et_second(15, 50)) & (pl.col("et_second") < _et_second(16, 0)))
        .sort("date", "ts_event", "intra_ts_rank")
        .collect(engine="streaming")
    )


def _macro_open_vwap_ticks(ticks: pl.DataFrame) -> pl.DataFrame:
    if ticks.is_empty():
        return ticks.with_columns(vwap_1550=pl.lit(None, dtype=pl.Float64))
    return (
        ticks.with_columns(
            pv=pl.col("price") * pl.col("size").cast(pl.Float64),
            is_1550=(pl.col("et_second") >= _et_second(15, 50)) & (pl.col("et_second") < _et_second(15, 51)),
        )
        .with_columns(
            pv_1550=pl.when(pl.col("is_1550")).then(pl.col("pv")).otherwise(0.0),
            size_1550=pl.when(pl.col("is_1550")).then(pl.col("size")).otherwise(0),
        )
        .with_columns(
            cum_pv_1550=pl.col("pv_1550").cum_sum().over("date"),
            cum_size_1550=pl.col("size_1550").cum_sum().over("date"),
        )
        .with_columns(
            vwap_1550=pl.when(pl.col("cum_size_1550") > 0)
            .then(pl.col("cum_pv_1550") / pl.col("cum_size_1550"))
            .otherwise(None)
        )
    )


def _target_from_ticks(day_ticks: pl.DataFrame, start_second: int, end_second: int) -> float | None:
    window = day_ticks.filter((pl.col("et_second") >= start_second) & (pl.col("et_second") < end_second)).sort("ts_event", "intra_ts_rank")
    if window.is_empty():
        return None
    return float(window.item(window.height - 1, "price") - window.item(0, "price"))


def _blank_tick_metrics(date: object, barrier_ts_utc: datetime | None = None) -> dict:
    row = {
        "date": date,
        "barrier_ts_utc": barrier_ts_utc,
        "post_barrier_tick_count_1550": None,
        "vwap_side_at_barrier": None,
        "vwap_dist_at_barrier_points": None,
        "vwap_side_at_1550_close": None,
        "vwap_dist_at_1550_close_points": None,
        "closed_wrong_side_1550": None,
        "closed_wrong_side_more_than_1tick": None,
        "closed_wrong_side_more_than_2pts": None,
        "closed_wrong_side_more_than_5pts": None,
        "worst_wrong_side_dist_points": None,
        "worst_wrong_side_dist_bps": None,
        "seconds_wrong_side_vwap": None,
        "wrong_side_share_1550": None,
        "target_1550_10s_1554_points": None,
        "target_1550_10s_1559_points": None,
        "target_1551_1559_points": None,
    }
    for prefix in ["target_1550_10s_1554", "target_1550_10s_1559", "target_1551_1559"]:
        row[f"{prefix}_sign"] = None
        row[f"{prefix}_state"] = None
    return row


def _tick_metrics_for_day(day: dict, day_ticks: pl.DataFrame) -> dict:
    date = day["date"]
    trend = day["macro_trend_state"]
    barrier_time = day["barrier_time"]
    day_ticks = day_ticks.sort("ts_event", "intra_ts_rank")
    barrier_ts = _barrier_ts_utc(date, barrier_time)
    if day_ticks.is_empty() or barrier_time is None:
        return _blank_tick_metrics(date, barrier_ts)

    barrier_second = _et_second(15, 50, int(barrier_time))
    post_barrier = day_ticks.filter((pl.col("et_second") >= barrier_second) & (pl.col("et_second") < _et_second(15, 51)))

    def signed_dist(price: float | None, vwap_value: float | None) -> float | None:
        if price is None or vwap_value is None:
            return None
        if trend == "bullish":
            return price - vwap_value
        if trend == "bearish":
            return vwap_value - price
        return None

    def raw_dist(price: float | None, vwap_value: float | None) -> float | None:
        if price is None or vwap_value is None:
            return None
        return price - vwap_value

    if post_barrier.is_empty():
        at_barrier = None
        at_close = None
        signed_values: list[float] = []
    else:
        at_barrier = post_barrier.row(0, named=True)
        at_close = post_barrier.row(post_barrier.height - 1, named=True)
        signed_values = [signed_dist(row["price"], row["vwap_1550"]) for row in post_barrier.iter_rows(named=True)]
        signed_values = [value for value in signed_values if value is not None]

    wrong_values = [max(-value, 0.0) for value in signed_values]
    worst_wrong = max(wrong_values) if wrong_values else None
    post_count = post_barrier.height
    wrong_seconds = None
    wrong_share = None
    if post_count:
        post_rows = list(post_barrier.iter_rows(named=True))
        close_ts = (barrier_ts.replace(minute=51, second=0, microsecond=0) if barrier_ts is not None else None)
        wrong_seconds = 0.0
        for idx, tick_row in enumerate(post_rows):
            value = signed_dist(tick_row["price"], tick_row["vwap_1550"])
            start_ts = tick_row["ts_event"]
            end_ts = post_rows[idx + 1]["ts_event"] if idx + 1 < len(post_rows) else close_ts
            if value is None or end_ts is None:
                continue
            elapsed = max((end_ts - start_ts).total_seconds(), 0.0)
            if value < 0:
                wrong_seconds += elapsed
        denom = max(60.0 - float(int(barrier_time)), 0.0)
        wrong_share = wrong_seconds / denom if denom else None
    close_signed = signed_dist(at_close["price"], at_close["vwap_1550"]) if at_close else None
    close_wrong = max(-close_signed, 0.0) if close_signed is not None else None
    barrier_signed = signed_dist(at_barrier["price"], at_barrier["vwap_1550"]) if at_barrier else None
    target_10s_1554 = _target_from_ticks(day_ticks, _et_second(15, 50, 10), _et_second(15, 55))
    target_10s_1559 = _target_from_ticks(day_ticks, _et_second(15, 50, 10), _et_second(16, 0))
    target_1551_1559 = _target_from_ticks(day_ticks, _et_second(15, 51), _et_second(16, 0))

    row = {
        "date": date,
        "barrier_ts_utc": barrier_ts,
        "post_barrier_tick_count_1550": post_count,
        "vwap_side_at_barrier": _side_from_raw_dist(raw_dist(at_barrier["price"], at_barrier["vwap_1550"]) if at_barrier else None),
        "vwap_dist_at_barrier_points": barrier_signed,
        "vwap_side_at_1550_close": _side_from_raw_dist(raw_dist(at_close["price"], at_close["vwap_1550"]) if at_close else None),
        "vwap_dist_at_1550_close_points": close_signed,
        "closed_wrong_side_1550": (close_wrong > 0) if close_wrong is not None else None,
        "closed_wrong_side_more_than_1tick": (close_wrong > 0.25) if close_wrong is not None else None,
        "closed_wrong_side_more_than_2pts": (close_wrong > 2.0) if close_wrong is not None else None,
        "closed_wrong_side_more_than_5pts": (close_wrong > 5.0) if close_wrong is not None else None,
        "worst_wrong_side_dist_points": worst_wrong,
        "worst_wrong_side_dist_bps": (worst_wrong / at_close["vwap_1550"] * 10000.0) if worst_wrong is not None and at_close and at_close["vwap_1550"] not in (None, 0) else None,
        "seconds_wrong_side_vwap": wrong_seconds,
        "wrong_side_share_1550": wrong_share,
        "target_1550_10s_1554_points": target_10s_1554,
        "target_1550_10s_1559_points": target_10s_1559,
        "target_1551_1559_points": target_1551_1559,
    }
    for prefix in ["target_1550_10s_1554", "target_1550_10s_1559", "target_1551_1559"]:
        sign, state = _sign_state(row[f"{prefix}_points"])
        row[f"{prefix}_sign"] = sign
        row[f"{prefix}_state"] = state
    return row


def _tick_context_metrics(barrier: pl.DataFrame, tick_path: str | Path) -> pl.DataFrame:
    target_dates = set(barrier["date"].to_list())
    ticks = _macro_open_vwap_ticks(_scan_macro_ticks(tick_path, target_dates))
    if ticks.is_empty():
        tick_frames: dict[object, pl.DataFrame] = {}
    else:
        tick_frames = {
            key[0] if isinstance(key, tuple) else key: frame
            for key, frame in ticks.partition_by("date", as_dict=True, maintain_order=True).items()
        }
    rows = [_tick_metrics_for_day(day, tick_frames.get(day["date"], _empty_tick_frame())) for day in barrier.iter_rows(named=True)]
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame({"date": []}, schema={"date": pl.Date})


def _transition_label(start: str, end: str) -> str:
    if start == "unknown" or end == "unknown":
        return "unknown"
    if "touch" in {start, end}:
        return "touch_mixed"
    return f"{start}_to_{end}"


def build_macro_vwap_barrier_context(barrier: pl.DataFrame, vwap: pl.DataFrame, tick_path: str | Path) -> pl.DataFrame:
    _validate_frame(barrier, BARRIER_REQUIRED_COLUMNS, "barrier")
    _validate_frame(vwap, VWAP_REQUIRED_COLUMNS, "VWAP")
    base = barrier.join(vwap, on="date", how="inner")
    if base.is_empty():
        return _empty_context_frame()
    tick_metrics = _tick_context_metrics(base, tick_path)
    out = base.join(tick_metrics, on="date", how="left")
    rows = []
    for row in out.iter_rows(named=True):
        vwap_10s = classify_constructive_side(row["macro_trend_state"], row["macro_1550_at_1550_10s_vwap_side"])
        vwap_1555 = classify_constructive_side(row["macro_trend_state"], row["macro_1550_at_1555_vwap_side"])
        result = {
            **row,
            "vwap_10s_side": row["macro_1550_at_1550_10s_vwap_side"],
            "vwap_10s_dist_points": row["macro_1550_at_1550_10s_vwap_dist_points"],
            "vwap_10s_dist_bps": row["macro_1550_at_1550_10s_vwap_dist_bps"],
            "vwap_10s_constructive": vwap_10s,
            "barrier_first10_and_vwap_constructive": bool(row["barrier_first10"] and vwap_10s in {"constructive", "touch"}),
            "vwap_1555_side": row["macro_1550_at_1555_vwap_side"],
            "vwap_1555_dist_points": row["macro_1550_at_1555_vwap_dist_points"],
            "vwap_1555_dist_bps": row["macro_1550_at_1555_vwap_dist_bps"],
            "vwap_1555_constructive": vwap_1555,
            "vwap_context_10s_to_1555": _transition_label(vwap_10s, vwap_1555),
            "barrier_holds_and_1555_constructive": bool(row["barrier_holds"] and vwap_1555 in {"constructive", "touch"}),
            "barrier_first10_and_1555_constructive": bool(row["barrier_first10"] and vwap_1555 in {"constructive", "touch"}),
        }
        rows.append(result)
    return pl.DataFrame(rows, infer_schema_length=None).select(MACRO_VWAP_BARRIER_CONTEXT_COLUMNS).sort("date")


def _pct(count: int, denom: int) -> float | None:
    return count / denom * 100.0 if denom else None


def _summary_row(df: pl.DataFrame, scope: str, bucket: str, target_name: str) -> dict:
    state_col = f"{target_name}_state"
    points_col = f"{target_name}_points"
    sample_size = df.height
    bullish_count = df.filter(pl.col(state_col) == "bullish").height if sample_size else 0
    bearish_count = df.filter(pl.col(state_col) == "bearish").height if sample_size else 0
    neutral_count = df.filter(pl.col(state_col) == "neutral").height if sample_size else 0
    return {
        "scope": scope,
        "bucket": bucket,
        "target_name": target_name,
        "sample_size": sample_size,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "bullish_pct": _pct(bullish_count, sample_size),
        "bearish_pct": _pct(bearish_count, sample_size),
        "neutral_pct": _pct(neutral_count, sample_size),
        "avg_target_points": df.select(pl.col(points_col).mean()).item() if sample_size else None,
        "median_target_points": df.select(pl.col(points_col).median()).item() if sample_size else None,
        "p10_target_points": df.select(pl.col(points_col).quantile(0.10)).item() if sample_size else None,
        "p25_target_points": df.select(pl.col(points_col).quantile(0.25)).item() if sample_size else None,
        "p75_target_points": df.select(pl.col(points_col).quantile(0.75)).item() if sample_size else None,
        "p90_target_points": df.select(pl.col(points_col).quantile(0.90)).item() if sample_size else None,
    }


def _deciled(df: pl.DataFrame, value_col: str) -> pl.DataFrame | None:
    non_null = df.filter(pl.col(value_col).is_not_null())
    if non_null.height < 10:
        return None
    return non_null.with_columns(
        (((pl.col(value_col).rank(method="ordinal") - 1) * 10 / non_null.height).floor().cast(pl.Int64).clip(0, 9) + 1)
        .cast(pl.String)
        .alias("_bucket")
    )


def summarize_macro_vwap_barrier_context(df: pl.DataFrame) -> pl.DataFrame:
    missing = _missing_columns(df, set(MACRO_VWAP_BARRIER_CONTEXT_COLUMNS))
    if missing:
        raise ValueError(f"Missing context columns: {missing}")
    rows = []
    close_wrong = pl.when(pl.col("vwap_dist_at_1550_close_points").is_not_null()).then((-pl.col("vwap_dist_at_1550_close_points")).clip(0.0)).otherwise(None)
    bucketed = df.with_columns(
        pl.when(close_wrong.is_null() | (close_wrong <= 0)).then(pl.lit("no_wrong_side_close"))
        .when(close_wrong <= 0.25).then(pl.lit("wrong_le_1tick"))
        .when(close_wrong <= 2.0).then(pl.lit("wrong_1tick_to_2pts"))
        .when(close_wrong <= 5.0).then(pl.lit("wrong_2_to_5pts"))
        .otherwise(pl.lit("wrong_gt_5pts")).alias("_bucket")
    )
    for target in TARGET_PREFIXES:
        scopes = [
            ("barrier_only", "first10_true", df.filter(pl.col("barrier_first10"))),
            ("barrier_only", "first10_false", df.filter(~pl.col("barrier_first10"))),
            ("barrier_only", "holds_true", df.filter(pl.col("barrier_holds"))),
            ("barrier_only", "holds_false", df.filter(~pl.col("barrier_holds"))),
            ("vwap_10s_only", "constructive", df.filter(pl.col("vwap_10s_constructive") == "constructive")),
            ("vwap_10s_only", "wrong", df.filter(pl.col("vwap_10s_constructive") == "wrong")),
            ("vwap_10s_only", "touch", df.filter(pl.col("vwap_10s_constructive") == "touch")),
            ("vwap_10s_only", "unknown", df.filter(pl.col("vwap_10s_constructive") == "unknown")),
        ]
        for scope, bucket, subset in scopes:
            rows.append(_summary_row(subset, scope, bucket, target))
        for state in ["constructive", "wrong", "touch", "unknown"]:
            rows.append(_summary_row(df.filter(pl.col("barrier_first10") & (pl.col("vwap_10s_constructive") == state)), "barrier_vwap_10s", f"first10_true_{state}", target))
            rows.append(_summary_row(df.filter(~pl.col("barrier_first10") & (pl.col("vwap_10s_constructive") == state)), "barrier_vwap_10s", f"first10_false_{state}", target))
            rows.append(_summary_row(df.filter(pl.col("barrier_holds") & (pl.col("vwap_10s_constructive") == state)), "barrier_vwap_10s", f"holds_true_{state}", target))
            rows.append(_summary_row(df.filter(~pl.col("barrier_holds") & (pl.col("vwap_10s_constructive") == state)), "barrier_vwap_10s", f"holds_false_{state}", target))
            rows.append(_summary_row(df.filter(pl.col("barrier_holds") & (pl.col("vwap_1555_constructive") == state)), "barrier_1555_context", f"holds_true_{state}", target))
            rows.append(_summary_row(df.filter(~pl.col("barrier_holds") & (pl.col("vwap_1555_constructive") == state)), "barrier_1555_context", f"holds_false_{state}", target))
        for bucket in ["no_wrong_side_close", "wrong_le_1tick", "wrong_1tick_to_2pts", "wrong_2_to_5pts", "wrong_gt_5pts"]:
            rows.append(_summary_row(bucketed.filter(pl.col("_bucket") == bucket), "wrong_side_close_bucket", bucket, target))
        for value_col, scope in [("wrong_side_share_1550", "wrong_side_share_decile"), ("worst_wrong_side_dist_points", "worst_wrong_side_dist_decile")]:
            deciled = _deciled(df, value_col)
            if deciled is not None:
                for decile in [str(i) for i in range(1, 11)]:
                    rows.append(_summary_row(deciled.filter(pl.col("_bucket") == decile), scope, decile, target))
        for bucket in sorted(df.select(pl.col("vwap_context_10s_to_1555").drop_nulls().unique()).to_series().to_list()):
            rows.append(_summary_row(df.filter(pl.col("vwap_context_10s_to_1555") == bucket), "vwap_1555_decision", bucket, target))
    return pl.DataFrame(rows, infer_schema_length=None).select(MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS)


def write_macro_vwap_barrier_context(
    tick_path: str | Path = TICK_INPUT_PATH,
    barrier_path: str | Path = BARRIER_INPUT_PATH,
    vwap_path: str | Path = VWAP_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    barrier = pl.read_parquet(barrier_path)
    vwap = pl.read_parquet(vwap_path)
    context = build_macro_vwap_barrier_context(barrier, vwap, tick_path)
    summary = summarize_macro_vwap_barrier_context(context)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    context.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    for path in [TICK_INPUT_PATH, BARRIER_INPUT_PATH, VWAP_INPUT_PATH]:
        if not data_sources.source_exists(path):
            print(f"[ERROR] Input not found: {path}", file=sys.stderr)
            sys.exit(1)
    output, summary = write_macro_vwap_barrier_context()
    print(f"[OK] Wrote macro VWAP barrier context -> {output}")
    print(f"[OK] Wrote macro VWAP barrier context summary -> {summary}")


if __name__ == "__main__":
    main()
