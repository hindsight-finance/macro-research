from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import polars as pl

from utils import data_sources
from utils.minute_bars import MARKET_TZ
from utils.tick_data import (
    TICK_PRICE_DENOMINATOR,
    get_tick_schema,
    open_parquet_file,
    scan_source,
)

INPUT_PATH = data_sources.tick_data_url()
DEFAULT_BARRIER_PATH = Path("outputs/nq_macro_1550_barrier.parquet")
PREMACRO_OUTPUT_PATH = Path("outputs/nq_macro_vwap_premacro.parquet")
PREMACRO_SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_vwap_premacro_summary.parquet")
INTRAMACRO_OUTPUT_PATH = Path("outputs/nq_macro_vwap_intramacro.parquet")
INTRAMACRO_SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_vwap_intramacro_summary.parquet")

UTC_NS = pl.Datetime("ns", time_zone="UTC")
TOUCH_THRESHOLD_POINTS = 0.25
_REQUIRED_TICK_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks", "size"}
_REQUIRED_BARRIER_COLUMNS = {
    "date",
    "macro_trend_state",
    "barrier_extreme",
    "barrier_price",
    "barrier_time",
    "barrier_first10",
    "barrier_is_macro_extreme",
    "barrier_holds",
    "edge_case",
}

TARGET_PREFIXES = ("target_1550_1554", "target_1555_1559", "target_1550_1559")
PREMACRO_FEATURE_PREFIXES = ("rth_0930", "pm_1300", "h3pm_1500")
INTRAMACRO_FEATURE_PREFIXES = (
    "macro_1550_at_1550_10s",
    "macro_1550_at_1555",
    "macro_1550_at_1600",
    "eoii_1555_at_1600",
)
BARRIER_COLUMNS = [
    "barrier_macro_trend_state",
    "barrier_extreme",
    "barrier_price",
    "barrier_time",
    "barrier_first10",
    "barrier_is_macro_extreme",
    "barrier_holds",
    "barrier_edge_case",
]

_TARGET_COLUMNS = [col for prefix in TARGET_PREFIXES for col in (f"{prefix}_points", f"{prefix}_sign", f"{prefix}_state")]
_PREMACRO_VWAP_COLUMNS = [col for prefix in PREMACRO_FEATURE_PREFIXES for col in (f"{prefix}_vwap", f"{prefix}_price", f"{prefix}_vwap_dist_points", f"{prefix}_vwap_dist_bps", f"{prefix}_vwap_side")]
_INTRAMACRO_VWAP_COLUMNS = [col for prefix in INTRAMACRO_FEATURE_PREFIXES for col in (f"{prefix}_vwap", f"{prefix}_price", f"{prefix}_vwap_dist_points", f"{prefix}_vwap_dist_bps", f"{prefix}_vwap_side")]

PREMACRO_COLUMNS = [
    "date",
    *_PREMACRO_VWAP_COLUMNS,
    "premacro_above_count",
    "premacro_below_count",
    "premacro_touch_count",
    "premacro_net_side_score",
    *_TARGET_COLUMNS,
]
INTRAMACRO_COLUMNS = [
    "date",
    *_INTRAMACRO_VWAP_COLUMNS,
    "intramacro_above_count",
    "intramacro_below_count",
    "intramacro_touch_count",
    "intramacro_net_side_score",
    *_TARGET_COLUMNS,
    *BARRIER_COLUMNS,
]
SUMMARY_COLUMNS = [
    "feature_set",
    "feature_name",
    "target_name",
    "scope",
    "bucket",
    "sample_size",
    "bullish_count",
    "bearish_count",
    "neutral_count",
    "bullish_pct",
    "bearish_pct",
    "neutral_pct",
    "avg_target_points",
    "median_target_points",
]


def _et_second(hour: int, minute: int, second: int = 0) -> int:
    return hour * 3600 + minute * 60 + second


PREMACRO_SPECS = {
    "rth_0930": (_et_second(9, 30), _et_second(15, 50)),
    "pm_1300": (_et_second(13, 0), _et_second(15, 50)),
    "h3pm_1500": (_et_second(15, 0), _et_second(15, 50)),
}
INTRAMACRO_SPECS = {
    "macro_1550_at_1550_10s": (_et_second(15, 50), _et_second(15, 50, 10)),
    "macro_1550_at_1555": (_et_second(15, 50), _et_second(15, 55)),
    "macro_1550_at_1600": (_et_second(15, 50), _et_second(16, 0)),
    "eoii_1555_at_1600": (_et_second(15, 55), _et_second(16, 0)),
}
TARGET_SPECS = {
    "target_1550_1554": (_et_second(15, 50), _et_second(15, 55)),
    "target_1555_1559": (_et_second(15, 55), _et_second(16, 0)),
    "target_1550_1559": (_et_second(15, 50), _et_second(16, 0)),
}


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = sorted(_REQUIRED_TICK_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def _scan_ticks(
    path: str | Path,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> pl.LazyFrame:
    _validate_tick_schema(path)
    ts_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ)
    lf = scan_source(path).select(
        pl.col("ts_event").cast(UTC_NS).alias("ts_event"),
        pl.col("intra_ts_rank").cast(pl.Int64),
        pl.col("price_ticks").cast(pl.Int64),
        pl.col("size").cast(pl.Int64),
    )
    if start_utc is not None:
        lf = lf.filter(pl.col("ts_event") >= pl.lit(start_utc))
    if end_utc is not None:
        lf = lf.filter(pl.col("ts_event") < pl.lit(end_utc))
    return (
        lf.with_columns(
            datetime_et=ts_et,
            date=ts_et.dt.date(),
            et_second=(
                ts_et.dt.hour().cast(pl.Int32) * 3600
                + ts_et.dt.minute().cast(pl.Int32) * 60
                + ts_et.dt.second().cast(pl.Int32)
            ),
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .with_columns(pv=pl.col("price") * pl.col("size").cast(pl.Float64))
        .filter((pl.col("et_second") >= _et_second(9, 30)) & (pl.col("et_second") < _et_second(16, 0)))
    )


def _safe_vwap_expr(prefix: str) -> pl.Expr:
    return pl.when(pl.col(f"{prefix}_total_size") > 0).then(pl.col(f"{prefix}_pv") / pl.col(f"{prefix}_total_size")).otherwise(None)


def _side_expr(prefix: str) -> pl.Expr:
    dist = pl.col(f"{prefix}_vwap_dist_points")
    return (
        pl.when(dist.is_null())
        .then(None)
        .when(dist.abs() <= TOUCH_THRESHOLD_POINTS)
        .then(pl.lit("touch"))
        .when(dist > TOUCH_THRESHOLD_POINTS)
        .then(pl.lit("above"))
        .otherwise(pl.lit("below"))
    )


def _add_distance_columns(lf: pl.LazyFrame, prefix: str) -> pl.LazyFrame:
    return (
        lf.with_columns(
            (pl.col(f"{prefix}_price") - pl.col(f"{prefix}_vwap")).alias(f"{prefix}_vwap_dist_points"),
            pl.when((pl.col(f"{prefix}_vwap").is_not_null()) & (pl.col(f"{prefix}_vwap") != 0))
            .then((pl.col(f"{prefix}_price") / pl.col(f"{prefix}_vwap") - 1.0) * 10000.0)
            .otherwise(None)
            .alias(f"{prefix}_vwap_dist_bps"),
        )
        .with_columns(_side_expr(prefix).alias(f"{prefix}_vwap_side"))
    )


def _state_expr(points_col: str) -> pl.Expr:
    points = pl.col(points_col)
    return (
        pl.when(points.is_null())
        .then(None)
        .when(points > 0)
        .then(pl.lit("bullish"))
        .when(points < 0)
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("neutral"))
    )


def _sign_expr(points_col: str) -> pl.Expr:
    points = pl.col(points_col)
    return pl.when(points.is_null()).then(None).when(points > 0).then(1).when(points < 0).then(-1).otherwise(0)


def _window_mask(start_second: int, end_second: int) -> pl.Expr:
    return (pl.col("et_second") >= start_second) & (pl.col("et_second") < end_second)


def _ordered_price(mask: pl.Expr, position: str) -> pl.Expr:
    if position == "first":
        ts_at_edge = pl.col("ts_event").filter(mask).min()
        rank_at_edge = pl.col("intra_ts_rank").filter(mask & (pl.col("ts_event") == ts_at_edge)).min()
        return pl.col("price").filter(
            mask
            & (pl.col("ts_event") == ts_at_edge)
            & (pl.col("intra_ts_rank") == rank_at_edge)
        ).first()
    if position == "last":
        ts_at_edge = pl.col("ts_event").filter(mask).max()
        rank_at_edge = pl.col("intra_ts_rank").filter(mask & (pl.col("ts_event") == ts_at_edge)).max()
        return pl.col("price").filter(
            mask
            & (pl.col("ts_event") == ts_at_edge)
            & (pl.col("intra_ts_rank") == rank_at_edge)
        ).last()
    raise ValueError(f"position must be 'first' or 'last', got {position!r}")


def _daily_vwap_target_frame(base: pl.LazyFrame, vwap_specs: dict[str, tuple[int, int]]) -> pl.LazyFrame:
    aggs: list[pl.Expr] = []
    for prefix, (anchor_second, checkpoint_second) in vwap_specs.items():
        mask = _window_mask(anchor_second, checkpoint_second)
        aggs.extend(
            [
                pl.col("pv").filter(mask).sum().alias(f"{prefix}_pv"),
                pl.col("size").filter(mask).sum().alias(f"{prefix}_total_size"),
                _ordered_price(mask, "last").alias(f"{prefix}_price"),
            ]
        )
    for prefix, (start_second, end_second) in TARGET_SPECS.items():
        mask = _window_mask(start_second, end_second)
        aggs.extend(
            [
                _ordered_price(mask, "first").alias(f"{prefix}_open"),
                _ordered_price(mask, "last").alias(f"{prefix}_close"),
            ]
        )

    out = base.group_by("date", maintain_order=True).agg(*aggs)

    for prefix in vwap_specs:
        out = out.with_columns(_safe_vwap_expr(prefix).alias(f"{prefix}_vwap"))
        out = _add_distance_columns(out, prefix)

    for prefix in TARGET_PREFIXES:
        points_col = f"{prefix}_points"
        out = out.with_columns((pl.col(f"{prefix}_close") - pl.col(f"{prefix}_open")).alias(points_col))
        out = out.with_columns(
            _sign_expr(points_col).cast(pl.Int8).alias(f"{prefix}_sign"),
            _state_expr(points_col).alias(f"{prefix}_state"),
        )

    return out


def _count_side_expr(prefixes: tuple[str, ...], side: str) -> pl.Expr:
    expr = pl.lit(0, dtype=pl.Int16)
    for prefix in prefixes:
        expr = expr + (pl.col(f"{prefix}_vwap_side") == side).cast(pl.Int16).fill_null(0)
    return expr


def _add_confluence(lf: pl.LazyFrame, prefixes: tuple[str, ...], output_prefix: str) -> pl.LazyFrame:
    return lf.with_columns(
        _count_side_expr(prefixes, "above").alias(f"{output_prefix}_above_count"),
        _count_side_expr(prefixes, "below").alias(f"{output_prefix}_below_count"),
        _count_side_expr(prefixes, "touch").alias(f"{output_prefix}_touch_count"),
    ).with_columns(
        (pl.col(f"{output_prefix}_above_count") - pl.col(f"{output_prefix}_below_count")).alias(f"{output_prefix}_net_side_score")
    )


def _empty_barrier_context() -> pl.LazyFrame:
    return pl.LazyFrame(
        schema={
            "date": pl.Date,
            "barrier_macro_trend_state": pl.String,
            "barrier_extreme": pl.String,
            "barrier_price": pl.Float64,
            "barrier_time": pl.Int64,
            "barrier_first10": pl.Boolean,
            "barrier_is_macro_extreme": pl.Boolean,
            "barrier_holds": pl.Boolean,
            "barrier_edge_case": pl.Boolean,
        }
    )


def _barrier_context(barrier_path: str | Path | None) -> pl.LazyFrame:
    if barrier_path is None:
        return _empty_barrier_context()
    path = Path(barrier_path)
    if not path.exists():
        return _empty_barrier_context()
    missing = sorted(_REQUIRED_BARRIER_COLUMNS - set(get_tick_schema(path).names))
    if missing:
        raise ValueError(f"Missing barrier columns: {missing}")
    return pl.scan_parquet(path).select(
        pl.col("date").cast(pl.Date),
        pl.col("macro_trend_state").alias("barrier_macro_trend_state"),
        "barrier_extreme",
        "barrier_price",
        "barrier_time",
        "barrier_first10",
        "barrier_is_macro_extreme",
        "barrier_holds",
        pl.col("edge_case").alias("barrier_edge_case"),
    )


def build_macro_vwap_premacro(
    path: str | Path = INPUT_PATH,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> pl.LazyFrame:
    base = _scan_ticks(path, start_utc=start_utc, end_utc=end_utc)
    out = _daily_vwap_target_frame(base, PREMACRO_SPECS)
    out = _add_confluence(out, PREMACRO_FEATURE_PREFIXES, "premacro")
    return out.select(PREMACRO_COLUMNS).sort("date")


def build_macro_vwap_intramacro(
    path: str | Path = INPUT_PATH,
    barrier_path: str | Path | None = DEFAULT_BARRIER_PATH,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> pl.LazyFrame:
    base = _scan_ticks(path, start_utc=start_utc, end_utc=end_utc)
    out = _daily_vwap_target_frame(base, INTRAMACRO_SPECS)
    out = _add_confluence(out, INTRAMACRO_FEATURE_PREFIXES, "intramacro")
    out = out.join(_barrier_context(barrier_path), on="date", how="left")
    return out.select(INTRAMACRO_COLUMNS).sort("date")


def _feature_prefixes(feature_set: str) -> tuple[str, ...]:
    if feature_set == "premacro":
        return PREMACRO_FEATURE_PREFIXES
    if feature_set == "intramacro":
        return INTRAMACRO_FEATURE_PREFIXES
    raise ValueError(f"feature_set must be 'premacro' or 'intramacro', got {feature_set!r}")


def _confluence_column(feature_set: str) -> str:
    return "premacro_net_side_score" if feature_set == "premacro" else "intramacro_net_side_score"


def _pct(count: int, denom: int) -> float | None:
    return (count / denom) * 100.0 if denom else None


def _summary_row(subset: pl.DataFrame, feature_set: str, feature_name: str, target_name: str, scope: str, bucket: str) -> dict:
    state_col = f"{target_name}_state"
    points_col = f"{target_name}_points"
    sample_size = subset.height
    bullish_count = subset.filter(pl.col(state_col) == "bullish").height if sample_size else 0
    bearish_count = subset.filter(pl.col(state_col) == "bearish").height if sample_size else 0
    neutral_count = subset.filter(pl.col(state_col) == "neutral").height if sample_size else 0
    return {
        "feature_set": feature_set,
        "feature_name": feature_name,
        "target_name": target_name,
        "scope": scope,
        "bucket": str(bucket),
        "sample_size": sample_size,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "bullish_pct": _pct(bullish_count, sample_size),
        "bearish_pct": _pct(bearish_count, sample_size),
        "neutral_pct": _pct(neutral_count, sample_size),
        "avg_target_points": subset.select(pl.col(points_col).mean()).item() if sample_size else None,
        "median_target_points": subset.select(pl.col(points_col).median()).item() if sample_size else None,
    }


def _bps_band_expr(prefix: str) -> pl.Expr:
    bps = pl.col(f"{prefix}_vwap_dist_bps")
    side = pl.col(f"{prefix}_vwap_side")
    abs_bps = bps.abs()
    return (
        pl.when(bps.is_null())
        .then(None)
        .when(side == "touch")
        .then(pl.lit("touch"))
        .when((bps < 0) & (abs_bps > 20))
        .then(pl.lit("below_gt_20"))
        .when((bps < 0) & (abs_bps > 10))
        .then(pl.lit("below_10_20"))
        .when((bps < 0) & (abs_bps > 5))
        .then(pl.lit("below_5_10"))
        .when((bps < 0) & (abs_bps > 2))
        .then(pl.lit("below_2_5"))
        .when(bps < 0)
        .then(pl.lit("below_0_2"))
        .when((bps > 0) & (abs_bps <= 2))
        .then(pl.lit("above_0_2"))
        .when((bps > 0) & (abs_bps <= 5))
        .then(pl.lit("above_2_5"))
        .when((bps > 0) & (abs_bps <= 10))
        .then(pl.lit("above_5_10"))
        .when((bps > 0) & (abs_bps <= 20))
        .then(pl.lit("above_10_20"))
        .otherwise(pl.lit("above_gt_20"))
    )


def _deciled_frame(df: pl.DataFrame, prefix: str) -> pl.DataFrame | None:
    value_col = f"{prefix}_vwap_dist_bps"
    non_null = df.filter(pl.col(value_col).is_not_null())
    if non_null.height < 10:
        return None
    if non_null.select(pl.col(value_col).n_unique()).item() < 10:
        return None
    return non_null.with_columns(
        (((pl.col(value_col).rank(method="ordinal") - 1) * 10 / non_null.height)
         .floor()
         .cast(pl.Int64)
         .clip(0, 9)
         + 1)
        .cast(pl.String)
        .alias("_bucket")
    )


def _target_available(df: pl.DataFrame, target_name: str) -> bool:
    return f"{target_name}_points" in df.columns and f"{target_name}_state" in df.columns


def summarize_macro_vwap_features(df: pl.DataFrame, feature_set: str) -> pl.DataFrame:
    prefixes = _feature_prefixes(feature_set)
    rows: list[dict] = []
    targets = [target for target in TARGET_PREFIXES if _target_available(df, target)]
    for prefix in prefixes:
        side_col = f"{prefix}_vwap_side"
        bps_col = f"{prefix}_vwap_dist_bps"
        if side_col not in df.columns or bps_col not in df.columns:
            continue
        banded = df.with_columns(_bps_band_expr(prefix).alias("_bucket"))
        deciled = _deciled_frame(df, prefix)
        for target in targets:
            for side in ["above", "below", "touch"]:
                subset = df.filter(pl.col(side_col) == side)
                rows.append(_summary_row(subset, feature_set, prefix, target, "side", side))
            for band in [
                "below_gt_20", "below_10_20", "below_5_10", "below_2_5", "below_0_2",
                "touch", "above_0_2", "above_2_5", "above_5_10", "above_10_20", "above_gt_20",
            ]:
                subset = banded.filter(pl.col("_bucket") == band)
                rows.append(_summary_row(subset, feature_set, prefix, target, "fixed_bps_band", band))
            if deciled is not None:
                for decile in [str(i) for i in range(1, 11)]:
                    rows.append(_summary_row(deciled.filter(pl.col("_bucket") == decile), feature_set, prefix, target, "decile", decile))

    confluence_col = _confluence_column(feature_set)
    if confluence_col in df.columns:
        for target in targets:
            for bucket in sorted(df.select(pl.col(confluence_col).drop_nulls().unique()).to_series().to_list()):
                subset = df.filter(pl.col(confluence_col) == bucket)
                rows.append(_summary_row(subset, feature_set, confluence_col, target, "confluence", str(bucket)))

    if feature_set == "intramacro" and "barrier_first10" in df.columns:
        for prefix in prefixes:
            side_col = f"{prefix}_vwap_side"
            if side_col not in df.columns:
                continue
            for target in targets:
                for flag_col, scope in [("barrier_first10", "barrier_first10_by_side"), ("barrier_holds", "barrier_holds_by_side")]:
                    if flag_col not in df.columns:
                        continue
                    for side in ["above", "below", "touch"]:
                        for flag_value in [True, False]:
                            subset = df.filter((pl.col(side_col) == side) & (pl.col(flag_col) == flag_value))
                            rows.append(_summary_row(subset, feature_set, prefix, target, scope, f"{side}_{str(flag_value).lower()}"))

    if not rows:
        return pl.DataFrame(schema={col: pl.Null for col in SUMMARY_COLUMNS})
    return pl.DataFrame(rows, infer_schema_length=None).select(SUMMARY_COLUMNS)




def _edge_exprs(prefix: str, mask: pl.Expr, edge: str) -> list[pl.Expr]:
    if edge == "first":
        ts_at_edge = pl.col("ts_event").filter(mask).min()
        rank_at_edge = pl.col("intra_ts_rank").filter(mask & (pl.col("ts_event") == ts_at_edge)).min()
    elif edge == "last":
        ts_at_edge = pl.col("ts_event").filter(mask).max()
        rank_at_edge = pl.col("intra_ts_rank").filter(mask & (pl.col("ts_event") == ts_at_edge)).max()
    else:
        raise ValueError(f"edge must be 'first' or 'last', got {edge!r}")
    edge_mask = mask & (pl.col("ts_event") == ts_at_edge) & (pl.col("intra_ts_rank") == rank_at_edge)
    price_expr = pl.col("price").filter(edge_mask).first() if edge == "first" else pl.col("price").filter(edge_mask).last()
    return [
        ts_at_edge.alias(f"{prefix}_{edge}_ts"),
        rank_at_edge.alias(f"{prefix}_{edge}_rank"),
        price_expr.alias(f"{prefix}_{edge}_price"),
    ]


def _batch_aggregate(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame()
    ts_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ)
    work = (
        df.select(
            pl.col("ts_event").cast(UTC_NS),
            pl.col("intra_ts_rank").cast(pl.Int64),
            pl.col("price_ticks").cast(pl.Int64),
            pl.col("size").cast(pl.Int64),
        )
        .with_columns(
            date=ts_et.dt.date(),
            et_second=(
                ts_et.dt.hour().cast(pl.Int32) * 3600
                + ts_et.dt.minute().cast(pl.Int32) * 60
                + ts_et.dt.second().cast(pl.Int32)
            ),
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .with_columns(pv=pl.col("price") * pl.col("size").cast(pl.Float64))
        .filter((pl.col("et_second") >= _et_second(9, 30)) & (pl.col("et_second") < _et_second(16, 0)))
    )
    if work.is_empty():
        return pl.DataFrame()

    all_specs = {**PREMACRO_SPECS, **INTRAMACRO_SPECS}
    aggs: list[pl.Expr] = []
    for prefix, (start_second, end_second) in all_specs.items():
        mask = _window_mask(start_second, end_second)
        aggs.extend([
            pl.col("pv").filter(mask).sum().alias(f"{prefix}_pv"),
            pl.col("size").filter(mask).sum().alias(f"{prefix}_total_size"),
            *_edge_exprs(prefix, mask, "last"),
        ])
    for prefix, (start_second, end_second) in TARGET_SPECS.items():
        mask = _window_mask(start_second, end_second)
        aggs.extend([
            *_edge_exprs(f"{prefix}_open", mask, "first"),
            *_edge_exprs(f"{prefix}_close", mask, "last"),
        ])
    return work.group_by("date").agg(*aggs)


def _is_better_edge(row: dict, state: dict, key: str, edge: str) -> bool:
    ts = row.get(f"{key}_{edge}_ts")
    rank = row.get(f"{key}_{edge}_rank")
    if ts is None or rank is None:
        return False
    old_ts = state.get(f"{key}_{edge}_ts")
    old_rank = state.get(f"{key}_{edge}_rank")
    if old_ts is None:
        return True
    if edge == "first":
        return (ts, rank) < (old_ts, old_rank)
    return (ts, rank) > (old_ts, old_rank)


def _update_edge(row: dict, state: dict, key: str, edge: str) -> None:
    if _is_better_edge(row, state, key, edge):
        state[f"{key}_{edge}_ts"] = row.get(f"{key}_{edge}_ts")
        state[f"{key}_{edge}_rank"] = row.get(f"{key}_{edge}_rank")
        state[f"{key}_{edge}_price"] = row.get(f"{key}_{edge}_price")


def _side_from_dist(dist: float | None) -> str | None:
    if dist is None:
        return None
    if abs(dist) <= TOUCH_THRESHOLD_POINTS:
        return "touch"
    return "above" if dist > TOUCH_THRESHOLD_POINTS else "below"


def _target_state(points: float | None) -> tuple[int | None, str | None]:
    if points is None:
        return None, None
    if points > 0:
        return 1, "bullish"
    if points < 0:
        return -1, "bearish"
    return 0, "neutral"


def _finalize_feature_row(date, state: dict, prefixes: tuple[str, ...], feature_set: str) -> dict:
    row = {"date": date}
    for prefix in prefixes:
        total_size = state.get(f"{prefix}_total_size", 0) or 0
        pv = state.get(f"{prefix}_pv", 0.0) or 0.0
        vwap = (pv / total_size) if total_size > 0 else None
        price = state.get(f"{prefix}_last_price")
        dist_points = (price - vwap) if price is not None and vwap is not None else None
        dist_bps = ((price / vwap - 1.0) * 10000.0) if price is not None and vwap not in (None, 0) else None
        row[f"{prefix}_vwap"] = vwap
        row[f"{prefix}_price"] = price
        row[f"{prefix}_vwap_dist_points"] = dist_points
        row[f"{prefix}_vwap_dist_bps"] = dist_bps
        row[f"{prefix}_vwap_side"] = _side_from_dist(dist_points)

    output_prefix = "premacro" if feature_set == "premacro" else "intramacro"
    sides = [row.get(f"{prefix}_vwap_side") for prefix in prefixes]
    row[f"{output_prefix}_above_count"] = sum(side == "above" for side in sides)
    row[f"{output_prefix}_below_count"] = sum(side == "below" for side in sides)
    row[f"{output_prefix}_touch_count"] = sum(side == "touch" for side in sides)
    row[f"{output_prefix}_net_side_score"] = row[f"{output_prefix}_above_count"] - row[f"{output_prefix}_below_count"]

    for prefix in TARGET_PREFIXES:
        open_price = state.get(f"{prefix}_open_first_price")
        close_price = state.get(f"{prefix}_close_last_price")
        points = (close_price - open_price) if open_price is not None and close_price is not None else None
        sign, trend_state = _target_state(points)
        row[f"{prefix}_points"] = points
        row[f"{prefix}_sign"] = sign
        row[f"{prefix}_state"] = trend_state
    return row


def _compute_streaming_features(input_path: str | Path, batch_size: int = 750_000) -> tuple[pl.DataFrame, pl.DataFrame]:
    _validate_tick_schema(input_path)
    states: dict[object, dict] = {}
    parquet_file = open_parquet_file(input_path)
    columns = ["ts_event", "intra_ts_rank", "price_ticks", "size"]
    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        batch_df = _batch_aggregate(pl.from_arrow(batch))
        if batch_df.is_empty():
            continue
        for row in batch_df.iter_rows(named=True):
            date = row["date"]
            state = states.setdefault(date, {})
            for prefix in (*PREMACRO_FEATURE_PREFIXES, *INTRAMACRO_FEATURE_PREFIXES):
                state[f"{prefix}_pv"] = (state.get(f"{prefix}_pv", 0.0) or 0.0) + (row.get(f"{prefix}_pv", 0.0) or 0.0)
                state[f"{prefix}_total_size"] = (state.get(f"{prefix}_total_size", 0) or 0) + (row.get(f"{prefix}_total_size", 0) or 0)
                _update_edge(row, state, prefix, "last")
            for prefix in TARGET_PREFIXES:
                _update_edge(row, state, f"{prefix}_open", "first")
                _update_edge(row, state, f"{prefix}_close", "last")

    premacro_rows = [_finalize_feature_row(date, state, PREMACRO_FEATURE_PREFIXES, "premacro") for date, state in states.items()]
    intramacro_rows = [_finalize_feature_row(date, state, INTRAMACRO_FEATURE_PREFIXES, "intramacro") for date, state in states.items()]
    premacro = pl.DataFrame(premacro_rows, infer_schema_length=None).select(PREMACRO_COLUMNS).sort("date") if premacro_rows else pl.DataFrame(schema={column: pl.Null for column in PREMACRO_COLUMNS})
    intramacro_cols = [column for column in INTRAMACRO_COLUMNS if column not in BARRIER_COLUMNS]
    intramacro = pl.DataFrame(intramacro_rows, infer_schema_length=None).select(intramacro_cols).sort("date") if intramacro_rows else pl.DataFrame(schema={column: pl.Null for column in intramacro_cols})
    return premacro, intramacro

def _month_start(value: datetime) -> datetime:
    return datetime(value.year, value.month, 1, tzinfo=timezone.utc)


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)


def _tick_month_ranges(path: str | Path) -> list[tuple[datetime, datetime]]:
    _validate_tick_schema(path)
    bounds = scan_source(path).select(
        pl.col("ts_event").cast(UTC_NS).min().alias("min_ts"),
        pl.col("ts_event").cast(UTC_NS).max().alias("max_ts"),
    ).collect()
    min_ts = bounds.item(0, "min_ts")
    max_ts = bounds.item(0, "max_ts")
    if min_ts is None or max_ts is None:
        return []
    cursor = _month_start(min_ts)
    end_limit = _next_month(_month_start(max_ts))
    ranges: list[tuple[datetime, datetime]] = []
    while cursor < end_limit:
        nxt = _next_month(cursor)
        ranges.append((cursor, nxt))
        cursor = nxt
    return ranges


def _collect_monthly(
    input_path: str | Path,
    feature_set: str,
    barrier_path: str | Path | None = DEFAULT_BARRIER_PATH,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for start_utc, end_utc in _tick_month_ranges(input_path):
        if feature_set == "premacro":
            lf = build_macro_vwap_premacro(input_path, start_utc=start_utc, end_utc=end_utc)
        elif feature_set == "intramacro":
            lf = build_macro_vwap_intramacro(
                input_path,
                barrier_path=barrier_path,
                start_utc=start_utc,
                end_utc=end_utc,
            )
        else:
            raise ValueError(f"feature_set must be 'premacro' or 'intramacro', got {feature_set!r}")
        frame = lf.collect(engine="streaming")
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        columns = PREMACRO_COLUMNS if feature_set == "premacro" else INTRAMACRO_COLUMNS
        return pl.DataFrame(schema={column: pl.Null for column in columns})
    return pl.concat(frames, how="vertical_relaxed").unique(subset=["date"], keep="first").sort("date")

def _write_df(path: str | Path, df: pl.DataFrame) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)
    return output


def write_macro_vwap_features(
    input_path: str | Path = INPUT_PATH,
    premacro_output_path: str | Path = PREMACRO_OUTPUT_PATH,
    premacro_summary_output_path: str | Path = PREMACRO_SUMMARY_OUTPUT_PATH,
    intramacro_output_path: str | Path = INTRAMACRO_OUTPUT_PATH,
    intramacro_summary_output_path: str | Path = INTRAMACRO_SUMMARY_OUTPUT_PATH,
    barrier_path: str | Path | None = DEFAULT_BARRIER_PATH,
) -> tuple[Path, Path, Path, Path]:
    premacro, intramacro_without_barrier = _compute_streaming_features(input_path)
    barrier = _barrier_context(barrier_path).collect(engine="streaming")
    intramacro = intramacro_without_barrier.join(barrier, on="date", how="left").select(INTRAMACRO_COLUMNS)
    premacro_summary = summarize_macro_vwap_features(premacro, feature_set="premacro")
    intramacro_summary = summarize_macro_vwap_features(intramacro, feature_set="intramacro")

    premacro_output = _write_df(premacro_output_path, premacro)
    premacro_summary_output = _write_df(premacro_summary_output_path, premacro_summary)
    intramacro_output = _write_df(intramacro_output_path, intramacro)
    intramacro_summary_output = _write_df(intramacro_summary_output_path, intramacro_summary)
    return premacro_output, premacro_summary_output, intramacro_output, intramacro_summary_output


def main() -> None:
    if not data_sources.source_exists(INPUT_PATH):
        print(f"[ERROR] Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    outputs = write_macro_vwap_features()
    print(f"[OK] Wrote macro VWAP premacro features -> {outputs[0]}")
    print(f"[OK] Wrote macro VWAP premacro summary -> {outputs[1]}")
    print(f"[OK] Wrote macro VWAP intramacro features -> {outputs[2]}")
    print(f"[OK] Wrote macro VWAP intramacro summary -> {outputs[3]}")


if __name__ == "__main__":
    main()
