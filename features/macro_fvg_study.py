from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
ALIGNMENT_BUCKET_ORDER = [
    "3_aligned",
    "2_aligned_1_opposite",
    "1_aligned_2_opposite",
    "contains_neutral",
]
SUMMARY_COLUMNS = [
    "summary_scope",
    "fvg_side",
    "n_total",
    "n_confirmable",
    "hold_rate",
    "retrace_rate",
    "untouched_rate",
    "invalidation_rate",
    "assigned_minute_hhmm",
    "assigned_minute_index",
    "bar2_volume_bucket",
]


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


def detect_macro_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    required = {"DateTime_ET", "Open", "High", "Low", "Close", "Volume", "window"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["DateTime_ET"] = pd.to_datetime(work["DateTime_ET"])
    work = work.sort_values("DateTime_ET").reset_index(drop=True)
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
                "is_confirmable_by_1559",
                "bar1_direction",
                "bar2_direction",
                "bar3_direction",
                "aligned_count",
                "opposite_count",
                "neutral_count",
                "alignment_bucket",
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
            "is_confirmable_by_1559",
            "bar1_direction",
            "bar2_direction",
            "bar3_direction",
            "aligned_count",
            "opposite_count",
            "neutral_count",
            "alignment_bucket",
        ]
    ].reset_index(drop=True)


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
    if "window" in work_bars.columns:
        work_bars = work_bars[work_bars["window"] == MACRO_WINDOW].copy()
    work_bars["date"] = work_bars["DateTime_ET"].dt.normalize()
    bars_by_date = {
        date: group.reset_index(drop=True)
        for date, group in work_bars.groupby("date", sort=False)
    }

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
                    "first_retrace_at": pd.NaT,
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
        first_invalidation_at = pd.NaT
        first_stage_2_retrace_at = pd.NaT
        first_stage_2_invalidation_at = pd.NaT

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

        event_dict.update(
            {
                "first_retrace_at": first_retrace_at,
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


def run_macro_fvg_study(
    input_path: Path = INPUT_PATH,
    events_output_path: Path = EVENTS_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = pd.read_parquet(input_path)
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
