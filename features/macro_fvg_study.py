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
    work["bar1_date"] = work["DateTime_ET"].shift(1).dt.normalize()
    work["bar3_date"] = work["DateTime_ET"].shift(-1).dt.normalize()
    work["bar1_high"] = work["High"].shift(1)
    work["bar1_low"] = work["Low"].shift(1)
    work["bar3_time"] = work["DateTime_ET"].shift(-1)
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
                "gap_bottom",
                "gap_top",
                "gap_size",
                "is_confirmable_by_1559",
            ]
        )

    event_rows["date"] = event_rows["DateTime_ET"].dt.normalize()
    event_rows["assigned_at"] = event_rows["DateTime_ET"]
    event_rows["confirmed_at"] = event_rows["bar3_time"] + pd.Timedelta(minutes=1)
    event_rows["assigned_stage"] = event_rows["assigned_at"].map(assign_stage)
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
    event_rows["is_confirmable_by_1559"] = (
        event_rows["confirmed_at"].dt.strftime("%H:%M:%S") <= FINAL_SCAN_TIME
    )

    return event_rows[
        [
            "date",
            "fvg_side",
            "assigned_at",
            "confirmed_at",
            "assigned_stage",
            "gap_bottom",
            "gap_top",
            "gap_size",
            "is_confirmable_by_1559",
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


def _build_scope_summary(
    events: pd.DataFrame,
    scope_name: str,
    retrace_col: str,
    invalidate_col: str,
    held_col: str,
    untouched_col: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "summary_scope",
                "fvg_side",
                "n_total",
                "n_confirmable",
                "hold_rate",
                "retrace_rate",
                "untouched_rate",
                "invalidation_rate",
            ]
        )

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

    return pd.DataFrame(rows)


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
        return _build_scope_summary(
            pd.DataFrame(),
            "stage_1",
            "retraced_by_1559",
            "invalidated_by_1559",
            "held_to_1559_close",
            "untouched_to_1559_close",
        )
    return pd.concat(non_empty_frames, ignore_index=True)


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


def plot_fvg_summary_figures(events: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    if events.empty:
        for filename, title in [
            ("hold_vs_invalidate_by_side.png", "Hold vs Invalidate by Side"),
            ("stage1_to_stage2_outcomes.png", "Stage 1 to Stage 2 Outcomes"),
            ("creation_minute_outcome_heatmap.png", "Creation Minute Outcome Heatmap"),
            ("gap_size_vs_outcome.png", "Gap Size vs Outcome"),
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


def run_macro_fvg_study(
    input_path: Path = INPUT_PATH,
    events_output_path: Path = EVENTS_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = pd.read_parquet(input_path)
    events = detect_macro_fvgs(bars)
    events = scan_fvg_outcomes_until_1559_close(events, bars)
    summary = build_stage_summary_tables(events)

    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_output_path, index=False)
    summary.to_parquet(summary_output_path, index=False)
    plot_fvg_summary_figures(events, summary, figures_dir)
    return events, summary


if __name__ == "__main__":
    run_macro_fvg_study()
