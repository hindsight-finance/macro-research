import pandas as pd

from features.macro_fvg_study import (
    build_stage_summary_tables,
    detect_macro_fvgs,
    run_macro_fvg_study,
    scan_fvg_outcomes_until_1559_close,
)


def make_bars(rows):
    df = pd.DataFrame(rows)
    df["DateTime_ET"] = pd.to_datetime(df["DateTime_ET"])
    return df


def test_detects_bearish_macro_fvg_and_stores_assigned_and_confirmed_times():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["fvg_side"] == "bearish"
    assert event["assigned_at"] == pd.Timestamp("2025-01-02 15:50:00")
    assert event["confirmed_at"] == pd.Timestamp("2025-01-02 15:52:00")
    assert event["assigned_stage"] == "stage_1"


def test_excludes_new_detection_assigned_at_1559():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:58:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 100.0,
                "Close": 101.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 103.0,
                "High": 104.0,
                "Low": 103.0,
                "Close": 104.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 16:00:00",
                "Open": 105.0,
                "High": 106.0,
                "Low": 105.0,
                "Close": 106.0,
                "window": "POST",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    assert events.empty


def test_does_not_detect_fvg_across_day_boundary():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-01 15:59:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    assert events.empty


def test_marks_retrace_without_invalidation():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 95.0,
                "High": 98.5,
                "Low": 95.0,
                "Close": 96.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 96.5,
                "High": 97.0,
                "Low": 96.0,
                "Close": 96.2,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    assert len(scanned) == 1
    event = scanned.iloc[0]
    assert event["first_retrace_at"] == pd.Timestamp("2025-01-02 15:52:00")
    assert pd.isna(event["first_invalidation_at"])
    assert event["retraced_by_1559"]
    assert not event["invalidated_by_1559"]
    assert event["held_to_1559_close"]
    assert not event["untouched_to_1559_close"]


def test_marks_same_bar_retrace_and_invalidation():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 95.0,
                "High": 100.0,
                "Low": 95.0,
                "Close": 99.5,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["first_retrace_at"] == pd.Timestamp("2025-01-02 15:52:00")
    assert event["first_invalidation_at"] == pd.Timestamp("2025-01-02 15:52:00")
    assert event["retraced_by_1559"]
    assert event["invalidated_by_1559"]
    assert not event["held_to_1559_close"]


def test_marks_unconfirmable_late_fvg():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:57:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:58:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    assert len(scanned) == 1
    event = scanned.iloc[0]
    assert not event["is_confirmable_by_1559"]
    assert pd.isna(event["first_retrace_at"])
    assert pd.isna(event["first_invalidation_at"])


def test_builds_stage_and_transition_summary_rates():
    events = pd.DataFrame(
        [
            {
                "fvg_side": "bullish",
                "assigned_stage": "stage_1",
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "invalidated_by_1559": False,
                "held_to_1559_close": True,
                "untouched_to_1559_close": False,
                "retraced_in_stage_2": True,
                "invalidated_in_stage_2": False,
                "held_through_stage_2": True,
                "untouched_through_stage_2": False,
            },
            {
                "fvg_side": "bearish",
                "assigned_stage": "stage_1",
                "is_confirmable_by_1559": True,
                "retraced_by_1559": False,
                "invalidated_by_1559": True,
                "held_to_1559_close": False,
                "untouched_to_1559_close": True,
                "retraced_in_stage_2": False,
                "invalidated_in_stage_2": True,
                "held_through_stage_2": False,
                "untouched_through_stage_2": True,
            },
            {
                "fvg_side": "bullish",
                "assigned_stage": "stage_2",
                "is_confirmable_by_1559": True,
                "retraced_by_1559": False,
                "invalidated_by_1559": False,
                "held_to_1559_close": True,
                "untouched_to_1559_close": True,
                "retraced_in_stage_2": False,
                "invalidated_in_stage_2": False,
                "held_through_stage_2": True,
                "untouched_through_stage_2": True,
            },
        ]
    )

    summary = build_stage_summary_tables(events)

    assert set(summary["summary_scope"]) == {"stage_1", "stage_2", "stage_1_to_stage_2"}

    stage_1_bull = summary[
        (summary["summary_scope"] == "stage_1") & (summary["fvg_side"] == "bullish")
    ].iloc[0]
    assert stage_1_bull["n_total"] == 1
    assert stage_1_bull["n_confirmable"] == 1
    assert stage_1_bull["hold_rate"] == 1.0
    assert stage_1_bull["retrace_rate"] == 1.0

    transition_bear = summary[
        (summary["summary_scope"] == "stage_1_to_stage_2")
        & (summary["fvg_side"] == "bearish")
    ].iloc[0]
    assert transition_bear["invalidation_rate"] == 1.0
    assert transition_bear["untouched_rate"] == 1.0


def test_tracks_stage_1_outcomes_during_stage_2_window():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 95.0,
                "High": 96.5,
                "Low": 95.0,
                "Close": 96.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:55:00",
                "Open": 96.0,
                "High": 98.0,
                "Low": 96.0,
                "Close": 97.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 97.5,
                "High": 98.0,
                "Low": 97.0,
                "Close": 97.2,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["retraced_in_stage_2"]
    assert not event["invalidated_in_stage_2"]
    assert event["held_through_stage_2"]
    assert not event["untouched_through_stage_2"]


def test_run_macro_fvg_study_writes_parquet_and_figures(tmp_path):
    input_path = tmp_path / "nq_1m.parquet"
    events_path = tmp_path / "nq_macro_fvg_events.parquet"
    summary_path = tmp_path / "nq_macro_fvg_summary.parquet"
    figures_dir = tmp_path / "figs" / "fvg"

    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 95.0,
                "High": 96.5,
                "Low": 95.0,
                "Close": 96.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:55:00",
                "Open": 96.0,
                "High": 98.0,
                "Low": 96.0,
                "Close": 97.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:57:00",
                "Open": 97.0,
                "High": 98.0,
                "Low": 96.0,
                "Close": 97.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:58:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 95.0,
                "High": 96.0,
                "Low": 94.0,
                "Close": 95.5,
                "window": "MACRO",
            },
        ]
    )
    bars.to_parquet(input_path, index=False)

    run_macro_fvg_study(
        input_path=input_path,
        events_output_path=events_path,
        summary_output_path=summary_path,
        figures_dir=figures_dir,
    )

    assert events_path.exists()
    assert summary_path.exists()
    assert (figures_dir / "hold_vs_invalidate_by_side.png").exists()
    assert (figures_dir / "stage1_to_stage2_outcomes.png").exists()
    assert (figures_dir / "creation_minute_outcome_heatmap.png").exists()
    assert (figures_dir / "gap_size_vs_outcome.png").exists()
