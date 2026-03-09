import pandas as pd

from features.macro_fvg_study import (
    build_bar2_volume_summary,
    build_creation_minute_summary,
    build_stage_summary_tables,
    detect_macro_fvgs,
    run_macro_fvg_study,
    scan_fvg_outcomes_until_1559_close,
)


def make_bars(rows):
    df = pd.DataFrame(rows)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    else:
        df["Volume"] = df["Volume"].fillna(0)
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


def test_detect_macro_fvg_stores_assigned_minute_and_bar2_volume():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Volume": 10,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "Volume": 25,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "Volume": 40,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["assigned_minute_hhmm"] == "15:50"
    assert event["assigned_minute_index"] == 0
    assert event["bar2_volume"] == 25


def test_detect_macro_fvg_stores_later_assigned_minute_index():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:56:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Volume": 10,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:57:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "Volume": 55,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:58:00",
                "Open": 96.0,
                "High": 97.0,
                "Low": 94.0,
                "Close": 95.0,
                "Volume": 80,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["assigned_minute_hhmm"] == "15:57"
    assert event["assigned_minute_index"] == 7
    assert event["bar2_volume"] == 55


def test_detect_macro_fvg_stores_alignment_bucket_for_three_aligned():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 99.0,
                "Volume": 10,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "Volume": 25,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 97.0,
                "High": 98.0,
                "Low": 95.0,
                "Close": 96.0,
                "Volume": 40,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["fvg_side"] == "bearish"
    assert event["bar1_direction"] == "bearish"
    assert event["bar2_direction"] == "bearish"
    assert event["bar3_direction"] == "bearish"
    assert event["aligned_count"] == 3
    assert event["opposite_count"] == 0
    assert event["neutral_count"] == 0
    assert event["alignment_bucket"] == "3_aligned"


def test_detect_macro_fvg_stores_alignment_bucket_for_two_aligned_one_opposite():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 101.0,
                "High": 101.5,
                "Low": 98.5,
                "Close": 99.0,
                "Volume": 100,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 98.0,
                "Volume": 110,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 95.0,
                "High": 96.5,
                "Low": 94.0,
                "Close": 96.0,
                "Volume": 120,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["fvg_side"] == "bearish"
    assert event["bar1_direction"] == "bearish"
    assert event["bar2_direction"] == "bearish"
    assert event["bar3_direction"] == "bullish"
    assert event["aligned_count"] == 2
    assert event["opposite_count"] == 1
    assert event["neutral_count"] == 0
    assert event["alignment_bucket"] == "2_aligned_1_opposite"


def test_detect_macro_fvg_stores_alignment_bucket_for_one_aligned_two_opposite():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 97.0,
                "High": 100.0,
                "Low": 96.0,
                "Close": 99.0,
                "Volume": 10,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 96.0,
                "High": 99.0,
                "Low": 95.0,
                "Close": 98.0,
                "Volume": 25,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 95.0,
                "High": 95.0,
                "Low": 93.0,
                "Close": 94.0,
                "Volume": 40,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["fvg_side"] == "bearish"
    assert event["bar1_direction"] == "bullish"
    assert event["bar2_direction"] == "bullish"
    assert event["bar3_direction"] == "bearish"
    assert event["aligned_count"] == 1
    assert event["opposite_count"] == 2
    assert event["neutral_count"] == 0
    assert event["alignment_bucket"] == "1_aligned_2_opposite"


def test_detect_macro_fvg_marks_contains_neutral_when_pattern_has_doji():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 101.0,
                "High": 101.5,
                "Low": 98.5,
                "Close": 99.0,
                "Volume": 100,
                "window": "H3PM",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 99.0,
                "High": 100.0,
                "Low": 97.0,
                "Close": 99.0,
                "Volume": 110,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 95.0,
                "High": 96.0,
                "Low": 94.0,
                "Close": 95.0,
                "Volume": 120,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["bar2_direction"] == "neutral"
    assert event["bar3_direction"] == "neutral"
    assert event["aligned_count"] == 1
    assert event["opposite_count"] == 0
    assert event["neutral_count"] == 2
    assert event["alignment_bucket"] == "contains_neutral"


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


def test_builds_creation_minute_summary():
    events = pd.DataFrame(
        [
            {
                "assigned_minute_index": 0,
                "assigned_minute_hhmm": "15:50",
                "bar2_volume": 100,
                "is_confirmable_by_1559": True,
                "held_to_1559_close": True,
                "invalidated_by_1559": False,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
            {
                "assigned_minute_index": 0,
                "assigned_minute_hhmm": "15:50",
                "bar2_volume": 200,
                "is_confirmable_by_1559": True,
                "held_to_1559_close": False,
                "invalidated_by_1559": True,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
        ]
    )

    minute_summary = build_creation_minute_summary(events)

    row = minute_summary.iloc[0]
    assert row["assigned_minute_hhmm"] == "15:50"
    assert row["assigned_minute_index"] == 0
    assert row["n_total"] == 2
    assert row["n_confirmable"] == 2
    assert row["hold_rate"] == 0.5
    assert row["invalidation_rate"] == 0.5


def test_builds_bar2_volume_bucket_summary():
    events = pd.DataFrame(
        [
            {
                "assigned_minute_index": 0,
                "assigned_minute_hhmm": "15:50",
                "bar2_volume": 100,
                "is_confirmable_by_1559": True,
                "held_to_1559_close": True,
                "invalidated_by_1559": False,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
            {
                "assigned_minute_index": 1,
                "assigned_minute_hhmm": "15:51",
                "bar2_volume": 200,
                "is_confirmable_by_1559": True,
                "held_to_1559_close": False,
                "invalidated_by_1559": True,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
            {
                "assigned_minute_index": 2,
                "assigned_minute_hhmm": "15:52",
                "bar2_volume": 300,
                "is_confirmable_by_1559": True,
                "held_to_1559_close": True,
                "invalidated_by_1559": False,
                "retraced_by_1559": False,
                "untouched_to_1559_close": True,
            },
            {
                "assigned_minute_index": 3,
                "assigned_minute_hhmm": "15:53",
                "bar2_volume": 400,
                "is_confirmable_by_1559": True,
                "held_to_1559_close": False,
                "invalidated_by_1559": True,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
        ]
    )

    summary = build_bar2_volume_summary(events, bucket_count=2)

    assert "bar2_volume_bucket" in summary.columns
    assert len(summary) == 2
    assert summary["n_total"].sum() == 4


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
    assert (figures_dir / "creation_minute_outcome_bars.png").exists()
    assert (figures_dir / "bar2_volume_bucket_outcomes.png").exists()
    assert (figures_dir / "creation_minute_avg_bar2_volume.png").exists()
    assert (figures_dir / "creation_minute_volume_heatmap.png").exists()
