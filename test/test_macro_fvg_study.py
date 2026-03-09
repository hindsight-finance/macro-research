import features.macro_fvg_study as macro_fvg_study
import pandas as pd
import pytest

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


def test_stores_first_retrace_candle_metadata_for_bullish_fvg():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 100.5,
                "High": 101.5,
                "Low": 100.0,
                "Close": 101.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 102.0,
                "High": 103.0,
                "Low": 102.0,
                "Close": 102.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:54:00",
                "Open": 102.5,
                "High": 102.8,
                "Low": 101.2,
                "Close": 101.8,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["first_retrace_candle_at"] == pd.Timestamp("2025-01-02 15:54:00")
    assert event["first_retrace_candle_open"] == 102.5
    assert event["first_retrace_candle_high"] == 102.8
    assert event["first_retrace_candle_low"] == 101.2
    assert event["first_retrace_candle_close"] == 101.8


def test_marks_later_same_side_4_bar_fvg_as_stacked_continuation():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 100.5,
                "High": 101.0,
                "Low": 100.0,
                "Close": 100.8,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 102.0,
                "High": 103.0,
                "Low": 102.0,
                "Close": 102.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 103.0,
                "High": 104.0,
                "Low": 103.0,
                "Close": 103.5,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)

    first_event = events[events["assigned_at"] == pd.Timestamp("2025-01-02 15:50:00")].iloc[0]
    second_event = events[events["assigned_at"] == pd.Timestamp("2025-01-02 15:51:00")].iloc[0]
    assert not first_event["stacked_continuation_fvg"]
    assert pd.isna(first_event["stack_predecessor_assigned_at"])
    assert second_event["stacked_continuation_fvg"]
    assert second_event["stack_predecessor_assigned_at"] == pd.Timestamp("2025-01-02 15:50:00")


def test_marks_bullish_fvg_successful_when_later_breaks_first_retrace_high():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 100.5,
                "High": 101.5,
                "Low": 100.0,
                "Close": 101.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 102.0,
                "High": 103.0,
                "Low": 102.0,
                "Close": 102.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:54:00",
                "Open": 102.5,
                "High": 102.8,
                "Low": 101.2,
                "Close": 101.8,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:56:00",
                "Open": 101.9,
                "High": 103.1,
                "Low": 101.7,
                "Close": 102.9,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["first_retrace_candle_high"] == 102.8
    assert event["success_reference_price"] == 102.8
    assert event["successful_by_1559"]
    assert event["success_break_at"] == pd.Timestamp("2025-01-02 15:56:00")


def test_marks_bearish_fvg_successful_when_later_breaks_first_retrace_low():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
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
            {
                "DateTime_ET": "2025-01-02 15:53:00",
                "Open": 96.5,
                "High": 98.5,
                "Low": 96.2,
                "Close": 97.8,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:55:00",
                "Open": 97.5,
                "High": 97.8,
                "Low": 95.8,
                "Close": 96.0,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["first_retrace_candle_low"] == 96.2
    assert event["success_reference_price"] == 96.2
    assert event["successful_by_1559"]
    assert event["success_break_at"] == pd.Timestamp("2025-01-02 15:55:00")


def test_retraced_fvg_can_remain_unsuccessful_by_1559():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 100.5,
                "High": 101.5,
                "Low": 100.0,
                "Close": 101.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 102.0,
                "High": 103.0,
                "Low": 102.0,
                "Close": 102.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:54:00",
                "Open": 102.5,
                "High": 102.8,
                "Low": 101.2,
                "Close": 101.8,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:56:00",
                "Open": 101.8,
                "High": 102.8,
                "Low": 101.4,
                "Close": 102.1,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 102.0,
                "High": 102.7,
                "Low": 101.6,
                "Close": 102.3,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["retraced_by_1559"]
    assert event["success_reference_price"] == 102.8
    assert not event["successful_by_1559"]
    assert pd.isna(event["success_break_at"])


def test_no_retrace_keeps_success_context_null():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 100.5,
                "High": 101.5,
                "Low": 100.0,
                "Close": 101.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 102.0,
                "High": 103.0,
                "Low": 102.0,
                "Close": 102.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:54:00",
                "Open": 102.5,
                "High": 103.4,
                "Low": 102.2,
                "Close": 103.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 103.0,
                "High": 103.5,
                "Low": 102.1,
                "Close": 102.8,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert pd.isna(event["first_retrace_candle_at"])
    assert pd.isna(event["success_reference_price"])
    assert not event["successful_by_1559"]
    assert pd.isna(event["success_break_at"])


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


def test_bullish_entry_trigger_and_excursions_use_bar3_high():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 100.5,
                "High": 101.5,
                "Low": 100.0,
                "Close": 101.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 102.0,
                "High": 103.0,
                "Low": 102.0,
                "Close": 102.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:54:00",
                "Open": 102.5,
                "High": 104.0,
                "Low": 103.0,
                "Close": 103.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 103.5,
                "High": 105.0,
                "Low": 100.5,
                "Close": 104.0,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["fvg_side"] == "bullish"
    assert event["entry_price"] == 103.0
    assert event["entry_triggered_by_1559"]
    assert event["first_entry_trigger_at"] == pd.Timestamp("2025-01-02 15:54:00")
    assert event["entry_trigger_minute_hhmm"] == "15:54"
    assert event["entry_trigger_minute_index"] == 4
    assert event["mfe_pct_to_1559"] == pytest.approx((105.0 - 103.0) / 103.0)
    assert event["mae_pct_to_1559"] == pytest.approx((103.0 - 100.5) / 103.0)


def test_bearish_entry_trigger_and_excursions_use_bar3_low():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:49:00",
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
            {
                "DateTime_ET": "2025-01-02 15:53:00",
                "Open": 95.5,
                "High": 95.5,
                "Low": 93.5,
                "Close": 94.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 94.0,
                "High": 96.0,
                "Low": 92.0,
                "Close": 93.0,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["fvg_side"] == "bearish"
    assert event["entry_price"] == 94.0
    assert event["entry_triggered_by_1559"]
    assert event["first_entry_trigger_at"] == pd.Timestamp("2025-01-02 15:53:00")
    assert event["entry_trigger_minute_hhmm"] == "15:53"
    assert event["entry_trigger_minute_index"] == 3
    assert event["mfe_pct_to_1559"] == pytest.approx((94.0 - 92.0) / 94.0)
    assert event["mae_pct_to_1559"] == pytest.approx((96.0 - 94.0) / 94.0)


def test_entry_excursion_fields_stay_null_when_entry_never_triggers():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 100.5,
                "High": 101.5,
                "Low": 100.0,
                "Close": 101.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 102.0,
                "High": 103.0,
                "Low": 102.0,
                "Close": 102.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:54:00",
                "Open": 102.5,
                "High": 102.8,
                "Low": 101.8,
                "Close": 102.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 102.0,
                "High": 102.9,
                "Low": 101.0,
                "Close": 101.5,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert not event["entry_triggered_by_1559"]
    assert pd.isna(event["first_entry_trigger_at"])
    assert pd.isna(event["entry_trigger_minute_hhmm"])
    assert pd.isna(event["entry_trigger_minute_index"])
    assert pd.isna(event["mfe_pct_to_1559"])
    assert pd.isna(event["mae_pct_to_1559"])


def test_entry_trigger_on_1559_sets_trigger_without_excursions():
    bars = make_bars(
        [
            {
                "DateTime_ET": "2025-01-02 15:50:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:51:00",
                "Open": 100.5,
                "High": 101.5,
                "Low": 100.0,
                "Close": 101.0,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:52:00",
                "Open": 102.0,
                "High": 103.0,
                "Low": 102.0,
                "Close": 102.5,
                "window": "MACRO",
            },
            {
                "DateTime_ET": "2025-01-02 15:59:00",
                "Open": 102.5,
                "High": 103.2,
                "Low": 102.0,
                "Close": 103.0,
                "window": "MACRO",
            },
        ]
    )

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["entry_triggered_by_1559"]
    assert event["first_entry_trigger_at"] == pd.Timestamp("2025-01-02 15:59:00")
    assert event["entry_trigger_minute_hhmm"] == "15:59"
    assert event["entry_trigger_minute_index"] == 9
    assert pd.isna(event["mfe_pct_to_1559"])
    assert pd.isna(event["mae_pct_to_1559"])


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


def test_builds_alignment_bucket_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "3_aligned",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": ">=2.25",
                "is_confirmable_by_1559": True,
                "held_to_1559_close": True,
                "invalidated_by_1559": False,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
            {
                "alignment_bucket": "3_aligned",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": ">=2.25",
                "is_confirmable_by_1559": True,
                "held_to_1559_close": False,
                "invalidated_by_1559": True,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
        ]
    )

    summary_builder = getattr(macro_fvg_study, "build_alignment_bucket_summary", None)
    assert summary_builder is not None

    summary = summary_builder(events)

    row = summary.iloc[0]
    assert row["summary_scope"] == "alignment_bucket"
    assert row["alignment_bucket"] == "3_aligned"
    assert row["n_total"] == 2
    assert row["hold_rate"] == 0.5
    assert row["invalidation_rate"] == 0.5


def test_builds_alignment_bucket_minute_block_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "2_aligned_1_opposite",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": "<2.25",
                "is_confirmable_by_1559": True,
                "held_to_1559_close": True,
                "invalidated_by_1559": False,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
            {
                "alignment_bucket": "2_aligned_1_opposite",
                "minute_block": "15:53-15:57",
                "gap_size_bucket_225": "<2.25",
                "is_confirmable_by_1559": True,
                "held_to_1559_close": False,
                "invalidated_by_1559": True,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
        ]
    )

    summary_builder = getattr(macro_fvg_study, "build_alignment_bucket_minute_block_summary", None)
    assert summary_builder is not None

    summary = summary_builder(events)

    assert set(summary["minute_block"]) == {"15:50-15:52", "15:53-15:57"}
    early_row = summary[summary["minute_block"] == "15:50-15:52"].iloc[0]
    assert early_row["summary_scope"] == "alignment_bucket_minute_block"
    assert early_row["alignment_bucket"] == "2_aligned_1_opposite"
    assert early_row["hold_rate"] == 1.0


def test_builds_alignment_bucket_gap_bucket_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "contains_neutral",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": "<2.25",
                "is_confirmable_by_1559": True,
                "held_to_1559_close": True,
                "invalidated_by_1559": False,
                "retraced_by_1559": False,
                "untouched_to_1559_close": True,
            },
            {
                "alignment_bucket": "contains_neutral",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": ">=2.25",
                "is_confirmable_by_1559": True,
                "held_to_1559_close": False,
                "invalidated_by_1559": True,
                "retraced_by_1559": True,
                "untouched_to_1559_close": False,
            },
        ]
    )

    summary_builder = getattr(macro_fvg_study, "build_alignment_bucket_gap_bucket_summary", None)
    assert summary_builder is not None

    summary = summary_builder(events)

    assert summary["summary_scope"].eq("alignment_bucket_gap_bucket").all()
    assert set(summary["gap_size_bucket_225"]) == {"<2.25", ">=2.25"}


def test_builds_entry_excursion_alignment_bucket_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "3_aligned",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": ">=2.25",
                "is_confirmable_by_1559": True,
                "entry_triggered_by_1559": True,
                "mfe_pct_to_1559": 0.012,
                "mae_pct_to_1559": 0.004,
            },
            {
                "alignment_bucket": "3_aligned",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": ">=2.25",
                "is_confirmable_by_1559": True,
                "entry_triggered_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
        ]
    )

    summary_builder = getattr(
        macro_fvg_study,
        "build_entry_excursion_alignment_bucket_summary",
        None,
    )
    assert summary_builder is not None

    summary = summary_builder(events)

    row = summary.iloc[0]
    assert row["summary_scope"] == "entry_excursion_alignment_bucket"
    assert row["alignment_bucket"] == "3_aligned"
    assert row["n_confirmable"] == 2
    assert row["n_triggered"] == 1
    assert row["entry_trigger_rate"] == 0.5
    assert row["mfe_pct_mean"] == 0.012
    assert row["mae_pct_mean"] == 0.004


def test_builds_entry_excursion_alignment_bucket_minute_block_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "2_aligned_1_opposite",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": "<2.25",
                "is_confirmable_by_1559": True,
                "entry_triggered_by_1559": True,
                "mfe_pct_to_1559": 0.010,
                "mae_pct_to_1559": 0.003,
            },
            {
                "alignment_bucket": "2_aligned_1_opposite",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": "<2.25",
                "is_confirmable_by_1559": True,
                "entry_triggered_by_1559": True,
                "mfe_pct_to_1559": 0.020,
                "mae_pct_to_1559": 0.005,
            },
            {
                "alignment_bucket": "2_aligned_1_opposite",
                "minute_block": "15:53-15:57",
                "gap_size_bucket_225": "<2.25",
                "is_confirmable_by_1559": True,
                "entry_triggered_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
        ]
    )

    summary_builder = getattr(
        macro_fvg_study,
        "build_entry_excursion_alignment_bucket_minute_block_summary",
        None,
    )
    assert summary_builder is not None

    summary = summary_builder(events)

    early_row = summary[summary["minute_block"] == "15:50-15:52"].iloc[0]
    assert early_row["summary_scope"] == "entry_excursion_alignment_bucket_minute_block"
    assert early_row["alignment_bucket"] == "2_aligned_1_opposite"
    assert early_row["n_confirmable"] == 2
    assert early_row["n_triggered"] == 2
    assert early_row["entry_trigger_rate"] == 1.0
    assert early_row["mfe_pct_mean"] == pytest.approx(0.015)
    assert early_row["mae_pct_mean"] == pytest.approx(0.004)


def test_builds_entry_excursion_gap_bucket_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "contains_neutral",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": "<2.25",
                "is_confirmable_by_1559": True,
                "entry_triggered_by_1559": True,
                "mfe_pct_to_1559": 0.008,
                "mae_pct_to_1559": 0.002,
            },
            {
                "alignment_bucket": "contains_neutral",
                "minute_block": "15:50-15:52",
                "gap_size_bucket_225": ">=2.25",
                "is_confirmable_by_1559": True,
                "entry_triggered_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
        ]
    )

    summary_builder = getattr(
        macro_fvg_study,
        "build_entry_excursion_gap_bucket_summary",
        None,
    )
    assert summary_builder is not None

    summary = summary_builder(events)

    assert summary["summary_scope"].eq("entry_excursion_gap_bucket").all()
    low_gap_row = summary[summary["gap_size_bucket_225"] == "<2.25"].iloc[0]
    assert low_gap_row["n_confirmable"] == 1
    assert low_gap_row["n_triggered"] == 1
    assert low_gap_row["entry_trigger_rate"] == 1.0
    assert low_gap_row["mfe_pct_mean"] == 0.008
    assert low_gap_row["mae_pct_mean"] == 0.002


def test_builds_success_context_alignment_bucket_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "3_aligned",
                "stacked_continuation_fvg": False,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": True,
                "mfe_pct_to_1559": 0.010,
                "mae_pct_to_1559": 0.002,
            },
            {
                "alignment_bucket": "3_aligned",
                "stacked_continuation_fvg": False,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": True,
                "mfe_pct_to_1559": 0.020,
                "mae_pct_to_1559": 0.006,
            },
            {
                "alignment_bucket": "3_aligned",
                "stacked_continuation_fvg": False,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
        ]
    )

    summary_builder = getattr(
        macro_fvg_study,
        "build_success_context_alignment_bucket_summary",
        None,
    )
    assert summary_builder is not None

    summary = summary_builder(events)

    row = summary.iloc[0]
    assert row["summary_scope"] == "success_context_alignment_bucket"
    assert row["alignment_bucket"] == "3_aligned"
    assert row["n_confirmable"] == 3
    assert row["n_retraced"] == 3
    assert row["n_successful"] == 2
    assert row["retrace_rate"] == 1.0
    assert row["success_after_retrace_rate"] == pytest.approx(2 / 3)
    assert row["successful_share_of_confirmable"] == pytest.approx(2 / 3)
    assert row["mfe_pct_mean"] == pytest.approx(0.015)
    assert row["mfe_pct_median"] == pytest.approx(0.015)
    assert row["mfe_pct_p75"] == pytest.approx(0.0175)
    assert row["mae_pct_mean"] == pytest.approx(0.004)
    assert row["mae_pct_median"] == pytest.approx(0.004)
    assert row["mae_pct_p75"] == pytest.approx(0.005)


def test_builds_success_context_stacked_flag_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "2_aligned_1_opposite",
                "stacked_continuation_fvg": True,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": True,
                "mfe_pct_to_1559": 0.009,
                "mae_pct_to_1559": 0.003,
            },
            {
                "alignment_bucket": "3_aligned",
                "stacked_continuation_fvg": True,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
            {
                "alignment_bucket": "contains_neutral",
                "stacked_continuation_fvg": False,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": False,
                "successful_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
        ]
    )

    summary_builder = getattr(
        macro_fvg_study,
        "build_success_context_stacked_flag_summary",
        None,
    )
    assert summary_builder is not None

    summary = summary_builder(events)

    stacked_row = summary[summary["stacked_continuation_fvg"]].iloc[0]
    assert stacked_row["summary_scope"] == "success_context_stacked_flag"
    assert stacked_row["n_confirmable"] == 2
    assert stacked_row["n_retraced"] == 2
    assert stacked_row["n_successful"] == 1
    assert stacked_row["retrace_rate"] == 1.0
    assert stacked_row["success_after_retrace_rate"] == 0.5
    assert stacked_row["successful_share_of_confirmable"] == 0.5
    assert stacked_row["mfe_pct_mean"] == 0.009
    assert stacked_row["mfe_pct_median"] == 0.009
    assert stacked_row["mfe_pct_p75"] == 0.009
    assert stacked_row["mae_pct_mean"] == 0.003
    assert stacked_row["mae_pct_median"] == 0.003
    assert stacked_row["mae_pct_p75"] == 0.003


def test_builds_success_context_alignment_bucket_stacked_flag_summary():
    events = pd.DataFrame(
        [
            {
                "alignment_bucket": "3_aligned",
                "stacked_continuation_fvg": True,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": True,
                "mfe_pct_to_1559": 0.008,
                "mae_pct_to_1559": 0.004,
            },
            {
                "alignment_bucket": "3_aligned",
                "stacked_continuation_fvg": True,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": False,
                "mfe_pct_to_1559": float("nan"),
                "mae_pct_to_1559": float("nan"),
            },
            {
                "alignment_bucket": "3_aligned",
                "stacked_continuation_fvg": False,
                "is_confirmable_by_1559": True,
                "retraced_by_1559": True,
                "successful_by_1559": True,
                "mfe_pct_to_1559": 0.006,
                "mae_pct_to_1559": 0.002,
            },
        ]
    )

    summary_builder = getattr(
        macro_fvg_study,
        "build_success_context_alignment_bucket_stacked_flag_summary",
        None,
    )
    assert summary_builder is not None

    summary = summary_builder(events)

    row = summary[
        (summary["alignment_bucket"] == "3_aligned")
        & (summary["stacked_continuation_fvg"])
    ].iloc[0]
    assert row["summary_scope"] == "success_context_alignment_bucket_stacked_flag"
    assert row["n_confirmable"] == 2
    assert row["n_retraced"] == 2
    assert row["n_successful"] == 1
    assert row["success_after_retrace_rate"] == 0.5
    assert row["successful_share_of_confirmable"] == 0.5
    assert row["mfe_pct_mean"] == 0.008
    assert row["mfe_pct_median"] == 0.008
    assert row["mfe_pct_p75"] == 0.008
    assert row["mae_pct_mean"] == 0.004
    assert row["mae_pct_median"] == 0.004
    assert row["mae_pct_p75"] == 0.004


def test_plot_successful_fvg_mae_by_alignment_bucket_overlays_success_rate(
    tmp_path,
    monkeypatch,
):
    summary = pd.DataFrame(
        [
            {
                "summary_scope": "success_context_alignment_bucket",
                "alignment_bucket": "3_aligned",
                "mae_pct_mean": 0.004,
                "mae_pct_median": 0.003,
                "mae_pct_p75": 0.006,
                "successful_share_of_confirmable": 0.40,
            },
            {
                "summary_scope": "success_context_alignment_bucket",
                "alignment_bucket": "2_aligned_1_opposite",
                "mae_pct_mean": 0.005,
                "mae_pct_median": 0.004,
                "mae_pct_p75": 0.007,
                "successful_share_of_confirmable": 0.25,
            },
        ]
    )

    captured = {}
    original_close = macro_fvg_study.plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else macro_fvg_study.plt.gcf()

    monkeypatch.setattr(macro_fvg_study.plt, "close", capture_close)

    macro_fvg_study.plot_successful_fvg_mae_by_alignment_bucket(summary, tmp_path)

    fig = captured["fig"]
    assert len(fig.axes) == 2
    secondary_ax = fig.axes[1]
    assert secondary_ax.get_ylabel() == "Success Rate"
    assert len(secondary_ax.lines) == 1
    assert list(secondary_ax.lines[0].get_ydata()) == pytest.approx([40.0, 25.0])
    original_close(fig)


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

    events = pd.read_parquet(events_path)
    summary = pd.read_parquet(summary_path)

    assert events_path.exists()
    assert summary_path.exists()
    assert "successful_by_1559" in events.columns
    assert "stacked_continuation_fvg" in events.columns
    assert "first_retrace_candle_at" in events.columns
    assert "success_context_alignment_bucket" in set(summary["summary_scope"])
    assert "success_context_stacked_flag" in set(summary["summary_scope"])
    assert (figures_dir / "hold_vs_invalidate_by_side.png").exists()
    assert (figures_dir / "stage1_to_stage2_outcomes.png").exists()
    assert (figures_dir / "creation_minute_outcome_heatmap.png").exists()
    assert (figures_dir / "gap_size_vs_outcome.png").exists()
    assert (figures_dir / "creation_minute_outcome_bars.png").exists()
    assert (figures_dir / "bar2_volume_bucket_outcomes.png").exists()
    assert (figures_dir / "creation_minute_avg_bar2_volume.png").exists()
    assert (figures_dir / "creation_minute_volume_heatmap.png").exists()
    assert (figures_dir / "alignment_bucket_outcomes.png").exists()
    assert (figures_dir / "alignment_bucket_by_minute_block.png").exists()
    assert (figures_dir / "alignment_bucket_by_gap_bucket.png").exists()
    assert (figures_dir / "alignment_bucket_counts.png").exists()
    assert (figures_dir / "entry_trigger_rate_by_alignment_bucket.png").exists()
    assert (figures_dir / "mfe_mae_pct_by_alignment_bucket.png").exists()
    assert (figures_dir / "mfe_pct_by_minute_block.png").exists()
    assert (figures_dir / "mfe_pct_by_gap_bucket.png").exists()
    assert (figures_dir / "successful_fvg_mae_by_alignment_bucket.png").exists()
    assert (figures_dir / "successful_fvg_mae_by_stacked_flag.png").exists()
    assert (figures_dir / "successful_fvg_mfe_by_alignment_bucket.png").exists()
    assert (figures_dir / "successful_fvg_mfe_by_stacked_flag.png").exists()
