import pandas as pd

from features.macro_fvg_study import detect_macro_fvgs, scan_fvg_outcomes_until_1559_close


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
