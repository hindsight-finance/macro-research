from __future__ import annotations

import pandas as pd

from features.trend.modeling.labels import assign_three_scalar_labels


def test_assign_three_scalar_labels_marks_uncertain_rows_without_dropping():
    frame = pd.DataFrame(
        {
            "trend_score": [0.85, 0.20, 0.20, 0.45],
            "containment_score": [0.20, 0.82, 0.20, 0.45],
            "chop_score": [0.20, 0.30, 0.86, 0.45],
        }
    )

    labeled = assign_three_scalar_labels(
        frame=frame,
        trend_high=0.70,
        containment_high=0.70,
        chop_high=0.70,
        low_cutoff=0.40,
        containment_chop_max=0.55,
    )

    assert labeled["label"].tolist() == ["trend", "containment", "chop", "uncertain"]


def test_assign_three_scalar_labels_can_drop_uncertain_rows_for_probe_use():
    frame = pd.DataFrame(
        {
            "trend_score": [0.85, 0.45],
            "containment_score": [0.20, 0.45],
            "chop_score": [0.20, 0.45],
        }
    )

    labeled = assign_three_scalar_labels(
        frame=frame,
        trend_high=0.70,
        containment_high=0.70,
        chop_high=0.70,
        low_cutoff=0.40,
        containment_chop_max=0.55,
        drop_uncertain=True,
    )

    assert labeled["label"].tolist() == ["trend"]
