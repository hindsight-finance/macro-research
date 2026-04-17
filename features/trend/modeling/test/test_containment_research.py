from __future__ import annotations

from datetime import date

import pandas as pd

from features.trend.modeling.containment_research import (
    run_classification_bakeoff,
    run_regression_bakeoff,
    run_three_way_probe,
)
from features.trend.modeling.labels import assign_three_scalar_labels


def _make_research_frame(session_name: str) -> pd.DataFrame:
    rows = []
    for idx in range(36):
        trade_date = pd.Timestamp("2022-01-03") + pd.DateOffset(months=idx)
        base = 0.2 + 0.01 * idx
        rows.append(
            {
                "trade_date": date(trade_date.year, trade_date.month, min(trade_date.day, 28)),
                "session_name": session_name,
                "feature_status": "ok",
                "containment_status": "ok",
                "mss": base,
                "er": 1.0 - base,
                "containment_range_stability": 0.3 + 0.5 * base,
                "containment_target": 0.25 + 0.4 * base,
                "descriptive_target": 0.55 - 0.25 * base,
                "trend_score": 0.55 - 0.25 * base,
                "containment_score": 0.25 + 0.4 * base,
                "chop_score": 0.15 + 0.25 * (1.0 - base),
            }
        )
    return pd.DataFrame(rows)


def test_run_regression_bakeoff_returns_ranked_rows():
    table = _make_research_frame("1pm-3pm")

    results = run_regression_bakeoff(
        table=table,
        session_name="1pm-3pm",
        feature_columns=("mss", "er", "containment_range_stability"),
        holdout_fraction=0.2,
        train_months=12,
        valid_months=3,
        step_months=3,
    )

    assert {"model", "holdout_r2", "oos_r2"} <= set(results.columns)
    assert {"ridge", "hist_gbm", "random_forest", "extra_trees"} <= set(results["model"])


def test_run_classification_bakeoff_returns_pr_metrics():
    table = _make_research_frame("1pm-3pm")

    results = run_classification_bakeoff(
        table=table,
        session_name="1pm-3pm",
        feature_columns=("mss", "er", "containment_range_stability"),
        target_column="containment_target",
        holdout_fraction=0.2,
        train_months=12,
        valid_months=3,
        step_months=3,
    )

    assert {"model", "pr_auc", "precision_at_10pct", "lift_at_10pct"} <= set(results.columns)
    assert {"logit", "hist_gbm", "random_forest"} <= set(results["model"])


def test_assign_three_scalar_labels_returns_three_classes():
    table = pd.DataFrame(
        {
            "trend_score": [0.8, 0.2, 0.2],
            "containment_score": [0.2, 0.8, 0.2],
            "chop_score": [0.2, 0.3, 0.8],
        }
    )

    labeled = assign_three_scalar_labels(
        frame=table,
        trend_high=0.7,
        containment_high=0.7,
        chop_high=0.7,
        low_cutoff=0.4,
    )

    assert {"trend", "containment", "chop"} == set(labeled["label"])


def test_assign_three_scalar_labels_excludes_high_chop_from_containment():
    table = pd.DataFrame(
        {
            "trend_score": [0.2, 0.2],
            "containment_score": [0.8, 0.8],
            "chop_score": [0.50, 0.65],
        }
    )

    labeled = assign_three_scalar_labels(
        frame=table,
        trend_high=0.7,
        containment_high=0.7,
        chop_high=0.7,
        low_cutoff=0.4,
        containment_chop_max=0.55,
        drop_uncertain=True,
    )

    assert labeled["label"].tolist() == ["containment"]


def test_run_three_way_probe_filters_uncertain_rows_before_training():
    table = pd.DataFrame(
        {
            "trade_date": pd.date_range("2022-01-03", periods=16, freq="MS").date,
            "session_name": ["1pm-3pm"] * 16,
            "feature_status": ["ok"] * 16,
            "containment_status": ["ok"] * 16,
            "target_status": ["ok"] * 16,
            "mss": [0.1] * 16,
            "er": [0.2] * 16,
            "containment_range_stability": [0.3] * 16,
            "trend_score": [0.85, 0.20, 0.20, 0.45] * 4,
            "containment_score": [0.20, 0.82, 0.20, 0.45] * 4,
            "chop_score": [0.20, 0.30, 0.86, 0.45] * 4,
        }
    )

    results = run_three_way_probe(
        table=table,
        session_feature_sets={"1pm-3pm": ("mss", "er", "containment_range_stability")},
        holdout_fraction=0.25,
        label_thresholds={"trend_high": 0.7, "containment_high": 0.7, "chop_high": 0.7, "low_cutoff": 0.4},
    )

    assert {"model", "macro_f1", "balanced_accuracy", "confusion_matrix"} <= set(results.columns)
    assert set(results["n_dev_labeled"]) == {9}
    assert set(results["n_holdout_labeled"]) == {3}


def test_run_three_way_probe_returns_macro_metrics():
    table = _make_research_frame("1pm-3pm")
    extra = table.copy()
    trend_pattern = [0.85, 0.20, 0.15] * 12
    containment_pattern = [0.20, 0.85, 0.20] * 12
    chop_pattern = [0.20, 0.25, 0.85] * 12
    extra["descriptive_target"] = trend_pattern
    extra["trend_score"] = extra["descriptive_target"]
    extra["containment_target"] = containment_pattern
    extra["containment_score"] = extra["containment_target"]
    extra["chop_score"] = chop_pattern

    results = run_three_way_probe(
        table=extra,
        session_feature_sets={"1pm-3pm": ("mss", "er", "containment_range_stability")},
        holdout_fraction=0.2,
        label_thresholds={"trend_high": 0.7, "containment_high": 0.7, "chop_high": 0.7, "low_cutoff": 0.4},
    )

    assert {"model", "macro_f1", "balanced_accuracy", "confusion_matrix"} <= set(results.columns)
    assert {"multinomial_logit", "hist_gbm", "random_forest"} <= set(results["model"])


def test_run_three_way_probe_skips_when_holdout_lacks_full_class_coverage():
    table = _make_research_frame("1pm-3pm")
    limited = table.copy()
    limited["descriptive_target"] = [0.85] * len(limited)
    limited["trend_score"] = limited["descriptive_target"]
    limited["containment_target"] = [0.20] * len(limited)
    limited["containment_score"] = limited["containment_target"]
    limited["chop_score"] = [0.20] * len(limited)

    results = run_three_way_probe(
        table=limited,
        session_feature_sets={"1pm-3pm": ("mss", "er", "containment_range_stability")},
        holdout_fraction=0.2,
        label_thresholds={"trend_high": 0.7, "containment_high": 0.7, "chop_high": 0.7, "low_cutoff": 0.4},
    )

    assert results.empty
