from __future__ import annotations

from datetime import date

import pandas as pd

from features.trend.modeling.containment_research import (
    run_classification_bakeoff,
    run_regression_bakeoff,
)


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
