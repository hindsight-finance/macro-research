from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.trend.modeling.registry import filter_table_for_era
from features.trend.modeling.walkforward import generate_walkforward_folds, reserve_final_holdout


DEFAULT_CONTAINMENT_FEATURE_SETS = {
    "1pm-3pm": (
        "mss",
        "adx_strength",
        "adx_persistence",
        "adx_crossover",
        "irr",
        "er",
        "log_vr",
        "containment_overshoot_ratio",
        "containment_range_stability",
        "containment_mid_cross_count",
        "containment_swing_symmetry",
        "containment_ib_extension_ratio",
        "containment_ib_asymmetry",
        "containment_bandwidth_squeeze",
        "containment_vwap_acceptance",
        "containment_excess_rejection",
    ),
    "3pm-3:50pm": (
        "mss",
        "adx_quality",
        "irr",
        "er",
        "log_vr",
        "dra",
        "containment_overshoot_ratio",
        "containment_range_stability",
        "containment_mid_cross_count",
        "containment_swing_symmetry",
        "containment_ib_extension_ratio",
        "containment_ib_asymmetry",
        "containment_bandwidth_squeeze",
        "containment_vwap_acceptance",
        "containment_excess_rejection",
    ),
}


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    left = pd.Series(a)
    right = pd.Series(b)
    if len(left) <= 1 or left.nunique(dropna=False) <= 1 or right.nunique(dropna=False) <= 1:
        return np.nan
    return float(left.corr(right, method=method))


def _prepare_session_frame(
    table: pd.DataFrame,
    session_name: str,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> pd.DataFrame:
    frame = table.copy()
    if "session_name" in frame.columns:
        frame = frame.loc[frame["session_name"] == session_name].copy()
    if "feature_status" in frame.columns:
        frame = frame.loc[frame["feature_status"] == "ok"].copy()
    if "containment_status" in frame.columns:
        frame = frame.loc[frame["containment_status"] == "ok"].copy()
    if "target_status" in frame.columns:
        frame = frame.loc[frame["target_status"] == "ok"].copy()

    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    required_columns = list(feature_columns) + [target_column]
    frame = frame.dropna(subset=required_columns).sort_values("trade_date").reset_index(drop=True)
    return frame


def _regression_models() -> dict[str, object]:
    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "hist_gbm": HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=250,
            min_samples_leaf=20,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


def _classification_models() -> dict[str, object]:
    return {
        "logit": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
            ]
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=250,
            min_samples_leaf=20,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }


def run_regression_bakeoff(
    table: pd.DataFrame,
    session_name: str,
    feature_columns: tuple[str, ...],
    target_column: str = "containment_target",
    holdout_fraction: float = 0.15,
    train_months: int = 24,
    valid_months: int = 3,
    step_months: int = 3,
) -> pd.DataFrame:
    frame = _prepare_session_frame(
        table=table,
        session_name=session_name,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    development_dates, holdout_dates = reserve_final_holdout(
        frame["trade_date"],
        holdout_fraction=holdout_fraction,
    )
    development = frame.loc[frame["trade_date"].isin(development_dates)].copy()
    holdout = frame.loc[frame["trade_date"].isin(holdout_dates)].copy()
    folds = generate_walkforward_folds(
        development_dates,
        train_months=train_months,
        valid_months=valid_months,
        step_months=step_months,
    )

    rows: list[dict] = []
    for model_name, model in _regression_models().items():
        oos_targets: list[float] = []
        oos_predictions: list[float] = []
        for fold in folds:
            train_df = development.loc[development["trade_date"].isin(fold.train_dates)].copy()
            valid_df = development.loc[development["trade_date"].isin(fold.validation_dates)].copy()
            fit_model = clone(model)
            fit_model.fit(train_df[list(feature_columns)], train_df[target_column])
            fold_prediction = fit_model.predict(valid_df[list(feature_columns)])
            oos_targets.extend(valid_df[target_column].tolist())
            oos_predictions.extend(fold_prediction.tolist())

        final_model = clone(model)
        final_model.fit(development[list(feature_columns)], development[target_column])
        holdout_prediction = final_model.predict(holdout[list(feature_columns)])

        rows.append(
            {
                "session": session_name,
                "model": model_name,
                "features": len(feature_columns),
                "oos_mae": float(mean_absolute_error(oos_targets, oos_predictions)),
                "oos_r2": float(r2_score(oos_targets, oos_predictions)),
                "oos_pearson": _corr(np.asarray(oos_targets), np.asarray(oos_predictions), "pearson"),
                "oos_spearman": _corr(np.asarray(oos_targets), np.asarray(oos_predictions), "spearman"),
                "holdout_mae": float(mean_absolute_error(holdout[target_column], holdout_prediction)),
                "holdout_r2": float(r2_score(holdout[target_column], holdout_prediction)),
                "holdout_pearson": _corr(holdout[target_column].to_numpy(), holdout_prediction, "pearson"),
                "holdout_spearman": _corr(holdout[target_column].to_numpy(), holdout_prediction, "spearman"),
                "n_dev": int(len(development)),
                "n_holdout": int(len(holdout)),
                "n_folds": int(len(folds)),
            }
        )

    return pd.DataFrame(rows).sort_values("holdout_r2", ascending=False).reset_index(drop=True)


def run_classification_bakeoff(
    table: pd.DataFrame,
    session_name: str,
    feature_columns: tuple[str, ...],
    target_column: str = "containment_target",
    holdout_fraction: float = 0.15,
    train_months: int = 24,
    valid_months: int = 3,
    step_months: int = 3,
    target_quantile: float = 0.75,
) -> pd.DataFrame:
    frame = _prepare_session_frame(
        table=table,
        session_name=session_name,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    development_dates, holdout_dates = reserve_final_holdout(
        frame["trade_date"],
        holdout_fraction=holdout_fraction,
    )
    development = frame.loc[frame["trade_date"].isin(development_dates)].copy()
    holdout = frame.loc[frame["trade_date"].isin(holdout_dates)].copy()
    threshold = float(development[target_column].quantile(target_quantile))
    y_dev = (development[target_column] >= threshold).astype(int)
    y_hold = (holdout[target_column] >= threshold).astype(int)
    prevalence = float(y_hold.mean())
    has_both_classes = y_hold.nunique() > 1

    rows: list[dict] = []
    for model_name, model in _classification_models().items():
        fit_model = clone(model)
        fit_model.fit(development[list(feature_columns)], y_dev)
        probability = fit_model.predict_proba(holdout[list(feature_columns)])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        top_k = max(1, int(np.ceil(len(probability) * 0.10)))
        top_indices = np.argsort(probability)[-top_k:]
        precision_at_10pct = float(y_hold.iloc[top_indices].mean())

        rows.append(
            {
                "session": session_name,
                "model": model_name,
                "threshold": threshold,
                "holdout_prevalence": prevalence,
                "pr_auc": float(average_precision_score(y_hold, probability)),
                "roc_auc": float(roc_auc_score(y_hold, probability)) if has_both_classes else np.nan,
                "precision_at_10pct": precision_at_10pct,
                "lift_at_10pct": float(precision_at_10pct / prevalence) if prevalence > 0 else np.nan,
                "precision_at_0_5": float(precision_score(y_hold, prediction, zero_division=0)),
                "recall_at_0_5": float(recall_score(y_hold, prediction, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_hold, prediction)) if has_both_classes else np.nan,
                "mcc": float(matthews_corrcoef(y_hold, prediction)) if has_both_classes else np.nan,
                "n_holdout": int(len(holdout)),
                "positives_holdout": int(y_hold.sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def run_default_containment_research(
    table: pd.DataFrame,
    output_dir: str | Path,
    table_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regression_frames: list[pd.DataFrame] = []
    classification_frames: list[pd.DataFrame] = []

    for session_name, feature_columns in DEFAULT_CONTAINMENT_FEATURE_SETS.items():
        regression_frames.append(
            run_regression_bakeoff(
                table=table,
                session_name=session_name,
                feature_columns=feature_columns,
            )
        )
        classification_frames.append(
            run_classification_bakeoff(
                table=table,
                session_name=session_name,
                feature_columns=feature_columns,
            )
        )

    regression_summary = pd.concat(regression_frames, ignore_index=True)
    classification_summary = pd.concat(classification_frames, ignore_index=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    regression_summary.to_csv(output / "regression_summary.csv", index=False)
    classification_summary.to_csv(output / "classification_summary.csv", index=False)
    (output / "regression_summary.json").write_text(regression_summary.to_json(orient="records", indent=2))
    (output / "classification_summary.json").write_text(classification_summary.to_json(orient="records", indent=2))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "table_label": table_label,
                "sessions": list(DEFAULT_CONTAINMENT_FEATURE_SETS.keys()),
                "regression_models": list(_regression_models().keys()),
                "classification_models": list(_classification_models().keys()),
            },
            indent=2,
        )
    )

    return regression_summary, classification_summary


def load_post_covid_table(table_path: str | Path) -> pd.DataFrame:
    table = pd.read_parquet(table_path)
    return filter_table_for_era(table, "post_covid")
