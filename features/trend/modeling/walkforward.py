from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from features.trend.modeling.registry import ExperimentSpec


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]

    def to_manifest(self) -> dict:
        return asdict(self)


def _coerce_trade_dates(trade_dates: Iterable[object]) -> pd.DatetimeIndex:
    normalized = pd.to_datetime(pd.Index(trade_dates)).normalize().unique().sort_values()
    return pd.DatetimeIndex(normalized)


def _build_model(model_spec: ExperimentSpec):
    if model_spec.model_name == "ridge":
        return Ridge(alpha=model_spec.alpha)
    if model_spec.model_name == "elastic_net":
        if model_spec.l1_ratio is None:
            raise ValueError("Elastic Net requires l1_ratio")
        return ElasticNet(alpha=model_spec.alpha, l1_ratio=model_spec.l1_ratio, max_iter=10000)
    raise ValueError(f"Unsupported model_name: {model_spec.model_name}")


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def reserve_final_holdout(
    trade_dates: Iterable[object],
    holdout_fraction: float = 0.15,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    dates = _coerce_trade_dates(trade_dates)
    if dates.empty:
        raise ValueError("trade_dates must not be empty")

    n_holdout = max(1, int(np.ceil(len(dates) * holdout_fraction)))
    holdout_dates = dates[-n_holdout:]
    development_dates = dates[:-n_holdout]

    return development_dates, holdout_dates


def generate_walkforward_folds(
    trade_dates: Iterable[object],
    train_months: int = 24,
    valid_months: int = 3,
    step_months: int = 3,
) -> list[WalkForwardFold]:
    _require_positive_int("train_months", train_months)
    _require_positive_int("valid_months", valid_months)
    _require_positive_int("step_months", step_months)

    dates = _coerce_trade_dates(trade_dates)
    if dates.empty:
        return []

    folds: list[WalkForwardFold] = []
    fold_id = 1
    validation_start = dates.min() + pd.DateOffset(months=train_months)
    last_date = dates.max()

    while validation_start <= last_date:
        train_start_boundary = validation_start - pd.DateOffset(months=train_months)
        validation_end_boundary = validation_start + pd.DateOffset(months=valid_months)

        train_mask = (dates >= train_start_boundary) & (dates < validation_start)
        valid_mask = (dates >= validation_start) & (dates < validation_end_boundary)

        train_dates = dates[train_mask]
        validation_dates = dates[valid_mask]

        if len(train_dates) and len(validation_dates):
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_dates.min(),
                    train_end=train_dates.max(),
                    validation_start=validation_dates.min(),
                    validation_end=validation_dates.max(),
                    train_dates=tuple(train_dates),
                    validation_dates=tuple(validation_dates),
                )
            )
            fold_id += 1

        validation_start = validation_start + pd.DateOffset(months=step_months)

    return folds


def fit_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_spec: ExperimentSpec,
) -> dict:
    required_columns = feature_columns + [target_column]
    missing_train = [column for column in required_columns if column not in train_df.columns]
    missing_valid = [column for column in required_columns if column not in valid_df.columns]
    if missing_train:
        raise ValueError(f"Train frame missing columns: {missing_train}")
    if missing_valid:
        raise ValueError(f"Validation frame missing columns: {missing_valid}")

    if train_df[required_columns].isna().any().any():
        raise ValueError("Train frame contains NaNs in required columns")
    if valid_df[required_columns].isna().any().any():
        raise ValueError("Validation frame contains NaNs in required columns")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns])
    x_valid = scaler.transform(valid_df[feature_columns])
    y_train = train_df[target_column].to_numpy()
    y_valid = valid_df[target_column].to_numpy()

    model = _build_model(model_spec)
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)

    prediction_frame = valid_df.copy()
    prediction_frame["prediction"] = predictions
    prediction_frame["target"] = y_valid
    prediction_frame["residual"] = prediction_frame["target"] - prediction_frame["prediction"]

    return {
        "predictions": prediction_frame,
        "coefficients": dict(zip(feature_columns, model.coef_, strict=True)),
        "intercept": float(model.intercept_),
        "scaler_mean": dict(zip(feature_columns, scaler.mean_, strict=True)),
        "scaler_scale": dict(zip(feature_columns, scaler.scale_, strict=True)),
    }


def _json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_experiment_artifacts(
    experiment_dir: Path,
    manifest: dict,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame | dict,
    metrics: dict,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)

    coefficients_frame = coefficients if isinstance(coefficients, pd.DataFrame) else pd.DataFrame(
        {
            "feature": list(coefficients.keys()),
            "coefficient": list(coefficients.values()),
        }
    )

    predictions.to_parquet(experiment_dir / "oos_predictions.parquet", index=False)
    coefficients_frame.to_parquet(experiment_dir / "fold_coefficients.parquet", index=False)
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))


def _compute_regression_metrics(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {
            "status": "skipped",
            "n_rows": 0,
            "n_dates": 0,
            "mae": np.nan,
            "r2": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
        }

    target = predictions["target"]
    prediction = predictions["prediction"]
    residual = target - prediction
    denominator = float(((target - target.mean()) ** 2).sum())

    metrics = {
        "status": "ok",
        "n_rows": int(len(predictions)),
        "n_dates": int(predictions["trade_date"].nunique()) if "trade_date" in predictions.columns else int(len(predictions)),
        "mae": float(np.abs(residual).mean()),
        "r2": float(1.0 - (residual.pow(2).sum() / denominator)) if denominator > 0.0 else np.nan,
        "pearson": float(target.corr(prediction, method="pearson")) if len(predictions) > 1 else np.nan,
        "spearman": float(target.corr(prediction, method="spearman")) if len(predictions) > 1 else np.nan,
    }
    return metrics


def _summarize_coefficient_stability(coefficients: pd.DataFrame) -> dict:
    if coefficients.empty:
        return {"status": "skipped", "features": {}, "mean_sign_consistency": np.nan}

    features: dict[str, dict] = {}
    sign_consistencies: list[float] = []

    for feature, feature_df in coefficients.groupby("feature", sort=True):
        values = feature_df["coefficient"]
        positive_share = float((values > 0).mean())
        negative_share = float((values < 0).mean())
        zero_share = float((values == 0).mean())
        sign_consistency = max(positive_share, negative_share, zero_share)
        sign_consistencies.append(sign_consistency)
        features[feature] = {
            "mean_coefficient": float(values.mean()),
            "median_coefficient": float(values.median()),
            "std_coefficient": float(values.std(ddof=0)),
            "sign_consistency": sign_consistency,
        }

    return {
        "status": "ok",
        "features": features,
        "mean_sign_consistency": float(np.mean(sign_consistencies)),
    }


def _write_json(output_path: Path, payload: dict) -> None:
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _prepare_experiment_frame(
    table: pd.DataFrame,
    model_spec: ExperimentSpec,
    target_column: str,
) -> pd.DataFrame:
    frame = table.copy()
    if "session_name" in frame.columns:
        frame = frame.loc[frame["session_name"] == model_spec.session_name].copy()
    if "feature_status" in frame.columns:
        frame = frame.loc[frame["feature_status"] == "ok"].copy()
    if "target_status" in frame.columns:
        frame = frame.loc[frame["target_status"] == "ok"].copy()

    required_columns = list(model_spec.feature_columns) + [target_column]
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame.dropna(subset=required_columns).sort_values("trade_date").reset_index(drop=True)
    return frame


def _build_skip_result(experiment_dir: Path, manifest: dict, reason: str) -> dict:
    metrics = {
        "status": "skipped",
        "reason": reason,
        "n_rows": 0,
        "n_dates": 0,
        "mae": np.nan,
        "r2": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
    }
    write_experiment_artifacts(
        experiment_dir=experiment_dir,
        manifest=manifest,
        predictions=pd.DataFrame(),
        coefficients=pd.DataFrame(columns=["feature", "coefficient"]),
        metrics=metrics,
    )
    _write_json(experiment_dir / "holdout_metrics.json", metrics)
    _write_json(experiment_dir / "stability_summary.json", {"status": "skipped", "reason": reason, "features": {}})
    _write_json(experiment_dir / "final_refit_coefficients.json", {})
    pd.DataFrame().to_parquet(experiment_dir / "holdout_predictions.parquet", index=False)
    pd.DataFrame().to_parquet(experiment_dir / "fold_metrics.parquet", index=False)
    return {
        "status": "skipped",
        "experiment_dir": experiment_dir,
        "metrics": metrics,
        "holdout_metrics": metrics,
    }


def run_walkforward_experiment(
    table: pd.DataFrame,
    model_spec: ExperimentSpec,
    output_root: str | Path,
    target_column: str = "descriptive_target",
    holdout_fraction: float = 0.15,
    train_months: int = 24,
    valid_months: int = 3,
    step_months: int = 3,
) -> dict:
    experiment_dir = Path(output_root) / model_spec.session_name / model_spec.experiment_id
    prepared = _prepare_experiment_frame(table=table, model_spec=model_spec, target_column=target_column)

    manifest = {
        "experiment_id": model_spec.experiment_id,
        "experiment_group": model_spec.experiment_group,
        "session_name": model_spec.session_name,
        "era_name": model_spec.era_name,
        "feature_set_name": model_spec.feature_set_name,
        "feature_columns": list(model_spec.feature_columns),
        "model_name": model_spec.model_name,
        "alpha": model_spec.alpha,
        "l1_ratio": model_spec.l1_ratio,
        "target_column": target_column,
        "holdout_fraction": holdout_fraction,
        "train_months": train_months,
        "valid_months": valid_months,
        "step_months": step_months,
        "n_rows_after_filter": int(len(prepared)),
        "n_dates_after_filter": int(prepared["trade_date"].nunique()) if not prepared.empty else 0,
    }

    if prepared.empty:
        return _build_skip_result(experiment_dir=experiment_dir, manifest=manifest, reason="no_rows_after_filter")

    development_dates, holdout_dates = reserve_final_holdout(prepared["trade_date"], holdout_fraction=holdout_fraction)
    if development_dates.empty or holdout_dates.empty:
        return _build_skip_result(experiment_dir=experiment_dir, manifest=manifest, reason="insufficient_dates_for_holdout")

    development = prepared.loc[prepared["trade_date"].isin(development_dates)].copy()
    holdout = prepared.loc[prepared["trade_date"].isin(holdout_dates)].copy()
    folds = generate_walkforward_folds(
        trade_dates=development_dates,
        train_months=train_months,
        valid_months=valid_months,
        step_months=step_months,
    )

    manifest["development_start"] = development_dates.min()
    manifest["development_end"] = development_dates.max()
    manifest["holdout_start"] = holdout_dates.min()
    manifest["holdout_end"] = holdout_dates.max()
    manifest["n_folds"] = len(folds)

    if not folds:
        return _build_skip_result(experiment_dir=experiment_dir, manifest=manifest, reason="no_walkforward_folds")

    prediction_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict] = []
    fold_metric_rows: list[dict] = []

    for fold in folds:
        train_df = development.loc[development["trade_date"].isin(fold.train_dates)].copy()
        valid_df = development.loc[development["trade_date"].isin(fold.validation_dates)].copy()
        fold_result = fit_fold(
            train_df=train_df,
            valid_df=valid_df,
            feature_columns=list(model_spec.feature_columns),
            target_column=target_column,
            model_spec=model_spec,
        )

        fold_predictions = fold_result["predictions"].copy()
        fold_predictions["fold_id"] = fold.fold_id
        prediction_frames.append(fold_predictions)

        for feature, coefficient in fold_result["coefficients"].items():
            coefficient_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "feature": feature,
                    "coefficient": coefficient,
                }
            )

        fold_metrics = _compute_regression_metrics(fold_predictions)
        fold_metrics.update(
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
            }
        )
        fold_metric_rows.append(fold_metrics)

    oos_predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficient_frame = pd.DataFrame(coefficient_rows)
    fold_metrics_frame = pd.DataFrame(fold_metric_rows)
    metrics = _compute_regression_metrics(oos_predictions)
    metrics["n_folds"] = int(len(folds))

    stability_summary = _summarize_coefficient_stability(coefficient_frame)

    final_fit = fit_fold(
        train_df=development,
        valid_df=holdout,
        feature_columns=list(model_spec.feature_columns),
        target_column=target_column,
        model_spec=model_spec,
    )
    holdout_predictions = final_fit["predictions"].copy()
    holdout_metrics = _compute_regression_metrics(holdout_predictions)

    write_experiment_artifacts(
        experiment_dir=experiment_dir,
        manifest=manifest,
        predictions=oos_predictions,
        coefficients=coefficient_frame,
        metrics=metrics,
    )
    fold_metrics_frame.to_parquet(experiment_dir / "fold_metrics.parquet", index=False)
    holdout_predictions.to_parquet(experiment_dir / "holdout_predictions.parquet", index=False)
    _write_json(experiment_dir / "holdout_metrics.json", holdout_metrics)
    _write_json(experiment_dir / "stability_summary.json", stability_summary)
    _write_json(experiment_dir / "final_refit_coefficients.json", final_fit["coefficients"])

    return {
        "status": "ok",
        "experiment_dir": experiment_dir,
        "metrics": metrics,
        "holdout_metrics": holdout_metrics,
        "stability_summary": stability_summary,
    }


def summarize_experiments(experiments_dir: str | Path) -> pd.DataFrame:
    experiments_path = Path(experiments_dir)
    rows: list[dict] = []

    for experiment_dir in sorted(path for path in experiments_path.iterdir() if path.is_dir()):
        manifest_path = experiment_dir / "manifest.json"
        metrics_path = experiment_dir / "metrics.json"
        holdout_metrics_path = experiment_dir / "holdout_metrics.json"
        stability_path = experiment_dir / "stability_summary.json"

        if not manifest_path.exists() or not metrics_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        holdout_metrics = json.loads(holdout_metrics_path.read_text()) if holdout_metrics_path.exists() else {}
        stability = json.loads(stability_path.read_text()) if stability_path.exists() else {}

        rows.append(
            {
                "experiment_id": manifest.get("experiment_id", experiment_dir.name),
                "session_name": manifest.get("session_name"),
                "era_name": manifest.get("era_name"),
                "feature_set_name": manifest.get("feature_set_name"),
                "model_name": manifest.get("model_name"),
                "status": metrics.get("status", "unknown"),
                "oos_mae": metrics.get("mae"),
                "oos_r2": metrics.get("r2"),
                "oos_pearson": metrics.get("pearson"),
                "oos_spearman": metrics.get("spearman"),
                "holdout_mae": holdout_metrics.get("mae"),
                "holdout_r2": holdout_metrics.get("r2"),
                "holdout_pearson": holdout_metrics.get("pearson"),
                "holdout_spearman": holdout_metrics.get("spearman"),
                "mean_sign_consistency": stability.get("mean_sign_consistency"),
            }
        )

    return pd.DataFrame(rows).sort_values("experiment_id").reset_index(drop=True) if rows else pd.DataFrame()
