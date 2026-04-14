from __future__ import annotations

from pathlib import Path

import pandas as pd

from features.trend.modeling.registry import ExperimentSpec
from features.trend.modeling.walkforward import (
    fit_fold,
    generate_walkforward_folds,
    reserve_final_holdout,
    run_walkforward_experiment,
    summarize_experiments,
    write_experiment_artifacts,
)


def test_reserve_final_holdout_keeps_last_fifteen_percent_of_dates():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")

    development_dates, holdout_dates = reserve_final_holdout(dates, holdout_fraction=0.15)

    assert len(holdout_dates) == 3
    assert holdout_dates.min() > development_dates.max()


def test_generate_walkforward_folds_emits_non_overlapping_validation_windows():
    trade_dates = pd.date_range("2018-01-01", periods=42, freq="MS")

    folds = generate_walkforward_folds(
        trade_dates,
        train_months=24,
        valid_months=3,
        step_months=3,
    )

    assert folds
    assert folds[0].train_end < folds[0].validation_start
    assert all(first.validation_end < second.validation_start for first, second in zip(folds, folds[1:]))


def test_fit_fold_returns_predictions_and_coefficients():
    train_df = pd.DataFrame(
        {
            "mss": [0.1, 0.2, 0.3, 0.4, 0.5],
            "adx_quality": [0.4, 0.45, 0.5, 0.55, 0.6],
            "descriptive_target": [0.2, 0.25, 0.3, 0.35, 0.4],
        }
    )
    valid_df = pd.DataFrame(
        {
            "mss": [0.25, 0.35],
            "adx_quality": [0.48, 0.58],
            "descriptive_target": [0.28, 0.38],
        }
    )
    model_spec = ExperimentSpec(
        experiment_id="EXP03_full_core5",
        experiment_group="representation_sweep",
        session_name="1pm-3pm",
        era_name="full_dev",
        feature_set_name="core5",
        feature_columns=("mss", "adx_quality"),
        model_name="ridge",
        alpha=1.0,
    )

    result = fit_fold(
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=list(model_spec.feature_columns),
        target_column="descriptive_target",
        model_spec=model_spec,
    )

    assert len(result["predictions"]) == len(valid_df)
    assert set(result["coefficients"]) == {"mss", "adx_quality"}
    assert len(result["scaler_mean"]) == 2


def test_write_experiment_artifacts_writes_expected_files(tmp_path: Path):
    experiment_dir = tmp_path / "experiment"
    manifest = {"experiment_id": "EXP03_full_core5"}
    predictions = pd.DataFrame({"prediction": [0.1, 0.2], "target": [0.15, 0.25]})
    coefficients = pd.DataFrame({"feature": ["mss"], "coefficient": [0.5]})
    metrics = {"mae": 0.1}

    write_experiment_artifacts(
        experiment_dir=experiment_dir,
        manifest=manifest,
        predictions=predictions,
        coefficients=coefficients,
        metrics=metrics,
    )

    assert (experiment_dir / "manifest.json").exists()
    assert (experiment_dir / "metrics.json").exists()
    assert (experiment_dir / "oos_predictions.parquet").exists()
    assert (experiment_dir / "fold_coefficients.parquet").exists()


def _make_modeling_table() -> pd.DataFrame:
    trade_dates = pd.date_range("2018-01-01", periods=48, freq="MS")
    idx = pd.Series(range(len(trade_dates)), dtype=float)

    return pd.DataFrame(
        {
            "instrument": "NQ",
            "trade_date": trade_dates.date,
            "session_name": "1pm-3pm",
            "mss": 0.2 + 0.01 * idx,
            "adx_quality": 0.3 + 0.015 * idx,
            "adx_strength": 0.4 + 0.010 * idx,
            "adx_persistence": 0.45 + 0.005 * idx,
            "adx_crossover": 0.2 - 0.002 * idx,
            "irr": 0.6 - 0.008 * idx,
            "er": 0.25 + 0.012 * idx,
            "log_vr": -0.2 + 0.01 * idx,
            "dra": 0.5 + 0.001 * idx,
            "descriptive_target": 0.15 + 0.02 * idx,
            "feature_status": "ok",
            "target_status": "ok",
        }
    )


def test_run_walkforward_experiment_writes_holdout_outputs(tmp_path: Path):
    table = _make_modeling_table()
    model_spec = ExperimentSpec(
        experiment_id="EXP03_full_core5",
        experiment_group="representation_sweep",
        session_name="1pm-3pm",
        era_name="full_dev",
        feature_set_name="core5",
        feature_columns=("mss", "adx_quality", "irr", "er", "log_vr"),
        model_name="ridge",
        alpha=1.0,
    )

    result = run_walkforward_experiment(
        table=table,
        model_spec=model_spec,
        output_root=tmp_path,
    )

    experiment_dir = result["experiment_dir"]

    assert result["status"] == "ok"
    assert result["metrics"]["n_rows"] > 0
    assert result["holdout_metrics"]["n_rows"] > 0
    assert (experiment_dir / "holdout_predictions.parquet").exists()
    assert (experiment_dir / "holdout_metrics.json").exists()
    assert (experiment_dir / "stability_summary.json").exists()
    assert (experiment_dir / "final_refit_coefficients.json").exists()


def test_summarize_experiments_collects_experiment_metrics(tmp_path: Path):
    table = _make_modeling_table()
    model_spec = ExperimentSpec(
        experiment_id="EXP03_full_core5",
        experiment_group="representation_sweep",
        session_name="1pm-3pm",
        era_name="full_dev",
        feature_set_name="core5",
        feature_columns=("mss", "adx_quality", "irr", "er", "log_vr"),
        model_name="ridge",
        alpha=1.0,
    )
    run_walkforward_experiment(
        table=table,
        model_spec=model_spec,
        output_root=tmp_path,
    )

    summary = summarize_experiments(tmp_path / "1pm-3pm")

    assert len(summary) == 1
    assert summary.loc[0, "experiment_id"] == "EXP03_full_core5"
    assert {"status", "oos_mae", "holdout_mae"} <= set(summary.columns)
