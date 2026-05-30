from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import data_sources
from utils.helper import merge_news_daily
from utils.minute_bars import build_market_time_columns, derive_session_window, normalize_minute_bars


RTH_OPEN_MINUTE = 9 * 60 + 30
PRE_MACRO_CLOSE_MINUTE = 15 * 60 + 49
MACRO_WINDOW_MINUTES = 10
LEAKY_REALIZED_MACRO_COLUMNS = {"macro_bar_count", "macro_high", "macro_low", "macro_open", "macro_close"}
RTH_PRE_MACRO_MASK = (pl.col("minute_of_day") >= RTH_OPEN_MINUTE) & (
    pl.col("minute_of_day") <= PRE_MACRO_CLOSE_MINUTE
)


def build_macro_range_table(minute_bars: pl.DataFrame, economic_events: pl.DataFrame) -> pl.DataFrame:
    bars = derive_session_window(build_market_time_columns(normalize_minute_bars(minute_bars))).with_columns(
        trade_date=pl.col("date_et"),
        minute_of_day=pl.col("minute_of_day_et"),
    )

    daily = (
        bars.group_by("trade_date")
        .agg(
            day_open=pl.col("Open").filter(pl.col("minute_of_day") == RTH_OPEN_MINUTE).first(),
            close_1549=pl.col("Close").filter(pl.col("minute_of_day") == PRE_MACRO_CLOSE_MINUTE).first(),
            rth_high=pl.col("High").filter(RTH_PRE_MACRO_MASK).max(),
            rth_low=pl.col("Low").filter(RTH_PRE_MACRO_MASK).min(),
            macro_bar_count=pl.col("minute_of_day").filter(pl.col("window") == "MACRO").n_unique(),
            macro_high=pl.col("High").filter(pl.col("window") == "MACRO").max(),
            macro_low=pl.col("Low").filter(pl.col("window") == "MACRO").min(),
            macro_open=pl.col("Open").filter(pl.col("window") == "MACRO").first(),
            macro_close=pl.col("Close").filter(pl.col("window") == "MACRO").last(),
        )
        .filter(
            (pl.col("macro_bar_count") == MACRO_WINDOW_MINUTES)
            & pl.col("day_open").is_not_null()
            & pl.col("close_1549").is_not_null()
            & pl.col("rth_high").is_not_null()
            & pl.col("rth_low").is_not_null()
            & pl.col("macro_high").is_not_null()
            & pl.col("macro_low").is_not_null()
        )
        .with_columns(
            macro_range_pct=(pl.col("macro_high") - pl.col("macro_low")) / pl.col("close_1549"),
            rth_range_pct=(pl.col("rth_high") - pl.col("rth_low")) / pl.col("close_1549"),
            open_to_1549_return_pct=(pl.col("close_1549") - pl.col("day_open")) / pl.col("day_open"),
            close_at_day_high_pct=(pl.col("rth_high") - pl.col("close_1549")) / pl.col("close_1549"),
            close_at_day_low_pct=(pl.col("close_1549") - pl.col("rth_low")) / pl.col("close_1549"),
        )
        .sort("trade_date")
    )

    calendar = merge_news_daily(daily.select(pl.col("trade_date").alias("date")), economic_events)
    out = daily.join(calendar, left_on="trade_date", right_on="date", how="left")
    return out.drop("date") if "date" in out.columns else out


def add_history_features(daily: pl.DataFrame) -> pl.DataFrame:
    prior_macro_range = pl.col("macro_range_pct").shift(1)
    return daily.sort("trade_date").with_columns(
        macro_range_pct_lag_1=prior_macro_range,
        macro_range_pct_mean_5=prior_macro_range.rolling_mean(window_size=5).round(12),
        macro_range_pct_mean_20=prior_macro_range.rolling_mean(window_size=20).round(12),
        macro_range_pct_std_20=prior_macro_range.rolling_std(window_size=20),
        macro_range_pct_q25_20=prior_macro_range.rolling_quantile(quantile=0.25, window_size=20),
        macro_range_pct_q50_20=prior_macro_range.rolling_quantile(quantile=0.50, window_size=20),
        macro_range_pct_q75_20=prior_macro_range.rolling_quantile(quantile=0.75, window_size=20),
    )


def fit_har_baseline(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    feature_columns = ["macro_range_pct_lag_1", "macro_range_pct_mean_5", "macro_range_pct_mean_20"]
    train_complete = train.select(feature_columns + ["macro_range_pct"]).drop_nulls()
    test_complete = test.select(feature_columns).drop_nulls()
    if train_complete.is_empty() or test_complete.is_empty():
        return np.array([])

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train_complete.select(feature_columns).to_numpy(), train_complete["macro_range_pct"].to_numpy())
    return model.predict(test_complete.select(feature_columns).to_numpy())


FORECAST_SCHEMA = {
    "trade_date": pl.Date,
    "experiment": pl.Utf8,
    "model_name": pl.Utf8,
    "quantile": pl.Float64,
    "prediction": pl.Float64,
    "target_macro_range_pct": pl.Float64,
}

SUMMARY_SCHEMA = {
    "experiment": pl.Utf8,
    "model_name": pl.Utf8,
    "quantile": pl.Float64,
    "n_test_dates": pl.Int64,
    "pinball_loss": pl.Float64,
    "coverage": pl.Float64,
    "mae": pl.Float64,
    "har_mae": pl.Float64,
    "har_rmse": pl.Float64,
}


def _empty_forecasts() -> pl.DataFrame:
    return pl.DataFrame(schema=FORECAST_SCHEMA)


def _empty_summary() -> pl.DataFrame:
    return pl.DataFrame(schema=SUMMARY_SCHEMA)


def _model_feature_columns(daily: pl.DataFrame, target_column: str) -> list[str]:
    columns: list[str] = []
    for column, dtype in zip(daily.columns, daily.dtypes):
        if column in {"trade_date", target_column} or column in LEAKY_REALIZED_MACRO_COLUMNS:
            continue
        if dtype == pl.Boolean or dtype.is_numeric():
            columns.append(column)
    return columns


def _fit_xgboost_quantile_model(
    train_df: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    quantiles: tuple[float, ...],
    *,
    xgb_device: str,
    xgb_tree_method: str,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
):
    if not feature_columns:
        return None

    train_complete = train_df.select(feature_columns + [target_column]).drop_nulls()
    if train_complete.height < 2:
        return None

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for quantile boosting. Install it in the active virtualenv "
            "or run with --boosting-backend none."
        ) from exc

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=list(quantiles),
        tree_method=xgb_tree_method,
        device=xgb_device,
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=7,
    )
    model.fit(train_complete.select(feature_columns).to_numpy(), train_complete[target_column].to_numpy())
    return model


def _predict_xgboost_quantiles(model, test_df: pl.DataFrame, feature_columns: list[str], quantiles: tuple[float, ...]) -> list[dict[str, float]]:
    predictions = np.asarray(model.predict(test_df.select(feature_columns).to_numpy()), dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if predictions.shape[0] == len(quantiles) and predictions.shape[1] == test_df.height:
        predictions = predictions.T
    if predictions.shape[1] != len(quantiles):
        raise ValueError(
            f"Expected XGBoost predictions for {len(quantiles)} quantiles, got shape {predictions.shape}"
        )

    rows: list[dict[str, float]] = []
    for row_idx in range(test_df.height):
        for quantile_idx, quantile in enumerate(quantiles):
            rows.append({"quantile": float(quantile), "prediction": float(predictions[row_idx, quantile_idx])})
    return rows


def _usable_model_feature_columns(train: pl.DataFrame, test: pl.DataFrame, candidates: list[str]) -> list[str]:
    usable: list[str] = []
    for column in candidates:
        if column not in train.columns or column not in test.columns:
            continue
        if test.select(pl.col(column).is_not_null().all()).item() is not True:
            continue
        if train.select(pl.col(column).is_not_null().sum()).item() == 0:
            continue
        usable.append(column)
    return usable


def _refit_block_key_expr(refit_frequency: str) -> pl.Expr:
    if refit_frequency == "daily":
        return pl.col("trade_date").cast(pl.String)
    if refit_frequency == "monthly":
        return pl.col("trade_date").dt.strftime("%Y-%m")
    raise ValueError("refit_frequency must be 'daily' or 'monthly'")


def _rolling_empirical_quantiles(train_target: pl.Series, quantiles: tuple[float, ...]) -> dict[float, float]:
    values = train_target.drop_nulls().to_numpy()
    if values.size == 0:
        return {}
    return {quantile: float(np.quantile(values, quantile)) for quantile in quantiles}


def _pinball_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = actual - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def _summarize_forecasts(forecasts: pl.DataFrame) -> pl.DataFrame:
    if forecasts.is_empty():
        return _empty_summary()

    rows = []
    for key, group in forecasts.group_by(["experiment", "model_name", "quantile"], maintain_order=True):
        experiment, model_name, quantile = key
        actual = group["target_macro_range_pct"].to_numpy()
        prediction = group["prediction"].to_numpy()
        mae = float(np.mean(np.abs(actual - prediction))) if quantile == 0.5 else None
        har_mae = float(np.mean(np.abs(actual - prediction))) if model_name == "har_rv" else None
        har_rmse = float(np.sqrt(np.mean((actual - prediction) ** 2))) if model_name == "har_rv" else None
        rows.append(
            {
                "experiment": experiment,
                "model_name": model_name,
                "quantile": float(quantile),
                "n_test_dates": int(group.height),
                "pinball_loss": _pinball_loss(actual, prediction, float(quantile)),
                "coverage": float(np.mean(actual <= prediction)),
                "mae": mae,
                "har_mae": har_mae,
                "har_rmse": har_rmse,
            }
        )
    return pl.DataFrame(rows, schema=SUMMARY_SCHEMA)


def run_experiment(
    daily: pl.DataFrame,
    experiment_name: str,
    train_window_years: int = 2,
    holdout_fraction: float = 0.20,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
    boosting_backend: str = "xgboost",
    refit_frequency: str = "monthly",
    xgb_device: str = "cuda",
    xgb_tree_method: str = "hist",
    xgb_n_estimators: int = 300,
    xgb_max_depth: int = 3,
    xgb_learning_rate: float = 0.05,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    target_column = "macro_range_pct"
    if boosting_backend not in {"xgboost", "none"}:
        raise ValueError("boosting_backend must be 'xgboost' or 'none'")
    _refit_block_key_expr(refit_frequency)
    if "trade_date" not in daily.columns or target_column not in daily.columns:
        return _empty_forecasts(), _empty_summary()

    candidate_feature_columns = _model_feature_columns(daily, target_column)

    clean = daily.sort("trade_date").drop_nulls(subset=["trade_date", target_column] + candidate_feature_columns)
    if clean.height < 2:
        return _empty_forecasts(), _empty_summary()

    holdout_rows = max(1, int(np.ceil(clean.height * holdout_fraction)))
    holdout_rows = min(holdout_rows, clean.height - 1)
    holdout = clean.tail(holdout_rows)
    rows = []

    for test_row in holdout.iter_rows(named=True):
        test_date = test_row["trade_date"]
        prior = clean.filter(pl.col("trade_date") < test_date)
        if prior.is_empty():
            continue

        window_start = (pd.Timestamp(test_date) - pd.DateOffset(years=train_window_years)).date()
        train = prior.filter(pl.col("trade_date") >= window_start)
        if train.height < 2:
            train = prior
        if train.is_empty():
            continue

        target_value = float(test_row[target_column])
        test_df = pl.DataFrame([test_row], schema=clean.schema)
        feature_columns = candidate_feature_columns

        for quantile, prediction in _rolling_empirical_quantiles(train[target_column], quantiles).items():
            rows.append(
                {
                    "trade_date": test_date,
                    "experiment": experiment_name,
                    "model_name": "rolling_quantiles",
                    "quantile": float(quantile),
                    "prediction": prediction,
                    "target_macro_range_pct": target_value,
                }
            )

        har_prediction = fit_har_baseline(train, test_df)
        if len(har_prediction) == 1:
            rows.append(
                {
                    "trade_date": test_date,
                    "experiment": experiment_name,
                    "model_name": "har_rv",
                    "quantile": 0.5,
                    "prediction": float(har_prediction[0]),
                    "target_macro_range_pct": target_value,
                }
            )


    if boosting_backend == "xgboost":
        holdout_with_blocks = holdout.with_columns(refit_block=_refit_block_key_expr(refit_frequency))
        for block_key in holdout_with_blocks["refit_block"].unique(maintain_order=True).to_list():
            block = holdout_with_blocks.filter(pl.col("refit_block") == block_key).drop("refit_block")
            block_start = block["trade_date"].min()
            prior = clean.filter(pl.col("trade_date") < block_start)
            if prior.is_empty():
                continue
            window_start = (pd.Timestamp(block_start) - pd.DateOffset(years=train_window_years)).date()
            train = prior.filter(pl.col("trade_date") >= window_start)
            if train.height < 2:
                train = prior
            boosting_feature_columns = _usable_model_feature_columns(train, block, candidate_feature_columns)
            model = _fit_xgboost_quantile_model(
                train,
                boosting_feature_columns,
                target_column,
                quantiles,
                xgb_device=xgb_device,
                xgb_tree_method=xgb_tree_method,
                xgb_n_estimators=xgb_n_estimators,
                xgb_max_depth=xgb_max_depth,
                xgb_learning_rate=xgb_learning_rate,
            )
            if model is None:
                continue
            prediction_rows = _predict_xgboost_quantiles(model, block, boosting_feature_columns, quantiles)
            block_dates = block["trade_date"].to_list()
            block_targets = block[target_column].to_list()
            for idx, prediction_row in enumerate(prediction_rows):
                row_idx = idx // len(quantiles)
                rows.append(
                    {
                        "trade_date": block_dates[row_idx],
                        "experiment": experiment_name,
                        "model_name": "quantile_boosting",
                        "quantile": prediction_row["quantile"],
                        "prediction": prediction_row["prediction"],
                        "target_macro_range_pct": float(block_targets[row_idx]),
                    }
                )

    if not rows:
        return _empty_forecasts(), _empty_summary()

    forecasts = pl.DataFrame(rows, schema=FORECAST_SCHEMA).sort(["trade_date", "model_name", "quantile"])
    return forecasts, _summarize_forecasts(forecasts)


def write_macro_range_forecasts(
    minute_path: str | Path = data_sources.minute_nq_url("outputs/nq_1m.parquet"),
    events_path: str | Path = data_sources.econ_events_url(),
    output_dir: str | Path = "outputs",
    holdout_fraction: float = 0.20,
    train_window_years: int = 2,
    boosting_backend: str = "xgboost",
    refit_frequency: str = "monthly",
    xgb_device: str = "cuda",
    xgb_tree_method: str = "hist",
    xgb_n_estimators: int = 300,
    xgb_max_depth: int = 3,
    xgb_learning_rate: float = 0.05,
) -> tuple[Path, Path]:
    minute_bars = pl.read_parquet(
        minute_path,
        storage_options=data_sources.storage_options() if data_sources.is_remote(minute_path) else None,
    )
    economic_events = pl.read_parquet(
        events_path,
        storage_options=data_sources.storage_options() if data_sources.is_remote(events_path) else None,
    )

    daily = add_history_features(build_macro_range_table(minute_bars, economic_events))

    experiment_frames: list[pl.DataFrame] = []
    summary_frames: list[pl.DataFrame] = []
    for experiment_name, slice_start in (
        ("full_history", None),
        ("post_covid", date(2020, 3, 16)),
    ):
        experiment_daily = daily if slice_start is None else daily.filter(pl.col("trade_date") >= slice_start)
        forecasts, summary = run_experiment(
            experiment_daily,
            experiment_name=experiment_name,
            train_window_years=train_window_years,
            holdout_fraction=holdout_fraction,
            boosting_backend=boosting_backend,
            refit_frequency=refit_frequency,
            xgb_device=xgb_device,
            xgb_tree_method=xgb_tree_method,
            xgb_n_estimators=xgb_n_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_learning_rate=xgb_learning_rate,
        )
        experiment_frames.append(forecasts)
        summary_frames.append(summary)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    forecast_out = output_path / "nq_macro_range_forecast.parquet"
    summary_out = output_path / "nq_macro_range_forecast_summary.parquet"

    pl.concat(experiment_frames, how="diagonal_relaxed").write_parquet(forecast_out)
    pl.concat(summary_frames, how="diagonal_relaxed").write_parquet(summary_out)
    return forecast_out, summary_out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Macro range forecasting study")
    parser.add_argument("--minute-path", default=data_sources.minute_nq_url("outputs/nq_1m.parquet"))
    parser.add_argument("--events-path", default=data_sources.econ_events_url())
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--train-window-years", type=int, default=2)
    parser.add_argument("--boosting-backend", choices=["xgboost", "none"], default="xgboost")
    parser.add_argument("--refit-frequency", choices=["daily", "monthly"], default="monthly")
    parser.add_argument("--xgb-device", default="cuda")
    parser.add_argument("--xgb-tree-method", default="hist")
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    args = parser.parse_args(argv)

    write_macro_range_forecasts(
        minute_path=args.minute_path,
        events_path=args.events_path,
        output_dir=args.output_dir,
        holdout_fraction=args.holdout_fraction,
        train_window_years=args.train_window_years,
        boosting_backend=args.boosting_backend,
        refit_frequency=args.refit_frequency,
        xgb_device=args.xgb_device,
        xgb_tree_method=args.xgb_tree_method,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
