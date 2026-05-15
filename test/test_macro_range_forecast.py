import sys
from types import SimpleNamespace

import numpy as np
import polars as pl

from features.macro_range_forecast import (
    _model_feature_columns,
    add_history_features,
    build_macro_range_table,
    fit_har_baseline,
    run_experiment,
    write_macro_range_forecasts,
)



class FakeXGBRegressor:
    fit_calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.quantiles = kwargs.get("quantile_alpha", [0.5])

    def fit(self, x_train, y_train):
        self.y_mean = float(np.mean(y_train))
        FakeXGBRegressor.fit_calls.append({"kwargs": self.kwargs, "n_train": len(y_train)})
        return self

    def predict(self, x_test):
        quantiles = self.quantiles if isinstance(self.quantiles, (list, tuple)) else [self.quantiles]
        return np.tile(np.array([self.y_mean + float(q) / 100.0 for q in quantiles]), (len(x_test), 1))


def _install_fake_xgboost(monkeypatch):
    FakeXGBRegressor.fit_calls = []
    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=FakeXGBRegressor))

def _fixture_minute_bars(start: str = "2020-09-01", days: int = 2) -> pl.DataFrame:
    rows = []
    for day_idx in range(days):
        # September 2020 is UTC-4 in New York. 13:30 UTC = 09:30 ET.
        date = pl.date(2020, 9, 1 + day_idx)
        date_str = f"2020-09-{1 + day_idx:02d}"
        base = 100.0 + day_idx * 10.0
        rows.append(
            {
                "datetime_utc": f"{date_str} 13:30:00+00:00",
                "Open": base - 1.0,
                "High": base + 0.5,
                "Low": base - 1.5,
                "Close": base,
                "Volume": 10,
            }
        )
        rows.append(
            {
                "datetime_utc": f"{date_str} 19:49:00+00:00",
                "Open": base,
                "High": base + 1.0,
                "Low": base - 1.0,
                "Close": base,
                "Volume": 20,
            }
        )
        for minute in range(50, 60):
            offset = minute - 50
            rows.append(
                {
                    "datetime_utc": f"{date_str} 19:{minute:02d}:00+00:00",
                    "Open": base,
                    "High": base + 3.0 + offset * 0.1,
                    "Low": base - 1.0 - offset * 0.1,
                    "Close": base + 0.5,
                    "Volume": 30 + offset,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))


def _fixture_events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime_utc": ["2020-09-01 14:00:00+00:00"],
            "currency": ["USD"],
            "impact": ["high"],
            "title": ["US ISM Manufacturing PMI"],
            "id": [11],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))


def test_build_macro_range_table_uses_1549_close_and_event_flags():
    minute_bars = _fixture_minute_bars(days=2)
    economic_events = _fixture_events()

    out = build_macro_range_table(minute_bars, economic_events)

    assert out.height == 2
    assert out.item(0, "trade_date").isoformat() == "2020-09-01"
    assert out.item(0, "close_1549") == 100.0
    assert out.item(0, "macro_high") == 103.9
    assert out.item(0, "macro_low") == 98.1
    assert abs(out.item(0, "macro_range_pct") - 0.058) < 1e-12
    assert out.item(0, "has_event_today")
    assert out.item(0, "has_ISM_MANUF_today")


def test_build_macro_range_table_requires_complete_macro_window_and_day_context():
    minute_bars = _fixture_minute_bars(days=1)
    incomplete_macro = minute_bars.filter(~pl.col("datetime_utc").dt.minute().is_in([55]))
    assert build_macro_range_table(incomplete_macro, _fixture_events()).is_empty()

    missing_rth_open = minute_bars.filter(pl.col("datetime_utc").dt.hour() != 13)
    assert build_macro_range_table(missing_rth_open, _fixture_events()).is_empty()


def test_model_feature_columns_excludes_realized_macro_target_columns():
    daily = pl.DataFrame(
        {
            "trade_date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 2), eager=True),
            "macro_range_pct": [0.01, 0.02],
            "macro_bar_count": [10, 10],
            "macro_high": [105.0, 106.0],
            "macro_low": [99.0, 98.0],
            "macro_open": [100.0, 101.0],
            "macro_close": [104.0, 102.0],
            "close_1549": [100.0, 101.0],
            "rth_range_pct": [0.02, 0.03],
            "has_event_today": [False, True],
        }
    )

    features = _model_feature_columns(daily, "macro_range_pct")

    assert "macro_bar_count" not in features
    assert "macro_high" not in features
    assert "macro_low" not in features
    assert "macro_open" not in features
    assert "macro_close" not in features
    assert "close_1549" in features
    assert "rth_range_pct" in features
    assert "has_event_today" in features


def test_add_history_features_creates_shifted_rolling_columns():
    daily = pl.DataFrame(
        {
            "trade_date": pl.date_range(pl.date(2020, 9, 1), pl.date(2020, 9, 10), eager=True),
            "macro_range_pct": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
        }
    )

    out = add_history_features(daily)

    assert out.item(1, "macro_range_pct_lag_1") == 0.01
    assert out.item(5, "macro_range_pct_mean_5") == 0.03
    assert out.item(5, "macro_range_pct_mean_20") is None


def test_fit_har_baseline_returns_point_forecast():
    train = pl.DataFrame(
        {
            "macro_range_pct_lag_1": [0.02, 0.03, 0.04, 0.05, 0.06],
            "macro_range_pct_mean_5": [0.02, 0.025, 0.03, 0.035, 0.04],
            "macro_range_pct_mean_20": [0.02, 0.025, 0.03, 0.035, 0.04],
            "macro_range_pct": [0.021, 0.031, 0.039, 0.051, 0.058],
        }
    )
    test = pl.DataFrame(
        {
            "macro_range_pct_lag_1": [0.06],
            "macro_range_pct_mean_5": [0.05],
            "macro_range_pct_mean_20": [0.04],
        }
    )

    pred = fit_har_baseline(train, test)
    assert len(pred) == 1
    assert pred[0] > 0.0


def test_run_experiment_emits_forecasts_and_summary(monkeypatch):
    _install_fake_xgboost(monkeypatch)
    daily = pl.DataFrame(
        {
            "trade_date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), eager=True),
            "macro_range_pct": [0.01 + (i % 10) * 0.005 for i in range(91)],
            "macro_range_pct_lag_1": [None] + [0.01 + (i % 10) * 0.005 for i in range(90)],
            "macro_range_pct_mean_5": [None] * 4 + [0.02] * 87,
            "macro_range_pct_mean_20": [None] * 19 + [0.03] * 72,
            "has_event_today": [False] * 91,
            "has_CPI_today": [False] * 91,
        }
    )

    forecasts, summary = run_experiment(
        daily,
        experiment_name="full_history",
        train_window_years=2,
        holdout_fraction=0.2,
        quantiles=(0.1, 0.5, 0.9),
        boosting_backend="xgboost",
        xgb_device="cpu",
        refit_frequency="daily",
    )

    assert set(["trade_date", "experiment", "model_name", "quantile", "prediction", "target_macro_range_pct"]).issubset(forecasts.columns)
    assert summary.height > 0
    assert "pinball_loss" in summary.columns


def test_run_experiment_monthly_refits_xgboost_once_per_holdout_month(monkeypatch):
    _install_fake_xgboost(monkeypatch)
    daily = pl.DataFrame(
        {
            "trade_date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 30), eager=True),
            "macro_range_pct": [0.01 + (i % 10) * 0.001 for i in range(121)],
            "macro_range_pct_lag_1": [0.01] * 121,
            "macro_range_pct_mean_5": [0.011] * 121,
            "macro_range_pct_mean_20": [0.012] * 121,
            "rth_range_pct": [0.02] * 121,
        }
    )

    forecasts, _ = run_experiment(
        daily,
        experiment_name="post_covid",
        train_window_years=2,
        holdout_fraction=0.25,
        quantiles=(0.1, 0.5, 0.9),
        boosting_backend="xgboost",
        xgb_device="cuda",
        xgb_n_estimators=17,
        refit_frequency="monthly",
    )

    boosting = forecasts.filter(pl.col("model_name") == "quantile_boosting")
    assert boosting.height > 0
    assert set(boosting["quantile"].unique()) == {0.1, 0.5, 0.9}
    assert len(FakeXGBRegressor.fit_calls) == 2
    assert all(call["kwargs"]["device"] == "cuda" for call in FakeXGBRegressor.fit_calls)
    assert all(call["kwargs"]["objective"] == "reg:quantileerror" for call in FakeXGBRegressor.fit_calls)
    assert all(call["kwargs"]["n_estimators"] == 17 for call in FakeXGBRegressor.fit_calls)


def test_write_macro_range_forecasts_creates_output_files(tmp_path, monkeypatch):
    _install_fake_xgboost(monkeypatch)
    minute_path = tmp_path / "nq_minute_base.parquet"
    events_path = tmp_path / "economic_events.parquet"

    minute_bars = _fixture_minute_bars(days=30)
    minute_bars.write_parquet(minute_path)
    economic_events = _fixture_events()
    economic_events.write_parquet(events_path)

    forecast_path, summary_path = write_macro_range_forecasts(
        minute_path=minute_path,
        events_path=events_path,
        output_dir=tmp_path,
        holdout_fraction=0.2,
        train_window_years=2,
        boosting_backend="xgboost",
        xgb_device="cpu",
        refit_frequency="monthly",
    )

    assert forecast_path.exists()
    assert summary_path.exists()
    forecasts = pl.read_parquet(forecast_path)
    summary = pl.read_parquet(summary_path)
    assert forecasts.height > 0
    assert summary.height > 0
    assert set(forecasts["experiment"]) == {"full_history", "post_covid"}
    assert set(summary["experiment"]) == {"full_history", "post_covid"}
    assert {"rolling_quantiles", "har_rv", "quantile_boosting"}.issubset(set(forecasts["model_name"]))
