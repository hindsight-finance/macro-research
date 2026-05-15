# Macro Range Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast the distribution of NQ 15:50–15:59 ET macro range as a percentage of the 15:49 close, using rolling quantile and HAR-RV baselines plus quantile boosting in full-history and post-COVID walk-forward experiments.

**Architecture:** Build one Polars-first study module that turns canonical minute bars plus the economic-events parquet into a daily modeling table, enriches it with same-day context, rolling history, and calendar flags, then runs a fixed 2-year walk-forward loop. Keep baselines and the main model in the same module so the target table, feature construction, forecasts, and summary metrics all stay aligned. Persist outputs in long-form parquet with an `experiment` column so full-history and post-COVID runs can be compared with the same downstream code.

**Tech Stack:** Polars, NumPy, pandas for date-window arithmetic only, scikit-learn, pytest, Parquet.

---

## File Structure

- Create: `features/macro_range_forecast.py`
  - Build the daily target table from minute bars
  - Merge economic calendar flags
  - Add same-day and rolling history features
  - Run rolling quantile, HAR-RV, and quantile-boosting forecasts
  - Write parquet outputs and expose `main()`
- Create: `test/test_macro_range_forecast.py`
  - Synthetic minute-bar fixtures
  - Calendar merge tests
  - Rolling-window and walk-forward tests
  - Writer / CLI smoke tests

Reuse without rewriting:

- `utils/minute_bars.py` for canonical UTC/ET normalization and macro-window tagging
- `utils/helper.py` for `merge_news_daily()` and event-type flags

---

### Task 1: Build the daily macro target table and same-day context features

**Files:**
- Create `features/macro_range_forecast.py`
- Create `test/test_macro_range_forecast.py`

- [ ] **Step 1: Write the failing test**

```python
import polars as pl

from features.macro_range_forecast import build_macro_range_table


def test_build_macro_range_table_uses_1549_close_and_event_flags():
    minute_bars = pl.DataFrame(
        {
            "datetime_utc": [
                "2020-09-01 19:49:00+00:00",
                "2020-09-01 19:50:00+00:00",
                "2020-09-01 19:59:00+00:00",
                "2020-09-01 20:00:00+00:00",
                "2020-09-02 19:49:00+00:00",
                "2020-09-02 19:50:00+00:00",
                "2020-09-02 19:59:00+00:00",
            ],
            "Open": [100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
            "High": [100.0, 103.0, 105.0, 101.0, 202.0, 201.0, 204.0],
            "Low": [100.0, 99.0, 101.0, 100.5, 199.0, 198.0, 197.0],
            "Close": [100.0, 102.0, 104.0, 100.8, 201.0, 200.5, 203.0],
            "Volume": [1, 2, 3, 4, 5, 6, 7],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))

    economic_events = pl.DataFrame(
        {
            "datetime_utc": ["2020-09-01 14:00:00+00:00"],
            "currency": ["USD"],
            "impact": ["high"],
            "title": ["US ISM Manufacturing PMI"],
            "id": [11],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))

    out = build_macro_range_table(minute_bars, economic_events)

    assert out.height == 2
    assert out.item(0, "trade_date").isoformat() == "2020-09-01"
    assert out.item(0, "close_1549") == 100.0
    assert out.item(0, "macro_high") == 105.0
    assert out.item(0, "macro_low") == 99.0
    assert out.item(0, "macro_range_pct") == 0.06
    assert out.item(0, "has_event_today")
    assert out.item(0, "has_ISM_MANUF_today")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_range_forecast.py::test_build_macro_range_table_uses_1549_close_and_event_flags -v
```

Expected: fail because `features.macro_range_forecast.build_macro_range_table` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from utils.helper import merge_news_daily
from utils.minute_bars import build_market_time_columns, derive_session_window, normalize_minute_bars


def build_macro_range_table(minute_bars: pl.DataFrame, economic_events: pl.DataFrame) -> pl.DataFrame:
    bars = derive_session_window(build_market_time_columns(normalize_minute_bars(minute_bars))).with_columns(
        trade_date=pl.col("date_et"),
        minute_of_day=pl.col("minute_of_day_et"),
    )

    daily = (
        bars.group_by("trade_date")
        .agg(
            day_open=pl.col("Open").filter(pl.col("minute_of_day") == 570).first(),
            close_1549=pl.col("Close").filter(pl.col("minute_of_day") == 949).first(),
            rth_high=pl.col("High").filter(pl.col("minute_of_day") >= 570).filter(pl.col("minute_of_day") <= 949).max(),
            rth_low=pl.col("Low").filter(pl.col("minute_of_day") >= 570).filter(pl.col("minute_of_day") <= 949).min(),
            macro_high=pl.col("High").filter(pl.col("window") == "MACRO").max(),
            macro_low=pl.col("Low").filter(pl.col("window") == "MACRO").min(),
            macro_open=pl.col("Open").filter(pl.col("window") == "MACRO").first(),
            macro_close=pl.col("Close").filter(pl.col("window") == "MACRO").last(),
        )
        .filter(pl.col("close_1549").is_not_null() & pl.col("macro_high").is_not_null() & pl.col("macro_low").is_not_null())
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
    return daily.join(calendar, left_on="trade_date", right_on="date", how="left").drop("date")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_range_forecast.py::test_build_macro_range_table_uses_1549_close_and_event_flags -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_range_forecast.py test/test_macro_range_forecast.py
git commit -m "feat: build macro range target table"
```

---

### Task 2: Add rolling history features and the HAR-RV baseline helpers

**Files:**
- Modify `features/macro_range_forecast.py`
- Modify `test/test_macro_range_forecast.py`

- [ ] **Step 1: Write the failing test**

```python
import polars as pl

from features.macro_range_forecast import add_history_features, fit_har_baseline


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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_range_forecast.py::test_add_history_features_creates_shifted_rolling_columns test/test_macro_range_forecast.py::test_fit_har_baseline_returns_point_forecast -v
```

Expected: fail because the helper functions do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def add_history_features(daily: pl.DataFrame) -> pl.DataFrame:
    return daily.sort("trade_date").with_columns(
        macro_range_pct_lag_1=pl.col("macro_range_pct").shift(1),
        macro_range_pct_mean_5=pl.col("macro_range_pct").shift(1).rolling_mean(5),
        macro_range_pct_mean_20=pl.col("macro_range_pct").shift(1).rolling_mean(20),
        macro_range_pct_std_20=pl.col("macro_range_pct").shift(1).rolling_std(20),
        macro_range_pct_q25_20=pl.col("macro_range_pct").shift(1).rolling_quantile(0.25),
        macro_range_pct_q50_20=pl.col("macro_range_pct").shift(1).rolling_quantile(0.50),
        macro_range_pct_q75_20=pl.col("macro_range_pct").shift(1).rolling_quantile(0.75),
    )


def fit_har_baseline(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    feature_columns = ["macro_range_pct_lag_1", "macro_range_pct_mean_5", "macro_range_pct_mean_20"]
    train_pd = train.select(feature_columns + ["macro_range_pct"]).drop_nulls().to_pandas()
    test_pd = test.select(feature_columns).drop_nulls().to_pandas()

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train_pd[feature_columns], train_pd["macro_range_pct"])
    return model.predict(test_pd[feature_columns])
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_range_forecast.py::test_add_history_features_creates_shifted_rolling_columns test/test_macro_range_forecast.py::test_fit_har_baseline_returns_point_forecast -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_range_forecast.py test/test_macro_range_forecast.py
git commit -m "feat: add macro range history features"
```

---

### Task 3: Implement fixed-window walk-forward quantile boosting and rolling-quantile forecasts

**Files:**
- Modify `features/macro_range_forecast.py`
- Modify `test/test_macro_range_forecast.py`

- [ ] **Step 1: Write the failing test**

```python
import polars as pl

from features.macro_range_forecast import run_experiment


def test_run_experiment_emits_forecasts_and_summary():
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
    )

    assert set(["trade_date", "experiment", "model_name", "quantile", "prediction", "target_macro_range_pct"]).issubset(forecasts.columns)
    assert summary.height > 0
    assert "pinball_loss" in summary.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_range_forecast.py::test_run_experiment_emits_forecasts_and_summary -v
```

Expected: fail because the walk-forward runner does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from sklearn.ensemble import GradientBoostingRegressor


def _fit_quantile_boosting(train_df: pl.DataFrame, feature_columns: list[str], target_column: str, quantiles: tuple[float, ...]) -> dict[float, GradientBoostingRegressor]:
    train_pd = train_df.select(feature_columns + [target_column]).drop_nulls().to_pandas()
    models: dict[float, GradientBoostingRegressor] = {}
    for quantile in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=7,
        )
        model.fit(train_pd[feature_columns], train_pd[target_column])
        models[quantile] = model
    return models


def _rolling_empirical_quantiles(train_target: pl.Series, quantiles: tuple[float, ...]) -> dict[float, float]:
    values = train_target.drop_nulls().to_numpy()
    return {q: float(np.quantile(values, q)) for q in quantiles}
```

Walk-forward rules to encode in the same task:

```python
def run_experiment(
    daily: pl.DataFrame,
    experiment_name: str,
    train_window_years: int = 2,
    holdout_fraction: float = 0.20,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    1. Keep only rows with complete target + feature columns.
    2. Split the final holdout fraction off the end of the experiment slice.
    3. For each holdout date, train on the previous 2 years only.
    4. Emit long-form forecast rows for:
       - rolling_quantiles
       - har_rv
       - quantile_boosting
    5. Return both the daily forecast table and the summary table.
    """
```

The summary table should contain, at minimum:

- `experiment`
- `model_name`
- `quantile`
- `n_test_dates`
- `pinball_loss`
- `coverage`
- `mae` for the median forecast
- `har_mae` / `har_rmse` for the HAR baseline rows

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_range_forecast.py::test_run_experiment_emits_forecasts_and_summary -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_range_forecast.py test/test_macro_range_forecast.py
git commit -m "feat: add macro range walk-forward modeling"
```

---

### Task 4: Add CLI, output writers, and end-to-end smoke coverage for both experiments

**Files:**
- Modify `features/macro_range_forecast.py`
- Modify `test/test_macro_range_forecast.py`

- [ ] **Step 1: Write the failing test**

```python
import polars as pl

from features.macro_range_forecast import write_macro_range_forecasts


def test_write_macro_range_forecasts_creates_output_files(tmp_path):
    minute_path = tmp_path / "nq_minute_base.parquet"
    events_path = tmp_path / "economic_events.parquet"

    minute_bars = pl.DataFrame(
        {
            "datetime_utc": [
                "2020-09-01 19:49:00+00:00",
                "2020-09-01 19:50:00+00:00",
                "2020-09-01 19:59:00+00:00",
                "2020-09-02 19:49:00+00:00",
                "2020-09-02 19:50:00+00:00",
                "2020-09-02 19:59:00+00:00",
            ],
            "Open": [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
            "High": [100.0, 103.0, 105.0, 202.0, 201.0, 204.0],
            "Low": [100.0, 99.0, 101.0, 199.0, 198.0, 197.0],
            "Close": [100.0, 102.0, 104.0, 201.0, 200.5, 203.0],
            "Volume": [1, 2, 3, 5, 6, 7],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))
    minute_bars.write_parquet(minute_path)

    economic_events = pl.DataFrame(
        {
            "datetime_utc": ["2020-09-01 14:00:00+00:00"],
            "currency": ["USD"],
            "impact": ["high"],
            "title": ["US ISM Manufacturing PMI"],
            "id": [11],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))
    economic_events.write_parquet(events_path)

    forecast_path, summary_path = write_macro_range_forecasts(
        minute_path=minute_path,
        events_path=events_path,
        output_dir=tmp_path,
        holdout_fraction=0.2,
        train_window_years=2,
    )

    assert forecast_path.exists()
    assert summary_path.exists()
    assert pl.read_parquet(forecast_path).height > 0
    assert pl.read_parquet(summary_path).height > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_range_forecast.py::test_write_macro_range_forecasts_creates_output_files -v
```

Expected: fail because the writer and CLI entry point do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
def write_macro_range_forecasts(
    minute_path: str | Path = "outputs/nq_minute_base.parquet",
    events_path: str | Path = "input-data/economic_events.parquet",
    output_dir: str | Path = "outputs",
    holdout_fraction: float = 0.20,
    train_window_years: int = 2,
) -> tuple[Path, Path]:
    minute_bars = pl.read_parquet(minute_path)
    economic_events = pl.read_parquet(events_path)

    daily = build_macro_range_table(minute_bars, economic_events)
    daily = add_history_features(daily)

    experiment_frames = []
    summary_frames = []
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
        )
        experiment_frames.append(forecasts)
        summary_frames.append(summary)

    forecast_out = output_dir / "nq_macro_range_forecast.parquet"
    summary_out = output_dir / "nq_macro_range_forecast_summary.parquet"
    pl.concat(experiment_frames, how="diagonal_relaxed").write_parquet(forecast_out)
    pl.concat(summary_frames, how="diagonal_relaxed").write_parquet(summary_out)
    return forecast_out, summary_out
```

CLI entry point to add in the same task:

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Macro range forecasting study")
    parser.add_argument("--minute-path", default="outputs/nq_minute_base.parquet")
    parser.add_argument("--events-path", default="input-data/economic_events.parquet")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--train-window-years", type=int, default=2)
    args = parser.parse_args(argv)
    write_macro_range_forecasts(
        minute_path=args.minute_path,
        events_path=args.events_path,
        output_dir=args.output_dir,
        holdout_fraction=args.holdout_fraction,
        train_window_years=args.train_window_years,
    )
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_range_forecast.py::test_write_macro_range_forecasts_creates_output_files -v
```

Expected: PASS. Then run the module once end-to-end:

```bash
.venv/bin/python -m features.macro_range_forecast
```

Expected: `outputs/nq_macro_range_forecast.parquet` and `outputs/nq_macro_range_forecast_summary.parquet` are written.

- [ ] **Step 5: Commit**

```bash
git add features/macro_range_forecast.py test/test_macro_range_forecast.py
git commit -m "feat: add macro range forecast writer"
```

---

## Coverage Check

This plan covers every requirement from the approved spec:

- normalized target as macro range % of the 15:49 close
- same-day pre-15:50 context
- rolling history features
- economic calendar features
- rolling quantile baseline
- HAR-RV baseline
- quantile boosting main model
- fixed 2-year walk-forward window
- full-history and post-COVID experiments
- long-form forecast and summary parquet outputs

The later 15:50–15:54 and 15:55–15:59 follow-ups remain intentionally out of scope.
