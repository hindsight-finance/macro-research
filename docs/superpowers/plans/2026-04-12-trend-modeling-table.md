# Trend Modeling Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a canonical trend-modeling table plus walk-forward training harness that calibrates usable feature weights for the regime filter from realized intraday price action.

**Architecture:** Add a small `features/trend/modeling/` package that does four things separately: build realized-window targets from raw OHLC data, build/cache a feature table from tagged minute bars, define experiment variants, and run grouped walk-forward training with saved artifacts. Keep the feature table as the reusable source of truth so representation changes (`core5`, `+DRA`, `adx_parts`) do not require rebuilding raw bars every time.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, pyarrow, pytest

---

## File Structure

- Create: `features/trend/modeling/__init__.py`
  Expose table builder, target builder, experiment registry, and walk-forward runner.
- Create: `features/trend/modeling/target.py`
  Compute the continuous descriptive regime target from realized OHLC windows.
- Create: `features/trend/modeling/table.py`
  Load tagged minute bars, derive per-session windows, compute features, and write cached parquet tables.
- Create: `features/trend/modeling/registry.py`
  Define session filters, feature sets, era filters, Ridge/Elastic Net parameter grids, and experiment ids.
- Create: `features/trend/modeling/walkforward.py`
  Reserve the final holdout, generate rolling folds, fit models, score OOS rows, and summarize coefficients.
- Create: `features/trend/modeling/cli.py`
  Add CLI entrypoints for `build-table`, `run-experiments`, and `summarize`.
- Create: `features/trend/modeling/test/test_target.py`
  Cover target components and bounded output.
- Create: `features/trend/modeling/test/test_table.py`
  Cover row grain, feature columns, and session extraction.
- Create: `features/trend/modeling/test/test_registry.py`
  Cover experiment definitions and era filters.
- Create: `features/trend/modeling/test/test_walkforward.py`
  Cover holdout reservation, fold generation, and coefficient output shape.
- Write generated artifacts to: `outputs/trend_modeling/`
  - `cache/<instrument>_trend_modeling_table.parquet`
  - `experiments/<session_name>/<experiment_id>/...`

## Canonical Table Contract

### Row Grain

One row per:
- `instrument`
- `trade_date`
- `session_name`

Recommended session names:
- `1pm-3pm`
- `3pm-3:50pm`
- `3:50pm-4pm`

The builder should emit a single long table with all supported session rows, but training should filter to one `session_name` at a time.

### Required Columns

Identity and grouping:
- `instrument`
- `trade_date`
- `session_name`
- `window_start_ts`
- `window_end_ts`
- `n_bars_raw`

Core primary features:
- `mss`
- `adx_quality`
- `irr`
- `er`
- `log_vr`

Optional/context feature:
- `dra`

Diagnostic columns to preserve:
- `adx_strength`
- `adx_persistence`
- `adx_crossover`
- `er_net_change`
- `er_path_length`
- `vr_raw`
- `vr_one_period_variance`
- `vr_multi_period_variance`

Target columns:
- `target_strength`
- `target_consistency`
- `target_smoothness`
- `target_retention`
- `descriptive_target`

Feature health columns:
- `feature_status`
- `target_status`

### Descriptive Target Definition

For a realized OHLC window:

```python
returns = np.diff(np.log(close))
nonzero_returns = returns[returns != 0]

strength_raw = abs(returns.sum()) / (returns.std(ddof=1) * np.sqrt(len(returns)) + 1e-12)
target_strength = 1.0 - np.exp(-strength_raw)

up_share = np.mean(nonzero_returns > 0) if len(nonzero_returns) else 0.5
down_share = np.mean(nonzero_returns < 0) if len(nonzero_returns) else 0.5
target_consistency = 2.0 * max(up_share, down_share) - 1.0

signs = np.sign(nonzero_returns)
sign_change_rate = np.mean(signs[1:] != signs[:-1]) if len(signs) > 1 else 0.0
target_smoothness = 1.0 - sign_change_rate

target_retention = np.clip(
    abs(close[-1] - open_[0]) / (high.max() - low.min() + 1e-12),
    0.0,
    1.0,
)

descriptive_target = (
    0.35 * target_strength
    + 0.25 * target_consistency
    + 0.20 * target_smoothness
    + 0.20 * target_retention
)
```

Do not derive the target from the candidate feature outputs. Build it only from raw realized path behavior.

## Modeling Decisions Locked In

- Use signed coefficients. Do not invert every feature to “higher is better” up front.
- Standardize features inside each training fold only.
- Start with `Ridge` as the default model.
- Run `Elastic Net` only as a confirmatory second pass on the best variants.
- Save both stitched OOS predictions and fold-by-fold coefficients.
- Treat `DRA` as an experiment switch, not as always-on baseline.

## Experiment Matrix

### Representation

- `core5`
  - `mss`
  - `adx_quality`
  - `irr`
  - `er`
  - `log_vr`
- `core5_dra`
  - `mss`
  - `adx_quality`
  - `irr`
  - `er`
  - `log_vr`
  - `dra`
- `adx_parts`
  - `mss`
  - `adx_strength`
  - `adx_persistence`
  - `adx_crossover`
  - `irr`
  - `er`
  - `log_vr`

### Era Filters

- `full_dev`
  Use all dates before the final untouched holdout.
- `pre_covid`
  `trade_date <= 2020-02-28`
- `post_covid`
  `trade_date >= 2020-07-01`

Exclude transition dates:
- `2020-03-01` through `2020-06-30`

### Walk-Forward Structure

- final untouched holdout: last `15%` of dates within the filtered era block
- rolling development folds:
  - train: trailing `24 months`
  - validate: next `3 months`
  - step: `3 months`

### Model Grid

Ridge alpha sweep:

```python
RIDGE_ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
```

Elastic Net confirmatory grid:

```python
ELASTIC_NET_ALPHAS = [0.03, 0.1, 0.3, 1.0]
ELASTIC_NET_L1_RATIOS = [0.1, 0.25, 0.5]
```

## Task 1: Scaffold the Modeling Package

**Files:**
- Create: `features/trend/modeling/__init__.py`
- Create: `features/trend/modeling/test/test_target.py`

- [ ] **Step 1: Write the failing target smoke test**

```python
from features.trend.modeling.target import build_descriptive_target


def test_build_descriptive_target_returns_bounded_components():
    result = build_descriptive_target(
        open_=np.array([100.0, 101.0, 102.0]),
        high=np.array([101.0, 102.0, 103.0]),
        low=np.array([99.5, 100.5, 101.5]),
        close=np.array([100.5, 101.5, 102.5]),
    )

    assert 0.0 <= result["descriptive_target"] <= 1.0
    assert set(result) == {
        "target_strength",
        "target_consistency",
        "target_smoothness",
        "target_retention",
        "descriptive_target",
        "target_status",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py::test_build_descriptive_target_returns_bounded_components -q`

Expected: `ImportError` or missing symbol failure because `features.trend.modeling.target` does not exist yet.

- [ ] **Step 3: Create minimal package exports**

```python
# features/trend/modeling/__init__.py
from .target import build_descriptive_target

__all__ = ["build_descriptive_target"]
```

- [ ] **Step 4: Run the single test again**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py::test_build_descriptive_target_returns_bounded_components -q`

Expected: still fails because `build_descriptive_target` is not implemented yet.

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/__init__.py features/trend/modeling/test/test_target.py
git commit -m "test: scaffold trend modeling target package"
```

## Task 2: Implement the Descriptive Target

**Files:**
- Create: `features/trend/modeling/target.py`
- Modify: `features/trend/modeling/test/test_target.py`

- [ ] **Step 1: Add failing tests for directional and choppy windows**

```python
def test_descriptive_target_scores_clean_trend_above_round_trip_chop():
    trend = build_descriptive_target(...)
    chop = build_descriptive_target(...)

    assert trend["descriptive_target"] > chop["descriptive_target"]


def test_descriptive_target_handles_zero_return_window():
    flat = build_descriptive_target(...)
    assert flat["target_status"] == "ok"
    assert flat["descriptive_target"] == pytest.approx(0.2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`

Expected: failures because the function returns placeholder values or is missing.

- [ ] **Step 3: Write the minimal implementation**

```python
def build_descriptive_target(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict:
    returns = np.diff(np.log(close.astype(float)))
    nonzero_returns = returns[returns != 0]
    # compute strength, consistency, smoothness, retention
    # blend into descriptive_target
    return {
        "target_strength": ...,
        "target_consistency": ...,
        "target_smoothness": ...,
        "target_retention": ...,
        "descriptive_target": ...,
        "target_status": "ok",
    }
```

- [ ] **Step 4: Run target tests**

Run: `python3 -m pytest features/trend/modeling/test/test_target.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/target.py features/trend/modeling/test/test_target.py
git commit -m "feat: add descriptive trend target builder"
```

## Task 3: Build the Canonical Modeling Table

**Files:**
- Create: `features/trend/modeling/table.py`
- Create: `features/trend/modeling/test/test_table.py`

- [ ] **Step 1: Write failing tests for row grain and columns**

```python
from features.trend.modeling.table import build_modeling_table


def test_build_modeling_table_emits_one_row_per_date_and_session(tmp_path):
    bars_path = tmp_path / "bars.parquet"
    sample_bars.to_parquet(bars_path, index=False)

    table = build_modeling_table(
        input_path=bars_path,
        instrument="NQ",
        session_names=["1pm-3pm"],
    )

    assert {"instrument", "trade_date", "session_name", "mss", "adx_quality", "irr", "er", "log_vr"} <= set(table.columns)
    assert table.groupby(["trade_date", "session_name"]).size().eq(1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py::test_build_modeling_table_emits_one_row_per_date_and_session -q`

Expected: fail because the builder does not exist yet.

- [ ] **Step 3: Implement a minimal table builder**

```python
def build_modeling_table(input_path: Path, instrument: str, session_names: list[str]) -> pd.DataFrame:
    bars = pd.read_parquet(input_path)
    normalized = normalize_ohlcv_columns(bars)
    session_windows = extract_session_windows(normalized, session_names=session_names)
    rows = [build_session_row(window_df, instrument=instrument, session_name=session_name) for ...]
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Add cache writer**

```python
def write_modeling_table_cache(table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
```

- [ ] **Step 5: Run table tests**

Run: `python3 -m pytest features/trend/modeling/test/test_table.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add features/trend/modeling/table.py features/trend/modeling/test/test_table.py
git commit -m "feat: add trend modeling table builder"
```

## Task 4: Add the Experiment Registry

**Files:**
- Create: `features/trend/modeling/registry.py`
- Create: `features/trend/modeling/test/test_registry.py`

- [ ] **Step 1: Write failing registry tests**

```python
from features.trend.modeling.registry import build_experiment_registry


def test_build_experiment_registry_contains_core_variants():
    registry = build_experiment_registry(session_name="1pm-3pm")
    ids = {experiment.experiment_id for experiment in registry}

    assert "EXP03_full_core5" in ids
    assert "EXP04_full_core5_dra" in ids
    assert "EXP05_full_adx_parts" in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_registry.py -q`

Expected: fail because registry module does not exist yet.

- [ ] **Step 3: Implement the registry**

```python
@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    session_name: str
    era_name: str
    feature_columns: tuple[str, ...]
    model_name: str
    alpha: float
    l1_ratio: float | None = None
```

```python
def build_experiment_registry(session_name: str, ridge_alpha: float = 1.0) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(...),
        ExperimentSpec(...),
    ]
```

- [ ] **Step 4: Run registry tests**

Run: `python3 -m pytest features/trend/modeling/test/test_registry.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/registry.py features/trend/modeling/test/test_registry.py
git commit -m "feat: add trend modeling experiment registry"
```

## Task 5: Implement Walk-Forward Training

**Files:**
- Create: `features/trend/modeling/walkforward.py`
- Create: `features/trend/modeling/test/test_walkforward.py`

- [ ] **Step 1: Write failing tests for holdout reservation and fold generation**

```python
from features.trend.modeling.walkforward import reserve_final_holdout, generate_walkforward_folds


def test_reserve_final_holdout_keeps_last_fifteen_percent_of_dates():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    dev_dates, holdout_dates = reserve_final_holdout(pd.Series(dates), holdout_fraction=0.15)

    assert len(holdout_dates) == 3
    assert holdout_dates.min() > dev_dates.max()


def test_generate_walkforward_folds_emits_non_overlapping_validation_windows():
    folds = generate_walkforward_folds(...)
    assert folds[0].train_end < folds[0].validation_start
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest features/trend/modeling/test/test_walkforward.py -q`

Expected: fail because walk-forward helpers do not exist yet.

- [ ] **Step 3: Implement fold generation and model fitting**

```python
def fit_fold(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_columns: list[str], target_column: str, model_spec: ExperimentSpec) -> dict:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns])
    x_valid = scaler.transform(valid_df[feature_columns])
    model = Ridge(alpha=model_spec.alpha)
    model.fit(x_train, train_df[target_column].to_numpy())
    predictions = model.predict(x_valid)
    return {
        "predictions": predictions,
        "coefficients": dict(zip(feature_columns, model.coef_)),
        "intercept": float(model.intercept_),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
```

- [ ] **Step 4: Add artifact writers**

```python
def write_experiment_artifacts(experiment_dir: Path, manifest: dict, predictions: pd.DataFrame, coefficients: pd.DataFrame, metrics: dict) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(experiment_dir / "oos_predictions.parquet", index=False)
    coefficients.to_parquet(experiment_dir / "fold_coefficients.parquet", index=False)
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
```

- [ ] **Step 5: Run walk-forward tests**

Run: `python3 -m pytest features/trend/modeling/test/test_walkforward.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add features/trend/modeling/walkforward.py features/trend/modeling/test/test_walkforward.py
git commit -m "feat: add walkforward trend model training"
```

## Task 6: Add a CLI for Table Build and Training

**Files:**
- Create: `features/trend/modeling/cli.py`

- [ ] **Step 1: Write a failing smoke test for CLI argument parsing**

```python
from features.trend.modeling.cli import build_parser


def test_build_parser_accepts_build_table_and_run_experiments():
    parser = build_parser()
    build_args = parser.parse_args(["build-table", "--input-path", "bars.parquet", "--instrument", "NQ"])
    run_args = parser.parse_args(["run-experiments", "--table-path", "table.parquet", "--session-name", "1pm-3pm"])

    assert build_args.command == "build-table"
    assert run_args.command == "run-experiments"
```

- [ ] **Step 2: Run smoke test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_registry.py features/trend/modeling/test/test_walkforward.py -q`

Expected: fail or import error because the CLI does not exist yet.

- [ ] **Step 3: Implement the parser and dispatch**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    # build-table
    # run-experiments
    # summarize
    return parser
```

- [ ] **Step 4: Verify CLI entrypoints manually**

Run:

```bash
python3 -m features.trend.modeling.cli build-table \
  --input-path /mnt/e/backup/code/Finance/research/macro/outputs/nq_1m.parquet \
  --instrument NQ \
  --output-path outputs/trend_modeling/cache/nq_trend_modeling_table.parquet
```

Expected: cache parquet written.

Run:

```bash
python3 -m features.trend.modeling.cli run-experiments \
  --table-path outputs/trend_modeling/cache/nq_trend_modeling_table.parquet \
  --session-name 1pm-3pm \
  --output-dir outputs/trend_modeling/experiments
```

Expected: experiment directories written under `outputs/trend_modeling/experiments/1pm-3pm/`.

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/cli.py
git commit -m "feat: add trend modeling cli"
```

## Task 7: Run the Baseline Experiment Set

**Files:**
- Use: `outputs/trend_modeling/cache/nq_trend_modeling_table.parquet`
- Write: `outputs/trend_modeling/experiments/...`

- [ ] **Step 1: Build the canonical table**

Run:

```bash
python3 -m features.trend.modeling.cli build-table \
  --input-path /mnt/e/backup/code/Finance/research/macro/outputs/nq_1m.parquet \
  --instrument NQ \
  --output-path outputs/trend_modeling/cache/nq_trend_modeling_table.parquet
```

Expected: parquet exists with one row per `trade_date x session_name`.

- [ ] **Step 2: Run Ridge alpha sweeps**

Run:

```bash
python3 -m features.trend.modeling.cli run-experiments \
  --table-path outputs/trend_modeling/cache/nq_trend_modeling_table.parquet \
  --session-name 1pm-3pm \
  --experiment-group ridge_alpha_sweep
```

Expected: alpha sweep summaries written.

- [ ] **Step 3: Run representation sweep**

Run:

```bash
python3 -m features.trend.modeling.cli run-experiments \
  --table-path outputs/trend_modeling/cache/nq_trend_modeling_table.parquet \
  --session-name 1pm-3pm \
  --experiment-group representation_sweep
```

Expected: `core5`, `core5_dra`, and `adx_parts` outputs written.

- [ ] **Step 4: Summarize coefficient stability**

Run:

```bash
python3 -m features.trend.modeling.cli summarize \
  --experiments-dir outputs/trend_modeling/experiments/1pm-3pm
```

Expected: a summary table showing mean coefficient, median coefficient, sign consistency, and OOS metrics for each experiment.

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling docs/superpowers/plans/2026-04-12-trend-modeling-table.md
git commit -m "feat: add trend modeling table and walkforward baseline"
```

## Review Notes for the Human Partner

These decisions are worth confirming before implementation:

1. **Per-session models vs one blended model**
   Recommendation: keep one canonical long table, but train separate models per `session_name`. Mixing `1pm-3pm`, `3pm-3:50pm`, and `3:50pm-4pm` into one fit is likely wrong because the windows have different lengths, ADX configs, and target distributions.

2. **Full realized session rows vs rolling 5-minute snapshots**
   Recommendation: start with full realized session rows. They align with the current feature modules, avoid overlapping-label leakage, and get you usable weights faster. Rolling windows can be a v2.

3. **DRA treatment**
   Recommendation: compute `dra` in the table, but do not make it part of the baseline feature set. Keep it as an experiment toggle until its coefficient stability is proven.

4. **Input path handling**
   Recommendation: require `--input-path` explicitly in the CLI. Ignored `outputs/` files do not appear in fresh worktrees, so hardcoded relative defaults will break in exactly the environment we are using.

5. **Production weight source**
   Recommendation: use post-COVID as the practical production weighting source if it agrees reasonably with the full-sample model on holdout. Use pre-COVID as a diagnostic, not as the default weight source.
