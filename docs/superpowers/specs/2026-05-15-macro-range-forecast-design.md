# Macro Range Forecast Design

## Goal

Build a distributional forecasting study for the **15:50–15:59 ET macro range** on NQ.

The primary target is:

`(macro_high - macro_low) / close_at_15:49`

The first version should forecast **quantiles** of that normalized range rather than a single point estimate.

## Scope

This study focuses on predicting the **full macro-window range before 15:50 ET**, using only information available up to **15:49 ET**.

In scope:

- Full-history experiment with a reserved out-of-sample block
- Post-COVID experiment beginning around **mid-March 2020**
- Fixed-window walk-forward evaluation
- Quantile regression / boosting main model
- Rolling quantile and HAR-RV baselines
- Economic calendar features
- Rolling history features

Out of scope for the first version:

- Predicting the intra-macro sub-windows separately
- Predicting directional outcome instead of range distribution
- Tick-level first-minute macro response modeling
- Price-path modeling beyond the range target

## Target Definition

For each trade date, define:

- `macro_range_pct = (macro_high - macro_low) / close_at_15_49`

Where:

- `macro_high` and `macro_low` are from the 15:50–15:59 ET window
- `close_at_15_49` is the 15:49 ET close on the same trade date

The target should be stable across price regimes and suitable for quantile forecasting.

## Experiments

### Experiment A: Full History

- Use the full available macro history
- Reserve the final portion of the sample as out-of-sample
- Use a fixed **2-year** training window in walk-forward evaluation

### Experiment B: Post-COVID

- Start the sample around **mid-March 2020**
- Motivation: structural changes in liquidity dynamics and options flow
- Reserve the final **20%** of the sample as out-of-sample
- Use the same fixed **2-year** training window in walk-forward evaluation

## Modeling Approach

### Baselines

1. **Rolling empirical quantiles**
   - Compute historical quantiles of macro range pct from the training window
   - Use these as a simple benchmark distribution

2. **HAR-RV baseline**
   - Forecast range-like volatility using multi-horizon realized volatility / range features
   - Use a HAR-style structure with daily, weekly, and monthly history
   - Treat this as the main classical benchmark

### Main Model

Use **quantile regression / gradient boosting** as the primary model family.

Predict multiple quantiles, such as:

- 10th
- 25th
- 50th
- 75th
- 90th

The model should learn from:

- same-day pre-15:50 context
- rolling history
- economic calendar flags
- realized-volatility style inputs

## Feature Families

### Same-Day Context

Features available by 15:49 ET only, such as:

- prior session structure
- opening behavior
- intraday trend / compression / expansion state
- pre-macro range position

### Rolling History

Features derived from prior days, such as:

- prior macro ranges
- rolling quantiles of macro range pct
- rolling realized volatility proxies
- range-based estimators over different lookbacks
- prior day/session stats

### Economic Calendar

Include calendar-aware features such as:

- event day flags
- event type
- release timing
- high-impact vs normal-impact classification

## Evaluation

Evaluate each experiment with:

- pinball loss for each quantile
- out-of-sample quantile coverage
- calibration by event-day and non-event-day subsets
- comparison against rolling quantile and HAR-RV baselines

Also inspect:

- how often the predicted bands contain the realized macro range
- whether high-event days are systematically under- or over-estimated
- whether the post-COVID subset behaves differently from the full-history run

## Walk-Forward Design

Use a **fixed 2-year training window**.

For each forecast date:

1. Collect the prior 2 years of eligible training observations
2. Fit the baseline/model on that window
3. Forecast the next day’s quantiles
4. Advance one day and repeat

This keeps the study closer to a live deployment setup and avoids older regimes dominating the forecast indefinitely.

## Outputs

Expected research outputs should live under `outputs/` and include:

- a daily forecast table with realized target and predicted quantiles
- a summary table of calibration and error metrics
- optional per-experiment comparison tables

Suggested naming:

- `outputs/nq_macro_range_forecast.parquet`
- `outputs/nq_macro_range_forecast_summary.parquet`

If the study is split by experiment, add experiment labels rather than separate code paths where practical.

## Future Follow-Ups

Keep these as separate later studies:

- predict **15:50–15:54** range from the first macro minute plus pre-15:50 context
- predict **15:55–15:59** from what happens in **15:50–15:54**

These are intentionally excluded from the first version to keep the target clean and pre-event.

## Non-Goals

- Do not use raw tick data directly in the first version
- Do not forecast returns or direction as the primary objective
- Do not optimize for intraday execution logic
- Do not add visualization work until the forecast table and metrics are stable
