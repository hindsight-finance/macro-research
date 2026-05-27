---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/h3pm, feature/outcome]
targets: macro_range_pct
predictors: pre-15:50 same-day context, rolling history, economic calendar flags
bins: 10/25/50/75/90 quantiles
lookback: 2y walk-forward
era_filters: full-history, post-COVID (mid-March 2020)
model: quantile gradient boosting vs rolling empirical quantiles, HAR-RV
---
# Macro Range Forecast

**Question.** Can the normalized 15:50–15:59 ET macro range be forecast distributionally (as quantiles) using only information available up to 15:49 ET?

**Scope.** New module `features/macro_range_forecast.py` builds a daily target `macro_range_pct = (macro_high - macro_low) / close_at_15_49` from canonical minute bars plus the economic-events parquet. Features: same-day pre-15:50 context, rolling history (lagged/rolling quantiles, realized-vol proxies), and calendar flags. Models: rolling empirical quantiles and HAR-RV baselines vs a quantile gradient-boosting model (10/25/50/75/90). Fixed 2-year walk-forward over full-history and post-COVID (~mid-March 2020) experiments; evaluated by pinball loss and coverage. Long-form forecast + summary parquet with an `experiment` column. No raw tick data in v1.

**Headline.** see source artifacts

**Concepts.** [[macro-window]] · [[macro-outcome]]

**Artifacts.** [[2026-05-15-macro-range-forecast-design]] · [[2026-05-15-macro-range-forecast]] · [[2026-05-15-macro-range-forecast-todo]]

**Related.** [[macro-tick-range-context]]
