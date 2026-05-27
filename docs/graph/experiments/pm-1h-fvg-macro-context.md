---
type: experiment
status: Exploratory
tags:
  - asset/nq
  - window/macro
  - window/pm
  - feature/fvg
predictors: "no-leak 1H FVG from h12 (12:00-13:00) & h14 (14:00-15:00); 13:00-15:00 imbalance from h13+h14"
prior_windows: "h12 12:00-13:00, h13 13:00-14:00, h14 14:00-15:00 ET; 15:00-16:00 excluded"
targets: "macro_dir_sign over 15:50-15:59"
bins: "fvg_direction, imbalance_direction, fvg_x_imbalance, fvg_size_bucket (none/small/medium/large)"
---
# PM 1H FVG Macro Context

**Question.** Does no-leak 1H context before 15:00 ET (a 12:00-15:00 FVG setup plus 13:00-15:00 imbalance) line up with the 15:50-15:59 ET closing macro direction?

**Scope.** Descriptive, no-ML, one row per day. Inputs: `outputs/nq_1m.parquet` and `outputs/nq_macro_outcomes.parquet`; ET derived from UTC via `utils.minute_bars.MARKET_TZ`. No-leak 1H FVG built from h12 (12:00-13:00) and h14 (14:00-15:00); 13:00-15:00 imbalance from h13+h14; the 15:00-16:00 candle is excluded to avoid overlapping the macro target. Target = `macro_dir_sign`; cohort summaries by FVG direction, imbalance direction, their cross, and FVG-size bucket.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[macro-window]] · [[session-windows]]

**Artifacts.** [[2026-05-13-pm-1h-fvg-macro-context-design]] · [[2026-05-13-pm-1h-fvg-macro-context]]

**Related.** [[macro-fvg-study]] · [[macro-outcome]]
