# Frozen Trendability Spec

## Purpose

Lock the current trendability model before adding containment/range work.

## Feature Contract

Use one common feature schema across supported sessions:

- `mss`
- `adx_strength`
- `adx_persistence`
- `adx_crossover`
- `irr`
- `er`
- `log_vr`

Do not drop features dynamically by session.

## ADX Definition

`adx_persistence` is now the default margin-weighted persistence metric.

ADX subfeatures are:

- `adx_strength`: normalized ADX/DX strength
- `adx_persistence`: margin-weighted DI dominance persistence
- `adx_crossover`: DI crossover smoothness score

## Target

Trendability target remains the existing descriptive realized-window target:

- `target_strength`
- `target_consistency`
- `target_smoothness`
- `target_retention`
- blended into `descriptive_target`

This target is a trend-vs-non-trend score, not a distinct range/chop classifier.

## Training Contract

- Train separate models per `session_name`
- Keep the same feature schema across sessions
- Standardize features within each training fold
- Default model: Ridge
- Main research era: post-COVID

## Current Interpretation

- `ER` and `MSS` carry most of the trend signal
- ADX subfeatures add useful structure
- `log_vr` is weaker, but remains in the frozen schema for consistency
- `IRR` remains included even though its incremental contribution is modest

## Next Step

Build a second target for containment/range behavior on top of the same modeling table and training harness.
