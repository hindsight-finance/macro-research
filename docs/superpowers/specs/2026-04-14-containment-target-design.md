# Containment Target Design

## Goal
Add a second descriptive target that scores clean, bounded, two-sided rotational auction behavior without collapsing it into trend or noisy diffusion.

## Scope
This target is independent of the existing trendability target.

- trendability stays as-is
- containment is a second realized-window target
- same feature table and walk-forward harness stay in place
- no new volatility-regime feature in v1

## Definition
High containment means:

- price remains bounded inside a usable realized box
- both sides of the box get used
- net displacement is modest relative to box width
- movement is orderly rather than wasteful/noisy
- brief edge interaction is allowed
- repeated overshoot / unstable breakouts are penalized

This is meant to score clean rotating range, not middle-of-box stagnation.

## Recommended V1 Formula
For one realized OHLC window:

```python
range_ = high.max() - low.min()
close_pos = (close - low.min()) / (range_ + 1e-12)
returns = np.diff(np.log(close))

# 1. low final displacement relative to total box width
displacement_score = 1.0 - np.clip(abs(close[-1] - open_[0]) / (range_ + 1e-12), 0.0, 1.0)

# 2. both upper and lower parts of the box get used
upper_share = np.mean(close_pos >= 2.0 / 3.0)
lower_share = np.mean(close_pos <= 1.0 / 3.0)
edge_balance = 2.0 * min(upper_share, lower_share)
edge_balance = np.clip(edge_balance, 0.0, 1.0)

# 3. closes stay inside the realized box interior rather than repeatedly breaking out
inside_share = np.mean((close_pos >= 0.05) & (close_pos <= 0.95))

# 4. noisy path waste should reduce containment
path_length = np.sum(np.abs(np.diff(close)))
path_waste = np.clip(path_length / (range_ + 1e-12), 0.0, 4.0) / 4.0
path_efficiency_penalty = path_waste

containment_target = (
    0.30 * displacement_score
    + 0.30 * edge_balance
    + 0.25 * inside_share
    + 0.15 * (1.0 - path_efficiency_penalty)
)
containment_target = np.clip(containment_target, 0.0, 1.0)
```

## Interpretation
- high trendability, low containment -> trend
- low trendability, high containment -> clean range / rotational auction
- low trendability, low containment -> diffusion / two-sided chop

## Why This Shape
- `displacement_score` stops directional sessions from scoring high
- `edge_balance` rewards actual rotation, not center-stick behavior
- `inside_share` rewards bounded behavior
- `path_efficiency_penalty` keeps ugly diffusion from looking like healthy range

## Known Limitations
- uses realized full-window bounds, so it is descriptive only
- still a single scalar, so some edge cases will remain mixed
- strong but orderly one-sided pullback sessions may sit between trend and containment
- dedicated diffusion/noise target may still be useful later

## Integration Plan
1. Add `build_containment_target(...)` in `features/trend/modeling/target.py`
2. Add new output columns to the modeling table
3. Reuse the same walk-forward harness with `target_column="containment_target"`
4. Compare trendability and containment jointly, not as replacements

## Success Criteria
- containment target is bounded `0-1`
- clean rotational ranges score above one-sided trends
- noisy two-sided diffusion scores below clean rotational ranges
- same training harness can fit containment without structural changes
