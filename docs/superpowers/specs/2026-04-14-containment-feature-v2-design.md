# Containment Feature V2 Design

## Goal

Improve containment regime labeling quality by adding a small set of high-value descriptive features that better separate:

- clean bounded rotational auctions
- messy two-sided chop / diffusion
- one-sided trend windows

This work is for historical labeling and backtesting, not live prediction.

## Context

Current containment experiments already learn useful signal from the frozen trendability feature family:

- `ER`, `MSS`, and `DRA` mainly identify `not trend`
- the remaining gap is `clean containment` versus `ugly two-sided chop`

So v2 should add features that capture:

- boundedness
- stability of the auction box
- orderly traversal through value
- symmetry of two-sided rotation

## Scope

In scope:

- add a small containment-focused descriptive feature layer to the modeling table
- keep the current containment target unchanged
- keep the current trendability target unchanged
- keep the current walk-forward harness unchanged
- run quick follow-up experiments only for:
  - `1pm-3pm`
  - `3pm-3:50pm`

Out of scope:

- live-safe / causal feature constraints
- redesign of the frozen trendability schema
- separate diffusion target
- new order-flow or footprint data requirements
- `3:50pm-4pm` experimentation

## Recommended V2 Feature Set

### 1. `containment_overshoot_ratio`

Definition:

- Use the realized session box `[low.min(), high.max()]`
- Measure overshoot pressure as total close-to-box-edge excursion beyond a tighter interior band
- Recommended implementation:
  - `close_pos = (close - low.min()) / (range_ + 1e-12)`
  - define interior as `[0.05, 0.95]`
  - `lower_overshoot = clip(0.05 - close_pos, 0, None)`
  - `upper_overshoot = clip(close_pos - 0.95, 0, None)`
  - `containment_overshoot_ratio = mean(lower_overshoot + upper_overshoot)`

Why:

- directly penalizes unstable breakout / leakage behavior
- complements `containment_inside_share` by measuring magnitude, not just incidence

### 2. `containment_range_stability`

Definition:

- Split the session into equal sequential blocks
- Compute each block's realized range normalized by full-session realized range
- Score stability as inverse dispersion of those block ranges
- Recommended implementation:
  - use `n_blocks = min(4, len(window_bars))`
  - split bars with `np.array_split` and ignore any empty splits
  - `block_range_i = (block_high.max() - block_low.min()) / (full_range + 1e-12)`
  - `dispersion = std(block_range_i, ddof=0)`
  - `containment_range_stability = 1 - clip(dispersion, 0, 1)`

Why:

- clean contained sessions keep a steadier auction envelope
- noisy chop tends to expand / contract erratically

### 3. `containment_mid_cross_count`

Definition:

- Use the realized session midrange:
  - `mid = (high.max() + low.min()) / 2`
- Count sign changes of `(close - mid)` after removing exact zeros
- Normalize by possible maximum crossings
- Recommended implementation:
  - `side = sign(close - mid)`
  - remove zero entries
  - `crosses = count_nonzero(side[1:] != side[:-1])`
  - `containment_mid_cross_count = crosses / max(len(side) - 1, 1)`

Why:

- clean containment should traverse both sides of value
- one-sided trends will have few crossings
- edge-hugging or single-auction sessions will also have fewer crossings

Note:

- this feature alone will not distinguish orderly rotation from frantic noise
- it is intended to work jointly with `range_stability` and `swing_symmetry`

### 4. `containment_swing_symmetry`

Definition:

- Use directional runs of close-to-close changes as simple swings
- Compare positive-run excursion sizes against negative-run excursion sizes
- Score higher when both sides contribute in a more balanced way
- Recommended implementation:
  - compute close-to-close deltas
  - collapse consecutive same-sign deltas into one swing magnitude
  - sum positive swing magnitudes and negative swing magnitudes
  - if either side is absent, return `0.0`
  - `containment_swing_symmetry = 1 - abs(pos_total - neg_total) / (pos_total + neg_total + 1e-12)`

Why:

- clean rotational auction tends to distribute excursion across both directions
- noisy chop can still be two-sided, but often with less balanced swing structure
- trends will collapse toward one-sided dominance

## Integration Shape

Add the new descriptive feature columns to the modeling table:

- `containment_overshoot_ratio`
- `containment_range_stability`
- `containment_mid_cross_count`
- `containment_swing_symmetry`

Do not alter:

- `containment_target`
- `containment_status`
- existing frozen trendability feature columns

## Experiment Plan

Run a quick v2 containment sweep only for:

- `1pm-3pm`
- `3pm-3:50pm`

Add minimal new registry variants that append the four containment-v2 features onto the currently relevant winning representations:

- `adx_parts + containment_v2`
- `core5_dra + containment_v2`

This keeps the comparison narrow and useful.

## Testing Plan

Use the same rigor as prior modeling work.

Test-first requirements:

- add table-level tests that require the four new feature columns
- add direct unit tests for the feature computations on synthetic windows
- verify ranking behavior on synthetic windows:
  - clean trend -> poor containment-v2 feature profile
  - clean rotating range -> strong containment-v2 feature profile
  - noisy two-sided chop -> weaker than clean rotating range on stability / symmetry / overshoot
- extend registry tests if new feature sets are added

Verification after implementation:

- `python3 -m pytest features/trend/modeling/test -q`
- `python3 -m pytest features/trend -q`
- rebuild containment table
- rerun quick containment experiments for `1pm-3pm` and `3pm-3:50pm`

## Success Criteria

- new feature columns are present and bounded where appropriate
- tests remain fully green
- quick rerun shows whether containment classification improves in the two relevant sessions
- implementation remains small and consistent with existing modeling patterns
