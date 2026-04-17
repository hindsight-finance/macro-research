# Three-Scalar Regime Design

## Goal

Move regime research from a single containment scalar toward a three-scalar descriptive regime system:

- `trend_score`
- `containment_score`
- `chop_score`

This is for historical labeling and backtesting research, not live production inference.

## Why This Change

Current state:

- `descriptive_target` already works as a frozen trendability scalar
- `containment_target` already works as a descriptive containment scalar
- current single containment-focused research still compresses:
  - clean rotational range
  - ugly two-sided chop
  - mixed / transitional sessions

So the next step is not one more single score.

The next step is an explicit three-scalar system where:

- trend stays trend
- containment stays clean bounded rotation
- chop becomes its own descriptive target

## Scope

In scope:

- preserve frozen trendability target
- preserve current containment target
- add a new realized `chop_score`
- add explicit scalar aliases:
  - `trend_score`
  - `containment_score`
  - `chop_score`
- derive 3-way labels later from scalar thresholds in the research runner

Out of scope:

- forcing the 3 scalars to sum to `1`
- redesigning the frozen trendability target
- replacing current containment target
- live / causal constraints

## Scalar Definitions

### 1. `trend_score`

Definition:

- `trend_score = descriptive_target`

Purpose:

- explicit naming for regime research
- preserves backward compatibility with the frozen trendability baseline

### 2. `containment_score`

Definition:

- `containment_score = containment_target`

Purpose:

- explicit naming for regime research
- preserves the current descriptive containment target unchanged

### 3. `chop_score`

Definition:

- realized noisy-two-sided score
- should be high when the path is wasteful, unstable, edge-leaking, and flip-heavy

This is not meant to score clean range.
It is meant to score diffusion / ugly chop.

## Chop Score Formula

For one realized OHLC window:

```python
range_ = high.max() - low.min()
close_pos = (close - low.min()) / (range_ + 1e-12)
returns = np.diff(np.log(close))
nonzero_returns = returns[returns != 0]

# 1. frequent sign flips -> choppy
sign_changes = np.count_nonzero(
    np.sign(nonzero_returns[1:]) != np.sign(nonzero_returns[:-1])
) if nonzero_returns.size > 1 else 0
flip_rate = sign_changes / max(nonzero_returns.size - 1, 1)

# 2. lots of path used relative to realized box -> waste
path_length = np.sum(np.abs(np.diff(close)))
path_waste = np.clip(path_length / (range_ + 1e-12), 0.0, 4.0) / 4.0

# 3. repeated edge/outside behavior -> unstable containment
outside_share = np.mean((close_pos < 0.05) | (close_pos > 0.95))

# 4. unstable subwindow range behavior -> disorder
block_ranges = [
    (block_high.max() - block_low.min()) / (range_ + 1e-12)
    for each non-empty block in np.array_split(np.arange(len(close)), min(4, len(close)))
]
instability = np.clip(np.std(block_ranges, ddof=0), 0.0, 1.0)

chop_score = (
    0.35 * path_waste
    + 0.30 * flip_rate
    + 0.20 * outside_share
    + 0.15 * instability
)
chop_score = np.clip(chop_score, 0.0, 1.0)
```

## New Output Columns

Add:

- `trend_score`
- `containment_score`
- `chop_flip_rate`
- `chop_path_waste`
- `chop_outside_share`
- `chop_instability`
- `chop_score`
- `chop_status`

Keep:

- `descriptive_target`
- `containment_target`
- current containment-v2 and v3 feature columns

## Interpretation

- high `trend_score`, low `containment_score`, low `chop_score`
  - clean trend
- low `trend_score`, high `containment_score`, low-mid `chop_score`
  - clean bounded range / rotational auction
- low `trend_score`, low `containment_score`, high `chop_score`
  - ugly two-sided chop / diffusion
- mixed scores
  - transitional or ambiguous sessions

This ambiguity is acceptable and desirable for research.

## Why Independent Scalars

Do not force scores to sum to `1`.

Reason:

- mixed sessions should remain mixed
- ambiguity should stay visible
- no fake certainty
- later state normalization / Markov handoff can be derived separately if needed

## Integration Shape

- add `build_chop_target(...)` in `features/trend/modeling/target.py`
- wire scalar aliases + chop outputs into `features/trend/modeling/table.py`
- keep walk-forward harness reusable via:
  - `target_column="trend_score"`
  - `target_column="containment_score"`
  - `target_column="chop_score"`

Do not add a permanent 3-way label column to the modeling table in this round.
Derive 3-way labels in the research runner from scalar thresholds.

## Research Evaluation Path

### Scalar evaluation

Run three scalar regressions separately:

- `trend_score`
- `containment_score`
- `chop_score`

Metrics:

- holdout `R²`
- holdout `Spearman`
- decile monotonicity / bucket spread where useful

### 3-way label evaluation

Derive labels from scalar thresholds in research code, not the table.

Initial threshold logic:

- `trend` if high `trend_score` and low `containment_score` and low `chop_score`
- `containment` if high `containment_score` and low `trend_score` and low-mid `chop_score`
- `chop` if high `chop_score` and low `trend_score` and low `containment_score`

Metrics:

- macro-F1
- balanced accuracy
- confusion matrix
- per-class precision / recall

## Testing Plan

Test-first:

- direct unit tests for `build_chop_target(...)`
- validate behavior on synthetic windows:
  - clean trend -> low `chop_score`
  - clean rotational range -> lower `chop_score` than ugly chop
  - ugly chop -> high `chop_score`
- table tests require scalar aliases + chop columns

Verification:

- `python3 -m pytest features/trend/modeling/test -q`
- `python3 -m pytest features/trend -q`

## Success Criteria

- all 3 scalars exist in the table
- chop score is bounded `0-1`
- chop score ranks ugly chop above clean rotating range and clean trend
- current frozen trend target remains unchanged
- research runner can use the 3 scalar system for multiclass probing
