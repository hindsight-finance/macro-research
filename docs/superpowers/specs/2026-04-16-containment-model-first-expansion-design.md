# Containment Model-First Expansion Design

## Goal

Push containment research forward with the highest signal-to-noise next step:

1. formalize model-family experimentation on current `containment_v2` table
2. add a small set of low-duplication containment features
3. rerun only the strongest model families on the expanded table

This remains historical labeling / backtest research, not live production inference.

## Why This Direction

Current state says:

- `containment_v2` feature work improved Ridge only modestly
- larger gains came from model family changes
- current best regression holdout results:
  - `1pm-3pm`: `ridge 0.5198` -> `hist_gbm 0.5591`
  - `3pm-3:50pm`: `ridge 0.4763` -> `extra_trees 0.5167`
- current binary clean-containment tests also show useful lift:
  - top-decile precision `0.7727` in both kept sessions

So next step should not be another blind feature pile-on.

The correct order is:

- first lock the best model families on current features
- then add only a few low-duplication features with strong conceptual support
- then rerun the winning model families

## Scope

In scope:

- keep work inside containment worktree / branch
- use only `1pm-3pm` and `3pm-3:50pm`
- keep existing containment target unchanged
- keep existing trendability target unchanged
- add narrow experimentation support for:
  - continuous containment regression
  - binary clean-containment classification
- add a small feature set:
  - `IB extension / asymmetry`
  - `BandWidth squeeze`
  - `VWAP acceptance`
  - `tail / excess rejection`

Out of scope:

- `3:50pm-4pm`
- 3-way `trend / clean containment / chop` relabeling in this round
- HMM / HSMM implementation in this round
- production / live-safe constraints
- redesign of existing frozen feature schema

## Batch A: Formal Model Bakeoff

Use current `containment_v2` table as-is.

Run regression bakeoff:

- `Ridge`
- `HistGradientBoostingRegressor`
- `RandomForestRegressor`
- `ExtraTreesRegressor`

Run binary clean-containment classification bakeoff:

- label = top quartile of `containment_target` within development sample
- models:
  - `LogisticRegression`
  - `HistGradientBoostingClassifier`
  - `RandomForestClassifier`

Report:

- regression:
  - holdout `R²`
  - holdout `Spearman`
  - `oos R²`
  - `oos Spearman`
- classification:
  - `PR-AUC`
  - `precision@10%`
  - `lift@10%`
  - `MCC`
  - `balanced accuracy`

Purpose:

- establish whether nonlinear model choice is now more valuable than further small linear feature tweaks

## Batch B: Low-Duplication Feature Additions

These are chosen because outside research + current repo overlap analysis both say they add information not already dominated by `ER`, `MSS`, `DRA`, `VR`, `IRR`, and current `containment_v2`.

### 1. `containment_ib_extension_ratio`

Definition:

- build initial balance from first `k` bars of each session
- recommended `k`:
  - `1pm-3pm`: first `15` bars
  - `3pm-3:50pm`: first `10` bars
- `ib_range = ib_high - ib_low`
- `ext_up = max(0, session_high - ib_high) / (ib_range + 1e-12)`
- `ext_dn = max(0, ib_low - session_low) / (ib_range + 1e-12)`
- `containment_ib_extension_ratio = ext_up + ext_dn`

Interpretation:

- low = contained auction staying near early balance
- high = initiative extension

### 2. `containment_ib_asymmetry`

Definition:

- reuse `ext_up` and `ext_dn`
- `containment_ib_asymmetry = abs(ext_up - ext_dn) / (ext_up + ext_dn + 1e-12)`

Interpretation:

- low = symmetric use of both sides
- high = one-sided extension / imbalance

### 3. `containment_bandwidth_squeeze`

Definition:

- compute rolling Bollinger BandWidth over session close series
- use simple implementation:
  - rolling mean over `n = 10`
  - rolling std over `n = 10`
  - `upper = mean + 2 * std`
  - `lower = mean - 2 * std`
  - `bandwidth = (upper - lower) / (abs(mean) + 1e-12)`
  - drop warmup `NaN` rows before aggregation
- feature = inverse average bandwidth:
  - `containment_bandwidth_squeeze = 1 / (1 + mean(bandwidth_valid))`

Interpretation:

- higher = tighter compression
- lower = wider / more expansive path

### 4. `containment_vwap_acceptance`

Definition:

- compute session VWAP from typical price and volume:
  - `tp = (high + low + close) / 3`
  - cumulative `vwap = cumsum(tp * volume) / cumsum(volume)`
- define `session_range = session_high - session_low`
- measure normalized distance of close from VWAP:
  - `dist = abs(close - vwap) / (session_range + 1e-12)`
- `containment_vwap_acceptance = 1 - clip(mean(dist), 0, 1)`

Interpretation:

- higher = price spends more time accepted near fair value
- lower = persistent rejection / migration away from value

### 5. `containment_excess_rejection`

Definition:

- per bar:
  - `upper_tail = (high - max(open, close)) / (high - low + 1e-12)`
  - `lower_tail = (min(open, close) - low) / (high - low + 1e-12)`
- define rejection bars as tails above threshold, recommended `0.4`
- score balanced excess:
  - `upper_reject_share = mean(upper_tail >= 0.4)`
  - `lower_reject_share = mean(lower_tail >= 0.4)`
  - `containment_excess_rejection = 2 * min(upper_reject_share, lower_reject_share)`

Interpretation:

- higher = repeated two-sided rejection at edges
- lower = little excess or one-sided dominance

## Batch C: Rerun Only Winning Model Families

After Batch B:

- rebuild containment table cache with added columns
- rerun only strongest families from Batch A
- expected shortlist:
  - regression: `HistGBM`, `ExtraTrees`, maybe `Ridge` as linear control
  - classification: best of `Logit`, `HistGBM`, `RF`

Do not run another broad kitchen-sink sweep unless results justify it.

## Test Strategy

Same rigor as prior work.

Test-first:

- direct unit tests for new feature computations on synthetic windows
- table tests for new columns
- CLI / registry tests if new experiment-group support is added

Verification:

- `python3 -m pytest features/trend/modeling/test -q`
- `python3 -m pytest features/trend -q`

Experiment verification:

- write fresh artifacts under new containment experiment outputs
- compare against current `containment_v2` baselines, not memory

## Success Criteria

- current model-family results are captured reproducibly
- new features are added with green tests
- at least one kept session improves beyond current best nonlinear baseline
- results clarify whether next bottleneck is:
  - feature set
  - model family
  - target / label design

## Expected Readout

At end of this round we should know:

1. whether nonlinear models remain clearly superior after small feature expansion
2. whether the new OHLCV-native containment features add real incremental signal
3. whether next branch should be:
   - more features
   - 3-way labels
   - regime-state layer (`HMM/HSMM`)
