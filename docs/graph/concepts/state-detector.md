---
type: concept
tags: [feature/trend, feature/trend/state-detector]
---
# State detector

The component that combines the trend indicators ([[adx]], [[atr-range-ratio]], [[dra]],
[[irr]], [[lag-autocorr-hurst]], [[mss]], [[swing-point-density]], [[efficiency-ratio]],
[[variance-ratio]]) into a single trend/regime state. Implementation:
`features/trend/state_detector.py`; each indicator module tracks its integration readiness.

**Related.** [[trend-regime]] · [[historical-regimes]] · [[trendability]]
**Used by.** [[three-scalar-regime]] · [[trend-modeling-table]]
