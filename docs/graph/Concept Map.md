---
type: hub
tags: [moc, index]
---
# Concept Map

Hub for the [[README|concept graph]]. Open `docs/` as an Obsidian vault and press
Ctrl/Cmd-G for the graph view. Layers — durable **concepts** (some with **variants**) and the
**experiments** that use them — wired by `[[wikilinks]]`, with a category→variant→param
hierarchy in tags + frontmatter (see [[README]]).

## Concepts

**Structure & infrastructure**
[[macro-window]] (→ [[macro-open-1550]] · [[macro-close-1559]]) · [[session-windows]] · [[time-handling]] · [[data-pipeline]] · [[tick-data]] · [[macro-outcome]] · [[remote-compute]]

**Microstructure features**
[[volume-delta]] · [[cumulative-delta-imbalance]] · [[anchored-vwap]] · [[fair-value-gap]] · [[tick-density]] · [[barrier-context]] · [[mae-mfe]]

**Trend & regime**
[[trend-regime]] · [[adx]] · [[atr-range-ratio]] · [[dra]] · [[irr]] · [[lag-autocorr-hurst]] · [[mss]] · [[swing-point-density]] · [[efficiency-ratio]] · [[variance-ratio]] · [[containment]] · [[trendability]] · [[state-detector]] · [[historical-regimes]]

## Experiments

Ordered roughly by thread. Status is the node's own; formal logs live in `docs/research_log.md`.

**Macro window — direction, timing, flow**
- [[0001-macro-vwap-features]] — anchored-VWAP context for macro direction *(formal log)*
- [[macro-vwap-barrier-context]] — first-10s barrier vs VWAP vs barrier+VWAP
- [[macro-1550-delta-impulse]] — does pre-15:50 delta predict the 15:50 opening impulse
- [[macro-delta-reversal]] — does pre-close delta predict reversal into the 15:59 close
- [[macro-bucket-path]] — early-10s conviction → continuation / fade / churn
- [[macro-tick-range-context]] — tick-range context around the macro window
- [[macro-range-forecast]] — forecasting macro-window range
- [[macro-regime-direction]] — prior regime/delta context vs macro direction

**Fair-value-gap family**
- [[macro-fvg-study]] · [[macro-fvg-alignment]] · [[macro-fvg-excursion]] · [[macro-fvg-success-context]] · [[macro-fvg-minute-volume]] · [[macro-fvg-volume-delta-dominance]] · [[pm-1h-fvg-macro-context]] · [[fvg-delta-mae-mfe-profiles]] · [[fvg-delta-time-basis]]

**Tick density & volume**
- [[expanded-macro-tick-density]] · [[tick-density-visualization]] · [[one-minute-total-size-bands]] · [[volume-delta-tick]]

**Trend modeling & regime**
- [[trend-modeling-table]] · [[containment-feature-v2]] · [[containment-model-first-expansion]] · [[post-adx-ablation]] · [[three-scalar-regime]] · [[historical-regimes-main-switch]]

**Data infrastructure**
- [[minute-base-utc-pipeline-refactor]] · [[polars-data-system]]

## Browse by tag

The category→variant hierarchy lives in nested tags (see the [[README#Tag taxonomy & params|tag registry]]).
Use the tag pane or color graph groups by family:

- `#feature/vwap` · `#feature/volume-delta` · `#feature/fvg` · `#feature/tick-density` · `#feature/trend`
- `#window/macro` (`/open-1550`, `/close-1559`) · `#window/pm` · `#window/h3pm`
- `#infra/time` · `#infra/pipeline` · `#infra/ticks` · `#infra/remote-compute`

With the Dataview plugin, list a family's studies and their params, e.g.:

```dataview
TABLE predictors, targets, bucket_size, sample_n
FROM #feature/volume-delta AND "graph/experiments"
SORT file.name
```
