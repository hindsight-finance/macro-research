---
type: concept
tags: [feature/vwap, feature/vwap/anchor-0930, feature/vwap/anchor-1300, feature/vwap/anchor-1500, feature/vwap/anchor-1550, feature/vwap/anchor-1555]
---
# Anchored VWAP

Volume-weighted average price anchored at a session point:
`sum((price_ticks/4.0) * size) / sum(size)` over ticks from the anchor to a checkpoint.
Price's **side** relative to the VWAP is `touch` when `abs(price - vwap) <= 0.25`, else
`above` / `below`.

Premacro anchors: `rth_0930`, `pm_1300`, `h3pm_1500` (measured <15:50). Intramacro anchors at
15:50 / 15:55 checkpoints. Confluence = `above_count − below_count` across checks.

**Related.** [[tick-data]] · [[barrier-context]] · [[macro-window]]
**Used by.** [[0001-macro-vwap-features]] · [[macro-vwap-barrier-context]]
