---
type: concept
tags: [feature/barrier, feature/barrier/break-direction]
---
# First-10s break direction (15:50 macro open)

The first side of the first-10-second barrier range `[low_10s, high_10s]` to **break strictly**
after 15:50:10 ET sets a causal directional **bias** — a high break → bullish, a low break →
bearish (first break wins on whipsaw). Unlike [[barrier-context]] (which reads *which side is
touched first* and the wrong-side behaviour of the range), this reads the first **break** as a
directional signal from which forward outcomes are measured. The candidate entry is a **retouch**
of the 15:50-anchored [[anchored-vwap]] (the frozen first-10s VWAP, or the rolling cumulative
VWAP) after the break; outcomes are measured from both the break and each retouch into the
[[macro-close-1559|15:59 close]].

**Variant of.** [[barrier-context]]
**Related.** [[anchored-vwap]] · [[macro-open-1550]] · [[macro-outcome]] · [[mae-mfe]]
**Used by.** [[macro-1550-vwap-retouch]]
