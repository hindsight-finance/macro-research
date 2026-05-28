---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/open-1550, window/macro/close-1559, feature/volume-delta, feature/volume-delta/bucket-5s]
candles: [k350, k359]
relative_buckets: "0-11"
residual_windows: [post_10s_to_30s, post_10s]
conviction: [signed_strong_weak, absolute_deciles]
sample_n: 841
---
# Macro bucket path

**Question.** Does first-10-second [[volume-delta]] conviction inside the 15:50 and 15:59
candles predict continuation, fade, or churn **after** the impulse?

**Scope.** NQ, 5-second buckets normalised to relative buckets 0..11; 841 complete candles per
candle. Key distinction: inclusive metrics include the early predictor; **residual** metrics
(`post_10s`, buckets 2..11) measure follow-through after the first 10s.

**Headline.** First-10s delta is a **strong inclusive anchor** of the full candle at 15:50
(high conviction → 91–96% same-signed full candle) and moderate at 15:59 — but **residual**
follow-through is much weaker (15:50 post-10s ≈ balanced; 15:59 modest continuation, strongest
for strong-negative early flow). Use residual metrics for "rest of candle" claims.

**Concepts.** [[macro-window]] · [[macro-open-1550]] · [[macro-close-1559]] · [[volume-delta]]

**Artifacts.** [[2026-05-14-macro-bucket-path-design]] · [[2026-05-14-macro-bucket-path]] · [[2026-05-14-macro-bucket-path-findings]]

**Related.** [[macro-1550-delta-impulse]] · [[macro-delta-reversal]]
