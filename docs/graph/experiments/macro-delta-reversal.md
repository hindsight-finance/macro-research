---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/close-1559, feature/volume-delta, feature/volume-delta/imbalance-pre59, feature/volume-delta/bucket-5s]
predictors: [eth_rth_pre59, eth_rth_macro_pre59, rth_macro_pre59, eth_rth_pre_35940]
targets: [k359, k359_00_29, k359_30_59, k359_45_59, k359_50_59]
bucket_size: 5s
sample_n: 841
---
# Macro delta reversal

**Question.** Does accumulated [[volume-delta]] before the 15:59 ET close candle predict
opposite-signed delta into the 15:59 close flow? (Includes a seconds-level extension.)

**Scope.** NQ, 5-second buckets; study shape (841, 211) after the seconds extension. Flow only.

**Headline.** Full-minute 15:59 has a weak reversal tilt; the effect is **concentrated in the
final 15/10 seconds** (15:59:50–59 opposite-rate ~62%, corr ≈ −0.25), while 15:59:00–29 is
noise. Cleanest when accumulated ETH+RTH+15:50–15:58 imbalance is strongly positive (decile 9/10).
Adding 15:59:00–39 into the predictor does not help.

**Concepts.** [[macro-window]] · [[macro-close-1559]] · [[volume-delta]] · [[cumulative-delta-imbalance]]

**Artifacts.** [[2026-05-13-macro-delta-reversal-design]] · [[2026-05-13-macro-delta-reversal-seconds-design]] · [[2026-05-13-macro-delta-reversal]] · [[2026-05-13-macro-delta-reversal-seconds]] · [[2026-05-13-macro-delta-reversal-findings]]

**Related.** [[macro-1550-delta-impulse]] · [[macro-bucket-path]]
