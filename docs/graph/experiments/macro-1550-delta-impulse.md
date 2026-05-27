---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/open-1550, feature/volume-delta, feature/volume-delta/imbalance-pre350, feature/volume-delta/bucket-5s]
predictors: [eth_only_pre350, rth_only_pre350, eth_rth_pre350]
targets: [k350_00_09]
support_windows: [k350_00_04, k350_05_09, k350_00_29, k350_00_59]
bucket_size: 5s
sample_n: 841
---
# Macro 15:50 delta impulse

**Question.** Does accumulated ETH-only / RTH-only / ETH+RTH [[volume-delta]] before 15:50 ET
predict the initial 15:50:00–15:50:09 opening impulse delta?

**Scope.** NQ, 5-second buckets; study shape (841, 110). No price target — flow only.

**Headline.** ETH-only adds ~no signal. RTH-only and ETH+RTH show a modest **opposite-signed
(reversal)** tendency (opposite-rate ~55%), persisting through the first 30–60s and cleanest in
the strongly-positive imbalance tail. A volume-flow context variable, not a standalone edge.

**Concepts.** [[macro-window]] · [[macro-open-1550]] · [[volume-delta]] · [[cumulative-delta-imbalance]]

**Artifacts.** [[2026-05-13-macro-1550-delta-impulse-design]] · [[2026-05-13-macro-1550-delta-impulse]] · [[2026-05-13-macro-1550-delta-impulse-findings]]

**Related.** [[macro-delta-reversal]] · [[macro-bucket-path]]
