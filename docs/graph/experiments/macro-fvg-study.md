---
type: experiment
status: Exploratory
tags:
  - asset/nq
  - window/macro
  - feature/fvg
  - feature/outcome
candles: "3-bar 1m FVG (bar1/bar2/bar3); assigned_at=bar2"
support_windows: "stage_1 15:50-15:54, stage_2 15:55-15:58"
targets: "retraced/invalidated/held/untouched by 15:59 close"
---
# Macro FVG Close-Window Event Study

**Question.** How do 1-minute fair value gaps created during the close macro window behave (hold vs. invalidate, retrace vs. untouched) before the 15:59 close?

**Scope.** NQ canonical 1-minute bars (`outputs/nq_1m.parquet`); each FVG is one independent event. Detection limited to patterns whose `assigned_at` (bar 2) falls inside the MACRO window 15:50-15:59; no new detection at 15:59; outcome scanning runs from `confirmed_at` to the 15:59 close. Stage 1 = 15:50-15:54, stage 2 = 15:55-15:58.

**Headline.** see source artifacts

**Concepts.** [[fair-value-gap]] · [[macro-window]]

**Artifacts.** [[2026-03-09-macro-fvg-design]] · [[2026-03-09-macro-fvg-study]]

**Related.** [[macro-fvg-minute-volume]] · [[macro-fvg-alignment]] · [[macro-fvg-excursion]] · [[macro-fvg-success-context]]
