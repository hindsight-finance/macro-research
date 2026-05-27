---
type: concept
tags: [window, window/macro, window/h3pm, window/post]
---
# Session windows

The fixed time-window and session vocabulary used across the workspace, derived in
`utils.minute_bars` (`build_market_time_columns`, `derive_session_window`).

- **Macro-relative windows:** `H3PM` 15:00–15:49 · `MACRO` 15:50–15:59 · `POST` 16:00–16:10 ET.
- **Sessions:** `ASIA`, `LONDON`, `NYAM`, `LUNCH`, `PM`, `OTHER`.

All labels are derived from UTC on demand — see [[time-handling]].

**Related.** [[macro-window]] · [[time-handling]] · [[macro-outcome]]
