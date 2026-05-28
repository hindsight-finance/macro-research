---
type: concept
tags: [infra/time]
---
# Time handling

The project keeps **UTC internally** (`datetime_utc`) and derives ET on demand via
`utils.minute_bars` (`MARKET_TZ = "America/New_York"`), so DST is handled correctly. Use
`build_market_time_columns` / `derive_session_window` rather than hand-rolling tz math.

Legacy ET inputs with ambiguous DST-fallback timestamps are rejected by
`normalize_minute_bars`; prefer `datetime_utc` inputs.

**Related.** [[session-windows]] · [[data-pipeline]] · [[minute-base-utc-pipeline-refactor]]
