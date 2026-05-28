---
type: concept
tags: [infra/ticks]
---
# Tick data (memory-safe access)

Tick parquet files (`input-data/merged_nq_ticks.parquet`) have hundreds of millions of rows
and must **never be eager-read**. Use `utils.tick_data`: `get_tick_schema()` to validate from
metadata, `scan_tick_data()` for a lazy frame over `TICK_COLUMNS`
(`ts_event, intra_ts_rank, side, price_ticks, size`), and `collect_tick_window()` which
*requires* bounded start/end UTC and collects with `engine="streaming"`.

Encoding: `side` `2`=buy, `1`=sell, `0`=none; price = `price_ticks / 4.0`. Filter to a bounded
window early; use `sink_parquet` for full-output pipelines; preserve empty 5-second buckets.

**Related.** [[data-pipeline]] · [[volume-delta]] · [[tick-density]] · [[anchored-vwap]]
