---
type: concept
tags: [infra/remote-compute]
runner: ubuntu-latest
runner_ram_gb: 16
data_store: cloudflare-r2
read_mode: range-reads
trigger: workflow_dispatch
---
# Remote compute (GitHub Actions + R2)

Offloads the repo's compute-heavy studies from a RAM-bottlenecked PC onto GitHub-hosted
runners, with source data served from **Cloudflare R2** instead of committed to the repo.

- **Range reads, not downloads.** The year-sharded tick files in the R2 `futures-data`
  lake are read in place via a `pl.scan_parquet` glob over `s3://`: predicate pushdown
  prunes row groups and fetches only the byte ranges a window filter touches — the files
  are never downloaded. Selection is env-driven (`utils/data_sources.py`), funnelled
  through the [[tick-data]] chokepoint (`scan_source` / `get_tick_schema` /
  `iter_tick_batches`), which also synthesizes `price_ticks` from the lake's float `price`
  (lossless on the 0.25 grid). With env unset, everything falls back to local
  `input-data/` so local runs are unchanged.
- **Outputs mirror.** The small derived `outputs/` tree (a [[data-pipeline]] stage) is
  `rclone`-synced down before a run and up after, so inter-script paths keep working.
- **Workflows.** `backtest.yml` runs one study on dispatch; `sweep.yml` fans a trend-harness
  ridge-alpha sweep across parallel matrix jobs.

**Operate it:** see [[github-actions-r2]] (bucket layout, one-time upload, secrets, triggers,
range-read verification).

**Related.** [[data-pipeline]] · [[tick-data]] · [[time-handling]]
