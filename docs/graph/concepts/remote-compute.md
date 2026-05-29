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

- **Range reads, not downloads.** The time-sorted tick parquet is read in place via
  `pl.scan_parquet` over an `s3://` R2 URL: predicate pushdown prunes row groups and
  fetches only the byte ranges a window filter touches — the 2.8 GB file is never
  downloaded. Selection is env-driven (`utils/data_sources.py`), funnelled through the
  [[tick-data]] chokepoint (`scan_source` / `get_tick_schema` / `open_parquet_file`);
  with env unset, everything falls back to local `input-data/` so local runs are unchanged.
- **Outputs mirror.** The small derived `outputs/` tree (a [[data-pipeline]] stage) is
  `rclone`-synced down before a run and up after, so inter-script paths keep working.
- **Workflows.** `backtest.yml` runs one study on dispatch; `sweep.yml` fans a trend-harness
  ridge-alpha sweep across parallel matrix jobs.

**Operate it:** see [[github-actions-r2]] (bucket layout, one-time upload, secrets, triggers,
range-read verification).

**Related.** [[data-pipeline]] · [[tick-data]] · [[time-handling]]
