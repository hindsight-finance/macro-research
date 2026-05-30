# Running research scripts on GitHub Actions (data served from Cloudflare R2)

The repo's compute-heavy studies can run on GitHub-hosted runners instead of a
RAM-bottlenecked local PC. Source data is read in place from the existing Cloudflare R2
**`futures-data` data lake** — the large tick files are never downloaded; polars fetches
only the parquet row groups a time-window filter touches via HTTP range requests.

## Why this works (feasibility)

- **Runner.** A public-repo `ubuntu-latest` runner is 4 vCPU / 16 GB RAM, unlimited minutes.
- **Memory / transfer.** The tick files are time-sorted, so a macro-window read touches a
  handful of row groups. Validated over R2: one day's 15:50–16:00 window across the 6
  year-sharded files = 17.7k rows in ~11 s; the full `macro_extreme_timing` study (all dates)
  ran in ~156 s — neither downloads the files.
- **Cost.** R2 has zero egress and a generous free tier; a run issues at most tens of
  thousands of range reads.

## Architecture — reading the lake in place

```
Cloudflare R2 bucket  futures-data   (the existing data lake; nothing re-uploaded)
├── NQ/tick/2020_merged_nq.parquet … 2025_merged_nq.parquet   ← year-sharded ticks, read via glob
│        schema: ts_event, intra_ts_rank, side, price (float), size
├── NQ/NQ-*.ohlcv-1m.parquet         ← canonical minute base (datetime_utc, OHLCV)
├── ES/ … RTY/ … YM/ … NIY/          ← other instruments (ES used by sweep.yml)
└── macro-research/outputs/          ← derived parquet/figs MIRROR (created by the workflow)
```

- **Schema bridge.** The lake stores a float `price`; the pipeline expects `price_ticks`
  (UInt32). `utils/tick_data.py` synthesizes `price_ticks = round(price*4)` (lossless — NQ
  trades sit on the 0.25 grid), so every study is layout-agnostic. Tick reads funnel through
  `scan_source` / `get_tick_schema` / `iter_tick_batches`; the minute base through
  `utils.minute_bars.load_minute_bars`. All are R2-aware and fall back to local files when the
  env vars below are unset (so **local runs are unchanged**).
- **Env wiring** (`utils/data_sources.py`): `TICK_DATA_URL` (glob), `MINUTE_NQ_URL`,
  `ECON_EVENTS_URL`, and `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_ENDPOINT_URL`
  → polars `storage_options`. The workflows set these from repo secrets.
- **Derived outputs** are `rclone`-synced down (prerequisites) before a run and back up
  (results) after, under the `macro-research/outputs` prefix so they don't clutter the lake.

## Setup status

Already done (this is recorded here for reference):
- **Repo is public** → 16 GB / 4-core runner, unlimited minutes.
- **Secrets set:** `R2_BUCKET` (=`futures-data`), `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
  `R2_ENDPOINT_URL`. Verify with `gh secret list -R hindsight-finance/macro-research`.
- **Data already in the lake** — no upload step.

Outstanding (only if you want these specific studies):
- **`macro_range_forecast`** needs an `economic_events.parquet`, which is **not in the lake**.
  Upload one (e.g. to `s3://futures-data/macro-research/economic_events.parquet`) and add a
  secret/env `ECON_EVENTS_URL` pointing at it before running that target.
- If the R2 credentials get rotated, re-set the two credential secrets (stage values in files
  and `gh secret set <NAME> < file`, never paste them into a shell command or chat).

## Running

**`backtest.yml`** — run any single study. Actions tab → *backtest* → *Run workflow*, pick a
`target`, optionally add `extra_args`. From the CLI:
```bash
gh workflow run backtest.yml -f target=macro_extreme_timing
gh workflow run backtest.yml -f target=macro_outcomes
gh workflow run backtest.yml -f target=tick_density
# macro_range_forecast (only after econ-events is uploaded):
gh workflow run backtest.yml -f target=macro_range_forecast -f extra_args="--xgb-device cpu"
```
Results upload as a `backtest-<target>-<run_id>` artifact and sync to `macro-research/outputs`.

**`sweep.yml`** — parallel ridge-alpha sweep of the trend harness (pulls the instrument's
ohlcv-1m from the lake, builds the table, fans out, summarizes):
```bash
gh workflow run sweep.yml -f instrument=NQ -f session_name=3:50pm-4pm \
  -f experiment_group=ridge_alpha_sweep -f ridge_alphas="0.3,1.0,3.0,10.0"
```

## Verifying the range-read path
```bash
TICK_DATA_URL='s3://futures-data/NQ/tick/*_merged_nq.parquet' \
R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_ENDPOINT_URL=https://<acct>.r2.cloudflarestorage.com \
POLARS_VERBOSE=1 .venv/bin/python -c "
from utils.tick_data import collect_tick_window
df = collect_tick_window(start_utc='2024-03-07T20:50:00Z', end_utc='2024-03-07T21:00:00Z')
print('rows:', df.height)"   # ~17.7k, a few MB transferred (not the full files)
```

## Notes & caveats

- **Year glob.** `TICK_DATA_URL` is a glob (`NQ/tick/*_merged_nq.parquet`); new yearly files
  are picked up automatically. `MINUTE_NQ_URL` is also a glob (`NQ/NQ-*.ohlcv-1m.parquet`) —
  it assumes exactly one NQ ohlcv-1m object exists (replace, don't accumulate, on refresh).
- **History.** The lake tick files span 2020–2025, but macro-window studies currently surface
  data from 2022-07 onward (the earlier years lack full key-minute coverage for the close
  window). Verify upstream coverage if you need 2020–2021.
- **ns-precision filters.** `ts_event` is `datetime[ns, UTC]`; `utils/tick_data._dt` casts
  bounds to ns so row-group pruning stays enabled over R2 (polars #25731).
- **xgboost device.** `macro_range_forecast` defaults to `--xgb-device cuda`; pass `cpu`.
- **Dependency DAG.** Most feature studies read derived `outputs/*.parquet` from upstream
  scripts; the workflow pulls the `macro-research/outputs` mirror first, so seed it by running
  upstream targets before the studies that depend on them.
