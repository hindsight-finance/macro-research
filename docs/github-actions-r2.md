# Running research scripts on GitHub Actions (data served from Cloudflare R2)

This repo can run its compute-heavy studies on GitHub-hosted runners instead of a
RAM-bottlenecked local PC. Source data lives in **Cloudflare R2** and is read in place;
the large tick parquet is never downloaded — polars fetches only the parquet row groups
a time-window filter touches, via HTTP range requests.

## Why this works (feasibility)

- **Runner.** A *public-repo* `ubuntu-latest` runner is 4 vCPU / 16 GB RAM, with
  unlimited Actions minutes. (Private would be 2 vCPU / 8 GB + 2,000 min/month — also
  fine, since reads are streamed and windowed, but public is a free 2× upgrade.)
- **Memory.** The tick parquet is time-sorted, so a macro-window read touches a handful
  of row groups. Measured locally: one day's 15:50–16:00 window = 17.7k rows in 0.31 s at
  ~150 MB RSS against the 2.8 GB file. R2 range reads transfer only those row groups.
- **Cost.** R2 free tier (10 GB storage, zero egress, 10 M Class-B reads/mo) covers this
  comfortably; a run issues at most tens of thousands of range reads.

## Architecture

```
Cloudflare R2 bucket <R2_BUCKET>
├── source/merged_nq_ticks.parquet   (2.8 GB) ← polars range-reads in place (no download)
├── source/economic_events.parquet   (60 KB)  ← tiny eager read
└── outputs/                          (~167 MB) ← derived parquet/figs mirror
        nq_minute_base.parquet, nq_macro_*.parquet, figs/, trend_modeling/, ...
```

- **Raw inputs** are redirected to R2 by env vars (`TICK_DATA_URL`, `ECON_EVENTS_URL`)
  plus credentials (`R2_*`). Resolution lives in `utils/data_sources.py`; the tick
  chokepoint is `utils/tick_data.py` (`scan_source`, `get_tick_schema`, `open_parquet_file`).
  With env unset, everything falls back to local `input-data/` — **local runs are unchanged.**
- **Derived `outputs/`** is `rclone`-synced down (prerequisites) before a run and back up
  (results) after, so every inter-script `outputs/...` path keeps working without edits.
- **The two 1m CSVs never go to Actions.** `session_tagger.py` runs locally to produce the
  minute-base parquet, which lives in the R2 `outputs/` mirror.

## One-time setup (run locally — the raw data only exists on your machine)

### 1. Create an R2 bucket + API token
In the Cloudflare dashboard: create a bucket (note its name = `<R2_BUCKET>`), then create
an **R2 API token** (Object Read & Write) scoped to that bucket. Note the Access Key ID,
Secret Access Key, and the S3 endpoint `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.

### 2. Configure rclone for R2
```bash
rclone config create r2 s3 \
  provider=Cloudflare \
  access_key_id=<R2_ACCESS_KEY_ID> \
  secret_access_key=<R2_SECRET_ACCESS_KEY> \
  endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com \
  region=auto
```

### 3. Upload source data + seed the outputs mirror
```bash
# raw sources
rclone copy input-data/merged_nq_ticks.parquet  r2:<R2_BUCKET>/source/ -P
rclone copy input-data/economic_events.parquet  r2:<R2_BUCKET>/source/ -P

# build the canonical minute parquet locally, store under the name macro_outcomes expects
.venv/bin/python session_tagger.py            # writes outputs/nq_1m.parquet (+ es)
cp outputs/nq_1m.parquet outputs/nq_minute_base.parquet
# (regenerate any other derived prerequisites you need, then seed the whole mirror)
rclone copy outputs/  r2:<R2_BUCKET>/outputs/ -P
```

### 4. Add repository secrets
```bash
gh secret set R2_BUCKET            --body '<R2_BUCKET>'
gh secret set R2_ACCESS_KEY_ID     --body '<R2_ACCESS_KEY_ID>'
gh secret set R2_SECRET_ACCESS_KEY --body '<R2_SECRET_ACCESS_KEY>'
gh secret set R2_ENDPOINT_URL      --body 'https://<ACCOUNT_ID>.r2.cloudflarestorage.com'
```

### 5. (Optional) Make the repo public for the bigger runner
Only the *code* becomes public — data stays private in R2, never committed. First skim
`ideadump.md`, `docs/`, `.planning/` for anything you would not want world-readable.
```bash
gh repo edit hindsight-finance/macro-research --visibility public --accept-visibility-change-consequences
```

## Running

**`backtest.yml`** — run any single study. Actions tab → *backtest* → *Run workflow*,
pick a `target`, optionally add `extra_args`. Or from the CLI:
```bash
gh workflow run backtest.yml -f target=macro_extreme_timing
gh workflow run backtest.yml -f target=macro_range_forecast -f extra_args="--xgb-device cpu"
```
Results are uploaded as a `backtest-<target>-<run_id>` artifact and pushed to the R2
`outputs/` mirror.

**`sweep.yml`** — parallel ridge-alpha sweep of the trend harness (build table → matrix →
summarize):
```bash
gh workflow run sweep.yml -f session_name=3:50pm-4pm -f experiment_group=ridge_alpha_sweep \
  -f ridge_alphas="0.3,1.0,3.0,10.0"
```

## Verifying the range-read path (after setup)
```bash
TICK_DATA_URL="s3://<R2_BUCKET>/source/merged_nq_ticks.parquet" \
R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_ENDPOINT_URL=https://<acct>.r2.cloudflarestorage.com \
POLARS_VERBOSE=1 .venv/bin/python -c "
from utils.tick_data import collect_tick_window
df = collect_tick_window(start_utc='2024-03-07T20:50:00Z', end_utc='2024-03-07T21:00:00Z')
print('rows:', df.height)"
```
Expect ~17.7k rows and a few MB transferred (check the R2 dashboard), not 2.8 GB.

## Notes & caveats

- **ns-precision filters.** `ts_event` is `datetime[ns, UTC]`; `utils/tick_data._dt` casts
  bounds to ns so row-group pruning stays enabled over R2 (polars #25731). Keep window
  bounds ns-compatible if you add new tick filters.
- **xgboost device.** `macro_range_forecast` defaults to `--xgb-device cuda`; runners are
  CPU-only, so pass `--xgb-device cpu`.
- **Dependency DAG.** Most feature studies read derived `outputs/*.parquet` produced by
  upstream scripts. The workflow pulls the `outputs/` mirror first, so prerequisites must
  already be in R2 (seed them in step 3, or run upstream targets first).
- **Naming drift.** `session_tagger.py` writes `nq_1m.parquet`; `macro_outcomes.py` reads
  `nq_minute_base.parquet` — store/copy under the latter name in the mirror (step 3).
