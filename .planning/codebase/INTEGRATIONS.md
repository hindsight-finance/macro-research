# External Integrations

**Analysis Date:** 2026-05-24

This is an offline, file-based quantitative research workspace. There are **no live external API calls, no network clients, no databases, and no auth providers** in the current codebase. All "integration" is via static data files dropped into `input-data/` and derived artifacts written to `outputs/`. The `README.md` references future plans to scrape economic calendars (Forex Factory / Investing.com via Selenium/API), but no scraping or HTTP code exists in source.

## APIs & External Services

**Live integrations:**
- None - No `requests`, `urllib`, `httpx`, `selenium`, `beautifulsoup4`, or any HTTP/scraping library is installed or imported

**Aspirational (documented, not implemented):**
- Economic calendar scraping (Forex Factory / Investing.com) - Mentioned in `README.md` (lines 56, 82, 119) and `roadmap.md` as a planned ingestion step. Currently fulfilled by a pre-supplied static file (see Data Storage below)

## Data Storage

**Databases:**
- None - No SQLite, Postgres, or any DB driver installed or referenced (despite `README.md` line 120 suggesting SQLite/Postgres as an option)

**File-based data inputs** (`input-data/`, gitignored, not committed):
- `input-data/nq_1m.csv` - NQ 1-minute OHLCV bars. Columns: `DateTime_ET, Open, High, Low, Close, Volume, DateTime_UTC` (data back to 2010-06-06)
- `input-data/es_1m.csv` - ES 1-minute OHLCV bars, same schema
- `input-data/merged_nq_ticks.parquet` - NQ trade-level tick data (hundreds of millions of rows). Schema documented in `input-data/merged_ticks_schema.txt`: `ts_event` (datetime[ns, UTC]), `intra_ts_rank` (UInt8), `side` (UInt8: 0=none, 1=sell aggressor, 2=buy aggressor), `price_ticks` (UInt32, actual_price = price_ticks / 4), `size` (UInt16). Derived from a Databento CSV export (`price_ticks = round(price * 4)`)
- `input-data/merged_ticks_example.csv` - Small sample tick CSV (same columns)
- `input-data/economic_events.parquet` - Economic calendar events. Schema: `datetime_utc` (timestamp[ns, UTC]), `currency` (str), `impact` (str), `title` (str), `id` (int64), `leaked` (bool). Consumed by `features/macro_range_forecast.py` (`build_macro_event_links`, `merge_news_daily` in `utils/helper.py`) and `test/cpi.py`. Provenance is upstream/manual — no generating code in this repo

**Output artifacts** (`outputs/`, gitignored):
- Canonical minute bars: `outputs/nq_1m.parquet`, `outputs/es_1m.parquet`, `outputs/nq_minute_base.parquet` (columns: `datetime_utc, Open, High, Low, Close, Volume`)
- Macro feature tables: `outputs/nq_macro_*.parquet` and `*_summary.parquet` (e.g. `nq_macro_outcomes`, `nq_macro_1550_barrier`, `nq_macro_extreme_timing`, `nq_macro_fvg_events`, `nq_macro_vwap_*`, `nq_macro_tick_density*`, `nq_macro_volume_delta_*`, `nq_macro_range_forecast`)
- Trend/regime modeling: `outputs/trend_modeling/` (cache, ablations, experiments, regime tables)
- Figures: `outputs/figs/<study>/` PNGs; logs in `outputs/logs/`

**Data provider (origin, external to repo):**
- Databento - The tick parquet originates from a Databento CSV export per `input-data/merged_ticks_schema.txt`. No Databento SDK or API client is used in code; data arrives pre-exported

**External sibling project:**
- `smt` editable package mapped to `/mnt/e/backup/code/Finance/Misc/SMT/smt` (outside this repo) via the venv `.pth` file. Not imported by this repo's source

**Caching:**
- `outputs/trend_modeling/cache/` - On-disk parquet cache for trend modeling experiments

## Authentication & Identity

- None - No auth provider, API keys, tokens, or credential handling. No `os.environ`/`getenv`/dotenv usage anywhere in source

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- Plain file logs written under `outputs/logs/` by pipeline scripts. No structured logging framework

## CI/CD & Deployment

**Hosting:**
- None - Offline research workspace, no deployment target

**CI Pipeline:**
- None detected - No `.github/workflows/`, CI config, or pipeline files. `AGENTS.md` recommends headless-safe (non-visual) test assertion paths so logic can run in CI/headless shells, but no CI is wired up

## Environment Configuration

**Required env vars:**
- None - All configuration is via `argparse` CLI flags and hardcoded `UPPER_CASE` path constants per script

**Secrets location:**
- Not applicable - no secrets consumed. `.gitignore` defensively ignores `.env`/`.env.*`, but no such files exist

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Data Flow Summary

Per `AGENTS.md`:
1. `input-data/nq_1m.csv` / `es_1m.csv` → normalized by `session_tagger.py` → `outputs/nq_1m.parquet` / `es_1m.parquet`
2. Canonical minute bars → `macro_outcomes.py` → `outputs/nq_macro_outcomes.parquet`
3. `input-data/merged_nq_ticks.parquet` → `tick_density.py` / `volume_delta.py` → tick density & volume-delta parquet
4. `features/*.py` → derived research tables in `outputs/`
5. `viz/*.py` → CSV summaries and figures under `outputs/figs/`
6. `input-data/economic_events.parquet` → joined into news/range studies via `utils/helper.py` and `features/macro_range_forecast.py`

---

*Integration audit: 2026-05-24*
