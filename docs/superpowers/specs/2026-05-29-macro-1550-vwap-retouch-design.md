# Design: First-10s VWAP-retouch event study (`macro_1550_vwap_retouch`)

## Status
Draft / Exploratory (descriptive event study, pre-implementation)

## Date
2026-05-29

## Motivation / thesis
The first 10 seconds of the 15:50 ET ("3:50 PM") macro-open candle carry a disproportionate
share of that minute's volume and act as a **barrier range** `[low_10s, high_10s]`. The trade
thesis being characterised:

1. **Causal direction signal:** whichever side of the first-10s barrier breaks *first* after
   15:50:10 sets the bias — break `high_10s` first → **bullish**, break `low_10s` first →
   **bearish**. (Working hypothesis: a low-break-first makes a bearish 3:50 candle / bearish
   macro significantly more likely.)
2. **Entry timing:** once biased, a **retouch of the first-10s VWAP** (price pulling back to the
   VWAP in the drive direction) is the candidate entry.
3. **Outcome:** forward continuation from the break and from the retouch.

Everything used to *form the signal and entry* is known by the retouch timestamp, so the study
is causal. The realised 15:50-candle direction and full macro direction are recorded only as
**outcomes to score the signal against**, never as inputs.

This is the descriptive **event study** stage. A simulated entry/exit backtest (target/stop,
win-rate, expectancy) is a deliberate later stage built on the same retouch events — see
Follow-ups.

## Scope
- **In:** per-day detection of the first-10s barrier range + both VWAPs; first barrier break
  (direction signal); first VWAP retouch against **two** references (frozen 10s level and rolling
  15:50-anchored VWAP); forward outcome distributions from break and from each retouch; signal
  validation vs the 15:50 candle and the macro direction; a conditional summary table.
- **Out (v1):** target/stop backtest and PnL; ES / cross-asset; visualisation figures;
  any use of `macro_trend_state` as a signal.
- **Instrument:** NQ only. Source: the tick parquet (local `input-data/merged_nq_ticks.parquet`
  or the R2 data lake — see Remote execution).

## Definitions & parameters
- **Time:** UTC internally, ET derived via `utils.minute_bars.MARKET_TZ`. All "seconds" below are
  ET seconds within the 15:50 minute unless stated.
- **First-10s window:** ET `[15:50:00, 15:50:10)`.
- **Macro window:** ET `[15:50:00, 16:00:00)` (minutes 50–59). The read also includes the
  **16:00 POST minute** `[16:00:00, 16:01:00)` purely to provide the 16:00 horizon print.
- `BARRIER_SECONDS = 10` (first-10s width).
- `TOUCH_THRESHOLD_POINTS = 0.25` (the repo-wide touch band; a retouch is "within 0.25 pt").
- `TICK_PRICE_DENOMINATOR = 4.0` → `price = price_ticks / 4.0`.
- **Tie-break / ordering:** `(ts_event, intra_ts_rank)`, ascending — matches the repo's
  first-touch convention.
- **Horizons (candle-close convention, matching existing `target_*` windows):**
  - `h1554` = last tick with `et_second < 15:55:00` (close of the 15:54 candle).
  - `h1559` = last tick with `et_second < 16:00:00` (close of the 15:59 candle = **macro close**).
  - `h1600` = last tick with `et_second < 16:01:00` (close of the 16:00 POST candle).
- **Signed-by-bias helper:** `signed(p_from, p_to) = (p_to − p_from)` if bias is bullish, else
  `(p_from − p_to)`. Positive = favourable to the bias.

## Per-day algorithm (pure core `detect_retouch_events`)
Input: one day's tick rows (`ts_event, intra_ts_rank, et_second, price, size`), sorted by
`(ts_event, intra_ts_rank)`. Output: a single dict/row (schema below).

1. **First-10s aggregates** over `[15:50:00, 15:50:10)`:
   - If no ticks in this window → emit a data-quality row (`has_first10 = false`, signal/retouch
     fields null). Otherwise:
   - `high_10s = max(price)`, `low_10s = min(price)`, `range_10s_points = high_10s − low_10s`.
   - `vwap_10s_frozen = Σ(price·size) / Σ(size)` over the window.
   - `open_1550 = ` first tick price at/after 15:50:00.
   - `vol_share_first10 = Σsize[first 10s] / Σsize[15:50:00–15:51:00)` (the "disproportionate
     volume" diagnostic; null if the 15:50 minute has no size).

2. **Rolling VWAP** anchored at 15:50:00: cumulative `Σ(price·size)/Σ(size)` evaluated at each
   tick across the macro window.

3. **Barrier break (direction signal)** — scan ticks with `et_second ≥ 15:50:10` and
   `< 16:00:00` in order:
   - First tick with `price > high_10s` → `break_side = "high"`, `bias = "bullish"`.
   - First tick with `price < low_10s` → `break_side = "low"`, `bias = "bearish"`.
   - **First to occur wins** (whipsaw → first break only). Break is a **strict** cross
     (`>` / `<`); touching the level is not a break.
   - Record `break_time_s` (seconds after 15:50:00), `break_ts_utc`, `break_price`.
   - If neither side breaks before 16:00:00 → `break_side = "none"`, `bias = "none"`,
     `trigger_state = "no_trigger"` (excluded from outcome statistics; counted in a coverage line).

4. **Retouch (entry candidate)** — for each reference `ref ∈ {frozen, rolling}`, scan ticks
   strictly after `break_ts_utc` (≤ 16:00:00):
   - Frozen: retouch = first tick where price returns within the touch band of `vwap_10s_frozen`
     from the displaced side — bullish: `price ≤ vwap_10s_frozen + 0.25`; bearish:
     `price ≥ vwap_10s_frozen − 0.25`.
   - Rolling: same test against the **current** rolling VWAP value at that tick.
   - **First retouch only** = the entry. Record `retouch_{ref}_occurred`, `retouch_{ref}_time_s`,
     `retouch_{ref}_ts_utc`, `retouch_{ref}_price`, `retouch_{ref}_lag_s`
     (`= retouch_time_s − break_time_s`); for rolling also `vwap_rolling_at_retouch`.
   - Break but no retouch before 16:00:00 → `retouch_{ref}_occurred = false` ("runaway");
     break-anchored outcomes still recorded, retouch-anchored outcomes null.
   - **Tiny-range edge:** when `range_10s_points` is within the touch band (≤ 0.25), the break
     price can already satisfy the retouch test on the first post-break tick (immediate retouch).
     This is legitimate but flagged by `range_10s_points` so such days can be filtered downstream.

5. **Forward outcomes** (signed by bias) from each anchor `a ∈ {break, retouch_frozen,
   retouch_rolling}` whose timestamp exists:
   - `fwd_{a}_1554_points = signed(a_price, h1554_price)`, likewise `_1559`, `_1600`.
     A horizon is **null when the anchor timestamp is ≥ that horizon's cutoff** (cannot measure
     a passed horizon — e.g. a 15:58 retouch → `_1554` null).
   - `mfe_{a}_points = max over ticks (a_ts → 16:00:00] of signed(a_price, tick_price)`
     (≥ 0 typically); `mae_{a}_points = min over the same of signed(a_price, tick_price)`
     (≤ 0). Excursions are measured to the **macro close** (16:00:00), not into POST.

6. **Validation fields** (outcomes, never signal), from the same tick window:
   - `candle_1550` open/close = first tick ≥ 15:50:00 / last tick `< 15:51:00`;
     `candle_1550_dir_points = close − open`, `candle_1550_dir_sign`, `candle_1550_state`.
   - `macro` open/close = first tick ≥ 15:50:00 / last tick `< 16:00:00`;
     `macro_dir_points`, `macro_dir_sign`, `macro_trend_state`.
   - `bias_matches_1550_candle` = (bias bullish & candle_1550 bullish) or (bias bearish &
     candle_1550 bearish); null when `bias = none` or candle neutral. `bias_matches_macro`
     analogous. These directly test the working hypothesis.

## Output schema
`outputs/nq_macro_1550_vwap_retouch.parquet` — one row per qualifying date:

- **Identity / quality:** `date`, `tick_count_macro`, `has_first10`, `trigger_state`.
- **First-10s:** `open_1550`, `high_10s`, `low_10s`, `range_10s_points`, `vwap_10s_frozen`,
  `vol_share_first10`.
- **Signal:** `break_side`, `bias`, `break_time_s`, `break_ts_utc`, `break_price`.
- **Retouch (×2 refs):** `retouch_frozen_occurred`, `retouch_frozen_time_s`,
  `retouch_frozen_ts_utc`, `retouch_frozen_price`, `retouch_frozen_lag_s`;
  `retouch_rolling_occurred`, `retouch_rolling_time_s`, `retouch_rolling_ts_utc`,
  `retouch_rolling_price`, `retouch_rolling_lag_s`, `vwap_rolling_at_retouch`.
- **Forward from break:** `fwd_break_1554_points`, `fwd_break_1559_points`,
  `fwd_break_1600_points`, `mfe_break_points`, `mae_break_points`.
- **Forward from frozen retouch:** `fwd_retouch_frozen_1554_points`, `_1559_points`,
  `_1600_points`, `mfe_retouch_frozen_points`, `mae_retouch_frozen_points`.
- **Forward from rolling retouch:** `fwd_retouch_rolling_1554_points`, `_1559_points`,
  `_1600_points`, `mfe_retouch_rolling_points`, `mae_retouch_rolling_points`.
- **Validation:** `candle_1550_dir_points`, `candle_1550_dir_sign`, `candle_1550_state`,
  `macro_dir_points`, `macro_dir_sign`, `macro_trend_state`, `bias_matches_1550_candle`,
  `bias_matches_macro`.

## Summary table
`outputs/nq_macro_1550_vwap_retouch_summary.parquet` — long form `(scope, bucket, metric…,
sample_size)`. Every rate carries `sample_size` (per AGENTS.md). Scopes:

- `signal_validation` — by `break_side`: P(candle_1550 bullish/bearish), P(macro bullish/bearish),
  `bias_matches_*` rates, with n. (Tests "low breaks first ⇒ bearish.")
- `coverage` — counts of `no_trigger`, runaway (break-no-retouch) per ref, `has_first10 = false`.
- `retouch_frequency` — % of triggered days with a frozen / rolling retouch; retouch-lag
  distribution (mean/median/p10–p90).
- `forward_outcome` — mean/median/p10/p25/p75/p90 of `fwd_*` and MFE/MAE, split by `bias`
  (bullish/bearish) × anchor (break / frozen retouch / rolling retouch) × horizon.
- Decile cross-tabs of forward outcome on `retouch_lag_s`, `range_10s_points`, `vol_share_first10`
  (mirrors the existing decile pattern in `macro_vwap_barrier_context`).

## Module structure (Approach A — standalone module, pure core)
New file `features/macro_1550_vwap_retouch.py`:

- `detect_retouch_events(day_ticks, *, barrier_seconds=10, touch=0.25) -> dict` — **pure**, no I/O.
- `_scan_macro_window(path) -> pl.LazyFrame` — `scan_source(path)` → select
  `ts_event(cast ns UTC), intra_ts_rank, price_ticks, size`, derive ET / `et_second` / `price`,
  filter to `[15:50:00, 16:01:00)` (macro + POH 16:00 minute), validate schema via
  `get_tick_schema`.
- `build_macro_1550_vwap_retouch(path) -> pl.DataFrame` — collect the bounded window
  (`engine="streaming"`), `partition_by("date")`, run `detect_retouch_events` per day, assemble.
- `summarize_macro_1550_vwap_retouch(df) -> pl.DataFrame`.
- `write_macro_1550_vwap_retouch(...)` and `main()` guarded by `data_sources.source_exists`.
- Output stems: `nq_macro_1550_vwap_retouch[.parquet]`, `nq_macro_1550_vwap_retouch_summary`.
- Run locally: `.venv/bin/python -m features.macro_1550_vwap_retouch`.

Mirrors `features/macro_extreme_timing.py` (R2-ready tick read) and the per-day-iteration style of
`features/macro_vwap_barrier_context.py`, but keeps the **causal** break signal in its own file —
distinct from the lookahead barrier in `macro_1550_barrier.py`.

## Remote execution (GitHub Actions + Cloudflare R2)
The study is built R2-ready from the start so it can run on a GitHub-hosted runner (16 GB / 4 vCPU)
when the local box is RAM-constrained.

- **Tick reads via the source helpers only** — `INPUT_PATH = data_sources.tick_data_url()` and
  `scan_source(INPUT_PATH)`; never raw `pl.scan_parquet` on the source. `scan_source` attaches R2
  `storage_options` for `s3://` URLs and synthesizes `price_ticks` from the lake's float `price`,
  so selecting `price_ticks` works against both layouts. ns-precision bounds keep row-group
  pruning enabled over R2.
- **Self-contained / no upstream dependency** — validation directions are derived from the same
  tick window (no join to `nq_macro_outcomes`), so the study needs nothing from the `outputs/`
  mirror; it only *writes* its two parquets, which the workflow rclone-syncs back to
  `macro-research/outputs` and uploads as an artifact.
- **Wiring** — add `macro_1550_vwap_retouch` to `.github/workflows/backtest.yml`'s `target`
  `options:` list and a `case` line: `macro_1550_vwap_retouch) cmd="python -m
  features.macro_1550_vwap_retouch" ;;`. Trigger:
  `gh workflow run backtest.yml -f target=macro_1550_vwap_retouch`.
- **Local unchanged** — with env vars unset, `tick_data_url()` returns the local path and
  `storage_options()` is `None`, so local development/tests are identical.
- **History caveat** — the R2 lake spans **2020-09 → 2025-11** vs the local file's ~2022-07
  onward, so a runner produces a longer history (more samples, but includes the 2020–2021 regime).
  Mark results with their actual date range.

## Testing (headless, no R2)
Unit tests on the pure `detect_retouch_events` with synthetic single-day frames:
- clean high-break → frozen + rolling retouch → favourable continuation;
- low-break (bearish) symmetric case;
- whipsaw (low then high) → first break wins;
- no break (price stays inside the 10s range) → `no_trigger`;
- break but no retouch → runaway (break outcomes present, retouch outcomes null);
- exact tie on `(ts_event, intra_ts_rank)`;
- DST day (ET derived correctly from UTC);
- late retouch (15:58) → `_1554` horizon null;
- empty first-10s → `has_first10 = false`.

Plus: a schema-failure test (missing tick column → `ValueError`), and a small `write_*`
smoke test over a **local fixture parquet** (with `price_ticks`, exercising `scan_source` →
build → summary). All headless; no plotting in v1.

## Follow-ups (after first run)
1. Run on GHA (`backtest.yml`), pull the artifact, sanity-check coverage & sample sizes.
2. Add the experiment log (`docs/research_log.md` index + `docs/experiments/NNNN-…md`) per the
   AGENTS protocol, marked Exploratory, with the actual date range and sample sizes.
3. Add concept-graph nodes (`experiments/macro-1550-vwap-retouch.md`, and a `concepts/` node for
   the causal first-10s-break signal) and link them into `docs/graph/Concept Map.md`.
4. Optional viz script mirroring `viz/macro_vwap_barrier_context_viz.py`.
5. Stage 2: the deferred entry/exit backtest (target/stop, expectancy) over these retouch events.
