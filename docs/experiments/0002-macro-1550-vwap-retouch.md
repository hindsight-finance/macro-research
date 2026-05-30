---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/open-1550, window/macro/close-1559, feature/barrier, feature/barrier/break-direction, feature/vwap, feature/vwap/anchor-1550, feature/vwap/retouch-1550, feature/mae-mfe, feature/outcome]
anchors: [macro_1550_break, retouch_frozen, retouch_rolling]
checkpoints: ["15:50:10", "15:55:00", "16:00:00", "16:01:00"]
barrier_seconds: 10
side_threshold_pts: 0.25
targets: [fwd_break_1554, fwd_break_1559, fwd_break_1600, fwd_retouch_frozen_1559, fwd_retouch_rolling_1559]
sample_n: 1252
---
# Experiment: First-10s VWAP-Retouch Event Study (15:50 Macro Open)

## Status
Exploratory

## Concepts
[[first-10s-break-direction]] · [[anchored-vwap]] · [[barrier-context]] · [[macro-open-1550]] · [[macro-window]] · [[macro-close-1559]] · [[macro-outcome]] · [[mae-mfe]] — see the [[Concept Map]] graph.

## Question
After the first 10 seconds of the 15:50 macro open form a barrier range `[low_10s, high_10s]`, does the **first side to break** (a causal directional bias) and a subsequent **retouch of the 15:50-anchored VWAP** (frozen first-10s VWAP, or the rolling cumulative VWAP) describe a usable forward path into the 15:59 macro close? This is a descriptive event study — it measures forward outcomes from the break and from each retouch; it is **not** a target/stop backtest.

## Dataset
- Asset(s): NQ
- Input minute file(s): none (self-contained; validation directions derived from the same tick window).
- Input tick file(s): Cloudflare R2 `futures-data` lake, `NQ/tick/*_merged_nq.parquet` (range-read in place on a GitHub-hosted runner; see [[remote-compute]]).
- Date range: 2020-09-01 through 2025-11-26 (1,252 ET macro dates).
- Timezone/session definition: UTC internally; ET market-time derived with `utils.minute_bars.MARKET_TZ` (`America/New_York`), DST-correct. Macro window is 15:50:00–15:59:59 ET.

## Study Universe
- Dates/events included: 1,252 ET dates with tick coverage in the 15:50:00–16:00:59 ET read window.
- Window(s): first-10s barrier `[15:50:00, 15:50:10)`; break search `[15:50:10, 16:00:00)`; retouch search strictly after the break, `< 16:00:00`; forward horizons read up to `< 16:01:00`.
- Completeness filters: `has_first10` requires ≥1 tick in `[15:50:00, 15:50:10)` (true on all 1,252 dates here). Days that never break the barrier are recorded as `no_trigger` with null signal/outcome fields.
- Regime/context joins: none.
- Alignment notes: the realised 15:50-minute candle direction and the full-macro (15:50→15:59) direction are recorded as **validation outcomes only**, never as signals. Forward points are signed favourable-to-bias; a horizon is null when the anchor is at/after that horizon's cutoff.

## Parameters
- `barrier_seconds`: 10 → first-10s range = first tick high/low over `[15:50:00, 15:50:10)`.
- Break rule: first **strict** cross (`price > high_10s` or `price < low_10s`) after 15:50:10; whichever side breaks first sets `bias` (`high`→`bullish`, `low`→`bearish`). First break wins on whipsaw.
- Frozen VWAP: `sum(price·size) / sum(size)` over the first-10s ticks.
- Rolling VWAP: 15:50-anchored cumulative `cumsum(price·size) / cumsum(size)`.
- Retouch rule (after break): bullish — first tick with `price <= vwap + 0.25`; bearish — first tick with `price >= vwap − 0.25`. `touch = 0.25 pts`.
- Forward horizons (last tick before the cutoff): `1554` (<15:55), `1559` (<16:00, the macro close), `1600` (<16:01).
- Excursion: MFE/MAE from the anchor to the macro close (`<16:00`), signed favourable-to-bias.
- Runtime method: bounded **lazy scan** via `utils.tick_data.scan_source` (projects 4 columns, derives ET, filters the 15:50–16:00:59 window early) + per-date `collect(engine="streaming")`; no eager tick read.

## Results

### Coverage (n = 1,252 dates)

| Bucket | Value | Of |
| --- | ---: | --- |
| `has_first10` | 100.0% (1,252) | all dates |
| `triggered` (barrier broke) | 99.9% (1,251) | all dates |
| `no_trigger` | 0.1% (1) | all dates |
| Break side `high` / `low` | 603 / 648 | triggered |
| Runaway (no **frozen** retouch) | 23.9% | triggered |
| Runaway (no **rolling** retouch) | 3.0% | triggered |

The barrier breaks on essentially every day, and a retouch of the 15:50-anchored VWAP is common — the **frozen** VWAP is retouched on 76.1% of triggered days, the **rolling** VWAP on 97.0%.

### Signal validation — does the first-break direction agree with realised direction? (triggered)

| Realised reference | Agreement with break bias | Sample |
| --- | ---: | ---: |
| 15:50 candle direction | 81.4% | 1,239 |
| Full macro (15:50→15:59) close direction | 68.9% | 1,250 |

The first-10s break side agrees with the eventual macro-close direction roughly **69%** of the time — a real but imperfect directional tilt.

### Forward move to the 15:59 macro close (mean points, signed favourable-to-bias; % of days favourable)

| Anchor | Bullish mean / fav% (n) | Bearish mean / fav% (n) |
| --- | --- | --- |
| Break | +3.68 / 56.6% (603) | +2.67 / 53.9% (648) |
| Frozen-VWAP retouch | +2.15 / 54.2% (450) | +2.11 / 55.2% (502) |
| Rolling-VWAP retouch | +3.25 / 56.9% (578) | +3.16 / 56.2% (635) |

### Excursion to the macro close (mean points, signed favourable-to-bias)

| Anchor | Bullish MFE / MAE | Bearish MFE / MAE |
| --- | --- | --- |
| Break | +22.27 / −19.42 | +22.23 / −20.23 |
| Frozen-VWAP retouch | +19.47 / −17.95 | +19.53 / −18.08 |

## Findings
- The first-10s barrier **almost always breaks** (99.9% of dates) and a 15:50-anchored VWAP retouch is the norm, not the exception (frozen 76%, rolling 97% of triggered days) — so "break then retouch" is a high-frequency event, suitable as an entry candidate for a later backtest.
- The first-break direction is a **modestly informative** directional signal: it matches the 15:50-minute candle 81% of the time and the full macro close ~69% of the time.
- Mean forward moves to the close are **positive and favourable-to-bias** from every anchor (+2 to +4 pts) with ~54–57% favourable-day rates — a small descriptive edge in the central tendency.
- But the moves are **noisy**: MFE/MAE are large (≈ ±20 pts) and roughly symmetric, so per-day dispersion dwarfs the mean tilt. The rolling-VWAP retouch shows a slightly better mean and favourable rate than the frozen retouch, and retouching gives up only a little of the break-anchor mean.

## Caveats
- **Exploratory and descriptive only** — no target/stop rule, no expectancy, no transaction costs, no walk-forward. The means are central tendencies over wide, near-symmetric distributions.
- The 15:50-candle and macro-close directions are outcomes, not predictors; do not read the 81%/69% agreement as a tradeable signal without an entry/exit rule.
- Single contract (NQ), single window (15:50 macro open); no regime split yet.
- Tick study: bounded lazy scan + streaming collect through `scan_source` (no eager tick read); ET derived from UTC so DST is handled.

## Output Files
- Generated parquet (gitignored; from GitHub Actions run 26674032664, 2026-05-30; copied to local `outputs/`):
  - `outputs/nq_macro_1550_vwap_retouch.parquet` — 1,252 rows × 49 cols (one row per macro date).
  - `outputs/nq_macro_1550_vwap_retouch_summary.parquet` — 451 rows (long-form `scope/bucket/metric/value/sample_size`).
- Reports/charts: none yet.
- Source scripts/specs:
  - `features/macro_1550_vwap_retouch.py`
  - `test/test_macro_1550_vwap_retouch.py`
  - `docs/superpowers/specs/2026-05-29-macro-1550-vwap-retouch-design.md`
  - `docs/superpowers/plans/2026-05-29-macro-1550-vwap-retouch.md`

## Follow-ups
- **Stage 2 (separate spec/plan):** entry/exit backtest over these retouch events — target/stop, expectancy, costs — to test whether the descriptive tilt survives as a rule.
- Split by break side, range size (`range_10s_points`), retouch lag, and first-10s volume share (`vol_share_first10`); the summary already emits decile cross-tabs on these.
- Compare frozen vs rolling retouch as the entry trigger (rolling retouches more often and shows a marginally better mean).
- Add distribution visualizations (forward-return histograms/ECDFs by bias × anchor; MFE/MAE scatter).
- Infra: the Actions R2 push-back step fails `AccessDenied` (read-only token) — results live in the run artifact, not the R2 `outputs/` mirror, until write creds are fixed.
