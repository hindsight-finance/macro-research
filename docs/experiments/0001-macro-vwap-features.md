---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/open-1550, feature/vwap, feature/vwap/anchor-0930, feature/vwap/anchor-1300, feature/vwap/anchor-1500, feature/vwap/anchor-1550, feature/vwap/anchor-1555, feature/barrier]
anchors: [rth_0930, pm_1300, h3pm_1500, macro_1550, eoii_1555]
checkpoints: ["15:50:10", "15:55:00", "16:00:00"]
side_threshold_pts: 0.25
targets: [target_1550_1554, target_1555_1559, target_1550_1559]
sample_n: 871
---
# Experiment: Macro VWAP Features

## Status
Exploratory

## Concepts
[[anchored-vwap]] · [[barrier-context]] · [[macro-window]] · [[macro-open-1550]] · [[macro-outcome]] · [[volume-delta]] — see the [[Concept Map]] graph.

## Question
Do tick-derived anchored VWAP context features help explain or frame NQ 15:50-15:59 ET macro-window direction, especially before macro, at the first 10 seconds, and at the 15:55 EOII/imbalance-open decision point?

## Dataset
- Asset(s): NQ
- Input minute file(s): none for feature generation; macro comparison context from existing derived outputs where noted.
- Input tick file(s): `input-data/merged_nq_ticks.parquet`
- Date range: 2022-07-12 through 2025-11-27
- Timezone/session definition: UTC timestamps internally; ET market-time windows derived with `utils.minute_bars.MARKET_TZ` (`America/New_York`). Macro window is 15:50-15:59 ET.

## Study Universe
- Dates/events included: 871 ET trade dates with available tick-derived macro VWAP outputs.
- Window(s): pre-macro anchors to `<15:50:00`; intramacro anchors/checkpoints at `<15:50:10`, `<15:55:00`, and `<16:00:00`.
- Completeness filters: rows with missing tick coverage remain in the dataset with null feature/target values; summary sample sizes exclude null feature buckets implicitly through side/bucket filters.
- Regime/context joins: optional join to `outputs/nq_macro_1550_barrier.parquet` for barrier context in intramacro summaries.
- Alignment notes: pre-macro features are available before 15:50; `macro_1550_at_1550_10s` is available after the first 10 seconds; `macro_1550_at_1555` is available at 15:55; `macro_1550_at_1600` and `eoii_1555_at_1600` are descriptive for full-window labeling and should not be treated as pre-entry signals for the full 15:50-15:59 target.

## Parameters
- Premacro anchors: `rth_0930`, `pm_1300`, `h3pm_1500`, all measured at last tick `<15:50:00 ET`.
- Intramacro anchors/checkpoints:
  - `macro_1550_at_1550_10s`: anchor 15:50, checkpoint `<15:50:10`.
  - `macro_1550_at_1555`: anchor 15:50, checkpoint `<15:55:00`.
  - `macro_1550_at_1600`: anchor 15:50, checkpoint `<16:00:00`.
  - `eoii_1555_at_1600`: anchor 15:55, checkpoint `<16:00:00`.
- Targets:
  - `target_1550_1554`: first tick `>=15:50:00` to last tick `<15:55:00`.
  - `target_1555_1559`: first tick `>=15:55:00` to last tick `<16:00:00`.
  - `target_1550_1559`: first tick `>=15:50:00` to last tick `<16:00:00`.
- VWAP formula: `sum((price_ticks / 4.0) * size) / sum(size)`.
- Side threshold: `touch` when `abs(price - vwap) <= 0.25`; otherwise `above` or `below`.
- Runtime method: final implementation uses PyArrow batch streaming for the full tick parquet to avoid eager full-file reads and full-file global sorting.

## Results

### Output shapes

| Output | Rows | Columns |
| --- | ---: | ---: |
| `outputs/nq_macro_vwap_premacro.parquet` | 871 | 29 |
| `outputs/nq_macro_vwap_premacro_summary.parquet` | 237 | 14 |
| `outputs/nq_macro_vwap_intramacro.parquet` | 871 | 42 |
| `outputs/nq_macro_vwap_intramacro_summary.parquet` | 459 | 14 |

### Baseline target distribution

| Target | Bullish | Bearish | Neutral | Null |
| --- | ---: | ---: | ---: | ---: |
| `15:50→15:54` | 446 (51.2%) | 393 (45.1%) | 2 (0.2%) | 30 (3.4%) |
| `15:55→15:59` | 426 (48.9%) | 409 (47.0%) | 6 (0.7%) | 30 (3.4%) |
| `15:50→15:59` | 419 (48.1%) | 422 (48.5%) | 0 (0.0%) | 30 (3.4%) |

### Premacro VWAP side, full `15:50→15:59` target

| Feature / side | Sample | Bullish % | Bearish % | Avg target pts | Median target pts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h3pm_1500 above` | 424 | 47.9 | 52.1 | +1.05 | -1.125 |
| `h3pm_1500 below` | 413 | 52.1 | 47.9 | +1.06 | +1.5 |
| `pm_1300 above` | 473 | 49.9 | 49.3 | +1.45 | +0.5 |
| `pm_1300 below` | 366 | 48.9 | 50.3 | +0.37 | -0.5 |
| `rth_0930 above` | 484 | 47.9 | 49.2 | +0.50 | -0.5 |
| `rth_0930 below` | 386 | 48.4 | 47.4 | +1.54 | +0.625 |

Premacro anchored VWAP side had weak full-window directional separation. There was a modest first-leg tilt where price below premacro VWAPs into 15:50 leaned slightly bullish for `15:50→15:54`.

### Intramacro VWAP side, actionable checkpoints

| Context | Target | Sample | Bullish % | Bearish % | Avg target pts | Median target pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `macro_1550_at_1550_10s above` | `15:50→15:59` | 441 | 60.5 | 39.5 | +8.00 | +7.5 |
| `macro_1550_at_1550_10s below` | `15:50→15:59` | 349 | 38.7 | 61.3 | -7.43 | -6.5 |
| `macro_1550_at_1555 above` | `15:55→15:59` | 428 | 50.7 | 48.8 | +0.39 | +0.625 |
| `macro_1550_at_1555 below` | `15:55→15:59` | 395 | 50.9 | 48.1 | -0.90 | +0.25 |
| `macro_1550_at_1555 above` | `15:50→15:59` | 428 | 69.9 | 30.1 | +15.63 | +15.625 |
| `macro_1550_at_1555 below` | `15:50→15:59` | 395 | 28.6 | 71.4 | -14.71 | -11.0 |

The 10-second macro VWAP side showed a meaningful early contextual tilt for the full macro path. The 15:55 macro VWAP side strongly describes the full path, but it is much less separative for the clean post-15:55 target alone.

### Intramacro confluence, full `15:50→15:59` target

Confluence score is `above_count - below_count` across all four intramacro VWAP checks. Scores using `<16:00` checkpoints are descriptive and not available as early signals.

| Net side score | Sample | Bullish % | Bearish % | Avg target pts | Median target pts |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -4 | 81 | 3.7 | 96.3 | -38.65 | -30.5 |
| -3 | 22 | 9.1 | 90.9 | -24.02 | -20.0 |
| -2 | 162 | 16.7 | 83.3 | -18.87 | -15.375 |
| -1 | 39 | 12.8 | 87.2 | -13.59 | -11.75 |
| 0 | 207 | 41.5 | 44.0 | -1.64 | -0.75 |
| +2 | 205 | 78.0 | 22.0 | +18.59 | +17.5 |
| +3 | 26 | 80.8 | 19.2 | +26.13 | +29.25 |
| +4 | 111 | 95.5 | 4.5 | +34.72 | +29.25 |

## Findings
- Premacro anchored VWAP (`09:30`, `13:00`, `15:00`) is weak as a standalone full macro direction feature.
- The 10-second macro VWAP side is a useful early context variable: above VWAP after the first 10 seconds leaned bullish, below VWAP leaned bearish for the full macro path.
- At 15:55, price relative to 15:50-anchored VWAP strongly describes the path already traveled and full-window state, but it does not by itself strongly separate the clean `15:55→15:59` target.
- Full intramacro confluence is highly directional, but components using `<16:00` are descriptive/labeling features, not pre-entry predictors.
- Barrier + VWAP context appears promising enough to justify a follow-up experiment focused on first-10-second barrier behavior, wrong-side VWAP closes in the 15:50 minute, and 15:55 decision context.

## Caveats
- Exploratory and in-sample only; no walk-forward validation.
- Some intramacro features use information at or near the target endpoint and must not be treated as actionable predictors for that same target.
- Null rows exist where tick coverage was incomplete for the required windows.
- Summaries are distribution diagnostics, not trading rules.
- Runtime initially caused memory pressure under a lazy full-file plan; implementation was changed to one-pass PyArrow batch streaming for tick parquet processing.

## Output Files
- Generated parquet:
  - `outputs/nq_macro_vwap_premacro.parquet`
  - `outputs/nq_macro_vwap_premacro_summary.parquet`
  - `outputs/nq_macro_vwap_intramacro.parquet`
  - `outputs/nq_macro_vwap_intramacro_summary.parquet`
- Reports/charts: none yet.
- Source scripts/specs:
  - `features/macro_vwap_features.py` on branch/worktree `feat/macro-vwap-features`
  - `test/test_macro_vwap_features.py` on branch/worktree `feat/macro-vwap-features`
  - `docs/superpowers/specs/2026-05-15-macro-vwap-features-design.md`
  - `docs/superpowers/plans/2026-05-15-macro-vwap-features.md`

## Follow-ups
- Build `macro_vwap_barrier_context` experiment to compare first-10-second barrier-only vs VWAP-only vs barrier+VWAP context.
- Measure post-barrier wrong-side VWAP behavior inside the 15:50 minute, including close distance, worst wrong-side distance, and seconds/share spent wrong-side.
- Add 15:55 decision-context distributions using only information available before `15:55:00` and target `15:55→15:59`.
- Add distribution visualizations: histograms, ECDFs, box/violin plots, target return distributions by VWAP bucket, and barrier/VWAP heatmaps.
