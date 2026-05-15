# Macro VWAP Features Design

## Purpose

Build tick-derived VWAP context features for NQ market-close macro research. The study has two related feature sets:

1. **Feature set A: pre-macro VWAP context** — price relation to anchored VWAPs before 15:50 ET, tested against multiple macro direction targets.
2. **Feature set B: intramacro VWAP context** — price relation to macro-anchored VWAPs during the 15:50-15:59 ET macro window, including the first-10-second barrier context, EOII/imbalance-open context, and 15:59 close context.

The feature should be exploratory and bias-light. VWAP is recorded as context, not as a hard confirmation rule. Summaries should expose observed relationships without imposing trade logic.

## Scope

### In scope

- Tick-derived VWAP features from `input-data/merged_nq_ticks.parquet`.
- Separate A/B daily feature outputs.
- Separate A/B diagnostic summary outputs.
- Three macro direction targets:
  - `15:50 -> 15:54`
  - `15:55 -> 15:59`
  - full `15:50 -> 15:59`
- Price-vs-VWAP distance in points and signed basis points.
- Price-vs-VWAP side with a one-tick touch threshold.
- Optional neutral join to the existing 15:50 barrier dataset when available.
- Unit tests for VWAP math, target construction, boundary behavior, touch classification, missing-data behavior, and writer outputs.

### Out of scope

- Predictive modeling or ML.
- Trade entry/exit rules.
- Visualization scripts.
- Minute-bar fallback implementation. Tick data is assumed dense enough for the primary study.
- Overwriting source files in `input-data/`.

## Inputs

Primary input:

- `input-data/merged_nq_ticks.parquet`

Required tick columns:

- `ts_event`
- `intra_ts_rank`
- `price_ticks`
- `size`

Existing optional context input for feature set B:

- `outputs/nq_macro_1550_barrier.parquet`

Relevant barrier columns:

- `date`
- `macro_trend_state`
- `barrier_extreme`
- `barrier_price`
- `barrier_time`
- `barrier_first10`
- `barrier_is_macro_extreme`
- `barrier_holds`
- `edge_case`

The barrier join is contextual only. Column names and summaries must avoid confirmation language.

## Timezone and boundary conventions

- Keep timestamps in UTC internally.
- Derive ET market time using `utils.minute_bars.MARKET_TZ` (`America/New_York`).
- Process ticks lazily with `pl.scan_parquet`.
- Filter bounded ET windows before collect/sink.
- Order ticks by `ts_event`, then `intra_ts_rank`.
- Checkpoint snapshots use the last tick strictly before the checkpoint.

Boundaries:

- Pre-macro checkpoint: last tick `< 15:50:00 ET`.
- Initial barrier checkpoint: last tick `< 15:50:10 ET`.
- EOII/imbalance-open checkpoint: last tick `< 15:55:00 ET`.
- Macro close checkpoint: last tick `< 16:00:00 ET`.

## VWAP computation

Use tick price and volume:

- `price = price_ticks / TICK_PRICE_DENOMINATOR`
- `pv = price * size`
- `vwap = sum(pv) / sum(size)` for `anchor_time <= ts_event < checkpoint_time`

If an anchor window has no ticks or zero total size, VWAP is null.

Snapshot price is the last tick before the checkpoint. If absent, snapshot price is null.

Distance columns:

- `*_vwap_dist_points = checkpoint_price - vwap`
- `*_vwap_dist_bps = (checkpoint_price / vwap - 1.0) * 10000.0`

Side classification:

- `touch` when `abs(dist_points) <= 0.25`
- `above` when `dist_points > 0.25`
- `below` when `dist_points < -0.25`
- null when either price or VWAP is null

## Macro direction targets

Targets are tick-derived from macro open/close ticks.

Target definitions:

- `target_1550_1554_points`: close at last tick `< 15:55:00 ET` minus first tick `>= 15:50:00 ET`.
- `target_1555_1559_points`: close at last tick `< 16:00:00 ET` minus first tick `>= 15:55:00 ET`.
- `target_1550_1559_points`: close at last tick `< 16:00:00 ET` minus first tick `>= 15:50:00 ET`.

Each target also gets:

- `*_sign`: `-1`, `0`, or `1`
- `*_state`: `bearish`, `neutral`, or `bullish`

If either endpoint is missing, target points/sign/state are null.

## Feature set A: pre-macro VWAP context

Output path:

- `outputs/nq_macro_vwap_premacro.parquet`

Summary path:

- `outputs/nq_macro_vwap_premacro_summary.parquet`

One row per ET trade date.

Anchors, all measured at the pre-macro checkpoint `< 15:50:00 ET`:

| Prefix | Anchor | Checkpoint |
| --- | --- | --- |
| `rth_0930` | `09:30:00 ET` | `< 15:50:00 ET` |
| `pm_1300` | `13:00:00 ET` | `< 15:50:00 ET` |
| `h3pm_1500` | `15:00:00 ET` | `< 15:50:00 ET` |

For each prefix, include:

- `{prefix}_vwap`
- `{prefix}_price`
- `{prefix}_vwap_dist_points`
- `{prefix}_vwap_dist_bps`
- `{prefix}_vwap_side`

Also include the three target groups described above.

A confluence feature should count side alignment across the three pre-macro anchors:

- `premacro_above_count`
- `premacro_below_count`
- `premacro_touch_count`
- `premacro_net_side_score = above_count - below_count`

## Feature set B: intramacro VWAP context

Output path:

- `outputs/nq_macro_vwap_intramacro.parquet`

Summary path:

- `outputs/nq_macro_vwap_intramacro_summary.parquet`

One row per ET trade date.

Anchors/checkpoints:

| Prefix | Anchor | Checkpoint | Research context |
| --- | --- | --- | --- |
| `macro_1550_at_1550_10s` | `15:50:00 ET` | `< 15:50:10 ET` | Initial 10-second barrier context |
| `macro_1550_at_1555` | `15:50:00 ET` | `< 15:55:00 ET` | EOII/imbalance-open context |
| `macro_1550_at_1600` | `15:50:00 ET` | `< 16:00:00 ET` | Full macro close context |
| `eoii_1555_at_1600` | `15:55:00 ET` | `< 16:00:00 ET` | 15:55 open VWAP holding into 15:59 |

For each prefix, include:

- `{prefix}_vwap`
- `{prefix}_price`
- `{prefix}_vwap_dist_points`
- `{prefix}_vwap_dist_bps`
- `{prefix}_vwap_side`

Also include the three target groups.

If `outputs/nq_macro_1550_barrier.parquet` exists, left-join by date and include neutral context columns:

- `barrier_macro_trend_state`
- `barrier_extreme`
- `barrier_price`
- `barrier_time`
- `barrier_first10`
- `barrier_is_macro_extreme`
- `barrier_holds`
- `barrier_edge_case`

Intramacro confluence features:

- `intramacro_above_count`
- `intramacro_below_count`
- `intramacro_touch_count`
- `intramacro_net_side_score = above_count - below_count`

## Diagnostic summaries

Summaries should be simple research diagnostics, not model training outputs.

For both A and B, emit long-form rows with:

- `feature_set`: `premacro` or `intramacro`
- `feature_name`
- `target_name`
- `scope`
- `bucket`
- `sample_size`
- `bullish_count`
- `bearish_count`
- `neutral_count`
- `bullish_pct`
- `bearish_pct`
- `neutral_pct`
- `avg_target_points`
- `median_target_points`

Summary scopes:

1. `side`
   - buckets: `above`, `below`, `touch`
2. `fixed_bps_band`
   - signed bps buckets, keeping `touch` separate:
     - `below_gt_20`
     - `below_10_20`
     - `below_5_10`
     - `below_2_5`
     - `below_0_2`
     - `touch`
     - `above_0_2`
     - `above_2_5`
     - `above_5_10`
     - `above_10_20`
     - `above_gt_20`
3. `decile`
   - per-feature quantile bucket from signed bps distance.
   - skip deciles for features with too few non-null unique values.
4. `confluence`
   - for A: `premacro_net_side_score`
   - for B: `intramacro_net_side_score`

For B only, add optional barrier context summary scopes when barrier columns are present:

- `barrier_first10_by_side`
- `barrier_holds_by_side`

These scopes should report target outcomes conditional on barrier context plus VWAP side. They must not claim confirmation.

## Proposed implementation shape

Create one focused module:

- `features/macro_vwap_features.py`

Public functions:

- `build_macro_vwap_premacro(path: str | Path = INPUT_PATH) -> pl.LazyFrame`
- `build_macro_vwap_intramacro(path: str | Path = INPUT_PATH, barrier_path: str | Path | None = DEFAULT_BARRIER_PATH) -> pl.LazyFrame`
- `summarize_macro_vwap_features(df: pl.DataFrame, feature_set: str) -> pl.DataFrame`
- `write_macro_vwap_features(...) -> tuple[Path, Path, Path, Path]`
- `main() -> None`

Internal helpers:

- schema validation with `get_tick_schema()`.
- required-column lazy scan selecting only required tick columns.
- ET date/time derivation.
- anchored VWAP aggregation.
- checkpoint price extraction.
- target construction.
- side classification.
- bps banding.
- decile summary construction.

Implementation should follow existing script-first project style used by `features/macro_1550_barrier.py`, `features/macro_extreme_timing.py`, `tick_density.py`, and `volume_delta.py`.

## Testing plan

Create:

- `test/test_macro_vwap_features.py`

Test cases:

1. **VWAP exact math**
   - Synthetic ticks from 09:30 to 15:49.
   - Assert `sum(price * size) / sum(size)` for each A anchor.
2. **Strict checkpoint boundaries**
   - Tick exactly at `15:50:00`, `15:50:10`, `15:55:00`, or `16:00:00` must not be included in the prior checkpoint.
3. **Multi-target signs**
   - Synthetic macro path where `15:50->15:54`, `15:55->15:59`, and full macro have different signs.
4. **Touch threshold**
   - `abs(dist_points) <= 0.25` maps to `touch`; larger distances map to `above`/`below`.
5. **Missing and zero-size windows**
   - Missing ticks or zero volume produce null VWAP/price/distance/side without crashing.
6. **DST sanity**
   - Summer-date ET window maps to the correct UTC hour.
7. **Barrier optional join**
   - Intramacro build works without a barrier path.
   - Intramacro build includes barrier context when the file exists.
8. **Writer outputs**
   - Writer creates all four parquet files and expected schemas.

Primary test command:

```bash
.venv/bin/python -m pytest test/test_macro_vwap_features.py -q
```

Runtime command:

```bash
.venv/bin/python -m features.macro_vwap_features
```

## Output files

Generated outputs:

- `outputs/nq_macro_vwap_premacro.parquet`
- `outputs/nq_macro_vwap_premacro_summary.parquet`
- `outputs/nq_macro_vwap_intramacro.parquet`
- `outputs/nq_macro_vwap_intramacro_summary.parquet`

Generated files should not overwrite source data in `input-data/`.

## Acceptance criteria

- All VWAP features are tick-derived.
- Full raw tick file is never eager-read.
- Feature set A and feature set B produce separate daily parquet files.
- Each feature set has a long-form diagnostic summary parquet.
- Targets cover `15:50->15:54`, `15:55->15:59`, and `15:50->15:59`.
- Price relation to VWAP includes points, signed bps, and side.
- Side uses one-tick touch threshold.
- Boundary behavior is strict: snapshot uses ticks before checkpoint, not at checkpoint.
- Tests cover math, boundaries, targets, nulls, DST, optional barrier join, and output writing.
