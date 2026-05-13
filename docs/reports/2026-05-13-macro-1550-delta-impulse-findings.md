# Macro 15:50 Delta Impulse Findings

Date: 2026-05-13
Branch/worktree: `feat/macro-1550-foundation`

## Study Scope

Question: does accumulated ETH-only, RTH-only, or ETH+RTH volume delta before 15:50 ET predict the initial 15:50:00-15:50:09 ET volume-delta impulse?

Inputs used:

- `outputs/nq_globex_volume_delta_1m.parquet`
- `outputs/nq_macro_volume_delta_5s.parquet`

Outputs:

- `outputs/nq_macro_1550_delta_impulse.parquet`
- `outputs/nq_macro_1550_delta_impulse_summary.parquet`

## Main Predictors

- `eth_only_pre350`: Globex/session minute index `0..929`.
- `rth_only_pre350`: session minute index `930..1309`, RTH through 15:49 ET.
- `eth_rth_pre350`: session minute index `0..1309`, full pre-15:50 accumulated delta.

## Primary Target

- `k350_00_09`: 15:50:00-15:50:09 ET, macro 5-second buckets `0..1`.

## Results

Runtime inputs were absent in this worktree, so output parquet files were not regenerated here. Fill this section from `outputs/nq_macro_1550_delta_impulse_summary.parquet` after runtime generation.

## Caveats

- Study uses 5-second volume-delta buckets, not raw order-type-level imbalance.
- No price target is included.
- Findings describe volume-flow relationships, not trade recommendations.
