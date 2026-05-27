---
type: concept
tags: [infra/pipeline]
---
# Data pipeline

The chain of parquet stages, each script consuming the previous stage's output:

1. `session_tagger.py` normalises `input-data/{nq,es}_1m.csv` → canonical minute parquet.
2. `macro_outcomes.py` computes the daily [[macro-outcome]] table from canonical bars.
3. Tick studies (`tick_density.py`, `volume_delta.py`, most `features/macro_*`) read the large
   tick parquet via [[tick-data]] helpers.
4. `features/` scripts produce derived tables (`outputs/nq_macro_<feature>.parquet`).
5. `viz/` turns derived parquet into CSV summaries + figures under `outputs/figs/`.

**Naming-drift gotcha:** `session_tagger.py` writes `outputs/nq_1m.parquet` but
`macro_outcomes.py` defaults to `outputs/nq_minute_base.parquet`.

**Related.** [[time-handling]] · [[tick-data]] · [[macro-outcome]] · [[polars-data-system]]
