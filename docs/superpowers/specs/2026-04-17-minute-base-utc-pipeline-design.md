# Minute Base UTC Pipeline Refactor Design

## Goal

Clean up the minute-bar pipeline so the repository has one canonical per-instrument base dataset, stored in UTC with a real datetime dtype, while downstream research outputs remain separate. The refactor removes persisted ET/session/window columns from the base parquet and makes scripts derive New York market time in memory only when needed.

## Current Problems

- The current workflow is split across multiple prep scripts before analysis can start.
- The current base parquet is not actually canonical: it stores `DateTime_ET`, a string-like `DateTime_UTC`, and precomputed `session` / `window`.
- Several consumers depend on ET-naive wall-clock assumptions.
- The current output name `outputs/nq_1m.parquet` implies a raw minute dataset, but it is really a tagged derivative.
- The current schema mixes durable data fields with derived analysis helpers.

## Design Decisions

### Canonical Base Contract

The canonical stored base dataset is one per-instrument parquet in `outputs/` with only durable minute-bar fields:

- `datetime_utc`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- optional `instrument`

Rules:

- `datetime_utc` is the canonical time column.
- `datetime_utc` is stored as UTC `datetime64`, not as a string.
- The base parquet does not persist `DateTime_ET`, `DateTime_UTC`, `datetime_et`, `session`, or `window`.
- User-facing market-time logic uses `America/New_York`, not fixed UTC-5. This is DST-safe.

### Base File Names

The new canonical per-instrument outputs are:

- `outputs/nq_minute_base.parquet`
- `outputs/es_minute_base.parquet`

The old tagged-base naming pattern such as `outputs/nq_1m.parquet` is retired for the canonical path.

### Entry Point

`session_tagger.py` remains the top-level entry point in this minimal patch, but its behavior changes from "tag and persist ET/session/window" to "build canonical UTC minute base parquet."

The script name is legacy and not ideal, but renaming it is not required for this patch. Behavior and schema matter more than the file name in this phase.

## Input Handling

The base builder accepts both `.csv` and `.parquet` input.

Accepted input timestamp columns:

1. `datetime_utc` preferred
2. `DateTime_UTC` accepted as legacy input
3. `DateTime_ET` accepted as legacy input and converted to UTC before writing

Input normalization rules:

- If the input provides UTC, normalize it to canonical `datetime_utc`.
- If the input provides ET-local timestamps, interpret them as New York market time and convert to UTC.
- Duplicate or ambiguous timestamp fields are resolved into a single canonical `datetime_utc` column before write.
- Required price/volume columns are validated before output is written.

## Shared Time Helpers

Add shared loader/time utilities under `utils/`.

The shared helper layer is responsible for:

- loading `.csv` or `.parquet`
- normalizing legacy timestamp schemas into canonical `datetime_utc`
- converting `datetime_utc` to in-memory New York local datetime when needed
- deriving `session` and `window` in memory when needed

The helper layer is the compatibility choke point. ET- and window-aware consumers should stop open-coding timestamp parsing and clock-window logic where possible.

## Downstream Consumer Behavior

### `macro_outcomes.py`

`macro_outcomes.py` must stop requiring persisted `window` and `DateTime_ET` from the base parquet.

New behavior:

- load canonical base data
- derive New York local datetime in memory
- derive the relevant `window` labels in memory
- compute macro and post-close slices from those derived labels
- write `outputs/nq_macro_outcomes.parquet` and `outputs/es_macro_outcomes.parquet`

The output parquet remains separate from the base parquet.

### `features/macro_fvg_study.py`

`features/macro_fvg_study.py` must stop assuming the base parquet already contains `window` and `DateTime_ET`.

New behavior:

- load canonical base data
- derive New York local datetime in memory
- derive `window == "MACRO"` in memory
- keep all FVG detection, staging, and scan logic anchored to New York market time
- continue writing separate event and summary parquets

Outputs remain:

- `outputs/nq_macro_fvg_events.parquet`
- `outputs/nq_macro_fvg_summary.parquet`

### `features/pm_3pm.py`

`features/pm_3pm.py` must support canonical `datetime_utc` input.

New behavior:

- load canonical base data or compatible legacy inputs
- derive New York local datetime in memory
- compute PM and HR3 masks from New York local time
- use derived `session` only in memory if that simplifies existing logic

Its output remains separate.

### `viz/macro_analysis.py`

`viz/macro_analysis.py` must support the new canonical base schema by converting UTC to New York local time before stage slicing.

This is a read-side compatibility update, not part of the base contract.

## Out of Scope

These items are explicitly out of scope for this patch:

- converting every trend/LRLR fixture in the repository to UTC-native schemas
- broad package restructuring
- merging all downstream feature outputs into one daily parquet
- changing downstream research questions or output semantics
- refactoring `features/trend/test/generate_regime_charts.py`

Trend/LRLR fixture updates are not part of this patch. If a touched path breaks them, that follow-up should be handled as a separate cleanup change rather than expanded into this refactor.

## Migration Strategy

### Canonical-First

Touched scripts should support the new canonical base schema first.

Compatibility rules:

- ingest paths may still accept legacy timestamp columns
- canonical write path always emits `datetime_utc`
- no new code should depend on persisted ET/session/window in the base parquet

### Legacy Read Compatibility

Where useful during the transition, shared helpers may accept both:

- new canonical `datetime_utc`
- old legacy `DateTime_ET` / `DateTime_UTC`

This compatibility is a migration bridge for readers, not a reason to preserve the old base schema.

## Error Handling

The refactor should fail early on schema errors.

Required validation:

- missing required OHLCV fields
- missing timestamp field with no supported fallback
- unparsable timestamps
- duplicate canonical timestamps after normalization
- empty macro-window slices where a script requires them

Error messages should name the missing or invalid columns directly.

## Testing Strategy

Testing for this patch focuses on touched pipeline layers first.

Required coverage:

- loader tests for UTC normalization and New York conversion
- `macro_outcomes.py` tests for UTC-backed window derivation
- `features/macro_fvg_study.py` tests updated to derive windows in memory instead of requiring persisted `window`
- `features/pm_3pm.py` tests or smoke coverage for UTC-backed PM/HR3 slicing

Verification targets:

- canonical base builder writes UTC datetime dtype, not string
- no persisted ET/session/window fields remain in the base parquet
- macro outcomes still group by the correct New York trading date
- FVG stage and scan boundaries remain correct across DST-safe New York conversion

## Risks

### ET-Naive Drift

The main correctness risk is applying New York window logic directly to UTC timestamps. All hour/minute filtering must happen after conversion to `America/New_York`.

### Mixed Legacy Paths

Some scripts, docs, and fixtures still assume `DateTime_ET`. The shared helper should isolate that transition so touched pipeline scripts do not each invent their own conversion rules.

### Naming Confusion

`session_tagger.py` becomes a base builder rather than a tag-persistence script. This is acceptable in the minimal patch, but the mismatch should be documented.

## Success Criteria

The refactor is successful when:

- one run of `session_tagger.py` produces per-instrument canonical UTC minute-base parquets
- those base parquets contain `datetime_utc` plus durable OHLCV fields only
- `macro_outcomes.py`, `features/macro_fvg_study.py`, and `features/pm_3pm.py` read the canonical base directly
- no analysis step requires a persisted ET/session/window base parquet
- downstream outputs remain separate and preserve their current roles
