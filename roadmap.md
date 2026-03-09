# Close Macro Research Roadmap

_Last reviewed: 2026-03-09_

## Current Position

The project is past the original data-foundation stage and partway through the feature/research stage.

- The tagged intraday data pipeline exists and has already produced `outputs/nq_1m.parquet`, `outputs/es_1m.parquet`, and `outputs/nq_macro_outcomes.parquet`.
- PM / 3 PM / macro interaction work is implemented in code, but not yet unified into one canonical daily dataset.
- A macro FVG research track is already further along than the original roadmap and is generating event, summary, and figure outputs.
- The main gaps are pipeline integration, news wiring, path cleanup, and consistent verification.

## Phase 1 - Data Foundation

Status: mostly done, with timestamp/path cleanup still open.

| Status | Task | Notes |
| --- | --- | --- |
| ✅ | Historical minute data in repo | `input-data/nq_1m.csv` and `input-data/es_1m.csv` exist. |
| ✅ | Session/window tagging | `session_tagger.py` writes tagged parquet outputs for NQ and ES. |
| 🔄 | UTC -> ET-naive flow | Current tagging workflow uses UTC -> ET-naive conversion before session/window tagging, but the roadmap still needs clearer wording on how much of that flow is verified end-to-end. |
| ⬜ | CLI to slice data by session | Not present in the current top-level workflow. |
| ⬜ | Boundary sanity checks | No explicit boundary-validation script or test coverage found. |

Current output: per-instrument tagged parquet exists, but the original "single session_tagger.parquet" wording is no longer accurate.

## Phase 2 - Core Features

Status: partially done; implementation exists, integration is incomplete.

| Status | Task | Notes |
| --- | --- | --- |
| ✅ | Macro outcomes | `macro_outcomes.py` produces `outputs/nq_macro_outcomes.parquet`. |
| ✅ | PM + 3 PM feature extraction | `features/pm_3pm.py` computes daily PM / HR3 feature sets. |
| ✅ | High/low timing tags | PM and HR3 high/low timing fields are already in `features/pm_3pm.py`. |
| 🔄 | PM / 3 PM -> macro integration | `features/pm_macro_interactions.py` merges PM/HR3 with macro outcomes, but its output is not present in `outputs/` and paths are still hard-coded. |
| ⬜ | Consolidation logic | No finished HR3 consolidation / mean-reversion label yet. |
| ⬜ | One canonical daily feature parquet | Still missing. The repo currently produces separate datasets, not one consolidated feature table. |

## Phase 3 - Research Modules

Status: started. Descriptive analysis exists, but the original roadmap questions are only partly covered.

| Status | Research Focus | Notes |
| --- | --- | --- |
| ⬜ | Markov chain of macro states | Not implemented yet. |
| 🔄 | Range clustering / regime context | `viz/macro_analysis.py` and `outputs/figs/ma/` cover descriptive distributions and rolling context, but not a finished clustering workflow. |
| ⬜ | PM -> next-day predictiveness | Not found as a completed study. |
| ⬜ | 3 PM consolidation effect | Blocked by missing consolidation labels. |
| 🔄 | News overlap impact | There is join logic and a CPI analysis script, but not a finished integrated study for the macro/3 PM windows. |
| ✅ | Macro FVG study (added beyond original roadmap) | `features/macro_fvg_study.py`, `outputs/nq_macro_fvg_events.parquet`, `outputs/nq_macro_fvg_summary.parquet`, and `outputs/figs/fvg/` already exist. |

## Phase 4 - News Integration

Status: partially done; the data and helper layer exist, but the main pipeline is not wired through.

| Status | Task | Notes |
| --- | --- | --- |
| 🔄 | ForexFactory scraper available externally | The scraper exists, but it lives in a different repo rather than this workspace. |
| ✅ | Economic events parquet | `input-data/economic_events.parquet` exists and currently contains 1,567 rows. |
| ✅ | News normalization / join helper | `utils/helper.py` classifies events and builds daily or event-level joins. |
| 🔄 | Drop irrelevant speeches / low-impact events | Implemented as filters in the helper layer, but not yet promoted into one canonical output dataset. |
| 🔄 | Merge events with session or macro data | Helper functions exist and `test/cpi.py` uses them, but the merge is not part of the main pipeline. |
| ⬜ | Persist `is_news_day` / `is_holiday` flags | Not found in the main outputs. |

## Phase 5 - Interpretation Layer

Status: started, but script-based rather than notebook/card-based.

| Status | Task | Notes |
| --- | --- | --- |
| 🔄 | Simple research summaries | Visual analysis scripts exist, but not the one-page notebook flow described in the original roadmap. |
| ✅ | Visualize findings | `viz/` scripts and generated figures already exist for macro stats and FVG work. |
| ⬜ | Insight cards | Not found. |
| ⬜ | Rank insights by stability / tradeability | Not found. |

## Phase 6 - Maintenance / Workflow

Status: active cleanup needed.

| Status | Task | Notes |
| --- | --- | --- |
| 🔄 | Keep structure tidy | The repo is usable, but path conventions are inconsistent: some scripts expect `data/`, others use `input-data/`, and some feature/viz scripts hard-code `../outputs/`. |
| ⬜ | Project tracking workflow | No repo-local evidence of a project board or issue workflow. |
| 🔄 | Summarize progress back into roadmap | This roadmap update does that, but it is not yet automated or habitual. |
| ⚠️ | Reproducibility hardening | Verification is not clean end-to-end: the close-macro FVG suite is close but currently has one failing figure expectation, and broader trend/LRLR tests fail in this environment due to missing dependencies/import path issues. |

## What The Repo Says We Are Doing Right Now

The practical current state is:

1. Stable enough tagged intraday data and macro outcomes already exist.
2. PM / 3 PM / macro interaction code exists but still needs to be made into one reproducible feature pipeline.
3. Macro FVG analysis is the most advanced active research branch beyond the original roadmap.
4. News data is available and helper logic exists, but it is not yet first-class in the main outputs.
5. The next real milestone is not "start research" but "consolidate the data products and make the research workflow reproducible."

## Immediate Next Steps

1. Finish the canonical daily dataset: tagged bars -> PM/HR3 features -> macro outcomes -> news flags.
2. Replace hard-coded relative paths with repo-root-safe paths so feature and viz scripts run consistently from the project root.
3. Decide whether trend/LRLR work belongs on this roadmap or should be tracked as a separate research track.
4. Promote one research question into a clean reproducible study output: Markov transitions, PM -> next-day behavior, or news segmentation.
