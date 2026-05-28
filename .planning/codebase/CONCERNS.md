# Codebase Concerns

**Analysis Date:** 2026-05-24

This is a script-first Python research workspace for NQ/ES intraday macro-window (15:50-16:00 ET) analysis. Concerns below are grounded in the actual code, the roadmap (`roadmap.md`), idea backlog (`ideadump.md`), and documented experiment caveats under `docs/`.

## Tech Debt

**Output path naming drift / pipeline disconnect:**
- Issue: `session_tagger.py` writes `outputs/nq_1m.parquet` (`build_output_path` strips `_1m` then re-adds it → `nq_1m.parquet`), but `macro_outcomes.py` reads `INPUT_PATH = Path("outputs/nq_minute_base.parquet")`. These do not match, so the default pipeline is not runnable end-to-end without overriding paths. Documented in `AGENTS.md` ("naming drift") and `roadmap.md` Phase 1/6.
- Files: `session_tagger.py:23-25`, `macro_outcomes.py:21`
- Impact: Running `session_tagger.py` then `macro_outcomes.py` from a clean state fails (input not found). Tests pass because they inject paths/frames directly.
- Fix approach: Standardize on one canonical minute-base filename and update both scripts (and `AGENTS.md`) to agree.

**Hard-coded `../outputs/` relative paths (break from repo root):**
- Issue: Several scripts assume they are run from inside `viz/` or `features/`, using `../outputs/...`. Running them from the repo root (as `AGENTS.md` instructs) writes/reads the wrong location or fails.
- Files: `features/pm_macro_interactions.py:3-5`, `viz/pm_macro_viz.py:6`, `viz/macro_high.py:5`, `viz/viz_outcome.py:5`
- Impact: Inconsistent reproducibility; `viz/macro_high.py` and `viz/viz_outcome.py` also call `plt.show()` and assume `viz/`-relative paths (noted in `AGENTS.md`).
- Fix approach: Replace with repo-root-safe paths (e.g., resolve relative to `Path(__file__).resolve().parents[N]`), as `roadmap.md` Immediate Next Step #2 requests.

**`pm_macro_interactions.py` reads a non-existent / mismatched macro file:**
- Issue: `MACRO_PATH = "../outputs/macro_outcomes.parquet"`, but `macro_outcomes.py` writes `outputs/nq_macro_outcomes.parquet`. The expected join input never exists under that name.
- Files: `features/pm_macro_interactions.py:4`
- Impact: PM/HR3 → macro integration cannot run as written; `roadmap.md` Phase 2 confirms its output is not present in `outputs/`.
- Fix approach: Point to the real macro outcomes filename and parametrize the path.

**FVG study uses Python row-dict processing instead of vectorized Polars:**
- Issue: `features/macro_fvg_study.py` (1,202 lines, the largest module) implements outcome scanning, excursion math, grouping, percentiles, and summaries over `list[dict]` rows and per-bar Python loops (`scan_fvg_outcomes_until_1559_close`, `_group_rows`, `_percentile`, `_mean`, etc.) rather than Polars expressions. This contradicts the `AGENTS.md` "Polars by default, small composable functions" guidance.
- Files: `features/macro_fvg_study.py:359-635`
- Impact: Slow on larger universes, harder to maintain/verify, easy to introduce subtle aggregation bugs.
- Fix approach: Migrate aggregation/grouping to Polars group-by/quantile expressions; keep Python only at plotting boundary.

**Mixed Pandas/Polars in processing paths:**
- Issue: `features/macro_range_forecast.py` mixes Pandas, NumPy, sklearn, and Polars; `features/trend/modeling/*.py`, `features/trend/**/test_*.py`, and `viz/tick_density_viz.py` read with `pd.read_parquet`/`pd.read_csv`. `AGENTS.md` asks to keep Pandas out of new processing paths.
- Files: `features/macro_range_forecast.py:9-13`, `features/trend/modeling/table.py:335`, `features/trend/modeling/cli.py:98`, `viz/tick_density_viz.py:88,228`
- Impact: Two dataframe paradigms increase cognitive load and conversion overhead.
- Fix approach: Restrict Pandas to the visualization/sklearn boundary; standardize processing on Polars.

## Known Bugs

**`features/trend/test/test_historical_regimes.py` fails (canonical timestamp contract):**
- Symptoms: `test_main_writes_scores_and_label_to_parquet` raises `ValueError: Input bars must contain canonical UTC timestamp column: datetime_utc.`
- Files: `features/trend/modeling/table.py:35`, `features/trend/test/test_historical_regimes.py`
- Trigger: Run `.venv/bin/python -m pytest features/trend -q`. The historical-regimes path feeds the modeling table bars without a `datetime_utc` column, but `table.py` now requires it.
- Workaround: None in repo. The roadmap (Phase 6, ⚠️ Reproducibility hardening) already flags trend/LRLR verification as not clean end-to-end.
- Fix approach: Either normalize the historical-regime input to the canonical `datetime_utc` schema before `build_modeling_table`, or relax/adapt the contract for legacy ET inputs.

**Pytest "return not None" warnings in LRLR tests:**
- Symptoms: `test_lrlr_detection` and `test_tick_equal_detection` return values (tuple/dict) instead of asserting; pytest warns and future versions may error.
- Files: `features/lrlr/test/test_lrlr.py`
- Impact: These tests double as runnable analysis scripts but do not assert through pytest cleanly.
- Fix approach: Convert returns to `assert` statements or split the script entrypoint from the test function.

## Data-Correctness Risks

**Lookahead / target-endpoint leakage in intramacro features (acknowledged):**
- Risk: Many VWAP/intramacro features use information at or near the target endpoint (`<16:00`) and are descriptive labels, not pre-entry predictors. Documented in `docs/experiments/0001-macro-vwap-features.md` ("Caveats") and findings under `docs/reports/`.
- Files: `features/macro_vwap_features.py`, `features/macro_vwap_barrier_context.py`
- Current mitigation: `features/macro_range_forecast.py` defines `LEAKY_REALIZED_MACRO_COLUMNS = {"macro_bar_count","macro_high","macro_low","macro_open","macro_close"}` and excludes them from forecast features (`macro_range_forecast.py:22,129`). `features/pm_1h_macro_context.py:77` explicitly documents a "no-leak" context. History features in `add_history_features` correctly use `.shift(1)` before rolling stats (`macro_range_forecast.py:71-81`).
- Recommendations: The leak-awareness is uneven across modules. Add an explicit per-feature "available_before" timestamp/flag convention so downstream modeling cannot silently pull a same-window-endpoint feature into a predictor set. Treat barrier/VWAP confluence features as labels until a walk-forward predictor study validates them.

**In-sample-only, no walk-forward in most studies:**
- Risk: Most experiment outputs are exploratory/in-sample. `docs/experiments/0001-macro-vwap-features.md` states "Exploratory and in-sample only; no walk-forward validation." Only `features/trend/modeling/walkforward.py` and `features/macro_range_forecast.py` implement out-of-sample evaluation.
- Files: all `features/macro_*` summary studies; `viz/macro_analysis.py`
- Current mitigation: `walkforward.py` exists for trend modeling.
- Recommendations: Before promoting any descriptive edge to "tradeable" (a `roadmap.md` Phase 5 goal), require walk-forward / out-of-sample splits and report sample sizes (already an `AGENTS.md` experiment-log rule).

**Timezone-conversion inconsistency in `volume_delta.py`:**
- Risk: The 1-minute path uses `pl.col("ts_event").dt.replace_time_zone("UTC").dt.convert_time_zone(ET_TZ)` (`volume_delta.py:87-89`), while the 5-second path uses `pl.col("ts_event").dt.convert_time_zone(ET_TZ)` directly (`volume_delta.py:119`) without first selecting/casting to a UTC-aware dtype. `_scan_required_tick_columns` in `volume_delta.py:45-47` does not cast `ts_event` to `UTC_NS` (unlike `tick_density.py:62-67` which does). It works today only because the source schema is already `datetime[ns, UTC]` (`input-data/merged_ticks_schema.txt`).
- Files: `volume_delta.py:45-47,87-89,119`
- Current mitigation: Source schema documented as UTC-aware.
- Recommendations: Make tick TZ handling uniform (always cast `ts_event` to `UTC_NS` at scan, then a single `convert_time_zone(ET_TZ)` helper) so a future non-UTC input cannot silently mis-bucket macro seconds. DST correctness depends entirely on this conversion being correct.

**DST / ambiguous-timestamp handling for legacy ET inputs:**
- Risk: `normalize_minute_bars` accepts legacy `DateTime_ET`/`datetime_et` columns and converts with `strict=True`, raising on ambiguous DST-fallback values (`utils/minute_bars.py:28-58`). This is a deliberate guard but means legacy ET CSVs spanning the fall-back hour will hard-fail rather than disambiguate.
- Files: `utils/minute_bars.py:28-35,46-58`
- Current mitigation: Clear error message instructs providing `datetime_utc`.
- Recommendations: Acceptable as-is; document that callers must supply UTC for the DST transition days.

**Macro-window completeness filtering is uneven:**
- Risk: `features/macro_range_forecast.py` enforces `macro_bar_count == 10` and non-null gates (`macro_range_forecast.py:47-55`), and `features/macro_extreme_timing.py` keeps only dates with a complete key-minute set (per `AGENTS.md`). But `macro_outcomes.py:102-159` computes outcomes for any date with at least one MACRO row and does not require all 10 minutes present, so partial macro windows produce silently shorter ranges.
- Files: `macro_outcomes.py:95-127`
- Current mitigation: None in `compute_macro_outcomes`.
- Recommendations: Add an explicit macro-bar-count completeness filter (or a `macro_complete` flag column) to `macro_outcomes.py` so partial sessions are excluded or labeled.

**NaN/null propagation via `math.nan` in outcome rows:**
- Risk: `macro_outcomes.py` emits `math.nan` for `skew_ratio`/`close_in_range`/`postclose_*` when ranges are zero or post window is empty (`macro_outcomes.py:30,41,119-154`). Mixing `math.nan` floats into Polars frames is workable but mixes null and NaN semantics; downstream `drop_nulls` will not drop NaN.
- Files: `macro_outcomes.py:29-31,41-42,119-122,149-154`
- Current mitigation: Explicit `np.isnan` checks before casting.
- Recommendations: Prefer Polars nulls over `math.nan` for "not computable", or ensure all downstream consumers use `drop_nans`/`is_nan` rather than null checks.

## Performance Bottlenecks

**Tick studies convert timezone across the full file before window filtering:**
- Problem: `tick_density.py`, `volume_delta.py`, and `features/macro_tick_range_context.py` compute ET minute-of-day on every row of the multi-hundred-million-row tick file and only then filter to the 15:40-16:10 ET window. There is no bounded UTC pre-filter to prune row groups before the per-row `convert_time_zone`.
- Files: `tick_density.py:82-90`, `volume_delta.py:84-111`, `features/macro_tick_range_context.py`
- Cause: Filter predicate is on a derived ET column, so parquet predicate pushdown / row-group skipping cannot prune on the raw `ts_event` UTC range. `utils/tick_data.py` provides `_bounded_filter` / `collect_tick_window` for exactly this, but the macro scripts do not use a coarse UTC bound.
- Improvement path: Add a coarse `ts_event` UTC band filter (e.g., daily 19:40-20:10 UTC equivalents, widened for DST) before TZ conversion to enable row-group pruning; keep the precise ET filter afterward. `features/macro_vwap_features.py:137-140` already supports optional `start_utc`/`end_utc` bounds — generalize that pattern.

**Whole-suite runtime is slow:**
- Problem: `.venv/bin/python -m pytest test -q` runs 191 tests in ~248s (4m 8s); trend/LRLR adds ~50s.
- Files: `test/` (multiple tick-fixture-heavy tests)
- Cause: Several tests build/scan tick fixtures and exercise streaming collection.
- Improvement path: Mark slow tick tests (`@pytest.mark.slow`) and provide a fast default subset for inner-loop development; cache fixtures where possible.

**`build_macro_event_links` uses per-date Python iteration:**
- Problem: `utils/helper.py:144-165` loops over every macro date and iterates news rows with `iter_rows`, appending dicts.
- Files: `utils/helper.py:156-165`
- Cause: Row-wise construction of event-link rows in Python.
- Improvement path: Replace with a Polars join on `date_et` and a vectorized minutes-from-close computation; acceptable for the current 1,567-row events table but will not scale.

## Fragile Areas

**`utils/minute_bars.py` timestamp normalization:**
- Files: `utils/minute_bars.py:13-79`
- Why fragile: Single choke point that all minute pipelines depend on; accepts four timestamp column spellings (`datetime_utc`, `DateTime_UTC`, `DateTime_ET`, `datetime_et`), enforces no-duplicate and no-null invariants, and silently chooses behavior by column presence. A column-name change anywhere upstream changes the conversion path.
- Safe modification: Keep the canonical-output contract (`BASE_COLUMNS`); add tests for each accepted input spelling before editing. Covered by `test/test_minute_bars.py`.
- Test coverage: Good for happy path; less coverage of DST-ambiguous legacy ET.

**Tick `side` encoding assumption:**
- Files: `volume_delta.py:54-66`, `tick_density.py:70-77`, `AGENTS.md`
- Why fragile: Buy/sell/none logic hard-codes `side == 2` (buy), `1` (sell), `0` (none). If a future tick export changes the encoding, all delta/density signs flip silently with no validation.
- Safe modification: Add an assertion that observed `side` values ⊆ {0,1,2} during schema validation.
- Test coverage: `test/test_volume_delta.py`, `test/test_tick_density.py` exist but assume the fixed encoding.

**Trend modeling subpackage schema contract:**
- Files: `features/trend/modeling/table.py:35`, `features/trend/test/test_historical_regimes.py`
- Why fragile: `build_modeling_table` now requires `datetime_utc`, but the historical-regimes feeder does not supply it (the active failing test). The contract change was not propagated to all callers.
- Safe modification: Audit every caller of `build_modeling_table` for the canonical-timestamp requirement before changing it again.

## Security Considerations

**No secrets, network, or dynamic-execution surface detected:**
- Risk: Low. No `requests`/`urllib`/`http`, no `os.system`/`subprocess`, no `eval`/`exec`/`pickle.load`, and no `api_key`/`token`/`password` literals were found in `features/`, `viz/`, `utils/`, or root scripts.
- Files: n/a
- Current mitigation: `.gitignore` excludes `.env`, `.env.*`, `input-data/`, `outputs/`, and virtualenvs (`.gitignore:14-34`). No `.env` file present in tree.
- Recommendations: The ForexFactory scraper lives in a separate repo (`roadmap.md` Phase 4); if it is ever vendored in, review it separately for network/credential handling. Keep `input-data/` untracked so raw market data is not committed.

## Dependencies at Risk

**Aggressive lower-bound pins on fast-moving libraries:**
- Risk: `requirements.txt` pins floors well above common stable releases: `polars>=1.40`, `pyarrow>=24`, `numpy>=2`, `scikit-learn>=1.8`, `scipy>=1.17`, `pytest>=9`, `xgboost>=2.0`. These are upper-edge/near-future versions; environments with older installs will not satisfy them, and there is no upper bound to protect against breaking majors.
- Files: `requirements.txt:1-9`
- Impact: Reproducibility depends on a narrow, recent toolchain; `roadmap.md` notes broader trend/LRLR tests "fail in this environment due to missing dependencies/import path issues."
- Migration plan: Pin tested exact versions (or add upper bounds) and capture a lockfile; document the verified interpreter (no `.python-version` is committed — it is gitignored).

## Missing Critical Features

**No single canonical daily feature dataset:**
- Problem: The repo produces many separate parquet products (macro outcomes, PM/HR3, FVG events, VWAP, delta, tick density) but no unified "tagged bars → PM/HR3 → macro outcomes → news flags" table. `roadmap.md` Phase 2 and Immediate Next Step #1 call this the next milestone.
- Blocks: Cross-feature modeling, consistent joins, and reproducible studies.

**News/`is_news_day`/`is_holiday` flags not persisted to main outputs:**
- Problem: `utils/helper.py` classifies events and builds joins, and `test/cpi.py` uses them, but `roadmap.md` Phase 4 confirms news flags are not first-class in `outputs/`.
- Blocks: News-segmented macro studies (a recurring `ideadump.md` theme).

**Open research items from `ideadump.md` not yet implemented:**
- Markov chain of macro/5m-candle states (`ideadump.md` 12-01, `roadmap.md` Phase 3) — not implemented.
- SMT-during-3PM directional skew study (`ideadump.md` 09-11) — not found.
- Contract expiry/rollover volatility effect on macro (`ideadump.md` 24-11) — not found.
- PM → next-day predictiveness; 3PM consolidation/mean-reversion labels (`roadmap.md` Phase 2/3) — blocked by missing consolidation labels.

## Test Coverage Gaps

**Untested integration glue:**
- What's not tested: `features/pm_macro_interactions.py` (hard-coded paths, no `test_pm_macro_interactions.py` present), and `viz/macro_high.py` / `viz/viz_outcome.py` (manual `plt.show()` scripts).
- Files: `features/pm_macro_interactions.py`, `viz/macro_high.py`, `viz/viz_outcome.py`
- Risk: The PM→macro merge can silently break (and currently points at a non-existent file) with no test to catch it.
- Priority: Medium.

**Macro-window completeness not asserted in `macro_outcomes`:**
- What's not tested: Behavior when the MACRO window has fewer than 10 minutes.
- Files: `macro_outcomes.py:95-127`, `test/test_macro_outcomes.py`
- Risk: Partial sessions produce undersized ranges with no warning.
- Priority: Medium.

**Trend/LRLR suite not green:**
- What's not tested cleanly: `features/trend/test/test_historical_regimes.py` fails (schema contract); LRLR tests emit return-value warnings.
- Files: `features/trend/test/`, `features/lrlr/test/test_lrlr.py`
- Risk: Regressions in the trend/regime stack go undetected because the suite is known-red and may be skipped.
- Priority: High (a red baseline erodes trust in the whole suite).

**Boundary / session-tagging sanity checks absent:**
- What's not tested: No explicit session/window boundary-validation script (`roadmap.md` Phase 1 lists "Boundary sanity checks" as ⬜ not done), e.g., verifying H3PM/MACRO/POST minute coverage per day across DST.
- Files: `utils/minute_bars.py:95-118`, `test/test_session_tagger.py`
- Risk: Off-by-one or DST-shifted window edges could mislabel macro minutes without detection.
- Priority: Medium.

---

*Concerns audit: 2026-05-24*
</content>
</invoke>
