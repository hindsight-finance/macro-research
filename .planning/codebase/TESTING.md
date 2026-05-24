# Testing Patterns

**Analysis Date:** 2026-05-24

## Test Framework

**Runner:**
- `pytest` (>=9, pinned in `requirements.txt`).
- No `pytest.ini`, `pyproject.toml`, `setup.cfg`, or `tox.ini` config detected.
  pytest runs with defaults; test paths are passed explicitly on the command
  line. The only pytest config artifact is the cache dir `.pytest_cache/`.

**Assertion Library:**
- Plain `assert` statements (pytest rewriting). No separate assertion library.
- `pytest.approx` for float comparisons (used ~115 times across the suite),
  e.g. `assert flat["descriptive_target"] == pytest.approx(0.2)`.

**Run Commands (always via the project virtualenv — never system python):**
```bash
.venv/bin/python -m pytest test -q                                              # main suite
.venv/bin/python -m pytest test/test_tick_data.py test/test_tick_density.py \
    test/test_volume_delta.py -q                                                # tick-data tests
.venv/bin/python -m pytest test/test_macro_extreme_timing.py \
    test/test_macro_extreme_timing_viz.py -q                                    # macro extreme timing
.venv/bin/python -m pytest features/lrlr/test features/trend -q                 # deeper feature packages
.venv/bin/python -m features.lrlr.test.test_lrlr --no-viz                       # LRLR detector check, no charts
```

No coverage tooling (`pytest-cov`, `coverage`) is configured.

## Test File Organization

**Location — two-tier layout:**
- `test/` at the repository root holds the main suite (26 files) for root entry
  scripts, `utils/`, top-level `features/` modules, and `viz/` scripts.
  Examples: `test/test_minute_bars.py`, `test/test_macro_outcomes.py`,
  `test/test_tick_density.py`, `test/test_tick_density_viz.py`.
- Deeper feature subpackages keep tests **beside the code they validate**, in a
  nested `test/` (or `testing/` / `Testing/`) directory. Examples:
  - `features/lrlr/test/test_lrlr.py` (+ fixture `features/lrlr/test/NQ.csv`)
  - `features/trend/modeling/test/test_target.py`,
    `test_table.py`, `test_labels.py`, `test_target.py`, `test_cli.py`,
    `test_walkforward.py`, `test_registry.py`, `test_init.py`,
    `test_containment_research.py`
  - `features/trend/ADX/testing/test_adx.py`,
    `features/trend/ATR Range/Testing/test_atr.py`,
    `features/trend/variance_ratio/test/test_vr.py`, etc.

**Naming:**
- `test_*.py` files; test functions are `test_*`. `AGENTS.md` requires
  `pytest`-compatible `test_*.py` files even when a test also doubles as a
  runnable analysis script.
- Test names are long and descriptive, encoding the asserted behavior:
  `test_build_macro_tick_density_covers_1540_to_1610_et_with_actual_et_minute_index`,
  `test_normalize_minute_bars_rejects_ambiguous_dst_fallback_et_inputs`.

**conftest:**
- A single `test/conftest.py` (21 lines) does environment setup only — it
  prefers the venv `mpl_toolkits.mplot3d` to avoid a namespace-package clash.
  It defines **no fixtures**.

## Test Structure

Tests are flat module-level functions (no test classes for grouping). The
dominant pattern is: build an in-memory Polars frame inline, call the function,
assert on the returned frame's `columns`, `height`, and specific cell values.

```python
def test_compute_macro_outcomes_uses_datetime_utc_and_derived_macro_window():
    bars = pl.DataFrame(
        {
            "datetime_utc": ["2020-09-01 19:49:00+00:00", "2020-09-01 19:50:00+00:00", ...],
            "Open": [100.0, 101.0, ...], "High": [...], "Low": [...],
            "Close": [...], "Volume": [...],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))

    out = compute_macro_outcomes(bars, macro_window_name="MACRO")

    assert isinstance(out, pl.DataFrame)
    assert out.height == 1
    assert out.item(0, "macro_open") == 101.0
    assert out.item(0, "postclose_range_points") == 1.5
```

**Assertion idioms (Polars):**
- `out.columns == MACRO_TICK_DENSITY_COLUMNS` — assert full output schema
  against the module's pinned `*_COLUMNS` constant.
- `out.item(row, col)`, `out.height`, `out.row(idx, named=True)`,
  `out.select(col).to_series().to_list()`, `out.select(col).unique().item()`.

## Fixtures and Factories

**No `@pytest.fixture` decorators are used anywhere in the suite.** Test data is
produced two ways:

1. **Inline literal Polars/Pandas frames** (the most common — see above).

2. **Local underscore-prefixed helper builders** that construct frames or write
   fixture parquet files:
   - `_write_ticks(path, rows)` writes a tick parquet with the correct
     UTC-ns timestamp dtype (`test/test_tick_density.py:16`,
     `test/test_macro_extreme_timing.py:14`).
   - `_make_ohlc(close_values, wick=0.1)` / `_make_ohlcv(...)` build numpy
     OHLC(V) arrays for numeric feature tests
     (`features/trend/modeling/test/test_target.py:13`).
   - `_fixture_minute_bars(start, days)` builds a synthetic minute-bar frame
     (`test/test_macro_range_forecast.py:39`).

**`tmp_path` is the standard filesystem fixture.** I/O round-trip tests take
`tmp_path: Path`, write inputs there, run the `write_*` function, and assert on
the returned `Path` and the re-read parquet:

```python
def test_write_macro_tick_density_writes_parquet_from_lazy_plan(tmp_path: Path):
    input_path = tmp_path / "ticks.parquet"
    output_path = tmp_path / "density.parquet"
    _write_ticks(input_path, {...})
    wrote = write_macro_tick_density(input_path, output_path)
    assert wrote == output_path
    out = pl.read_parquet(output_path)
    assert out.columns == MACRO_TICK_DENSITY_COLUMNS
```

**Committed fixture files** live beside their tests, e.g.
`features/lrlr/test/NQ.csv`. Most tests generate fixtures into `tmp_path`
rather than committing them.

## Mocking

Mocking is rare — only 3 files use it
(`test/test_macro_fvg_study.py`, `test/test_macro_range_forecast.py`,
`features/lrlr/test/test_lrlr.py`) and the tool is always pytest's
**`monkeypatch`** fixture (no `unittest.mock` / `Mock` / `patch`).

**Patterns observed:**
- Inject a fake heavy dependency by swapping the module:
  ```python
  def _install_fake_xgboost(monkeypatch):
      FakeXGBRegressor.fit_calls = []
      monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=FakeXGBRegressor))
  ```
  A hand-rolled `FakeXGBRegressor` class records `fit_calls` so tests can assert
  call counts (`test/test_macro_range_forecast.py:18-37`), e.g. that monthly
  walk-forward refits XGBoost once per holdout month.
- Patch matplotlib to capture plotting without rendering:
  `monkeypatch.setattr(macro_fvg_study.plt, "close", capture_close)`
  (`test/test_macro_fvg_study.py:1842`).

**What to mock:** expensive/optional ML libs (xgboost) and the matplotlib
boundary. **What NOT to mock:** Polars/NumPy data transforms and the real
parquet round-trip — these are exercised directly with `tmp_path`.

## Coverage / What Tests Assert (per `AGENTS.md`)

For new data-processing functions, cover **both**:
- Happy-path calculations (exact expected values).
- Schema failures — `pytest.raises(ValueError, match=...)` asserting the
  validation message. `pytest.raises` is used ~22 times;
  `pytest.mark.parametrize` (4 uses, e.g.
  `features/trend/modeling/test/test_target.py:72`) sweeps invalid inputs like
  zero/negative/NaN/inf OHLC.

For **tick studies**, tests must also cover:
- **DST handling** — assert UTC hour shifts between a winter and summer date
  (`test_build_macro_tick_density_uses_utc_only_and_handles_dst_macro_hour`).
- **Bounded-window filtering** — only rows inside the 15:40–16:10 ET window
  survive (`..._covers_1540_to_1610_et_...`).
- **First-touch / tie behavior** — earliest-UTC extreme, ties resolved
  `high_ts <= low_ts` (`test_build_macro_extreme_timing_uses_first_touch_seconds...`).
- **Empty-bucket preservation** — full 12-bucket / 24-bucket grids retained even
  when empty (`..._includes_empty_buckets_without_macro_minute_column`).

## Test Types

**Unit / behavioral:** the vast majority — pure functions over small synthetic
frames, asserting exact values and schema.

**I/O round-trip:** `write_*` functions write to `tmp_path` and re-read with
`pl.read_parquet` / `pd.read_parquet`.

**Visualization tests:** `viz/*_viz.py` tests run the plot pipeline headless
(Matplotlib `Agg`) and assert that expected figure/CSV artifacts exist and that
summary stats are correct, rather than inspecting pixels:
```python
def test_process_dataset_writes_band_outputs_only(tmp_path: Path):
    process_dataset(path, out_dir=out_dir)
    assert (out_dir / "sample_bands.png").exists()
    assert (out_dir / "sample_band_stats.csv").exists()
    assert not (out_dir / "sample_hist.png").exists()
```
`AGENTS.md` requires: *if a change produces plots, add a non-visual assertion
path so the logic can run in CI or headless shells.*

**E2E pipeline tests:** none. Tests target individual importable functions, not
full `input-data → outputs` script runs. Real tick inputs are never read in
tests.

## Common Patterns

**Async testing:** Not applicable — codebase is synchronous.

**Error testing:**
```python
with pytest.raises(ValueError, match="Missing tick columns"):
    build_macro_tick_density(path)

with pytest.raises(ValueError, match="close must contain only positive values"):
    build_descriptive_target(open_=..., high=..., low=..., close=np.array([100.0, 0.0, 101.0]))
```

**Parametrized invalid input:**
```python
@pytest.mark.parametrize("close", [np.array([100.0, 0.0, 101.0]), np.array([100.0, -1.0, 101.0])])
def test_descriptive_target_rejects_non_positive_close(close):
    with pytest.raises(ValueError, match="close must contain only positive values"):
        build_descriptive_target(..., close=close)
```

---

*Testing analysis: 2026-05-24*
