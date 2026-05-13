# FVG Delta MAE/MFE Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add useful MAE/MFE profile outputs for the delta-dominance features, covering both touched/entry-triggered FVGs and successful FVGs without producing excessive charts.

**Architecture:** Reuse existing macro FVG summary machinery in `features/macro_fvg_study.py`. Add entry-excursion scopes for absolute delta dominance by quantile, minute block, and creation minute; success-context scopes already carry MFE/MAE metrics. Add a focused profile export function that writes three CSV files and two PNG figures from the summary table.

**Tech Stack:** Python, Polars, Matplotlib Agg, pytest. Use `.venv/bin/python`.

---

## File Structure

- Modify: `features/macro_fvg_study.py`
  - Add entry-excursion summary builders for absolute delta dominance.
  - Add CSV export helpers and two focused plot helpers.
  - Call exports from `plot_fvg_summary_figures()`.
- Modify: `test/test_macro_fvg_study.py`
  - Add tests for new summary scopes and CSV/figure outputs.

## Scope

Include only absolute delta dominance because earlier analysis showed it was cleaner than aligned delta. Include:

- `entry_excursion_abs_delta_imbalance_quantile`
- `entry_excursion_minute_block_abs_delta_imbalance_quantile`
- `entry_excursion_creation_minute_abs_delta_imbalance_quantile`

Export CSV profiles:

- `delta_dominance_mae_mfe_abs_quantile.csv`
- `delta_dominance_mae_mfe_by_block.csv`
- `delta_dominance_mae_mfe_by_creation_minute.csv`

Export figures:

- `delta_abs_quantile_mfe_mae_profile.png`
- `early_block_abs_delta_mfe_mae_profile.png`

### Task 1: Add failing tests for entry-excursion delta profile scopes

**Files:**
- Modify: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Add a fixture helper for MAE/MFE profile events**

Append after `make_delta_5s_for_dominance()`:

```python
def make_delta_profile_events_for_excursion():
    return enrich_fvg_events_with_delta_dominance(
        make_delta_events_for_dominance(),
        make_delta_5s_for_dominance(),
    ).with_columns(
        pl.lit("stage_1").alias("assigned_stage"),
        pl.Series("assigned_minute_index", [0, 0, 5, 5]),
        pl.Series("assigned_minute_hhmm", ["15:50", "15:50", "15:55", "15:55"]),
        pl.lit(100).alias("bar2_volume"),
        pl.lit("3_aligned").alias("alignment_bucket"),
        pl.Series("minute_block", ["15:50-15:52", "15:50-15:52", "15:53-15:57", "15:53-15:57"]),
        pl.lit(">=2.25").alias("gap_size_bucket_225"),
        pl.Series("entry_triggered_by_1559", [True, True, True, False]),
        pl.Series("mfe_pct_to_1559", [0.010, 0.020, 0.030, float("nan")]),
        pl.Series("mae_pct_to_1559", [0.002, 0.003, 0.004, float("nan")]),
        pl.lit(True).alias("held_to_1559_close"),
        pl.lit(False).alias("invalidated_by_1559"),
        pl.lit(False).alias("untouched_to_1559_close"),
        pl.lit(False).alias("retraced_in_stage_2"),
        pl.lit(False).alias("invalidated_in_stage_2"),
        pl.lit(True).alias("held_through_stage_2"),
        pl.lit(False).alias("untouched_through_stage_2"),
        pl.lit(False).alias("stacked_continuation_fvg"),
    )
```

- [ ] **Step 2: Add a failing summary-scope test**

Append before `test_run_macro_fvg_study_writes_parquet_and_figures`:

```python
def test_build_summary_tables_includes_delta_mae_mfe_entry_excursion_profiles():
    summary = macro_fvg_study.build_summary_tables(make_delta_profile_events_for_excursion())
    scopes = set(summary["summary_scope"].to_list())

    assert "entry_excursion_abs_delta_imbalance_quantile" in scopes
    assert "entry_excursion_minute_block_abs_delta_imbalance_quantile" in scopes
    assert "entry_excursion_creation_minute_abs_delta_imbalance_quantile" in scopes

    abs_row = filter_one(
        summary,
        (pl.col("summary_scope") == "entry_excursion_abs_delta_imbalance_quantile")
        & (pl.col("abs_delta_imbalance_quantile") == "q4_highest"),
    )
    assert abs_row["n_confirmable"] == 1
    assert abs_row["n_triggered"] == 1
    assert abs_row["entry_trigger_rate"] == 1.0
    assert abs_row["mfe_pct_mean"] == pytest.approx(0.020)
    assert abs_row["mae_pct_mean"] == pytest.approx(0.003)

    block_row = filter_one(
        summary,
        (pl.col("summary_scope") == "entry_excursion_minute_block_abs_delta_imbalance_quantile")
        & (pl.col("minute_block") == "15:53-15:57")
        & (pl.col("abs_delta_imbalance_quantile") == "q3"),
    )
    assert block_row["n_triggered"] == 1
    assert block_row["mfe_pct_mean"] == pytest.approx(0.030)
    assert block_row["mae_pct_mean"] == pytest.approx(0.004)
```

- [ ] **Step 3: Run the new summary test and verify it fails**

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py::test_build_summary_tables_includes_delta_mae_mfe_entry_excursion_profiles -q
```

Expected: FAIL because the new entry-excursion scopes are not present.

### Task 2: Implement entry-excursion summary scopes

**Files:**
- Modify: `features/macro_fvg_study.py`

- [ ] **Step 1: Add builder functions**

Insert after `build_entry_excursion_alignment_bucket_minute_block_summary()`:

```python
def build_entry_excursion_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_entry_excursion_stats(
        _filter_non_null(events, column),
        [column],
        "entry_excursion_abs_delta_imbalance_quantile",
    )


def build_entry_excursion_minute_block_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_entry_excursion_stats(
        _filter_non_null(events, column),
        ["minute_block", column],
        "entry_excursion_minute_block_abs_delta_imbalance_quantile",
    )


def build_entry_excursion_creation_minute_abs_delta_imbalance_quantile_summary(events: pl.DataFrame) -> pl.DataFrame:
    column = "abs_delta_imbalance_quantile"
    return _group_entry_excursion_stats(
        _filter_non_null(events, column),
        ["assigned_minute_index", "assigned_minute_hhmm", column],
        "entry_excursion_creation_minute_abs_delta_imbalance_quantile",
    )
```

- [ ] **Step 2: Add builders to `build_summary_tables()`**

In the `frames` list, immediately after `build_entry_excursion_alignment_bucket_minute_block_summary(events),` add:

```python
        build_entry_excursion_abs_delta_imbalance_quantile_summary(events),
        build_entry_excursion_minute_block_abs_delta_imbalance_quantile_summary(events),
        build_entry_excursion_creation_minute_abs_delta_imbalance_quantile_summary(events),
```

- [ ] **Step 3: Run the summary test and verify it passes**

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py::test_build_summary_tables_includes_delta_mae_mfe_entry_excursion_profiles -q
```

Expected: PASS.

### Task 3: Add useful CSV and minimal figure outputs

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Add a failing output test**

Append before `test_run_macro_fvg_study_writes_parquet_and_figures`:

```python
def test_export_delta_dominance_mae_mfe_profiles_writes_useful_csvs_and_figures(tmp_path):
    events = make_delta_profile_events_for_excursion()
    summary = macro_fvg_study.build_summary_tables(events)

    macro_fvg_study.export_delta_dominance_mae_mfe_profiles(summary, tmp_path)

    expected_files = [
        "delta_dominance_mae_mfe_abs_quantile.csv",
        "delta_dominance_mae_mfe_by_block.csv",
        "delta_dominance_mae_mfe_by_creation_minute.csv",
        "delta_abs_quantile_mfe_mae_profile.png",
        "early_block_abs_delta_mfe_mae_profile.png",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists(), filename

    csv = pl.read_csv(tmp_path / "delta_dominance_mae_mfe_abs_quantile.csv")
    assert {"profile_type", "summary_scope", "abs_delta_imbalance_quantile", "mfe_pct_mean", "mae_pct_mean"}.issubset(csv.columns)
    assert set(csv["profile_type"].to_list()) == {"entry_triggered", "successful_only"}
```

- [ ] **Step 2: Run output test and verify it fails**

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py::test_export_delta_dominance_mae_mfe_profiles_writes_useful_csvs_and_figures -q
```

Expected: FAIL with missing `export_delta_dominance_mae_mfe_profiles`.

- [ ] **Step 3: Implement export helpers**

Insert before `plot_fvg_summary_figures()`:

```python
PROFILE_EXPORT_COLUMNS = [
    "profile_type",
    "summary_scope",
    "minute_block",
    "assigned_minute_index",
    "assigned_minute_hhmm",
    "abs_delta_imbalance_quantile",
    "n_confirmable",
    "n_triggered",
    "n_retraced",
    "n_successful",
    "entry_trigger_rate",
    "retrace_rate",
    "success_after_retrace_rate",
    "successful_share_of_confirmable",
    "mfe_pct_mean",
    "mfe_pct_median",
    "mfe_pct_p75",
    "mfe_pct_p90",
    "mae_pct_mean",
    "mae_pct_median",
    "mae_pct_p75",
    "mae_pct_p90",
]


def _profile_scope(summary: pl.DataFrame, scope: str, profile_type: str) -> pl.DataFrame:
    frame = _summary_scope(summary, scope)
    if frame.is_empty():
        return pl.DataFrame({column: [] for column in PROFILE_EXPORT_COLUMNS})
    return frame.with_columns(pl.lit(profile_type).alias("profile_type")).select(
        [pl.col(column) if column in frame.columns or column == "profile_type" else pl.lit(None).alias(column) for column in PROFILE_EXPORT_COLUMNS]
    )


def _write_profile_csv(summary: pl.DataFrame, output_path: Path, scope_pairs: list[tuple[str, str]]) -> pl.DataFrame:
    frames = [_profile_scope(summary, scope, profile_type) for scope, profile_type in scope_pairs]
    non_empty = [frame for frame in frames if not frame.is_empty()]
    output = pl.concat(non_empty, how="diagonal_relaxed") if non_empty else pl.DataFrame({column: [] for column in PROFILE_EXPORT_COLUMNS})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(output_path)
    return output


def _plot_delta_profile(frame: pl.DataFrame, x_col: str, filename: str, title: str, figures_dir: Path, order: list[str] | None = None) -> None:
    entry = frame.filter(pl.col("profile_type") == "entry_triggered")
    if entry.is_empty():
        _save_placeholder_figure(figures_dir / filename, title)
        return
    records = _ordered_records(entry, x_col, order)
    labels = [str(record[x_col]) for record in records]
    x = np.arange(len(labels))
    width = 0.35
    mfe = [0 if _is_null(record.get("mfe_pct_mean")) else float(record["mfe_pct_mean"]) * 100 for record in records]
    mae = [0 if _is_null(record.get("mae_pct_mean")) else float(record["mae_pct_mean"]) * 100 for record in records]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, mfe, width, label="entry MFE mean")
    ax.bar(x + width / 2, mae, width, label="entry MAE mean")
    success = frame.filter(pl.col("profile_type") == "successful_only")
    if not success.is_empty():
        lookup = {row[x_col]: row for row in success.to_dicts()}
        ax2 = ax.twinx()
        ax2.plot(
            x,
            [0 if _is_null(lookup.get(record[x_col], {}).get("successful_share_of_confirmable")) else float(lookup[record[x_col]]["successful_share_of_confirmable"]) * 100 for record in records],
            color="#252525",
            marker="o",
            linewidth=2,
            label="success rate",
        )
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_ylim(0, 100)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")
    else:
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_ylabel("Mean Excursion (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / filename)
    plt.close(fig)


def export_delta_dominance_mae_mfe_profiles(summary: pl.DataFrame, figures_dir: Path) -> None:
    abs_profile = _write_profile_csv(
        summary,
        figures_dir / "delta_dominance_mae_mfe_abs_quantile.csv",
        [
            ("entry_excursion_abs_delta_imbalance_quantile", "entry_triggered"),
            ("success_context_abs_delta_imbalance_quantile", "successful_only"),
        ],
    )
    block_profile = _write_profile_csv(
        summary,
        figures_dir / "delta_dominance_mae_mfe_by_block.csv",
        [
            ("entry_excursion_minute_block_abs_delta_imbalance_quantile", "entry_triggered"),
            ("success_context_minute_block_abs_delta_imbalance_quantile", "successful_only"),
        ],
    )
    _write_profile_csv(
        summary,
        figures_dir / "delta_dominance_mae_mfe_by_creation_minute.csv",
        [
            ("entry_excursion_creation_minute_abs_delta_imbalance_quantile", "entry_triggered"),
            ("success_context_creation_minute_abs_delta_imbalance_quantile", "successful_only"),
        ],
    )
    _plot_delta_profile(
        abs_profile,
        "abs_delta_imbalance_quantile",
        "delta_abs_quantile_mfe_mae_profile.png",
        "Delta Dominance MAE/MFE Profile by Abs Quantile",
        figures_dir,
        ["q1_lowest", "q2", "q3", "q4_highest"],
    )
    early = block_profile.filter(pl.col("minute_block") == "15:50-15:52") if "minute_block" in block_profile.columns else block_profile.clear()
    _plot_delta_profile(
        early,
        "abs_delta_imbalance_quantile",
        "early_block_abs_delta_mfe_mae_profile.png",
        "Early Block Delta Dominance MAE/MFE Profile",
        figures_dir,
        ["q1_lowest", "q2", "q3", "q4_highest"],
    )
```

- [ ] **Step 4: Call export from figure generation**

At the end of `plot_fvg_summary_figures()`, after the existing successful FVG plot calls, add:

```python
    export_delta_dominance_mae_mfe_profiles(summary, figures_dir)
```

- [ ] **Step 5: Run output test and verify pass**

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py::test_export_delta_dominance_mae_mfe_profiles_writes_useful_csvs_and_figures -q
```

Expected: PASS.

### Task 4: Full verification and regeneration

**Files:**
- Runtime outputs under `outputs/`

- [ ] **Step 1: Run focused tests**

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run related tests**

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py test/test_volume_delta.py test/test_tick_density.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Regenerate macro FVG outputs in the main workspace**

```bash
.venv/bin/python -m features.macro_fvg_study
```

Expected: writes parquet outputs plus the three CSV profile exports and two focused PNGs under `outputs/figs/fvg/`.

- [ ] **Step 4: Inspect profile CSVs**

```bash
.venv/bin/python - <<'PY'
import polars as pl
for path in [
    'outputs/figs/fvg/delta_dominance_mae_mfe_abs_quantile.csv',
    'outputs/figs/fvg/delta_dominance_mae_mfe_by_block.csv',
    'outputs/figs/fvg/delta_dominance_mae_mfe_by_creation_minute.csv',
]:
    df = pl.read_csv(path)
    print(path, df.shape)
    print(df.select(['profile_type', 'summary_scope', 'minute_block', 'assigned_minute_hhmm', 'abs_delta_imbalance_quantile', 'n_confirmable', 'n_triggered', 'n_successful', 'mfe_pct_mean', 'mae_pct_mean']).head(12))
PY
```

Expected: CSVs contain both `entry_triggered` and `successful_only` profile rows where applicable.

- [ ] **Step 5: Commit source changes**

```bash
git add docs/superpowers/plans/2026-05-13-fvg-delta-mae-mfe-profiles.md features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add fvg delta mae mfe profiles"
```
