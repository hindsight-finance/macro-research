# Post ADX Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable post-COVID ADX ablation sweep for the current winning trendability representation.

**Architecture:** Extend the modeling registry with a narrow `post_adx_ablation` experiment group and expose it through the existing CLI. Reuse the current walk-forward runner and summaries unchanged so ablations remain comparable to the existing representation sweep.

**Tech Stack:** Python 3, pandas, scikit-learn, pytest

---

### Task 1: Add Ablation Registry Coverage

**Files:**
- Modify: `features/trend/modeling/registry.py`
- Test: `features/trend/modeling/test/test_registry.py`

- [ ] **Step 1: Write the failing registry test**

```python
def test_build_post_adx_ablation_registry_contains_reduced_variants():
    registry = build_post_adx_ablation_registry(session_name="1pm-3pm")
    ids = {experiment.experiment_id for experiment in registry}

    assert "EXP20_post_adx_parts_base" in ids
    assert "EXP21_post_adx_parts_minus_persistence" in ids
    assert "EXP22_post_adx_parts_minus_crossover" in ids
    assert "EXP23_post_adx_parts_minus_log_vr" in ids
    assert "EXP24_post_adx_parts_minus_irr" in ids
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_registry.py -q`
Expected: import or missing symbol failure for `build_post_adx_ablation_registry`.

- [ ] **Step 3: Implement the minimal ablation registry**

```python
ADX_PARTS_MINUS_PERSISTENCE_FEATURE_COLUMNS = (
    "mss", "adx_strength", "adx_crossover", "irr", "er", "log_vr",
)
```

```python
def build_post_adx_ablation_registry(session_name: str, ridge_alpha: float = 1.0) -> list[ExperimentSpec]:
    return [
        ExperimentSpec("EXP20_post_adx_parts_base", "post_adx_ablation", session_name, "post_covid", "adx_parts", ADX_PARTS_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ...
    ]
```

- [ ] **Step 4: Run the registry tests**

Run: `python3 -m pytest features/trend/modeling/test/test_registry.py -q`
Expected: PASS

### Task 2: Expose Ablations Through The CLI

**Files:**
- Modify: `features/trend/modeling/cli.py`
- Test: `features/trend/modeling/test/test_cli.py`

- [ ] **Step 1: Write the failing CLI parse test**

```python
def test_build_parser_accepts_post_adx_ablation_group():
    parser = build_parser()
    args = parser.parse_args([
        "run-experiments",
        "--table-path", "table.parquet",
        "--session-name", "1pm-3pm",
        "--output-dir", "outputs/trend_modeling/experiments",
        "--experiment-group", "post_adx_ablation",
    ])

    assert args.experiment_group == "post_adx_ablation"
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_cli.py -q`
Expected: argparse choice failure until the new group is wired.

- [ ] **Step 3: Implement the parser and dispatch update**

```python
run_parser.add_argument(
    "--experiment-group",
    default="representation_sweep",
    choices=("representation_sweep", "ridge_alpha_sweep", "post_adx_ablation"),
)
```

```python
if args.experiment_group == "post_adx_ablation":
    return build_post_adx_ablation_registry(session_name=args.session_name, ridge_alpha=args.ridge_alpha)
```

- [ ] **Step 4: Run the CLI and modeling tests**

Run: `python3 -m pytest features/trend/modeling/test/test_cli.py features/trend/modeling/test/test_registry.py -q`
Expected: PASS

### Task 3: Verify End-To-End Ablation Runs

**Files:**
- Use: `outputs/trend_modeling/cache/nq_trend_modeling_table.parquet`
- Write: `outputs/trend_modeling/experiments/<session_name>/EXP20...EXP24...`

- [ ] **Step 1: Run the ablation sweep for 1pm-3pm**

Run: `python3 -m features.trend.modeling.cli run-experiments --table-path outputs/trend_modeling/cache/nq_trend_modeling_table.parquet --session-name 1pm-3pm --output-dir outputs/trend_modeling/experiments --experiment-group post_adx_ablation`
Expected: experiment directories plus summary output for base and reduced post-COVID variants.

- [ ] **Step 2: Run the ablation sweep for 3pm-3:50pm**

Run: `python3 -m features.trend.modeling.cli run-experiments --table-path outputs/trend_modeling/cache/nq_trend_modeling_table.parquet --session-name 3pm-3:50pm --output-dir outputs/trend_modeling/experiments --experiment-group post_adx_ablation`
Expected: matching ablation outputs for the second session.

- [ ] **Step 3: Re-run the full modeling suite**

Run: `python3 -m pytest features/trend/modeling/test -q`
Expected: PASS
