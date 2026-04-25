# Historical Regimes Main Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the main historical regime classification path with the three-scalar regime stack (`trend_score`, `containment_score`, `chop_score`, `label`) and stop using the legacy `StateDetector` ensemble for this workflow.

**Architecture:** Keep `features/trend/modeling/table.py` as the canonical session-level score builder, extract discrete label assignment into a new `features/trend/modeling/labels.py` module, and add a `features/trend/historical_regimes.py` entry script that builds scores then assigns labels. Treat `StateDetector` as legacy research infrastructure only; do not remove wrapper imports in this patch.

**Tech Stack:** Python, argparse, pandas, parquet/csv I/O, pytest, existing `features.trend.modeling` modules

---

## Locked Decisions

- Unmatched rows stay in the output and get `label="uncertain"`.
- `features/trend/historical_regimes.py` always writes the scalar columns and a `label` column; no `--no-label` flag in this patch.
- `--output-path` chooses format by suffix: `.parquet` or `.csv`.
- `state_detector.py` gets docstring deprecation notes only; no runtime `DeprecationWarning` in this patch.

## File Map

- Create: `features/trend/historical_regimes.py`
  Purpose: package-local module/CLI entrypoint for canonical historical regime classification.
- Create: `features/trend/modeling/labels.py`
  Purpose: canonical threshold defaults and `assign_three_scalar_labels(...)`.
- Create: `features/trend/modeling/test/test_labels.py`
  Purpose: unit coverage for label assignment, uncertain-row behavior, and drop/filter behavior.
- Create: `features/trend/test/test_historical_regimes.py`
  Purpose: parser and end-to-end output coverage for the new historical regimes script.
- Modify: `features/trend/modeling/containment_research.py`
  Purpose: import the label helper from `labels.py` and keep probe behavior by filtering out `uncertain` rows before model fitting.
- Modify: `features/trend/modeling/__init__.py`
  Purpose: expose label helper and default thresholds as part of the modeling surface.
- Modify: `features/trend/modeling/table.py`
  Purpose: add a short docstring clarifying that `build_modeling_table(...)` is the canonical historical regime feature/score builder.
- Modify: `features/trend/modeling/test/test_containment_research.py`
  Purpose: keep research tests aligned with new helper behavior and probe filtering.
- Modify: `features/trend/state_detector.py`
  Purpose: mark the ensemble classifier as legacy in module/class docstrings.

### Task 1: Extract Canonical Label Helper

**Files:**
- Create: `features/trend/modeling/labels.py`
- Create: `features/trend/modeling/test/test_labels.py`
- Modify: `features/trend/modeling/__init__.py`

- [ ] **Step 1: Write the failing label tests**

```python
from __future__ import annotations

import pandas as pd

from features.trend.modeling.labels import assign_three_scalar_labels


def test_assign_three_scalar_labels_marks_uncertain_rows_without_dropping():
    frame = pd.DataFrame(
        {
            "trend_score": [0.85, 0.20, 0.20, 0.45],
            "containment_score": [0.20, 0.82, 0.20, 0.45],
            "chop_score": [0.20, 0.30, 0.86, 0.45],
        }
    )

    labeled = assign_three_scalar_labels(
        frame=frame,
        trend_high=0.70,
        containment_high=0.70,
        chop_high=0.70,
        low_cutoff=0.40,
        containment_chop_max=0.55,
    )

    assert labeled["label"].tolist() == ["trend", "containment", "chop", "uncertain"]


def test_assign_three_scalar_labels_can_drop_uncertain_rows_for_probe_use():
    frame = pd.DataFrame(
        {
            "trend_score": [0.85, 0.45],
            "containment_score": [0.20, 0.45],
            "chop_score": [0.20, 0.45],
        }
    )

    labeled = assign_three_scalar_labels(
        frame=frame,
        trend_high=0.70,
        containment_high=0.70,
        chop_high=0.70,
        low_cutoff=0.40,
        containment_chop_max=0.55,
        drop_uncertain=True,
    )

    assert labeled["label"].tolist() == ["trend"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_labels.py -q`
Expected: FAIL because `features.trend.modeling.labels` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from __future__ import annotations

import pandas as pd

THREE_SCALAR_LABELS = ("trend", "containment", "chop")

DEFAULT_LABEL_THRESHOLDS = {
    "trend_high": 0.70,
    "containment_high": 0.70,
    "chop_high": 0.70,
    "low_cutoff": 0.40,
    "containment_chop_max": 0.55,
}


def assign_three_scalar_labels(
    frame: pd.DataFrame,
    trend_high: float,
    containment_high: float,
    chop_high: float,
    low_cutoff: float,
    containment_chop_max: float | None = None,
    uncertain_label: str = "uncertain",
    drop_uncertain: bool = False,
) -> pd.DataFrame:
    chop_cap = containment_chop_max if containment_chop_max is not None else 0.55
    labels = pd.Series(uncertain_label, index=frame.index, dtype="object")

    labels.loc[
        (frame["trend_score"] >= trend_high)
        & (frame["containment_score"] <= low_cutoff)
        & (frame["chop_score"] <= low_cutoff)
    ] = "trend"
    labels.loc[
        (frame["containment_score"] >= containment_high)
        & (frame["trend_score"] <= low_cutoff)
        & (frame["chop_score"] <= chop_cap)
    ] = "containment"
    labels.loc[
        (frame["chop_score"] >= chop_high)
        & (frame["trend_score"] <= low_cutoff)
        & (frame["containment_score"] <= low_cutoff)
    ] = "chop"

    labeled = frame.copy()
    labeled["label"] = labels.to_numpy()
    if drop_uncertain:
        labeled = labeled.loc[labeled["label"].isin(THREE_SCALAR_LABELS)].copy()
    return labeled
```

```python
from .labels import DEFAULT_LABEL_THRESHOLDS, THREE_SCALAR_LABELS, assign_three_scalar_labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_labels.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/labels.py features/trend/modeling/test/test_labels.py features/trend/modeling/__init__.py
git commit -m "feat: extract historical regime labels"
```

### Task 2: Repoint Research Code To The New Label Module

**Files:**
- Modify: `features/trend/modeling/containment_research.py`
- Modify: `features/trend/modeling/test/test_containment_research.py`

- [ ] **Step 1: Write the failing research compatibility test**

```python
def test_run_three_way_probe_filters_uncertain_rows_before_training():
    table = pd.DataFrame(
        {
            "trade_date": pd.date_range("2022-01-03", periods=16, freq="MS").date,
            "session_name": ["1pm-3pm"] * 16,
            "feature_status": ["ok"] * 16,
            "containment_status": ["ok"] * 16,
            "target_status": ["ok"] * 16,
            "mss": [0.1] * 16,
            "er": [0.2] * 16,
            "containment_range_stability": [0.3] * 16,
            "trend_score": [0.45] * 12 + [0.85, 0.20, 0.20, 0.45],
            "containment_score": [0.45] * 12 + [0.20, 0.82, 0.20, 0.45],
            "chop_score": [0.45] * 12 + [0.20, 0.30, 0.86, 0.45],
        }
    )

    results = run_three_way_probe(
        table=table,
        session_feature_sets={"1pm-3pm": ("mss", "er", "containment_range_stability")},
        holdout_fraction=0.25,
        label_thresholds={"trend_high": 0.7, "containment_high": 0.7, "chop_high": 0.7, "low_cutoff": 0.4},
    )

    assert {"model", "macro_f1", "balanced_accuracy", "confusion_matrix"} <= set(results.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/modeling/test/test_containment_research.py -q`
Expected: FAIL once the helper starts returning `uncertain` rows and `run_three_way_probe(...)` has not been updated yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from features.trend.modeling.labels import THREE_SCALAR_LABELS, assign_three_scalar_labels
```

```python
development = assign_three_scalar_labels(frame=development, **thresholds, drop_uncertain=True)
hold_labeled = assign_three_scalar_labels(frame=holdout, **thresholds, drop_uncertain=True)

if development.empty or hold_labeled.empty:
    continue
if set(hold_labeled["label"].unique()) != set(THREE_SCALAR_LABELS):
    continue
```

```python
def test_assign_three_scalar_labels_excludes_high_chop_from_containment():
    table = pd.DataFrame(
        {
            "trend_score": [0.2, 0.2],
            "containment_score": [0.8, 0.8],
            "chop_score": [0.50, 0.65],
        }
    )

    labeled = assign_three_scalar_labels(
        frame=table,
        trend_high=0.7,
        containment_high=0.7,
        chop_high=0.7,
        low_cutoff=0.4,
        containment_chop_max=0.55,
        drop_uncertain=True,
    )
    assert labeled["label"].tolist() == ["containment"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/modeling/test/test_containment_research.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/containment_research.py features/trend/modeling/test/test_containment_research.py
git commit -m "refactor: point probe code at canonical labels module"
```

### Task 3: Add The Canonical Historical Regimes Script

**Files:**
- Create: `features/trend/historical_regimes.py`
- Create: `features/trend/test/test_historical_regimes.py`

- [ ] **Step 1: Write the failing parser and end-to-end tests**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.trend.historical_regimes import build_parser, main


def _make_intraday_bars(trade_date: str, start_time: str, periods: int) -> pd.DataFrame:
    timestamp = pd.date_range(f"{trade_date} {start_time}", periods=periods, freq="min")
    base = 100.0 + np.arange(periods) * 0.05
    open_ = base
    close = base + 0.02
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    return pd.DataFrame(
        {
            "DateTime_ET": timestamp,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 100,
        }
    )


def test_build_parser_accepts_session_names_and_thresholds():
    args = build_parser().parse_args(
        [
            "--input-path",
            "outputs/nq_1m.parquet",
            "--instrument",
            "NQ",
            "--output-path",
            "outputs/nq_regimes.parquet",
            "--session-name",
            "1pm-3pm",
            "--session-name",
            "3pm-3:50pm",
            "--trend-high",
            "0.8",
        ]
    )

    assert args.session_names == ["1pm-3pm", "3pm-3:50pm"]
    assert args.trend_high == 0.8


def test_main_writes_scores_and_label_to_parquet(tmp_path: Path):
    bars = pd.concat(
        [
            _make_intraday_bars("2024-01-02", "13:00", 120),
            _make_intraday_bars("2024-01-03", "13:00", 120),
        ],
        ignore_index=True,
    )
    input_path = tmp_path / "bars.parquet"
    output_path = tmp_path / "historical_regimes.parquet"
    bars.to_parquet(input_path, index=False)

    exit_code = main(
        [
            "--input-path",
            str(input_path),
            "--instrument",
            "NQ",
            "--output-path",
            str(output_path),
            "--session-name",
            "1pm-3pm",
        ]
    )

    result = pd.read_parquet(output_path)
    assert exit_code == 0
    assert {"trend_score", "containment_score", "chop_score", "label"} <= set(result.columns)
    assert set(result["label"]) <= {"trend", "containment", "chop", "uncertain"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest features/trend/test/test_historical_regimes.py -q`
Expected: FAIL because `features/trend/historical_regimes.py` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from features.trend.modeling import build_modeling_table
from features.trend.modeling.labels import DEFAULT_LABEL_THRESHOLDS, assign_three_scalar_labels
from features.trend.modeling.table import DEFAULT_SESSION_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build historical regime scores and labels")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--session-name",
        dest="session_names",
        action="append",
        choices=DEFAULT_SESSION_NAMES,
        help="Limit the output to one or more session windows.",
    )
    parser.add_argument("--trend-high", type=float, default=DEFAULT_LABEL_THRESHOLDS["trend_high"])
    parser.add_argument("--containment-high", type=float, default=DEFAULT_LABEL_THRESHOLDS["containment_high"])
    parser.add_argument("--chop-high", type=float, default=DEFAULT_LABEL_THRESHOLDS["chop_high"])
    parser.add_argument("--low-cutoff", type=float, default=DEFAULT_LABEL_THRESHOLDS["low_cutoff"])
    parser.add_argument(
        "--containment-chop-max",
        type=float,
        default=DEFAULT_LABEL_THRESHOLDS["containment_chop_max"],
    )
    return parser


def _write_output(frame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        frame.to_parquet(output, index=False)
        return
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
        return
    raise ValueError("output-path must end with .parquet or .csv")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table = build_modeling_table(
        input_path=args.input_path,
        instrument=args.instrument,
        session_names=args.session_names or DEFAULT_SESSION_NAMES,
    )
    labeled = assign_three_scalar_labels(
        frame=table,
        trend_high=args.trend_high,
        containment_high=args.containment_high,
        chop_high=args.chop_high,
        low_cutoff=args.low_cutoff,
        containment_chop_max=args.containment_chop_max,
    )
    _write_output(labeled, args.output_path)
    print(f"Wrote {len(labeled)} rows to {args.output_path}")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest features/trend/test/test_historical_regimes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/historical_regimes.py features/trend/test/test_historical_regimes.py
git commit -m "feat: add historical regimes entry script"
```

### Task 4: Mark The Canonical Builder And Deprecate The Legacy Ensemble Path

**Files:**
- Modify: `features/trend/modeling/table.py`
- Modify: `features/trend/state_detector.py`

- [ ] **Step 1: Reuse the existing import smoke tests**

Use:
- `features/trend/modeling/test/test_init.py`
- `features/trend/test/test_state_detector.py`

- [ ] **Step 2: Run targeted import smoke**

Run: `python3 -m pytest features/trend/modeling/test/test_init.py features/trend/test/test_state_detector.py -q`
Expected: PASS before and after the docstring edits.

- [ ] **Step 3: Write the minimal implementation**

```python
def build_modeling_table(
    input_path: str | Path,
    instrument: str,
    session_names: Iterable[str] = DEFAULT_SESSION_NAMES,
) -> pd.DataFrame:
    """Canonical historical regime feature/score builder."""
```

```python
"""
Legacy ensemble-based market state detector.

Historical regime classification now lives under `features.trend.modeling`
via `build_modeling_table(...)` plus `assign_three_scalar_labels(...)`.
This module remains because indicator wrappers are still imported by
`features.trend.modeling.table`.
"""
```

```python
class StateDetector:
    """
    Legacy weighted indicator ensemble for research only.

    Use `features.trend.modeling.table.build_modeling_table(...)` and the
    historical regime labeling helpers for the canonical historical path.
    """
```

- [ ] **Step 4: Run targeted import smoke again**

Run: `python3 -m pytest features/trend/modeling/test/test_init.py features/trend/test/test_state_detector.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/trend/modeling/table.py features/trend/state_detector.py
git commit -m "docs: mark historical regime path as canonical"
```

### Task 5: Run Full Verification

**Files:**
- Modify: none

- [ ] **Step 1: Run focused tests**

Run: `python3 -m pytest features/trend/modeling/test/test_labels.py features/trend/modeling/test/test_containment_research.py features/trend/test/test_historical_regimes.py -q`
Expected: PASS

- [ ] **Step 2: Run the modeling suite**

Run: `python3 -m pytest features/trend/modeling/test -q`
Expected: PASS

- [ ] **Step 3: Run the broader trend suite**

Run: `python3 -m pytest features/trend -q`
Expected: PASS

- [ ] **Step 4: Run the canonical CLI smoke command**

Run:

```bash
python3 -m features.trend.historical_regimes \
  --input-path outputs/nq_1m.parquet \
  --instrument NQ \
  --output-path outputs/nq_regimes.parquet
```

Expected: console prints a single success line beginning with `Wrote ` and ending with `outputs/nq_regimes.parquet`

- [ ] **Step 5: Commit**

```bash
git add features/trend/historical_regimes.py features/trend/modeling features/trend/test docs/superpowers/plans/2026-04-17-historical-regimes-main-switch.md
git commit -m "feat: switch historical regime workflow to three-scalar labels"
```

## Spec Coverage Check

- New canonical main script: covered by Task 3.
- Move label helper into cleaner module: covered by Task 1.
- Keep `modeling/table.py` as canonical score builder: covered by Task 4 docstring note and Task 3 usage.
- Deprecate old ensemble classifier without deleting wrappers: covered by Task 4.
- Optional later wrapper extraction intentionally excluded from this patch: preserved by architecture and file map.
- Tests for synthetic labels, script outputs, and uncertain rows: covered by Tasks 1, 2, and 3.
- Final historical regimes workflow: covered by Task 5 smoke command.

## Notes For Execution

- Do not duplicate parquet reading in `features/trend/historical_regimes.py`; `build_modeling_table(...)` already owns that read path.
- Keep `containment_research.py` importing the helper from `labels.py` so old research code still finds the symbol there if the import path is preserved.
- Do not delete `StateDetector` or indicator wrapper classes in this patch.
