# Minute Base UTC Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tagged ET/session/window base parquet with a canonical UTC minute-base parquet, then update the core feature scripts to derive New York time and session/window labels in memory.

**Architecture:** Add one shared minute-bar loader that normalizes legacy schemas into canonical `datetime_utc`, plus small helper functions for `datetime_et`, `session`, and `window` derivation. Keep `session_tagger.py` as the base writer entrypoint, then refactor `macro_outcomes.py`, `features/macro_fvg_study.py`, `features/pm_3pm.py`, and `viz/macro_analysis.py` to consume the canonical base instead of persisted ET/session/window fields.

**Tech Stack:** Python 3, pandas, pytest, parquet IO

---

### Task 1: Add Shared UTC Minute-Bar Helpers

**Files:**
- Create: `utils/minute_bars.py`
- Create: `test/test_minute_bars.py`

- [ ] **Step 1: Write the failing loader/unit tests**

```python
import pandas as pd

from utils.minute_bars import (
    build_market_time_columns,
    derive_session_window,
    normalize_minute_bars,
)


def test_normalize_minute_bars_prefers_datetime_utc_and_keeps_utc_dtype():
    raw = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2020-09-01 19:50:00", "2020-09-01 19:51:00"],
                utc=True,
            ),
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 11],
        }
    )

    out = normalize_minute_bars(raw)

    assert list(out.columns) == ["datetime_utc", "Open", "High", "Low", "Close", "Volume"]
    assert str(out["datetime_utc"].dtype) == "datetime64[ns, UTC]"


def test_normalize_minute_bars_accepts_legacy_datetime_et_and_converts_to_utc():
    raw = pd.DataFrame(
        {
            "DateTime_ET": pd.to_datetime(["2020-09-01 15:50:00", "2020-09-01 15:51:00"]),
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 11],
        }
    )

    out = normalize_minute_bars(raw)

    assert out["datetime_utc"].iloc[0] == pd.Timestamp("2020-09-01 19:50:00+00:00")
    assert out["datetime_utc"].iloc[1] == pd.Timestamp("2020-09-01 19:51:00+00:00")


def test_build_market_time_columns_derives_new_york_datetime_and_date():
    raw = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(["2020-09-01 19:50:00"], utc=True),
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [10],
        }
    )

    out = build_market_time_columns(raw)

    assert out["datetime_et"].iloc[0] == pd.Timestamp("2020-09-01 15:50:00")
    assert out["date_et"].iloc[0] == pd.Timestamp("2020-09-01")


def test_derive_session_window_marks_macro_hour_without_persisting_it():
    raw = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2020-09-01 19:49:00", "2020-09-01 19:50:00", "2020-09-01 20:00:00"],
                utc=True,
            ),
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 11, 12],
        }
    )

    out = derive_session_window(build_market_time_columns(raw))

    assert out["window"].tolist() == ["H3PM", "MACRO", "POST"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_minute_bars.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing function errors for `utils.minute_bars`.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

MARKET_TZ = "America/New_York"
UTC = "UTC"
CANONICAL_COLUMNS = ["datetime_utc", "Open", "High", "Low", "Close", "Volume"]


def _read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def normalize_minute_bars(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "datetime_utc" in work.columns:
        ts = pd.to_datetime(work["datetime_utc"], utc=True, errors="coerce")
    elif "DateTime_UTC" in work.columns:
        ts = pd.to_datetime(work["DateTime_UTC"], utc=True, errors="coerce")
    elif "DateTime_ET" in work.columns:
        ts = (
            pd.to_datetime(work["DateTime_ET"], errors="coerce")
            .dt.tz_localize(MARKET_TZ)
            .dt.tz_convert(UTC)
        )
    else:
        raise ValueError("Expected one of: datetime_utc, DateTime_UTC, DateTime_ET")

    if ts.isna().any():
        raise ValueError("Timestamp column contains unparsable values")

    work["datetime_utc"] = ts
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = work[["datetime_utc", "Open", "High", "Low", "Close", "Volume"]].copy()
    out = out.sort_values("datetime_utc").reset_index(drop=True)
    if out["datetime_utc"].duplicated().any():
        raise ValueError("Duplicate datetime_utc values after normalization")
    return out


def load_minute_bars(path: str | Path) -> pd.DataFrame:
    return normalize_minute_bars(_read_any(path))


def build_market_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "datetime_utc" not in work.columns:
        raise ValueError("Expected canonical datetime_utc column")
    datetime_et = pd.to_datetime(work["datetime_utc"], utc=True).dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    work["datetime_et"] = datetime_et
    work["date_et"] = datetime_et.dt.normalize()
    work["time_et"] = datetime_et.dt.time
    work["minute_of_day_et"] = datetime_et.dt.hour * 60 + datetime_et.dt.minute
    return work


def derive_session_window(df: pd.DataFrame) -> pd.DataFrame:
    work = build_market_time_columns(df) if "minute_of_day_et" not in df.columns else df.copy()
    mins = work["minute_of_day_et"]
    work["session"] = "OTHER"
    work.loc[(mins >= 19 * 60) & (mins < 24 * 60), "session"] = "ASIA"
    work.loc[(mins >= 2 * 60) & (mins < 5 * 60), "session"] = "LONDON"
    work.loc[(mins >= 9 * 60 + 30) & (mins < 11 * 60), "session"] = "NYAM"
    work.loc[(mins >= 12 * 60) & (mins < 13 * 60), "session"] = "LUNCH"
    work.loc[(mins >= 13 * 60) & (mins < 15 * 60), "session"] = "PM"
    work["window"] = "NONE"
    work.loc[(mins >= 15 * 60) & (mins <= 15 * 60 + 49), "window"] = "H3PM"
    work.loc[(mins >= 15 * 60 + 50) & (mins <= 15 * 60 + 59), "window"] = "MACRO"
    work.loc[(mins >= 16 * 60) & (mins <= 16 * 60 + 10), "window"] = "POST"
    return work
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test/test_minute_bars.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_minute_bars.py utils/minute_bars.py
git commit -m "feat: add utc minute bar normalization helpers"
```

### Task 2: Refactor `session_tagger.py` Into Canonical Base Writer

**Files:**
- Modify: `session_tagger.py`
- Create: `test/test_session_tagger.py`

- [ ] **Step 1: Write the failing base-writer tests**

```python
from pathlib import Path

import pandas as pd

from session_tagger import process_file


def test_process_file_writes_minute_base_with_datetime_utc(tmp_path: Path):
    input_path = tmp_path / "nq_1m.csv"
    output_dir = tmp_path / "outputs"
    pd.DataFrame(
        {
            "DateTime_ET": ["2020-09-01 15:50:00", "2020-09-01 15:51:00"],
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 11],
        }
    ).to_csv(input_path, index=False)

    out_path = process_file(str(input_path), str(output_dir))
    out = pd.read_parquet(out_path)

    assert out_path.name == "nq_minute_base.parquet"
    assert list(out.columns) == ["datetime_utc", "Open", "High", "Low", "Close", "Volume"]
    assert str(out["datetime_utc"].dtype) == "datetime64[ns, UTC]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_session_tagger.py -q`
Expected: FAIL because `session_tagger.py` still writes tagged ET/session/window parquet.

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path

from utils.minute_bars import load_minute_bars


def process_file(input_path: str, output_dir: str):
    df = load_minute_bars(input_path)
    stem = Path(input_path).stem.replace("_1m", "")
    out_path = Path(output_dir) / f"{stem}_minute_base.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test/test_session_tagger.py test/test_minute_bars.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add session_tagger.py test/test_session_tagger.py
git commit -m "feat: write canonical utc minute base parquets"
```

### Task 3: Refactor `macro_outcomes.py` To Derive ET Windows In Memory

**Files:**
- Modify: `macro_outcomes.py`
- Create: `test/test_macro_outcomes.py`

- [ ] **Step 1: Write the failing macro outcome tests**

```python
import pandas as pd

from macro_outcomes import compute_macro_outcomes


def test_compute_macro_outcomes_uses_datetime_utc_and_derived_macro_window():
    bars = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2020-09-01 19:49:00", "2020-09-01 19:50:00", "2020-09-01 19:59:00", "2020-09-01 20:00:00"],
                utc=True,
            ),
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [100.5, 102.5, 103.5, 104.0],
            "Low": [99.5, 100.5, 101.5, 102.5],
            "Close": [100.0, 102.0, 103.0, 103.5],
            "Volume": [1, 2, 3, 4],
        }
    )

    out = compute_macro_outcomes(bars, macro_window_name="MACRO")

    assert len(out) == 1
    assert out.loc[0, "date"] == pd.Timestamp("2020-09-01")
    assert out.loc[0, "macro_open"] == 101.0
    assert out.loc[0, "postclose_range_points"] == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_macro_outcomes.py -q`
Expected: FAIL because `macro_outcomes.py` currently requires `DateTime_ET` and persisted `window`.

- [ ] **Step 3: Write minimal implementation**

```python
from utils.minute_bars import build_market_time_columns, derive_session_window, load_minute_bars


def compute_macro_outcomes(df: pd.DataFrame, macro_window_name: str) -> pd.DataFrame:
    work = derive_session_window(build_market_time_columns(df))
    macro = work[work["window"] == macro_window_name].copy()
    post = work[work["window"].isin(POST_WINDOW_CANDIDATES)].copy()
    work["date"] = work["date_et"]
    macro["date"] = macro["date_et"]
    post["date"] = post["date_et"]
    # keep the existing aggregation logic, but use datetime_et for all time comparisons
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test/test_macro_outcomes.py test/test_minute_bars.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add macro_outcomes.py test/test_macro_outcomes.py
git commit -m "feat: derive macro windows from utc minute base"
```

### Task 4: Refactor Macro FVG Detection To Use Canonical Base

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Write the failing UTC-backed FVG tests**

```python
import pandas as pd

from features.macro_fvg_study import detect_macro_fvgs


def make_bars(rows):
    df = pd.DataFrame(rows)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def test_detect_macro_fvg_derives_macro_window_from_datetime_utc():
    bars = make_bars(
        [
            {"datetime_utc": "2025-01-02 20:49:00+00:00", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1},
            {"datetime_utc": "2025-01-02 20:50:00+00:00", "Open": 99.0, "High": 100.0, "Low": 97.0, "Close": 98.0, "Volume": 2},
            {"datetime_utc": "2025-01-02 20:51:00+00:00", "Open": 96.0, "High": 97.0, "Low": 94.0, "Close": 95.0, "Volume": 3},
        ]
    )

    events = detect_macro_fvgs(bars)

    assert len(events) == 1
    assert events.iloc[0]["assigned_at"] == pd.Timestamp("2025-01-02 15:50:00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_macro_fvg_study.py -q`
Expected: FAIL because `detect_macro_fvgs()` still requires `DateTime_ET` and `window`.

- [ ] **Step 3: Write minimal implementation**

```python
from utils.minute_bars import build_market_time_columns, derive_session_window, load_minute_bars


def detect_macro_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    work = derive_session_window(build_market_time_columns(df))
    work["DateTime_ET"] = work["datetime_et"]
    work = work.sort_values("DateTime_ET").reset_index(drop=True)
    # keep existing FVG detection logic, but rely on derived in-memory `window`


def run_macro_fvg_study(input_path: Path = INPUT_PATH):
    bars = load_minute_bars(input_path)
    events = detect_macro_fvgs(bars)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test/test_macro_fvg_study.py test/test_minute_bars.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: derive fvg macro windows from utc base"
```

### Task 5: Refactor PM/HR3 and Macro Viz Readers To Support Canonical Base

**Files:**
- Modify: `features/pm_3pm.py`
- Modify: `viz/macro_analysis.py`
- Create: `test/test_pm_3pm.py`

- [ ] **Step 1: Write the failing UTC-backed PM/HR3 test**

```python
import pandas as pd

from features.pm_3pm import _load_minutes


def test_load_minutes_accepts_datetime_utc_and_derives_new_york_time(tmp_path):
    path = tmp_path / "nq_minute_base.parquet"
    pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2020-09-01 17:00:00", "2020-09-01 19:00:00", "2020-09-01 19:49:00"],
                utc=True,
            ),
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 11, 12],
        }
    ).to_parquet(path, index=False)

    out = _load_minutes([str(path)])

    assert "timestamp" in out.columns
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-09-01 13:00:00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_pm_3pm.py -q`
Expected: FAIL because `_load_minutes()` does not know `datetime_utc`.

- [ ] **Step 3: Write minimal implementation**

```python
from utils.minute_bars import build_market_time_columns, load_minute_bars


def _load_minutes(paths: List[str]) -> pd.DataFrame:
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    dfs = []
    for f in sorted(files):
        base = load_minute_bars(f)
        work = build_market_time_columns(base)
        work["timestamp"] = work["datetime_et"]
        work["date"] = work["datetime_et"].dt.date
        work["time"] = work["datetime_et"].dt.time
        work["__srcfile"] = os.path.basename(f)
        dfs.append(work)
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)
```

- [ ] **Step 4: Update viz reader to support canonical base**

```python
def read_bars(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file extension. Use .csv or .parquet")

    if "datetime_utc" in df.columns:
        dt = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        ci = {c.lower(): c for c in df.columns}
        dt_col = _find_col(ci, "datetime_et", "date_time_et", "datetime", "timestamp", "time_et", "dt_et", "dt", "time", "date_time")
        if dt_col is None:
            raise ValueError("Could not find a datetime column.")
        dt = pd.to_datetime(df[dt_col], errors="coerce")
        if pd.api.types.is_datetime64tz_dtype(dt):
            dt = dt.dt.tz_convert("America/New_York").dt.tz_localize(None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest test/test_pm_3pm.py test/test_minute_bars.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add features/pm_3pm.py test/test_pm_3pm.py viz/macro_analysis.py
git commit -m "feat: support utc minute base in pm and viz readers"
```

### Task 6: Final Verification

**Files:**
- Modify: `session_tagger.py`
- Modify: `macro_outcomes.py`
- Modify: `features/macro_fvg_study.py`
- Modify: `features/pm_3pm.py`
- Modify: `viz/macro_analysis.py`
- Create: `utils/minute_bars.py`
- Create: `test/test_minute_bars.py`
- Create: `test/test_session_tagger.py`
- Create: `test/test_macro_outcomes.py`
- Create: `test/test_pm_3pm.py`
- Modify: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Run focused regression suite**

Run: `python3 -m pytest test/test_minute_bars.py test/test_session_tagger.py test/test_macro_outcomes.py test/test_pm_3pm.py test/test_macro_fvg_study.py -q`
Expected: PASS

- [ ] **Step 2: Smoke-run the canonical base builder**

Run: `python3 session_tagger.py`
Expected: writes `outputs/*_minute_base.parquet` with `datetime_utc` and no persisted ET/session/window fields.

- [ ] **Step 3: Smoke-run the two main downstream studies**

Run: `python3 macro_outcomes.py`
Expected: writes macro outcomes from canonical base without needing a tagged parquet.

Run: `python3 features/macro_fvg_study.py`
Expected: writes FVG event and summary outputs from canonical base without needing persisted `window`.

- [ ] **Step 4: Commit**

```bash
git add session_tagger.py macro_outcomes.py features/macro_fvg_study.py features/pm_3pm.py \
  viz/macro_analysis.py utils/minute_bars.py test/test_minute_bars.py test/test_session_tagger.py \
  test/test_macro_outcomes.py test/test_pm_3pm.py test/test_macro_fvg_study.py
git commit -m "feat: migrate minute pipeline to canonical utc base"
```
