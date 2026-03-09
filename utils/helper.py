# src/utils/news_joiner.py
from __future__ import annotations
import pandas as pd
import re
from zoneinfo import ZoneInfo
from typing import Iterable, List, Tuple

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---- Event classifier (lightweight; tweak as you like) ----------------------
_EVENT_PATTERNS: List[Tuple[str, Iterable[str]]] = [
    ("FOMC_RATE",    [r"\b(fomc|fed funds|rate decision|interest rate decision)\b"]),
    ("FOMC_MINUTES", [r"\b(fomc minutes)\b"]),
    ("FOMC_PRESSER", [r"\b(press conference|powell speaks|fed chair .*press)\b"]),
    ("CPI",          [r"\b(cpi|consumer price)\b"]),
    ("PCE",          [r"\b(pce|personal consumption expenditures)\b"]),
    ("NFP",          [r"\b(nonfarm|non-farm|payroll)\b"]),
    ("PPI",          [r"\b(ppi|producer price)\b"]),
    ("ISM_MANUF",    [r"\b(ism).*manufacturing\b"]),
    ("ISM_SERVICES", [r"\b(ism).*services\b"]),
    ("RETAIL_SALES", [r"\b(retail sales)\b"]),
    ("GDP",          [r"\b(gdp)\b"]),
    ("HOLIDAY",      [r"\b(holiday|market holiday|bank holiday)\b"]),
]

def classify_event(title: str) -> str:
    if not isinstance(title, str):
        return "OTHER"
    t = title.lower()
    for label, pats in _EVENT_PATTERNS:
        for pat in pats:
            if re.search(pat, t):
                return label
    return "OTHER"

# ---- Robust news normalization ----------------------------------------------
def _prep_news(
    news_df: pd.DataFrame,
    impacts: Iterable[str] = ("high",),
    title_exclude_substrings: Iterable[str] = ("speaks",),
) -> pd.DataFrame:
    """
    Accepts any of:
      - 'datetime_utc' (preferred)      -> UTC timestamp per row
      - 'dt_utc' (already combined)     -> UTC timestamp per row
      - 'date' + 'time_utc'             -> will be combined, interpreted in UTC
    Required cols: impact, title
    Adds: dt_utc (tz-aware UTC), dt_et, date_et, event_type, id
    """
    n = news_df.copy()

    # Build dt_utc regardless of input schema
    if "datetime_utc" in n.columns:
        dt = pd.to_datetime(n["datetime_utc"], errors="coerce")
        if dt.dt.tz is None:  # naive -> localize as UTC
            dt = dt.dt.tz_localize(UTC)
        else:
            dt = dt.dt.tz_convert(UTC)
        n["dt_utc"] = dt

    elif "dt_utc" in n.columns:
        dt = pd.to_datetime(n["dt_utc"], errors="coerce")
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(UTC)
        else:
            dt = dt.dt.tz_convert(UTC)
        n["dt_utc"] = dt

    elif {"date", "time_utc"}.issubset(n.columns):
        n["dt_utc"] = pd.to_datetime(n["date"].astype(str) + " " + n["time_utc"], utc=True, errors="coerce")
    else:
        raise ValueError("news_df must contain one of: ['datetime_utc'] or ['dt_utc'] or ['date','time_utc'].")

    # Basic validation
    for col in ("impact", "title"):
        if col not in n.columns:
            raise ValueError(f"news_df missing required column '{col}'")

    # Filter impacts if provided
    if impacts is not None:
        n = n[n["impact"].isin(impacts)]

    # Drop generic speeches if desired
    for sub in (title_exclude_substrings or []):
        n = n[~n["title"].str.contains(sub, case=False, na=False)]

    # Classify
    n["event_type"] = n["title"].map(classify_event)

    # IDs
    if "id" not in n.columns:
        n["id"] = n.index.astype(str)

    # ET projections
    n["dt_et"] = n["dt_utc"].dt.tz_convert(ET)
    n["date_et"] = n["dt_et"].dt.date

    return n.sort_values("dt_utc").reset_index(drop=True)

# replace your existing _macro_close_dt_utc with this
from zoneinfo import ZoneInfo
import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

def _macro_close_dt_utc(date_et) -> pd.Timestamp:
    """
    Given an ET trading date (date/datetime-like), return the UTC timestamp
    for macro close (15:50 ET) on that date. DST-safe.
    """
    # ensure plain date
    d = pd.to_datetime(date_et).date()
    # midnight ET on that date
    midnight_et = pd.Timestamp(d).tz_localize(ET)
    # add 15:50
    mc_et = midnight_et + pd.Timedelta(hours=15, minutes=50)
    # convert to UTC
    return mc_et.tz_convert(UTC)

# ---- Daily (wide) -----------------------------------------------------------
def merge_news_daily(
    macro_df: pd.DataFrame,
    news_df: pd.DataFrame,
    impacts: Iterable[str] = ("high",),
    include_nextday_premarket: bool = True,
    premarket_end_et: str = "09:30",
) -> pd.DataFrame:
    """
    Adds day-level fields for TODAY and NEXT-PREMARKET relative to ET trading date.
    Compatible with news that stores 'datetime_utc'.
    """
    m = macro_df.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.date

    n = _prep_news(news_df, impacts=impacts)

    pre_cut = pd.to_datetime(premarket_end_et).time()
    n["is_premarket"] = n["dt_et"].dt.time < pre_cut

    # Today aggregation
    today = n.groupby("date_et").agg(
        events_today=("id", list),
        titles_today=("title", list),
        types_today=("event_type", list),
        num_today=("id", "count"),
    ).rename_axis("date").reset_index()

    # Next-day premarket linked back to previous ET date
    if include_nextday_premarket:
        ndp = n[n["is_premarket"]].copy()
        ndp["prev_date"] = (pd.to_datetime(ndp["date_et"]) - pd.Timedelta(days=1)).dt.date
        next_pre = ndp.groupby("prev_date").agg(
            events_next_premarket=("id", list),
            titles_next_premarket=("title", list),
            types_next_premarket=("event_type", list),
            num_next_premarket=("id", "count"),
        ).rename_axis("date").reset_index().rename(columns={"prev_date":"date"})
    else:
        next_pre = pd.DataFrame({"date": []})

    out = m.merge(today, on="date", how="left").merge(next_pre, on="date", how="left")

    # Fill list and counts
    for col in ["events_today","titles_today","types_today",
                "events_next_premarket","titles_next_premarket","types_next_premarket"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: x if isinstance(x, list) else [])
    out["num_today"] = out.get("num_today", 0).fillna(0).astype(int)
    out["num_next_premarket"] = out.get("num_next_premarket", 0).fillna(0).astype(int)
    out["num_total"] = out["num_today"] + out["num_next_premarket"]

    out["has_event_today"] = out["num_today"] > 0
    out["has_event_next_premarket"] = out["num_next_premarket"] > 0
    out["has_event_total"] = out["num_total"] > 0

    # Per-class booleans
    all_types = sorted(set(n["event_type"]))
    def has_type(lst, t): return t in set(lst)
    for t in all_types:
        out[f"has_{t}_today"] = out["types_today"].apply(lambda xs, t=t: has_type(xs, t))
        out[f"has_{t}_next_premarket"] = out["types_next_premarket"].apply(lambda xs, t=t: has_type(xs, t))
        out[f"has_{t}_total"] = out[f"has_{t}_today"] | out[f"has_{t}_next_premarket"]

    return out

# ---- Long links for causal work ---------------------------------------------
def build_macro_event_links(
    macro_df: pd.DataFrame,
    news_df: pd.DataFrame,
    impacts: Iterable[str] = ("high",),
    include_nextday_premarket: bool = True,
    premarket_end_et: str = "09:30",
) -> pd.DataFrame:
    """
    One row per (macro_date, event):
      relation ∈ {"today","next_premarket"}
      minutes_from_macro_close is signed (event - macro_close).
    Works with news that store 'datetime_utc'.
    """
    m = macro_df.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.date

    n = _prep_news(news_df, impacts=impacts)
    pre_cut = pd.to_datetime(premarket_end_et).time()
    n["is_premarket"] = n["dt_et"].dt.time < pre_cut

    rows = []
    for d in m["date"].unique():
        mc_utc = _macro_close_dt_utc(d)

        # today
        td = n[n["date_et"] == d]
        for _, r in td.iterrows():
            rows.append({
                "macro_date": d,
                "macro_close_dt_utc": mc_utc,
                "relation": "today",
                "event_id": r["id"],
                "event_dt_utc": r["dt_utc"],
                "event_dt_et": r["dt_et"],
                "impact": r["impact"],
                "title": r["title"],
                "event_type": r["event_type"],
                "minutes_from_macro_close": (r["dt_utc"] - mc_utc).total_seconds() / 60.0,
            })

        # next-day premarket -> link back to d
        if include_nextday_premarket:
            d1 = (pd.to_datetime(d) + pd.Timedelta(days=1)).date()
            pre = n[(n["date_et"] == d1) & (n["is_premarket"])]
            for _, r in pre.iterrows():
                rows.append({
                    "macro_date": d,
                    "macro_close_dt_utc": mc_utc,
                    "relation": "next_premarket",
                    "event_id": r["id"],
                    "event_dt_utc": r["dt_utc"],
                    "event_dt_et": r["dt_et"],
                    "impact": r["impact"],
                    "title": r["title"],
                    "event_type": r["event_type"],
                    "minutes_from_macro_close": (r["dt_utc"] - mc_utc).total_seconds() / 60.0,
                })

    links = pd.DataFrame(rows)
    return links.sort_values(["macro_date","relation","event_dt_utc"])
