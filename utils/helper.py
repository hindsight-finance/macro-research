from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Iterable, List, Tuple
from zoneinfo import ZoneInfo

import polars as pl

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

_EVENT_PATTERNS: List[Tuple[str, Iterable[str]]] = [
    ("FOMC_RATE", [r"\b(fomc|fed funds|rate decision|interest rate decision)\b"]),
    ("FOMC_MINUTES", [r"\b(fomc minutes)\b"]),
    ("FOMC_PRESSER", [r"\b(press conference|powell speaks|fed chair .*press)\b"]),
    ("CPI", [r"\b(cpi|consumer price)\b"]),
    ("PCE", [r"\b(pce|personal consumption expenditures)\b"]),
    ("NFP", [r"\b(nonfarm|non-farm|payroll)\b"]),
    ("PPI", [r"\b(ppi|producer price)\b"]),
    ("ISM_MANUF", [r"\b(ism).*manufacturing\b"]),
    ("ISM_SERVICES", [r"\b(ism).*services\b"]),
    ("RETAIL_SALES", [r"\b(retail sales)\b"]),
    ("GDP", [r"\b(gdp)\b"]),
    ("HOLIDAY", [r"\b(holiday|market holiday|bank holiday)\b"]),
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


def _ensure_polars(df) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def _prep_news(
    news_df,
    impacts: Iterable[str] = ("high",),
    title_exclude_substrings: Iterable[str] = ("speaks",),
) -> pl.DataFrame:
    n = _ensure_polars(news_df).clone()

    if "datetime_utc" in n.columns:
        n = n.with_columns(dt_utc=pl.col("datetime_utc").cast(pl.String).str.to_datetime(time_zone="UTC", strict=False))
    elif "dt_utc" in n.columns:
        n = n.with_columns(dt_utc=pl.col("dt_utc").cast(pl.String).str.to_datetime(time_zone="UTC", strict=False))
    elif {"date", "time_utc"}.issubset(n.columns):
        n = n.with_columns(dt_utc=(pl.col("date").cast(pl.String) + pl.lit(" ") + pl.col("time_utc").cast(pl.String)).str.to_datetime(time_zone="UTC", strict=False))
    else:
        raise ValueError("news_df must contain one of: ['datetime_utc'] or ['dt_utc'] or ['date','time_utc'].")

    for col in ("impact", "title"):
        if col not in n.columns:
            raise ValueError(f"news_df missing required column '{col}'")

    if impacts is not None:
        n = n.filter(pl.col("impact").is_in(list(impacts)))
    for sub in (title_exclude_substrings or []):
        n = n.filter(~pl.col("title").str.contains(sub, literal=True).fill_null(False))

    if "id" not in n.columns:
        n = n.with_row_index("id").with_columns(pl.col("id").cast(pl.String))

    return (
        n.with_columns(
            event_type=pl.col("title").map_elements(classify_event, return_dtype=pl.String),
            dt_et=pl.col("dt_utc").dt.convert_time_zone("America/New_York"),
        )
        .with_columns(date_et=pl.col("dt_et").dt.date())
        .sort("dt_utc")
    )


def _macro_close_dt_utc(date_et) -> datetime:
    d = date_et if isinstance(date_et, date) and not isinstance(date_et, datetime) else date.fromisoformat(str(date_et)[:10])
    mc_et = datetime.combine(d, time(15, 50), tzinfo=ET)
    return mc_et.astimezone(UTC)


def merge_news_daily(
    macro_df,
    news_df,
    impacts: Iterable[str] = ("high",),
    include_nextday_premarket: bool = True,
    premarket_end_et: str = "09:30",
) -> pl.DataFrame:
    m = _ensure_polars(macro_df).with_columns(pl.col("date").cast(pl.Date))
    n = _prep_news(news_df, impacts=impacts)
    pre_h, pre_m = map(int, premarket_end_et.split(":"))
    n = n.with_columns(is_premarket=(pl.col("dt_et").dt.time() < time(pre_h, pre_m)))

    today = n.group_by("date_et").agg(
        events_today=pl.col("id"),
        titles_today=pl.col("title"),
        types_today=pl.col("event_type"),
        num_today=pl.len(),
    ).rename({"date_et": "date"})

    if include_nextday_premarket:
        next_pre = (
            n.filter(pl.col("is_premarket"))
            .with_columns(date=(pl.col("date_et") - pl.duration(days=1)).cast(pl.Date))
            .group_by("date")
            .agg(
                events_next_premarket=pl.col("id"),
                titles_next_premarket=pl.col("title"),
                types_next_premarket=pl.col("event_type"),
                num_next_premarket=pl.len(),
            )
        )
    else:
        next_pre = pl.DataFrame({"date": []}, schema={"date": pl.Date})

    out = m.join(today, on="date", how="left").join(next_pre, on="date", how="left")
    list_cols = ["events_today", "titles_today", "types_today", "events_next_premarket", "titles_next_premarket", "types_next_premarket"]
    for col in list_cols:
        if col not in out.columns:
            out = out.with_columns(pl.lit([]).alias(col))
    out = out.with_columns(
        pl.col("num_today").fill_null(0).cast(pl.Int64),
        pl.col("num_next_premarket").fill_null(0).cast(pl.Int64),
    ).with_columns(
        num_total=pl.col("num_today") + pl.col("num_next_premarket"),
        has_event_today=pl.col("num_today") > 0,
        has_event_next_premarket=pl.col("num_next_premarket") > 0,
    ).with_columns(has_event_total=pl.col("num_total") > 0)

    for t in sorted(n["event_type"].unique().to_list()):
        out = out.with_columns(
            pl.col("types_today").list.contains(t).fill_null(False).alias(f"has_{t}_today"),
            pl.col("types_next_premarket").list.contains(t).fill_null(False).alias(f"has_{t}_next_premarket"),
        ).with_columns((pl.col(f"has_{t}_today") | pl.col(f"has_{t}_next_premarket")).alias(f"has_{t}_total"))
    return out


def build_macro_event_links(
    macro_df,
    news_df,
    impacts: Iterable[str] = ("high",),
    include_nextday_premarket: bool = True,
    premarket_end_et: str = "09:30",
) -> pl.DataFrame:
    m = _ensure_polars(macro_df).with_columns(pl.col("date").cast(pl.Date))
    n = _prep_news(news_df, impacts=impacts)
    pre_h, pre_m = map(int, premarket_end_et.split(":"))
    n = n.with_columns(is_premarket=(pl.col("dt_et").dt.time() < time(pre_h, pre_m)))

    rows = []
    for d in m["date"].unique().sort().to_list():
        mc_utc = _macro_close_dt_utc(d)
        for r in n.filter(pl.col("date_et") == d).iter_rows(named=True):
            rows.append({"macro_date": d, "macro_close_dt_utc": mc_utc, "relation": "today", "event_id": r["id"], "event_dt_utc": r["dt_utc"], "event_dt_et": r["dt_et"], "impact": r["impact"], "title": r["title"], "event_type": r["event_type"], "minutes_from_macro_close": (r["dt_utc"] - mc_utc).total_seconds() / 60.0})
        if include_nextday_premarket:
            d1 = d + timedelta(days=1)
            for r in n.filter((pl.col("date_et") == d1) & pl.col("is_premarket")).iter_rows(named=True):
                rows.append({"macro_date": d, "macro_close_dt_utc": mc_utc, "relation": "next_premarket", "event_id": r["id"], "event_dt_utc": r["dt_utc"], "event_dt_et": r["dt_et"], "impact": r["impact"], "title": r["title"], "event_type": r["event_type"], "minutes_from_macro_close": (r["dt_utc"] - mc_utc).total_seconds() / 60.0})
    return pl.DataFrame(rows).sort(["macro_date", "relation", "event_dt_utc"]) if rows else pl.DataFrame()
