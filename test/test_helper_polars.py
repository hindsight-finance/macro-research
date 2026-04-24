import polars as pl

from utils.helper import build_macro_event_links, merge_news_daily


def test_merge_news_daily_returns_polars_frame_with_today_and_next_premarket_flags():
    macro = pl.DataFrame({"date": ["2025-01-02"]}).with_columns(pl.col("date").str.to_date())
    news = pl.DataFrame(
        {
            "datetime_utc": ["2025-01-02T13:30:00Z", "2025-01-03T13:30:00Z"],
            "impact": ["high", "high"],
            "title": ["CPI release", "Nonfarm payrolls"],
        }
    )

    out = merge_news_daily(macro, news)

    assert isinstance(out, pl.DataFrame)
    assert out.item(0, "has_event_today")
    assert out.item(0, "has_event_next_premarket")
    assert out.item(0, "num_total") == 2


def test_build_macro_event_links_returns_polars_frame():
    macro = pl.DataFrame({"date": ["2025-01-02"]}).with_columns(pl.col("date").str.to_date())
    news = pl.DataFrame(
        {
            "datetime_utc": ["2025-01-02T13:30:00Z"],
            "impact": ["high"],
            "title": ["CPI release"],
        }
    )

    links = build_macro_event_links(macro, news)

    assert isinstance(links, pl.DataFrame)
    assert links.height == 1
    assert links.item(0, "relation") == "today"
    assert links.item(0, "event_type") == "CPI"
