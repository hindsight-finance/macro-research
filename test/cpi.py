# src/research/news_effects_cpi.py
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np

# relies on your earlier helper file:
#   src/utils/news_joiner.py with build_macro_event_links()
from utils.helper import build_macro_event_links

def permutation_test_mean_diff(a: np.ndarray, b: np.ndarray, iters: int = 100_000, rng: int = 42) -> float:
    """
    Two-sample permutation test (difference in means, A - B).
    Returns a two-sided p-value. No external deps.
    """
    rng = np.random.default_rng(rng)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan

    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(iters):
        rng.shuffle(pool)
        diff = pool[:n_a].mean() - pool[n_a:].mean()
        if abs(diff) >= abs(obs):
            count += 1
    return (count + 1) / (iters + 1)  # add-1 smoothing

def summarize(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    return dict(n=len(s), mean=float(s.mean()) if len(s) else np.nan,
                median=float(s.median()) if len(s) else np.nan,
                std=float(s.std(ddof=1)) if len(s) > 1 else np.nan)

def build_cohorts(macro_df: pd.DataFrame, links_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds CPI flags to macro_df:
      has_CPI_today, has_CPI_next_premarket, has_CPI_total, cohort label
    """
    # Only CPI events (impact filtered is handled inside build_macro_event_links call)
    cpi = links_df[links_df["event_type"] == "CPI"].copy()

    # Per-date flags
    flags = (
        cpi.groupby(["macro_date", "relation"])
            .size()
            .unstack(fill_value=0)
            .rename_axis(None, axis=1)
            .reset_index()
    )
    flags = flags.rename(columns={
        "macro_date": "date",
        "today": "num_CPI_today",
        "next_premarket": "num_CPI_next_premarket",
    })
    if "num_CPI_today" not in flags.columns:
        flags["num_CPI_today"] = 0
    if "num_CPI_next_premarket" not in flags.columns:
        flags["num_CPI_next_premarket"] = 0

    out = macro_df.merge(flags, on="date", how="left")
    out[["num_CPI_today", "num_CPI_next_premarket"]] = out[
        ["num_CPI_today", "num_CPI_next_premarket"]
    ].fillna(0).astype(int)

    out["has_CPI_today"] = out["num_CPI_today"] > 0
    out["has_CPI_next_premarket"] = out["num_CPI_next_premarket"] > 0
    out["has_CPI_total"] = out["has_CPI_today"] | out["has_CPI_next_premarket"]

    # Label cohorts (mutually exclusive: today > next_premarket > control)
    def label_row(r):
        if r["has_CPI_today"]:
            return "CPI_today"
        if r["has_CPI_next_premarket"]:
            return "CPI_next_premarket"
        return "Control"

    out["cohort"] = out.apply(label_row, axis=1)
    return out

def main():
    p = argparse.ArgumentParser(description="CPI causality quick test")
    p.add_argument("--metrics", nargs="+", default=["macro_range_pct"],
                   help="Macro outcome columns to test (e.g., macro_range pm_range hr3_range)")
    p.add_argument("--premarket_end_et", default="09:30", help="Premarket cutoff ET for next-day link (default 09:30)")
    p.add_argument("--iters", type=int, default=100_000, help="Permutation iterations (default 100k)")
    args = p.parse_args()

    # --- Load data
    macro = pd.read_parquet("data/macro_outcomes.parquet")
    news = pd.read_parquet("data/economic_events.parquet")

    # Dates as ET trading dates
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    # --- Build links (only high impact; DST-safe handled in helper)
    links = build_macro_event_links(
        macro_df=macro[["date"]],
        news_df=news,
        impacts=("high",),                   # keep holidays out of the driver set
        include_nextday_premarket=True,
        premarket_end_et=args.premarket_end_et,
    )

    # --- CPI cohorts
    df = build_cohorts(macro, links)

    # OPTIONAL: if you carry a holiday flag in macro, you can drop those days:
    # df = df[~df["is_bank_holiday_today"]]

    # --- Report
    print("\n=== Cohort sizes ===")
    print(df["cohort"].value_counts().to_string())

    # --- Tests per metric
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"\n[WARN] Metric '{metric}' not found in macro df. Skipping.")
            continue

        print(f"\n=== Metric: {metric} ===")
        # Cohort splits
        a = df.loc[df["cohort"] == "CPI_today", metric]
        b = df.loc[df["cohort"] == "CPI_next_premarket", metric]
        c = df.loc[df["cohort"] == "Control", metric]

        # Summaries
        summ = {
            "CPI_today": summarize(a),
            "CPI_next_premarket": summarize(b),
            "Control": summarize(c),
        }
        for k, v in summ.items():
            print(f"{k:>18}: n={v['n']:>4}  mean={v['mean']:.4f}  median={v['median']:.4f}  std={v['std']:.4f}")

        # Permutation tests (two-sided)
        # Today vs Control
        p_tc = permutation_test_mean_diff(a.values, c.values, iters=args.iters)
        # NextPremarket vs Control
        p_nc = permutation_test_mean_diff(b.values, c.values, iters=args.iters)
        # Today vs NextPremarket
        p_tn = permutation_test_mean_diff(a.values, b.values, iters=args.iters)

        print("\nPermutation test p-values (two-sided, mean difference):")
        print(f"  CPI_today vs Control        p = {p_tc:.5f}")
        print(f"  CPI_next_premarket vs Ctrl  p = {p_nc:.5f}")
        print(f"  CPI_today vs CPI_next_pm    p = {p_tn:.5f}")

        # Simple effect sizes (Cohen's d)
        def cohens_d(x, y):
            x = x.dropna().astype(float).values
            y = y.dropna().astype(float).values
            if len(x) < 2 or len(y) < 2: return np.nan
            nx, ny = len(x), len(y)
            vx, vy = x.var(ddof=1), y.var(ddof=1)
            sp = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
            if sp == 0: return np.nan
            return (x.mean() - y.mean()) / sp

        d_tc = cohens_d(a, c)
        d_nc = cohens_d(b, c)
        d_tn = cohens_d(a, b)
        print("\nEffect sizes (Cohen's d):")
        print(f"  CPI_today vs Control        d = {d_tc:.3f}")
        print(f"  CPI_next_premarket vs Ctrl  d = {d_nc:.3f}")
        print(f"  CPI_today vs CPI_next_pm    d = {d_tn:.3f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
