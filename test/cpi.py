# src/research/news_effects_cpi.py
from __future__ import annotations
import argparse
import numpy as np
import polars as pl

from utils.helper import build_macro_event_links


def permutation_test_mean_diff(a: np.ndarray, b: np.ndarray, iters: int = 100_000, rng: int = 42) -> float:
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
    return (count + 1) / (iters + 1)


def _finite_values(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def summarize(values) -> dict:
    s = _finite_values(values)
    return {
        "n": len(s),
        "mean": float(s.mean()) if len(s) else np.nan,
        "median": float(np.median(s)) if len(s) else np.nan,
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
    }


def build_cohorts(macro_df: pl.DataFrame, links_df: pl.DataFrame) -> pl.DataFrame:
    cpi = links_df.filter(pl.col("event_type") == "CPI")
    if cpi.is_empty():
        flags = pl.DataFrame(schema={"date": pl.Date, "num_CPI_today": pl.Int64, "num_CPI_next_premarket": pl.Int64})
    else:
        flags = (
            cpi.group_by(["macro_date", "relation"])
            .len()
            .pivot(index="macro_date", on="relation", values="len")
            .rename({"macro_date": "date"})
        )
        if "today" not in flags.columns:
            flags = flags.with_columns(pl.lit(0).alias("today"))
        if "next_premarket" not in flags.columns:
            flags = flags.with_columns(pl.lit(0).alias("next_premarket"))
        flags = flags.rename({"today": "num_CPI_today", "next_premarket": "num_CPI_next_premarket"})

    return (
        macro_df.join(flags, on="date", how="left")
        .with_columns(
            pl.col("num_CPI_today").fill_null(0).cast(pl.Int64),
            pl.col("num_CPI_next_premarket").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            has_CPI_today=pl.col("num_CPI_today") > 0,
            has_CPI_next_premarket=pl.col("num_CPI_next_premarket") > 0,
        )
        .with_columns(has_CPI_total=pl.col("has_CPI_today") | pl.col("has_CPI_next_premarket"))
        .with_columns(
            cohort=pl.when(pl.col("has_CPI_today"))
            .then(pl.lit("CPI_today"))
            .when(pl.col("has_CPI_next_premarket"))
            .then(pl.lit("CPI_next_premarket"))
            .otherwise(pl.lit("Control"))
        )
    )


def cohens_d(x, y) -> float:
    x = _finite_values(x)
    y = _finite_values(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    return float((x.mean() - y.mean()) / sp)


def main():
    p = argparse.ArgumentParser(description="CPI causality quick test")
    p.add_argument("--metrics", nargs="+", default=["macro_range_pct"])
    p.add_argument("--premarket_end_et", default="09:30")
    p.add_argument("--iters", type=int, default=100_000)
    args = p.parse_args()

    macro = pl.read_parquet("data/macro_outcomes.parquet").with_columns(pl.col("date").cast(pl.Date))
    news = pl.read_parquet("data/economic_events.parquet")
    links = build_macro_event_links(macro_df=macro.select("date"), news_df=news, impacts=("high",), include_nextday_premarket=True, premarket_end_et=args.premarket_end_et)
    df = build_cohorts(macro, links)

    print("\n=== Cohort sizes ===")
    print(df.group_by("cohort").len().sort("cohort"))

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"\n[WARN] Metric '{metric}' not found in macro df. Skipping.")
            continue
        print(f"\n=== Metric: {metric} ===")
        a = df.filter(pl.col("cohort") == "CPI_today")[metric].to_numpy()
        b = df.filter(pl.col("cohort") == "CPI_next_premarket")[metric].to_numpy()
        c = df.filter(pl.col("cohort") == "Control")[metric].to_numpy()
        for label, values in {"CPI_today": a, "CPI_next_premarket": b, "Control": c}.items():
            v = summarize(values)
            print(f"{label:>18}: n={v['n']:>4}  mean={v['mean']:.4f}  median={v['median']:.4f}  std={v['std']:.4f}")
        print("\nPermutation test p-values (two-sided, mean difference):")
        print(f"  CPI_today vs Control        p = {permutation_test_mean_diff(a, c, iters=args.iters):.5f}")
        print(f"  CPI_next_premarket vs Ctrl  p = {permutation_test_mean_diff(b, c, iters=args.iters):.5f}")
        print(f"  CPI_today vs CPI_next_pm    p = {permutation_test_mean_diff(a, b, iters=args.iters):.5f}")
        print("\nEffect sizes (Cohen's d):")
        print(f"  CPI_today vs Control        d = {cohens_d(a, c):.3f}")
        print(f"  CPI_next_premarket vs Ctrl  d = {cohens_d(b, c):.3f}")
        print(f"  CPI_today vs CPI_next_pm    d = {cohens_d(a, b):.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
