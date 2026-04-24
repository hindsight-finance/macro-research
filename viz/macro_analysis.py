from pathlib import Path
from datetime import datetime, time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

INPUT_BARS = "outputs/nq_minute_base.parquet"
OUT_DIR = "outputs/figs/ma"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_col(ci_names, *candidates):
    for cand in candidates:
        if cand in ci_names:
            return ci_names[cand]
    return None


def read_bars(path: str) -> pl.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        df = pl.read_parquet(path)
    elif ext == ".csv":
        df = pl.read_csv(path, try_parse_dates=True)
    else:
        raise ValueError("Unsupported file extension. Use .csv or .parquet")
    ci = {c.lower(): c for c in df.columns}
    dt_col = _find_col(ci, "datetime_utc", "datetime_et", "date_time_et", "datetime", "timestamp", "time_et", "dt_et", "dt", "time", "date_time")
    if dt_col == ci.get("datetime_utc"):
        dt_expr = pl.col(dt_col).cast(pl.String).str.to_datetime(time_zone="UTC", strict=False).dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
    elif dt_col is not None:
        dt_expr = pl.col(dt_col).cast(pl.String).str.to_datetime(strict=False)
    else:
        date_col = _find_col(ci, "date", "trade_date", "session_date")
        time_col = _find_col(ci, "time", "time_et", "bar_time")
        if not (date_col and time_col):
            raise ValueError("Could not find a datetime column.")
        dt_expr = (pl.col(date_col).cast(pl.String) + pl.lit(" ") + pl.col(time_col).cast(pl.String)).str.to_datetime(strict=False)
    O_col = _find_col(ci, "open", "o")
    H_col = _find_col(ci, "high", "h")
    L_col = _find_col(ci, "low", "l")
    C_col = _find_col(ci, "close", "c", "last")
    missing = [name for name, col in [["open", O_col], ["high", H_col], ["low", L_col], ["close", C_col]] if col is None]
    if missing:
        raise ValueError(f"Missing required OHLC column(s): {', '.join(missing)}")
    out = df.select(
        DateTime_ET=dt_expr,
        Open=pl.col(O_col).cast(pl.Float64),
        High=pl.col(H_col).cast(pl.Float64),
        Low=pl.col(L_col).cast(pl.Float64),
        Close=pl.col(C_col).cast(pl.Float64),
    ).drop_nulls("DateTime_ET")
    return out.with_columns(date=pl.col("DateTime_ET").dt.date(), time=pl.col("DateTime_ET").dt.time()).select(["DateTime_ET", "date", "time", "Open", "High", "Low", "Close"])


def skew_from_ohlc(o, h, l, c):
    rng = h - l
    return 0.0 if rng == 0 else float((c - o) / rng)


def pct_range(points, ref):
    return np.nan if ref == 0 or np.isnan(ref) else (points / ref) * 100.0


def stage_slice(df_day: pl.DataFrame, start_str: str, end_str: str) -> pl.DataFrame:
    d = df_day.row(0, named=True)["date"]
    t0 = datetime.combine(d, time.fromisoformat(start_str))
    t1 = datetime.combine(d, time.fromisoformat(end_str))
    return df_day.filter((pl.col("DateTime_ET") >= t0) & (pl.col("DateTime_ET") <= t1))


def candle_at(df_day: pl.DataFrame, time_str: str) -> pl.DataFrame:
    d = df_day.row(0, named=True)["date"]
    t = datetime.combine(d, time.fromisoformat(time_str))
    return df_day.filter(pl.col("DateTime_ET") == t)


def added_range(prev_low, prev_high, new_low, new_high):
    return max(0.0, prev_low - new_low) + max(0.0, new_high - prev_high)


def _first_last_high_low(df: pl.DataFrame, ref_open: float | None = None) -> dict:
    if df.is_empty():
        return dict(range_pts=np.nan, range_pct=np.nan, dir_pts=np.nan, dir_pct=np.nan, skew=np.nan, O=np.nan, C=np.nan, H=np.nan, L=np.nan)
    rows = df.to_dicts()
    o, c = float(rows[0]["Open"]), float(rows[-1]["Close"])
    h, l = float(df["High"].max()), float(df["Low"].min())
    r_pts, d_pts = h - l, c - o
    ref = o if ref_open is None else ref_open
    return dict(range_pts=r_pts, range_pct=pct_range(r_pts, ref), dir_pts=d_pts, dir_pct=pct_range(d_pts, ref), skew=skew_from_ohlc(o, h, l, c), O=o, C=c, H=h, L=l)


def compute_day_stats(df_day: pl.DataFrame) -> dict | None:
    macro = stage_slice(df_day, "15:50:00", "15:59:00")
    if macro.is_empty() or macro.height < 5:
        return None
    m = _first_last_high_low(macro)
    O, C, H, L = m["O"], m["C"], m["H"], m["L"]
    macro_range_pts, macro_dir_pts = H - L, C - O
    s1 = _first_last_high_low(stage_slice(df_day, "15:50:00", "15:54:00"), O)
    s2 = _first_last_high_low(stage_slice(df_day, "15:55:00", "15:59:00"), O)
    k350 = _first_last_high_low(candle_at(df_day, "15:50:00"), O)
    k355 = _first_last_high_low(candle_at(df_day, "15:55:00"), O)
    k359 = _first_last_high_low(candle_at(df_day, "15:59:00"), O)
    macro_dir = np.sign(macro_dir_pts) if macro_dir_pts != 0 else 0
    s1_dir = np.sign(s1["dir_pts"]) if not np.isnan(s1["dir_pts"]) else 0
    s2_dir = np.sign(s2["dir_pts"]) if not np.isnan(s2["dir_pts"]) else 0
    s1_contrib_pts = s1["range_pts"] if not np.isnan(s1["range_pts"]) else np.nan
    pre_env_low, pre_env_high = (s1["L"], s1["H"]) if not np.isnan(s1_contrib_pts) else (np.inf, -np.inf)
    s2_contrib_pts = added_range(pre_env_low, pre_env_high, s2["L"], s2["H"]) if not np.isnan(s1_contrib_pts) and not np.isnan(s2["range_pts"]) else s2["range_pts"]
    c350_contrib_pts = k350["range_pts"] if not np.isnan(k350["range_pts"]) else np.nan
    env_low, env_high = (k350["L"], k350["H"]) if not np.isnan(c350_contrib_pts) else (np.inf, -np.inf)
    c355_contrib_pts = added_range(env_low, env_high, k355["L"], k355["H"]) if not np.isnan(c350_contrib_pts) and not np.isnan(k355["range_pts"]) else np.nan
    if not np.isnan(c355_contrib_pts):
        env_low, env_high = min(env_low, k355["L"]), max(env_high, k355["H"])
    c359_contrib_pts = added_range(env_low, env_high, k359["L"], k359["H"]) if env_low != np.inf and not np.isnan(k359["range_pts"]) else np.nan
    pre59, c359 = stage_slice(df_day, "15:50:00", "15:58:00"), candle_at(df_day, "15:59:00")
    if not pre59.is_empty() and not c359.is_empty():
        pre59_H, pre59_L = float(pre59["High"].max()), float(pre59["Low"].min())
        macro_dir_sign = int(np.sign(macro_dir_pts))
        expanded_up = k359["C"] > pre59_H if macro_dir_sign >= 0 else False
        expanded_down = k359["C"] < pre59_L if macro_dir_sign <= 0 else False
        c359_expanded_dir = "up" if expanded_up else ("down" if expanded_down else "none")
        retraced_inside = (min(k359["H"], pre59_H) - max(k359["L"], pre59_L)) > 0.0
        push_past_pts = max(0.0, k359["H"] - pre59_H) if c359_expanded_dir == "up" else (max(0.0, pre59_L - k359["L"]) if c359_expanded_dir == "down" else 0.0)
        retrace_depth_pts = max(0.0, pre59_H - k359["L"]) if c359_expanded_dir == "up" else (max(0.0, k359["H"] - pre59_L) if c359_expanded_dir == "down" else 0.0)
        c359_behavior = "expand_and_retrace" if c359_expanded_dir != "none" and retraced_inside else ("expand_only" if c359_expanded_dir != "none" else ("band_only" if retraced_inside else "none"))
    else:
        c359_behavior, c359_expanded_dir, retraced_inside, push_past_pts, retrace_depth_pts = "unknown", "none", False, 0.0, 0.0
    return dict(
        date=df_day.row(0, named=True)["date"].isoformat(), macro_open=O, macro_close=C, macro_high=H, macro_low=L,
        macro_range_points=macro_range_pts, macro_range_pct=pct_range(macro_range_pts, O), macro_dir_points=macro_dir_pts, macro_dir_pct=pct_range(macro_dir_pts, O), macro_skew=skew_from_ohlc(O, H, L, C),
        stage1_range_pct=s1["range_pct"], stage1_skew=s1["skew"], stage2_range_pct=s2["range_pct"], stage2_skew=s2["skew"], k350_range_pct=k350["range_pct"], k350_skew=k350["skew"], k355_range_pct=k355["range_pct"], k355_skew=k355["skew"], k359_range_pct=k359["range_pct"], k359_skew=k359["skew"],
        s1_dir=int(s1_dir), s2_dir=int(s2_dir), macro_dir=int(macro_dir), s1_contrib_pct=pct_range(s1_contrib_pts, O) if not np.isnan(s1_contrib_pts) else np.nan, s2_contrib_pct=pct_range(s2_contrib_pts, O) if not np.isnan(s2_contrib_pts) else np.nan, k350_contrib_pct=pct_range(c350_contrib_pts, O) if not np.isnan(c350_contrib_pts) else np.nan, k355_contrib_pct=pct_range(c355_contrib_pts, O) if not np.isnan(c355_contrib_pts) else np.nan, k359_contrib_pct=pct_range(c359_contrib_pts, O) if not np.isnan(c359_contrib_pts) else np.nan,
        c359_behavior=c359_behavior, c359_expanded_dir=c359_expanded_dir, c359_retraced_inside=bool(retraced_inside), c359_push_past_points=float(push_past_pts), c359_push_past_pct=float(pct_range(push_past_pts, O)), c359_retrace_depth_points=float(retrace_depth_pts), c359_retrace_depth_pct=float(pct_range(retrace_depth_pts, O))
    )


def _clean(series) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    return arr[np.isfinite(arr)]


def hist_save(series, title, out_path, bins=40):
    s = _clean(series)
    if len(s) == 0:
        return
    plt.figure(); plt.hist(s, bins=bins); plt.title(title); plt.xlabel("Value"); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(out_path); plt.close()


def bar_save(labels, values, title, out_path):
    if len(values) == 0:
        return
    plt.figure(); x = np.arange(len(labels)); plt.bar(x, values); plt.xticks(x, labels, rotation=0); plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()


def stacked_bar_save(labels, stacks, stack_labels, title, out_path):
    if not stacks or len(stacks[0]) == 0:
        return
    plt.figure(); x = np.arange(len(labels)); bottom = np.zeros(len(labels))
    for arr, lab in zip(stacks, stack_labels):
        plt.bar(x, arr, bottom=bottom, label=lab); bottom = bottom + np.array(arr)
    plt.xticks(x, labels, rotation=0); plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()


def main():
    outdir = Path(OUT_DIR); ensure_dir(outdir)
    df = read_bars(INPUT_BARS)
    macro_days = df.filter((pl.col("DateTime_ET").dt.hour() == 15) & (pl.col("DateTime_ET").dt.minute().is_between(50, 59)))["date"].unique().sort().to_list()
    results = []
    for d in macro_days:
        rec = compute_day_stats(df.filter(pl.col("date") == d).sort("DateTime_ET"))
        if rec is not None:
            results.append(rec)
    if not results:
        raise SystemExit("No macro records computed; check your input data covers 15:50–15:59 ET.")
    res = pl.DataFrame(results)
    res.write_csv(outdir / "macro_stage_stats.csv")
    for title, col, fn in [
        ("Total Macro Range %", "macro_range_pct", "hist_macro_range_pct.png"), ("Stage 1 Range % (15:50–15:54)", "stage1_range_pct", "hist_stage1_range_pct.png"), ("Stage 2 Range % (15:55–15:59)", "stage2_range_pct", "hist_stage2_range_pct.png"),
        ("3:50 Candle Range %", "k350_range_pct", "hist_k350_range_pct.png"), ("3:55 Candle Range %", "k355_range_pct", "hist_k355_range_pct.png"), ("3:59 Candle Range %", "k359_range_pct", "hist_k359_range_pct.png"),
        ("Total Macro Skew", "macro_skew", "hist_macro_skew.png"), ("Stage 1 Skew", "stage1_skew", "hist_stage1_skew.png"), ("Stage 2 Skew", "stage2_skew", "hist_stage2_skew.png"),
        ("3:50 Candle Skew", "k350_skew", "hist_k350_skew.png"), ("3:55 Candle Skew", "k355_skew", "hist_k355_skew.png"), ("3:59 Candle Skew", "k359_skew", "hist_k359_skew.png"),
        ("3:59 Push Past % (beyond pre-59 edge)", "c359_push_past_pct", "hist_c359_push_past_pct.png"), ("3:59 Retrace Depth % (back inside band)", "c359_retrace_depth_pct", "hist_c359_retrace_depth_pct.png"),
    ]:
        hist_save(res[col].to_numpy(), title, outdir / fn)
    s = res.drop_nulls(["c359_push_past_pct", "c359_retrace_depth_pct"])
    if not s.is_empty():
        plt.figure(); plt.scatter(s["c359_push_past_pct"].to_numpy(), s["c359_retrace_depth_pct"].to_numpy()); plt.xlabel("3:59 Push Past %"); plt.ylabel("3:59 Retrace Depth %"); plt.title("3:59 Push vs Retrace"); plt.tight_layout(); plt.savefig(outdir / "scatter_c359_push_vs_retrace.png"); plt.close()
    stacked_bar_save(["Avg Contribution"], [[float(np.nanmean(res["s1_contrib_pct"].to_numpy()))], [float(np.nanmean(res["s2_contrib_pct"].to_numpy()))]], ["Stage1", "Stage2"], "Average Stage Contributions (added range %, normalized by macro open)", outdir / "avg_stage_contributions.png")
    c_means = [float(np.nanmean(res[c].to_numpy())) for c in ["k350_contrib_pct", "k355_contrib_pct", "k359_contrib_pct"]]
    bar_save(["15:50", "15:55", "15:59"], c_means, "Average Candle Contributions (added range %, normalized by macro open)", outdir / "avg_candle_contributions.png")
    bar_save(["S1==Macro", "S2==Macro", "S1→S2 Continue", "S1→S2 Reverse"], [float((res["s1_dir"] == res["macro_dir"]).mean() * 100.0), float((res["s2_dir"] == res["macro_dir"]).mean() * 100.0), float((res["s1_dir"] == res["s2_dir"]).mean() * 100.0), 100.0 - float((res["s1_dir"] == res["s2_dir"]).mean() * 100.0)], "Directional Alignment (%)", outdir / "directional_alignment.png")
    bc = res.group_by("c359_behavior").len().sort("c359_behavior")
    bar_save(bc["c359_behavior"].to_list(), [float(x) for x in bc["len"].to_list()], "15:59 Candle: Behavior Classification (counts)", outdir / "c359_behavior_counts.png")
    hist_save(res.filter(pl.col("macro_dir") > 0)["macro_range_pct"].to_numpy(), "Macro Range % (Bullish Macro Only)", outdir / "hist_macro_range_pct_bull.png")
    hist_save(res.filter(pl.col("macro_dir") < 0)["macro_range_pct"].to_numpy(), "Macro Range % (Bearish Macro Only)", outdir / "hist_macro_range_pct_bear.png")
    res = res.with_columns(pl.col("macro_range_pct").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True).alias("macro_range_decile"))
    hist_save(res.filter(pl.col("macro_range_decile") == res["macro_range_decile"].min())["stage1_range_pct"].to_numpy(), "Stage1 Range % (Small Macro Days)", outdir / "hist_stage1_small_days.png")
    hist_save(res.filter(pl.col("macro_range_decile") == res["macro_range_decile"].max())["stage1_range_pct"].to_numpy(), "Stage1 Range % (Large Macro Days)", outdir / "hist_stage1_large_days.png")
    corr = res.select(["stage1_skew", "stage2_skew", "macro_skew"]).drop_nulls().corr()
    corr.write_csv(outdir / "skew_correlations.csv")
    pct_rows = []
    for q in [0.1, 0.5, 0.9, 0.95, 0.99]:
        pct_rows.append({"percentile": f"p{int(q*100)}", **{col: res[col].quantile(q) for col in ["macro_range_pct", "stage1_range_pct", "stage2_range_pct", "k359_range_pct"]}})
    pl.DataFrame(pct_rows).write_csv(outdir / "percentiles_table.csv")
    print("Saved outputs to:", outdir.resolve())


if __name__ == "__main__":
    main()
