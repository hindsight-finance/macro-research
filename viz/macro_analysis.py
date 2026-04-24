
import os
from pathlib import Path
import numpy as np
import polars as pl
pd = __import__("pandas")
import matplotlib.pyplot as plt

# =============================
# HARD-CODED PATHS (edit these)
# =============================
INPUT_BARS = "outputs/nq_minute_base.parquet"   # <-- set to your bars file (.csv or .parquet)
OUT_DIR    = "outputs/figs/ma"                  # <-- set to your desired output folder

# ---------- Helpers ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _find_col(ci_names, *candidates):
    for cand in candidates:
        if cand in ci_names:
            return ci_names[cand]
    return None

def _coerce_datetime(series):
    try:
        return pd.to_datetime(series, errors="raise")
    except Exception:
        return None

def read_bars(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        df = pl.read_parquet(path).to_pandas()
    elif ext == ".csv":
        df = pl.read_csv(path).to_pandas()
    else:
        raise ValueError("Unsupported file extension. Use .csv or .parquet")

    ci = {c.lower(): c for c in df.columns}

    dt_col = _find_col(ci, "datetime_utc", "datetime_et", "date_time_et", "datetime", "timestamp", "time_et", "dt_et", "dt", "time", "date_time")
    if dt_col == ci.get("datetime_utc"):
        dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        if dt_col is None:
            date_col = _find_col(ci, "date", "trade_date", "session_date")
            time_col = _find_col(ci, "time", "time_et", "bar_time")
            if date_col and time_col:
                dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
            else:
                dt = None
                for col in df.columns:
                    cand = _coerce_datetime(df[col])
                    if cand is not None and cand.notna().mean() > 0.95:
                        dt = cand
                        break
                if dt is None:
                    raise ValueError("Could not find a datetime column.")
        else:
            dt = pd.to_datetime(df[dt_col], errors="coerce")

    if dt.isna().all():
        raise ValueError(f"Datetime column '{dt_col}' could not be parsed.")

    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        dt = dt.dt.tz_convert("America/New_York").dt.tz_localize(None)

    O_col = _find_col(ci, "open", "o")
    H_col = _find_col(ci, "high", "h")
    L_col = _find_col(ci, "low", "l")
    C_col = _find_col(ci, "close", "c", "last")
    missing = [name for name, col in [["open", O_col], ["high", H_col], ["low", L_col], ["close", C_col]] if col is None]
    if missing:
        raise ValueError(f"Missing required OHLC column(s): {', '.join(missing)}")

    out = pd.DataFrame({
        "DateTime_ET": dt,
        "Open": df[O_col].astype(float),
        "High": df[H_col].astype(float),
        "Low":  df[L_col].astype(float),
        "Close":df[C_col].astype(float),
    }).dropna(subset=["DateTime_ET"])

    out["date"] = out["DateTime_ET"].dt.date
    out["time"] = out["DateTime_ET"].dt.time
    return out[["DateTime_ET", "date", "time", "Open", "High", "Low", "Close"]]

def skew_from_ohlc(o, h, l, c):
    rng = h - l
    if rng == 0:
        return 0.0
    return float((c - o) / rng)

def pct_range(points, ref):
    if ref == 0 or np.isnan(ref):
        return np.nan
    return (points / ref) * 100.0

def stage_slice(df_day: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    t0 = pd.to_datetime(df_day["date"].iloc[0].strftime("%Y-%m-%d") + " " + start_str)
    t1 = pd.to_datetime(df_day["date"].iloc[0].strftime("%Y-%m-%d") + " " + end_str)
    return df_day[(df_day["DateTime_ET"] >= t0) & (df_day["DateTime_ET"] <= t1)]

def candle_at(df_day: pd.DataFrame, time_str: str) -> pd.DataFrame:
    t = pd.to_datetime(df_day["date"].iloc[0].strftime("%Y-%m-%d") + " " + time_str)
    return df_day[df_day["DateTime_ET"] == t]

def added_range(prev_low, prev_high, new_low, new_high):
    add_low = max(0.0, prev_low - new_low)
    add_high = max(0.0, new_high - prev_high)
    return add_low + add_high

# ---------- Core per-day computation ----------

def compute_day_stats(df_day: pd.DataFrame) -> dict:
    macro = stage_slice(df_day, "15:50:00", "15:59:00")
    if macro.empty or len(macro) < 5:
        return None

    O = float(macro["Open"].iloc[0])
    C = float(macro["Close"].iloc[-1])
    H = float(macro["High"].max())
    L = float(macro["Low"].min())

    macro_range_pts = H - L
    macro_dir_pts = C - O
    macro_range_pct = pct_range(macro_range_pts, O)
    macro_dir_pct = pct_range(macro_dir_pts, O)
    macro_skew = skew_from_ohlc(O, H, L, C)

    stage1 = stage_slice(df_day, "15:50:00", "15:54:00")
    stage2 = stage_slice(df_day, "15:55:00", "15:59:00")

    def stage_stats(stage_df: pd.DataFrame):
        if stage_df.empty:
            return dict(range_pts=np.nan, range_pct=np.nan, dir_pts=np.nan, dir_pct=np.nan, skew=np.nan,
                        O=np.nan, C=np.nan, H=np.nan, L=np.nan)
        o = float(stage_df["Open"].iloc[0])
        c = float(stage_df["Close"].iloc[-1])
        h = float(stage_df["High"].max())
        l = float(stage_df["Low"].min())
        r_pts = h - l
        d_pts = c - o
        return dict(
            range_pts=r_pts,
            range_pct=pct_range(r_pts, O),
            dir_pts=d_pts,
            dir_pct=pct_range(d_pts, O),
            skew=skew_from_ohlc(o, h, l, c),
            O=o, C=c, H=h, L=l
        )

    s1 = stage_stats(stage1)
    s2 = stage_stats(stage2)

    c350 = candle_at(df_day, "15:50:00")
    c355 = candle_at(df_day, "15:55:00")
    c359 = candle_at(df_day, "15:59:00")

    def candle_stats(cdf: pd.DataFrame):
        if cdf.empty:
            return dict(range_pts=np.nan, range_pct=np.nan, skew=np.nan, O=np.nan, C=np.nan, H=np.nan, L=np.nan)
        row = cdf.iloc[0]
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        r_pts = h - l
        return dict(
            range_pts=r_pts,
            range_pct=pct_range(r_pts, O),
            skew=skew_from_ohlc(o, h, l, c),
            O=o, C=c, H=h, L=l
        )

    k350 = candle_stats(c350)
    k355 = candle_stats(c355)
    k359 = candle_stats(c359)

    macro_dir = np.sign(macro_dir_pts) if macro_dir_pts != 0 else 0
    s1_dir = np.sign(s1["dir_pts"]) if not np.isnan(s1["dir_pts"]) else 0
    s2_dir = np.sign(s2["dir_pts"]) if not np.isnan(s2["dir_pts"]) else 0

    # Contributions
    s1_contrib_pts = s1["range_pts"] if not np.isnan(s1["range_pts"]) else np.nan
    pre_env_low, pre_env_high = (s1["L"], s1["H"]) if not np.isnan(s1_contrib_pts) else (np.inf, -np.inf)

    s2_contrib_pts = np.nan
    if not np.isnan(s1_contrib_pts) and not np.isnan(s2["range_pts"]):
        s2_contrib_pts = added_range(pre_env_low, pre_env_high, s2["L"], s2["H"])
    elif np.isnan(s1_contrib_pts) and not np.isnan(s2["range_pts"]):
        s2_contrib_pts = s2["range_pts"]

    c350_contrib_pts = k350["range_pts"] if not np.isnan(k350["range_pts"]) else np.nan
    env_low, env_high = (k350["L"], k350["H"]) if not np.isnan(c350_contrib_pts) else (np.inf, -np.inf)

    c355_contrib_pts = np.nan
    if not np.isnan(c350_contrib_pts) and not np.isnan(k355["range_pts"]):
        c355_contrib_pts = added_range(env_low, env_high, k355["L"], k355["H"])
        env_low = min(env_low, k355["L"])
        env_high = max(env_high, k355["H"])

    c359_contrib_pts = np.nan
    if env_low != np.inf and not np.isnan(k359["range_pts"]):
        c359_contrib_pts = added_range(env_low, env_high, k359["L"], k359["H"])

    # 15:59 behavior (directional expansion by close; retrace = wick back inside band)
    pre59 = stage_slice(df_day, "15:50:00", "15:58:00")
    if not pre59.empty and not c359.empty:
        pre59_H = float(pre59["High"].max())
        pre59_L = float(pre59["Low"].min())
        c59 = k359

        macro_dir_sign = int(np.sign(macro_dir_pts))  # +1 bull, -1 bear, 0 neutral
        if macro_dir_sign > 0:
            expanded_up   = (c59["C"] > pre59_H)
            expanded_down = False
        elif macro_dir_sign < 0:
            expanded_up   = False
            expanded_down = (c59["C"] < pre59_L)
        else:
            expanded_up   = (c59["C"] > pre59_H)
            expanded_down = (c59["C"] < pre59_L)

        if expanded_up:
            c359_expanded_dir = "up"
        elif expanded_down:
            c359_expanded_dir = "down"
        else:
            c359_expanded_dir = "none"

        overlap_low  = max(c59["L"], pre59_L)
        overlap_high = min(c59["H"], pre59_H)
        retraced_inside = (overlap_high - overlap_low) > 0.0

        if c359_expanded_dir == "up":
            push_past_pts = max(0.0, c59["H"] - pre59_H)
        elif c359_expanded_dir == "down":
            push_past_pts = max(0.0, pre59_L - c59["L"])
        else:
            push_past_pts = 0.0

        if c359_expanded_dir == "up":
            retrace_depth_pts = max(0.0, pre59_H - c59["L"])
        elif c359_expanded_dir == "down":
            retrace_depth_pts = max(0.0, c59["H"] - pre59_L)
        else:
            retrace_depth_pts = 0.0

        if c359_expanded_dir != "none" and retraced_inside:
            c359_behavior = "expand_and_retrace"
        elif c359_expanded_dir != "none" and not retraced_inside:
            c359_behavior = "expand_only"
        elif c359_expanded_dir == "none" and retraced_inside:
            c359_behavior = "band_only"
        else:
            c359_behavior = "none"
    else:
        c359_behavior = "unknown"
        c359_expanded_dir = "none"
        retraced_inside = False
        push_past_pts = 0.0
        retrace_depth_pts = 0.0

    push_past_pct = pct_range(push_past_pts, O)
    retrace_depth_pct = pct_range(retrace_depth_pts, O)

    return dict(
        date=df_day["date"].iloc[0].isoformat(),
        macro_open=O, macro_close=C, macro_high=H, macro_low=L,
        macro_range_points=macro_range_pts, macro_range_pct=macro_range_pct,
        macro_dir_points=macro_dir_pts, macro_dir_pct=macro_dir_pct, macro_skew=macro_skew,
        stage1_range_pct=s1["range_pct"], stage1_skew=s1["skew"],
        stage2_range_pct=s2["range_pct"], stage2_skew=s2["skew"],
        k350_range_pct=k350["range_pct"], k350_skew=k350["skew"],
        k355_range_pct=k355["range_pct"], k355_skew=k355["skew"],
        k359_range_pct=k359["range_pct"], k359_skew=k359["skew"],
        s1_dir=int(s1_dir), s2_dir=int(s2_dir), macro_dir=int(macro_dir),
        s1_contrib_pct=pct_range(s1_contrib_pts, O) if not np.isnan(s1_contrib_pts) else np.nan,
        s2_contrib_pct=pct_range(s2_contrib_pts, O) if not np.isnan(s2_contrib_pts) else np.nan,
        k350_contrib_pct=pct_range(c350_contrib_pts, O) if not np.isnan(c350_contrib_pts) else np.nan,
        k355_contrib_pct=pct_range(c355_contrib_pts, O) if not np.isnan(c355_contrib_pts) else np.nan,
        k359_contrib_pct=pct_range(c359_contrib_pts, O) if not np.isnan(c359_contrib_pts) else np.nan,
        c359_behavior=c359_behavior,
        c359_expanded_dir=c359_expanded_dir,
        c359_retraced_inside=bool(retraced_inside),
        c359_push_past_points=float(push_past_pts),
        c359_push_past_pct=float(push_past_pct),
        c359_retrace_depth_points=float(retrace_depth_pts),
        c359_retrace_depth_pct=float(retrace_depth_pct)
    )

# ---------- Plotting ----------

def hist_save(series, title, out_path, bins=40):
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return
    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def bar_save(labels, values, title, out_path):
    if len(values) == 0:
        return
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def stacked_bar_save(labels, stacks, stack_labels, title, out_path):
    if not stacks or len(stacks[0]) == 0:
        return
    plt.figure()
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for arr, lab in zip(stacks, stack_labels):
        plt.bar(x, arr, bottom=bottom, label=lab)
        bottom = bottom + np.array(arr)
    plt.xticks(x, labels, rotation=0)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    outdir = Path(OUT_DIR)
    ensure_dir(outdir)

    df = read_bars(INPUT_BARS)

    macro_mask = (df["DateTime_ET"].dt.hour == 15) & (df["DateTime_ET"].dt.minute.between(50, 59))
    days = sorted(df.loc[macro_mask, "date"].unique())

    results = []
    for d in days:
        day_df = df[df["date"] == d].sort_values("DateTime_ET")
        rec = compute_day_stats(day_df)
        if rec is not None:
            results.append(rec)

    if not results:
        raise SystemExit("No macro records computed; check your input data covers 15:50–15:59 ET.")

    res = pd.DataFrame(results)

    # Save raw results
    res_path = outdir / "macro_stage_stats.csv"
    res.to_csv(res_path, index=False)

    # Histograms
    histograms = [
        ("Total Macro Range %", res["macro_range_pct"], "hist_macro_range_pct.png"),
        ("Stage 1 Range % (15:50–15:54)", res["stage1_range_pct"], "hist_stage1_range_pct.png"),
        ("Stage 2 Range % (15:55–15:59)", res["stage2_range_pct"], "hist_stage2_range_pct.png"),
        ("3:50 Candle Range %", res["k350_range_pct"], "hist_k350_range_pct.png"),
        ("3:55 Candle Range %", res["k355_range_pct"], "hist_k355_range_pct.png"),
        ("3:59 Candle Range %", res["k359_range_pct"], "hist_k359_range_pct.png"),
        ("Total Macro Skew", res["macro_skew"], "hist_macro_skew.png"),
        ("Stage 1 Skew", res["stage1_skew"], "hist_stage1_skew.png"),
        ("Stage 2 Skew", res["stage2_skew"], "hist_stage2_skew.png"),
        ("3:50 Candle Skew", res["k350_skew"], "hist_k350_skew.png"),
        ("3:55 Candle Skew", res["k355_skew"], "hist_k355_skew.png"),
        ("3:59 Candle Skew", res["k359_skew"], "hist_k359_skew.png"),
    ]
    for title, series, fn in histograms:
        hist_save(series, title, outdir / fn)

    # Additional histograms for 3:59 magnitudes
    hist_save(res["c359_push_past_pct"], "3:59 Push Past % (beyond pre-59 edge)", outdir / "hist_c359_push_past_pct.png")
    hist_save(res["c359_retrace_depth_pct"], "3:59 Retrace Depth % (back inside band)", outdir / "hist_c359_retrace_depth_pct.png")

    # Scatter: push vs retrace
    plt.figure()
    s = res.dropna(subset=["c359_push_past_pct", "c359_retrace_depth_pct"])
    if len(s) > 0:
        plt.scatter(s["c359_push_past_pct"], s["c359_retrace_depth_pct"])
        plt.xlabel("3:59 Push Past %")
        plt.ylabel("3:59 Retrace Depth %")
        plt.title("3:59 Push vs Retrace")
        plt.tight_layout()
        plt.savefig(outdir / "scatter_c359_push_vs_retrace.png")
        plt.close()

    # Relative contribution (avg % of macro-open)
    s1_avg = float(np.nanmean(res["s1_contrib_pct"]))
    s2_avg = float(np.nanmean(res["s2_contrib_pct"]))
    stacked_bar_save(
        ["Avg Contribution"],
        [[s1_avg], [s2_avg]],
        ["Stage1", "Stage2"],
        "Average Stage Contributions (added range %, normalized by macro open)",
        outdir / "avg_stage_contributions.png"
    )

    c_means = [
        float(np.nanmean(res["k350_contrib_pct"])),
        float(np.nanmean(res["k355_contrib_pct"])),
        float(np.nanmean(res["k359_contrib_pct"])),
    ]
    bar_save(["15:50", "15:55", "15:59"], c_means, "Average Candle Contributions (added range %, normalized by macro open)", outdir / "avg_candle_contributions.png")

    # Directional alignment
    same_s1_macro = float(np.mean(res["s1_dir"] == res["macro_dir"]) * 100.0)
    same_s2_macro = float(np.mean(res["s2_dir"] == res["macro_dir"]) * 100.0)
    s1_s2_continuation = float(np.mean(res["s1_dir"] == res["s2_dir"]) * 100.0)
    s1_s2_reversal = 100.0 - s1_s2_continuation

    bar_save(
        ["S1==Macro", "S2==Macro", "S1→S2 Continue", "S1→S2 Reverse"],
        [same_s1_macro, same_s2_macro, s1_s2_continuation, s1_s2_reversal],
        "Directional Alignment (%)",
        outdir / "directional_alignment.png"
    )

    # 3:59 behavior counts
    behavior_counts = res["c359_behavior"].value_counts()
    labels = list(behavior_counts.index)
    vals = [float(behavior_counts[l]) for l in labels]
    bar_save(labels, vals, "15:59 Candle: Behavior Classification (counts)", outdir / "c359_behavior_counts.png")

    # Context: macro direction
    bull = res[res["macro_dir"] > 0]
    bear = res[res["macro_dir"] < 0]

    hist_save(bull["macro_range_pct"], "Macro Range % (Bullish Macro Only)", outdir / "hist_macro_range_pct_bull.png")
    hist_save(bear["macro_range_pct"], "Macro Range % (Bearish Macro Only)", outdir / "hist_macro_range_pct_bear.png")

    # Deciles
    res["macro_range_decile"] = pd.qcut(res["macro_range_pct"], 10, labels=False, duplicates="drop")
    small = res[res["macro_range_decile"] == res["macro_range_decile"].min()]
    large = res[res["macro_range_decile"] == res["macro_range_decile"].max()]
    hist_save(small["stage1_range_pct"], "Stage1 Range % (Small Macro Days)", outdir / "hist_stage1_small_days.png")
    hist_save(large["stage1_range_pct"], "Stage1 Range % (Large Macro Days)", outdir / "hist_stage1_large_days.png")

    # Skew correlations
    corr_df = res[["stage1_skew", "stage2_skew", "macro_skew"]].dropna()
    corr = corr_df.corr(method="pearson")
    corr_path = outdir / "skew_correlations.csv"
    corr.to_csv(corr_path, index=True)

    # Tail percentiles
    pct_tbl = pd.DataFrame({
        "macro_range_pct": pd.Series(res["macro_range_pct"]).dropna().quantile([0.1, 0.5, 0.9, 0.95, 0.99]),
        "stage1_range_pct": pd.Series(res["stage1_range_pct"]).dropna().quantile([0.1, 0.5, 0.9, 0.95, 0.99]),
        "stage2_range_pct": pd.Series(res["stage2_range_pct"]).dropna().quantile([0.1, 0.5, 0.9, 0.95, 0.99]),
        "k359_range_pct": pd.Series(res["k359_range_pct"]).dropna().quantile([0.1, 0.5, 0.9, 0.95, 0.99]),
    })
    pct_tbl.index = [f"p{int(p*100)}" for p in pct_tbl.index]
    pct_tbl_path = outdir / "percentiles_table.csv"
    pct_tbl.to_csv(pct_tbl_path)

    print("Saved outputs to:", outdir.resolve())

if __name__ == "__main__":
    main()
