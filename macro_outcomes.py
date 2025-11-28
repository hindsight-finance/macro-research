# macro_outcomes.py
# Computes daily macro-window outcomes (15:50–15:59 ET) from session-tagged minute data.
# Works with window names: H3PM / MACRO / POST
# Fixes date dtype issues for Parquet writing.

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ─── FILE PATHS ───────────────────────────────────────────────────────────────
INPUT_PATH = Path("outputs/nq_1m.parquet")      # session-tagged data
OUTPUT_PATH = Path("outputs/nq_macro_outcomes.parquet")
MACRO_WINDOW_NAME = "MACRO"
# ──────────────────────────────────────────────────────────────────────────────

POST_WINDOW_CANDIDATES = {"post", "postclose", "post_close", "postmacro", "POST", "Post", "PostClose"}

def _pct(numer, denom):
    return np.where(denom != 0, (numer / denom) * 100.0, np.nan)

def compute_macro_outcomes(df: pd.DataFrame, macro_window_name: str) -> pd.DataFrame:
    required = {"DateTime_ET", "window", "Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["DateTime_ET"] = pd.to_datetime(df["DateTime_ET"])
    df = df.sort_values("DateTime_ET")

    # Normalize to midnight ET for grouping
    try:
        df["date"] = (
            df["DateTime_ET"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
    except Exception:
        df["date"] = df["DateTime_ET"].dt.normalize()

    macro = df[df["window"] == macro_window_name].copy()
    if macro.empty:
        raise ValueError(f'No rows found where window == "{macro_window_name}".')

    post = df[df["window"].isin(POST_WINDOW_CANDIDATES)].copy()

    out_rows = []

    for d, g in macro.groupby("date"):
        g = g.sort_values("DateTime_ET")
        if g.empty:
            continue

        O = g["Open"].iloc[0]
        C = g["Close"].iloc[-1]
        H = g["High"].max()
        L = g["Low"].min()
        rng = H - L
        dir_pts = C - O

        rng_pct = _pct(rng, O)
        dir_pct = _pct(dir_pts, O)

        if rng > 0:
            up_imp = H - O
            dn_imp = O - L
            skew_ratio = (up_imp - dn_imp) / rng
            close_in_range = (C - L) / rng
        else:
            skew_ratio = np.nan
            close_in_range = np.nan

        t0 = g["DateTime_ET"].iloc[0]
        idx_high = g["High"].idxmax()
        idx_low = g["Low"].idxmin()
        t_high = g.loc[idx_high, "DateTime_ET"]
        t_low = g.loc[idx_low, "DateTime_ET"]
        macro_high_time = int((t_high - t0).total_seconds() // 60)
        macro_low_time = int((t_low - t0).total_seconds() // 60)

        post_rng_pts = np.nan
        post_rng_pct = np.nan
        post_d = post[post["date"] == d]
        if not post_d.empty:
            H_post = post_d["High"].max()
            L_post = post_d["Low"].min()
            post_rng_pts = H_post - L_post
            post_rng_pct = _pct(post_rng_pts, O)
        else:
            day_all = df[df["date"] == d].copy()
            day_all["hhmm"] = day_all["DateTime_ET"].dt.strftime("%H:%M")
            post_guess = day_all[(day_all["hhmm"] >= "16:00") & (day_all["hhmm"] <= "16:10")]
            if not post_guess.empty:
                H_post = post_guess["High"].max()
                L_post = post_guess["Low"].min()
                post_rng_pts = H_post - L_post
                post_rng_pct = _pct(post_rng_pts, O)

        out_rows.append(
            {
                "date": d,
                "macro_open": float(O),
                "macro_close": float(C),
                "macro_high": float(H),
                "macro_low": float(L),
                "macro_range_points": float(rng),
                "macro_range_pct": float(rng_pct),
                "macro_dir_points": float(dir_pts),
                "macro_dir_pct": float(dir_pct),
                "skew_ratio": float(skew_ratio) if pd.notna(skew_ratio) else np.nan,
                "close_in_range": float(close_in_range) if pd.notna(close_in_range) else np.nan,
                "macro_high_time": int(macro_high_time),
                "macro_low_time": int(macro_low_time),
                "postclose_range_points": float(post_rng_pts) if pd.notna(post_rng_pts) else np.nan,
                "postclose_range_pct": float(post_rng_pct) if pd.notna(post_rng_pct) else np.nan,
                "macro_type": "UNLABELED",
            }
        )

    feats = pd.DataFrame(out_rows)
    feats["date"] = pd.to_datetime(feats["date"])
    feats["macro_type"] = feats["macro_type"].astype("category")
    return feats.sort_values("date").reset_index(drop=True)

def main():
    in_path = INPUT_PATH
    out_path = OUTPUT_PATH
    macro_window = MACRO_WINDOW_NAME

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    feats = compute_macro_outcomes(df, macro_window_name=macro_window)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"[OK] Wrote macro outcomes → {out_path}  (rows={len(feats)})")

if __name__ == "__main__":
    main()
