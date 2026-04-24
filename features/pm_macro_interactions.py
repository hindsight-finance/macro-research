import polars as pl

PM_PATH = "../outputs/pm_3pm.parquet"
MACRO_PATH = "../outputs/macro_outcomes.parquet"
OUT_PATH = "../outputs/pm_macro_interaction.parquet"


def build_pm_macro_interaction(print_summary: bool = True):
    pm = pl.read_parquet(PM_PATH)
    macro = pl.read_parquet(MACRO_PATH)
    df = pm.join(macro, on="date", how="inner", suffix="_macro")

    combined_low = pl.min_horizontal("hr3_low", "macro_low")
    combined_high = pl.max_horizontal("hr3_high", "macro_high")
    combined_range = combined_high - combined_low
    safe_den = pl.when(combined_range != 0).then(combined_range).otherwise(None)

    df = df.with_columns(
        threepm_macro_combined_range_points=combined_range,
        threepm_range_macro_excl_pct=(pl.col("hr3_range") / safe_den) * 100,
        threepm_range_macro_incl_pct=pl.when(combined_range.is_not_null() & (combined_range != 0)).then(100.0).otherwise(None),
        macro_range_pct_of_threepm_macro=(pl.col("macro_range_points") / safe_den) * 100,
        pm_dir_points=pl.col("pm_dir"),
        threepm_dir_points=pl.col("hr3_dir"),
    ).with_columns(
        pm_dir_sign=pl.col("pm_dir_points").sign().fill_null(0).cast(pl.Int8),
        threepm_dir_sign=pl.col("threepm_dir_points").sign().fill_null(0).cast(pl.Int8),
        macro_dir_sign=pl.col("macro_dir_points").sign().fill_null(0).cast(pl.Int8),
        macro_took_threepm_high=pl.col("macro_high") > pl.col("hr3_high"),
        macro_took_threepm_low=pl.col("macro_low") < pl.col("hr3_low"),
        threepm_high_time_min_from_15=pl.col("hr3_high_time"),
        threepm_low_time_min_from_15=pl.col("hr3_low_time"),
        macro_high_time_min_from_15=pl.col("macro_high_time") + 50,
        macro_low_time_min_from_15=pl.col("macro_low_time") + 50,
    ).with_columns(
        macro_took_threepm_both=pl.col("macro_took_threepm_high") & pl.col("macro_took_threepm_low"),
        threepm_high_bucket_5m=(pl.col("threepm_high_time_min_from_15") // 5) * 5,
        threepm_low_bucket_5m=(pl.col("threepm_low_time_min_from_15") // 5) * 5,
    )

    df.write_parquet(OUT_PATH)
    print(f"Saved pm–macro interaction features to: {OUT_PATH}")
    print(f"Rows: {df.height}")

    if print_summary:
        print_summary_stats(df)
    return df


def print_summary_stats(df: pl.DataFrame):
    bull = df.filter(pl.col("macro_dir_sign") > 0)
    bear = df.filter(pl.col("macro_dir_sign") < 0)

    def _print_hist(label, frame: pl.DataFrame, col: str):
        counts = frame.group_by(col).len().sort(col).drop_nulls(col)
        if counts.is_empty():
            print(f"{label}: (no data)")
            return
        print(f"\n{label}:")
        for row in counts.iter_rows(named=True):
            b = int(row[col])
            print(f"  {b:02d}–{b+5:02d}  -> {row['len']}")

    _print_hist("Bullish macro – when 3pm HIGH forms", bull, "threepm_high_bucket_5m")
    _print_hist("Bullish macro – when 3pm LOW forms", bull, "threepm_low_bucket_5m")
    _print_hist("Bearish macro – when 3pm HIGH forms", bear, "threepm_high_bucket_5m")
    _print_hist("Bearish macro – when 3pm LOW forms", bear, "threepm_low_bucket_5m")

    means = df.select(
        pl.col("threepm_range_macro_excl_pct").mean().round(2),
        pl.col("macro_range_pct_of_threepm_macro").mean().round(2),
    )
    print("\nMean 3pm/macro range shares (% of combined 3pm+macro range):")
    print(means)


if __name__ == "__main__":
    build_pm_macro_interaction(print_summary=True)
