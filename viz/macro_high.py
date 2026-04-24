import polars as pl
import matplotlib.pyplot as plt


df = pl.read_parquet("../outputs/nq_macro_outcomes.parquet").with_columns(
    pl.col("macro_high_time").cast(pl.Float64, strict=False),
    pl.col("macro_low_time").cast(pl.Float64, strict=False),
)

plt.hist(df["macro_high_time"].drop_nulls().to_numpy(), bins=20, alpha=0.6, label="High")
plt.hist(df["macro_low_time"].drop_nulls().to_numpy(), bins=20, alpha=0.6, label="Low")
plt.legend()
plt.title("Macro High/Low Timing")
plt.show()
