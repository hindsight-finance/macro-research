import polars as pl
import matplotlib.pyplot as plt


df = pl.read_parquet("../outputs/nq_macro_outcomes.parquet")

plt.hist(df["macro_range_points"].drop_nulls().to_numpy(), bins=50)
plt.title("Macro Range Distribution")
plt.show()
