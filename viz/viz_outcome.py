# viz_macro_outcomes.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")

# === Load your data ===
df = pd.read_parquet("../outputs/nq_macro_outcomes.parquet")

# Basic datetime breakdowns
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["dow"] = df["date"].dt.day_name()

df = df[df["macro_range_points"] >= 1].copy()



# === 1. Macro Range Distribution ===
plt.figure(figsize=(8,4))
sns.histplot(df["macro_range_points"], bins=50, color="royalblue", edgecolor=None)
plt.title("Macro Range Distribution (points)")
plt.xlabel("Range (points)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === 2. Macro Range % Distribution ===
plt.figure(figsize=(8,4))
sns.histplot(df["macro_range_pct"], bins=50, color="mediumseagreen", edgecolor=None)
plt.title("Macro Range Distribution (%)")
plt.xlabel("Range (%)")
plt.tight_layout()
plt.show()

# === 3. Monthly Boxplot ===
plt.figure(figsize=(10,5))
sns.boxplot(x="month", y="macro_range_pct", data=df, color="skyblue")
plt.title("Macro Range % by Month")
plt.xlabel("Month")
plt.ylabel("Macro Range (%)")
plt.tight_layout()
plt.show()

# === 4. High/Low Timing Densities ===
plt.figure(figsize=(8,4))
sns.kdeplot(df["macro_high_time"], fill=True, label="High time")
sns.kdeplot(df["macro_low_time"], fill=True, label="Low time")
plt.title("Macro High vs Low Formation Timing")
plt.xlabel("Minutes after 15:50 ET")
plt.legend()
plt.tight_layout()
plt.show()

# === 5. Relationship: Range vs Skew ===
plt.figure(figsize=(6,5))
sns.scatterplot(x="skew_ratio", y="macro_range_points", data=df, alpha=0.5)
plt.title("Range vs Skew Ratio")
plt.xlabel("Skew Ratio")
plt.ylabel("Range (points)")
plt.tight_layout()
plt.show()

# === 6. Rolling 20-day macro range mean ===
df_sorted = df.sort_values("date")
df_sorted["macro_range_roll20"] = df_sorted["macro_range_points"].rolling(20).mean()

plt.figure(figsize=(10,5))
plt.plot(df_sorted["date"], df_sorted["macro_range_roll20"], color="orange")
plt.title("Rolling 20-Day Mean Macro Range (points)")
plt.xlabel("Date")
plt.ylabel("Mean Range (points)")
plt.tight_layout()
plt.show()

# === 7. Correlation heatmap ===
corr_cols = ["macro_range_points","macro_range_pct","macro_dir_points",
             "macro_dir_pct","skew_ratio","close_in_range","postclose_range_points"]
corr = df[corr_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
