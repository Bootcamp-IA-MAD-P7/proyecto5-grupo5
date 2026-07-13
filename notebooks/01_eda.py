# %%
# %% [markdown]
# # Exploratory Data Analysis

# %% [markdown]
# ## 1. Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from matplotlib.lines import Line2D
from IPython.display import display

from src.config import PROCESSED_DIR, REPORTS_DIR, REPORT_PATH, CLEAN_PATH, TARGET
from src.data import load_and_clean_raw_data
from src.eda import plot_numerical_feature, plot_categorical_feature

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 2. Load and clean dataset

# %%
df = load_and_clean_raw_data()
df.head()

# %% [markdown]
# ## 3. Initial inspection

# %%
print(f"Shape: {df.shape}")
print()

print("Info:")
df.info()
print()

print("Numerical features describe:")
display(df.describe())
print()

print("Categorical features describe:")
display(df.describe(include="object"))

# %%
print("Data types:")
print(df.dtypes)
print()

print("Target distribution:")
print(df[TARGET].value_counts())
print()

print("Missing values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing)
else:
    print("No missing values found in the raw dataset.")
print()

print(f"Duplicated rows: {df.duplicated().sum()}")

# %% [markdown]
# ### 3.1 Cardinality analysis

# %%
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
print("=== Cardinality Analysis ===\n")
for col in cat_cols:
    n_unique = df[col].nunique()
    top_val = df[col].value_counts().index[0]
    top_pct = df[col].value_counts(normalize=True).iloc[0] * 100
    print(
        f"  {col:30s} -> {n_unique:3d} unique | Most frequent: {str(top_val):30s} ({top_pct:.1f}%)"
    )

print(f"\nTotal categorical columns: {len(cat_cols)}")
high_card_cols = [c for c in cat_cols if df[c].nunique() > 20]
if high_card_cols:
    print(f"High-cardinality columns: {high_card_cols}")
else:
    print("No high-cardinality categorical columns found.")

# %% [markdown]
# ## 4. Data cleaning verification

# %%
print(f"\nShape after cleaning: {df.shape}")

print("\nCategorical columns after cleaning:")
for c in df.select_dtypes(include="object").columns:
    vals = df[c].unique()
    tag = "binary" if len(vals) <= 2 else f"{len(vals)} categories"
    print(f"  {c}: {sorted(vals)} ({tag})")

print(f"\nFinal shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# %% [markdown]
# ### 4.1 Dominant category & duplicated column analysis

# %%
cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print("=== Dominant Category Analysis ===\n")
dominant_found = False
for col in cat_cols:
    top_pct = df[col].value_counts(normalize=True).iloc[0] * 100
    if top_pct > 95:
        dominant_found = True
        top_val = df[col].value_counts().index[0]
        print(f"  {col}: '{top_val}' appears in {top_pct:.1f}% of rows")
if not dominant_found:
    print("  No columns with a dominant category (>95%) found.")

print("\n=== Duplicated Column Analysis ===")
dup_col_pairs = []
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for i in range(len(numerical_cols)):
    for j in range(i + 1, len(numerical_cols)):
        c1, c2 = numerical_cols[i], numerical_cols[j]
        if c1 in df.columns and c2 in df.columns:
            corr_val = df[c1].corr(df[c2])
            if abs(corr_val) > 0.9999:
                dup_col_pairs.append((c1, c2, corr_val))
if dup_col_pairs:
    print("  Numerical columns with near-perfect correlation (|r| > 0.9999):")
    for c1, c2, r in dup_col_pairs:
        print(f"    {c1} <-> {c2} (r = {r:.4f})")
else:
    print("  No duplicated numerical columns detected.")

# %% [markdown]
# ## 5. Exploratory Data Analysis

# %% [markdown]
# ### 5.1 Target analysis

# %%
target_counts = df[TARGET].value_counts()
target_pct = df[TARGET].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = target_counts.plot(kind="bar", ax=axes[0], color=["#3498db", "#e74c3c"])
axes[0].set_title("Churn Distribution")
axes[0].set_ylabel("Count")
axes[0].set_xlabel("Churn")
axes[0].tick_params(axis="x", rotation=0)
for p in ax.containers[0]:
    ax.annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
    )

ax = target_pct.plot(kind="bar", ax=axes[1], color=["#3498db", "#e74c3c"])
axes[1].set_title("Churn Percentage")
axes[1].set_ylabel("Percentage (%)")
axes[1].set_xlabel("Churn")
axes[1].tick_params(axis="x", rotation=0)
for p in ax.containers[0]:
    ax.annotate(
        f"{p.get_height():.1f}%",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

print(f"Churn rate: {target_pct[1]:.2f}%")

# %% [markdown]
# ### 5.2 Numerical variables

# %%
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != TARGET]
print(f"Numerical features: {num_cols}")

for col in num_cols:
    plot_numerical_feature(df, col, TARGET)

# %% [markdown]
# ### 5.3 Categorical variables

# %%
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
print(f"Categorical features: {cat_cols}")

for col in cat_cols:
    plot_categorical_feature(df, col, TARGET)

# %% [markdown]
# ### 5.4 Correlation analysis

# %%
corr = df.select_dtypes(include=[np.number]).corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Correlation Matrix — Numerical Features")
plt.tight_layout()
plt.show()

print("\nHighly correlated pairs (|r| > 0.90):")
upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
high_corr_pairs = []
for col1 in upper.columns:
    for col2 in upper.columns:
        if upper.loc[col1, col2] > 0.90:
            high_corr_pairs.append((col1, col2, round(upper.loc[col1, col2], 3)))
if high_corr_pairs:
    for c1, c2, r in high_corr_pairs:
        print(f"  {c1} <-> {c2}: r = {r}")
else:
    print("  No pairs with |r| > 0.90 found.")

# %% [markdown]
# ### 5.5 Additional visualizations

# %%
# Contract vs Churn
print("Contract vs Churn (crosstab):")
ct = pd.crosstab(df["Contract"], df[TARGET].map({0: "No", 1: "Yes"}), margins=True)
print(ct)

ct_pct = (
    pd.crosstab(df["Contract"], df[TARGET].map({0: "No", 1: "Yes"}), normalize="index")
    * 100
)
fig, ax = plt.subplots(figsize=(8, 5))
ct_pct.plot(kind="bar", stacked=True, ax=ax, color=["#3498db", "#e74c3c"])
ax.set_title("Contract Type vs Churn")
ax.set_ylabel("Percentage")
ax.set_xlabel("Contract")
ax.legend(title="Churn")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()

# %%
# Payment Method vs Churn
print("Payment Method vs Churn (crosstab):")
ct = pd.crosstab(
    df["Payment Method"], df[TARGET].map({0: "No", 1: "Yes"}), margins=True
)
print(ct)

ct_pct = (
    pd.crosstab(
        df["Payment Method"], df[TARGET].map({0: "No", 1: "Yes"}), normalize="index"
    )
    * 100
)
fig, ax = plt.subplots(figsize=(10, 5))
ct_pct.plot(kind="bar", stacked=True, ax=ax, color=["#3498db", "#e74c3c"])
ax.set_title("Payment Method vs Churn")
ax.set_ylabel("Percentage")
ax.legend(title="Churn")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %%
# Tenure distribution by Contract type
fig, ax = plt.subplots(figsize=(10, 5))
for contract in df["Contract"].unique():
    subset = df[df["Contract"] == contract]
    sns.kdeplot(subset["Tenure Months"], label=contract, ax=ax)
ax.set_title("Tenure Distribution by Contract Type")
ax.set_xlabel("Tenure (Months)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Monthly Charges vs Total Charges scatter colored by Churn
fig, ax = plt.subplots(figsize=(10, 6))
colors = df[TARGET].map({0: "#3498db", 1: "#e74c3c"})
ax.scatter(df["Monthly Charges"], df["Total Charges"], c=colors, alpha=0.4, s=10)
ax.set_title("Monthly Charges vs Total Charges (by Churn)")
ax.set_xlabel("Monthly Charges")
ax.set_ylabel("Total Charges")
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#3498db",
        markersize=8,
        label="No Churn",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#e74c3c",
        markersize=8,
        label="Churn",
    ),
]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Sweetviz report

# %%
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

report = sv.analyze(df)

report.show_html(
    filepath=str(REPORT_PATH),
    open_browser=False,
)

print(f"Report generated: {REPORT_PATH.resolve()}")

# %% [markdown]
# ## 7. Save cleaned dataset

# %%
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df.to_parquet(CLEAN_PATH, index=False)
print(f"Cleaned dataset saved: {CLEAN_PATH.resolve()}")
print(f"Shape: {df.shape}")
