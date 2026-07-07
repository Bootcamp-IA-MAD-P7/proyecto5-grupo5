# %%
# ruff: noqa: E402
# %% [markdown]
# # Feature Engineering

# %% [markdown]
# ## 1. Load cleaned dataset

# %%
import sys
from pathlib import Path

# Ensure src/ is importable (works from both notebooks/ and project root)
for p in [Path.cwd()] + list(Path.cwd().parents):
    if (p / "pyproject.toml").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from src.config import PROCESSED_DIR, CLEAN_PATH, ML_PATH, ML_SCALED_PATH, TARGET
from src.features import (
    add_num_services,
    add_protection_services,
    add_streaming_services,
    add_tenure_group,
    add_avg_monthly_spend,
    add_has_internet_service,
)
from src.encoding import (
    ORDINAL_ENCODINGS,
    apply_binary_encoding,
    apply_ordinal_encoding,
    classify_columns,
)

sns.set_theme(style="whitegrid")

# %%
df = pd.read_parquet(CLEAN_PATH)
print(f"Loaded shape: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Verify cleaned data

# %%
df_work = df.copy()
print(f"Working copy shape: {df_work.shape}")
print(f"Missing values: {df_work.isnull().sum().sum()}")
print(f"Data types:\n{df_work.dtypes.to_string()}")

# %% [markdown]
# ## 3. Feature engineering

# %% [markdown]
# ### 3.1 NumServices

# %%
df_work = add_num_services(df_work)
print(
    f"NumServices created. Distribution:\n{df_work['NumServices'].value_counts().sort_index()}"
)

# %% [markdown]
# ### 3.2 ProtectionServices

# %%
df_work = add_protection_services(df_work)
print(
    f"ProtectionServices created. Distribution:\n{df_work['ProtectionServices'].value_counts().sort_index()}"
)

# %% [markdown]
# ### 3.3 StreamingServices

# %%
df_work = add_streaming_services(df_work)
print(
    f"StreamingServices created. Distribution:\n{df_work['StreamingServices'].value_counts().sort_index()}"
)

# %% [markdown]
# ### 3.4 TenureGroup

# %%
df_work = add_tenure_group(df_work)
print(
    f"TenureGroup created. Distribution:\n{df_work['TenureGroup'].value_counts().sort_index()}"
)

# %% [markdown]
# ### 3.5 AvgMonthlySpend

# %%
df_work = add_avg_monthly_spend(df_work)
print("AvgMonthlySpend created.")
print(
    f"Min: {df_work['AvgMonthlySpend'].min():.2f}, "
    f"Max: {df_work['AvgMonthlySpend'].max():.2f}, "
    f"Mean: {df_work['AvgMonthlySpend'].mean():.2f}"
)

# %% [markdown]
# ### 3.6 HasInternetService

# %%
df_work = add_has_internet_service(df_work)
print(
    f"HasInternetService created. Value counts:\n{df_work['HasInternetService'].value_counts()}"
)

# %% [markdown]
# ## 4. Feature selection
#
# %% [markdown]
# ### 4.1 Correlation analysis

# %%
original_num = ["Tenure Months", "Monthly Charges", "Total Charges"]
new_feats = [
    "NumServices",
    "ProtectionServices",
    "StreamingServices",
    "TenureGroup",
    "AvgMonthlySpend",
    "HasInternetService",
]

df_eval = df_work.copy()
if "TenureGroup" in df_eval.columns and df_eval["TenureGroup"].dtype.name == "category":
    df_eval["TenureGroup"] = df_eval["TenureGroup"].cat.codes

existing_new = [c for c in new_feats if c in df_eval.columns]
existing_orig = [c for c in original_num if c in df_eval.columns]

corr_matrix = df_eval[existing_orig + existing_new].corr()
print("Correlation of new features with original numerical features:\n")
print(corr_matrix.loc[existing_new, existing_orig].to_string())
print()

print("Inter-correlation among new features:\n")
print(corr_matrix.loc[existing_new, existing_new].to_string())
print()

# %% [markdown]
# ### 4.2 Mutual Information with target

# %%
df_eval = df_work.copy()
cat_cols_eval = df_eval.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()
for col in cat_cols_eval:
    df_eval[col] = pd.factorize(df_eval[col])[0]

X_eval = df_eval.drop(columns=[TARGET])
y_eval = df_eval[TARGET]
mi_scores = mutual_info_classif(X_eval, y_eval, random_state=42)
mi_df = pd.DataFrame({"Feature": X_eval.columns, "MutualInformation": mi_scores})
mi_df = mi_df.sort_values("MutualInformation", ascending=False).reset_index(drop=True)
mi_df["MI_Rank"] = range(1, len(mi_df) + 1)

print("=== Mutual Information with Target ===\n")
print(mi_df.to_string(index=False))

# %% [markdown]
# ### 4.3 Feature retention decisions

# %%
features_to_drop = []

print("=== Feature Retention Analysis ===\n")

for feat in existing_new:
    if feat == "HasInternetService":
        print(
            f"  KEEP    {feat:25s} | Encoding simplification (prevents collinearity with OHE)"
        )
        continue

    orig_available = [c for c in original_num if c in df_work.columns]
    if orig_available:
        max_corr = corr_matrix.loc[feat, orig_available].abs().max()
        high_corr_with = corr_matrix.loc[feat, orig_available].abs().idxmax()
    else:
        max_corr = 0.0
        high_corr_with = ""

    mi_val = mi_df.loc[mi_df["Feature"] == feat, "MutualInformation"].values[0]

    # Check redundancy with other new features (exclude HasInternetService which is kept for encoding)
    other_new = [c for c in existing_new if c != feat and c != "HasInternetService"]
    max_new_corr = 0.0
    redundant_new_with = ""
    if other_new and feat in corr_matrix.index:
        max_new_corr = corr_matrix.loc[feat, other_new].abs().max()
        redundant_new_with = corr_matrix.loc[feat, other_new].abs().idxmax()

    if max_corr > 0.95:
        print(
            f"  REMOVE  {feat:25s} | r={max_corr:.3f} with {high_corr_with} (redundant with original)"
        )
        features_to_drop.append(feat)
        continue

    # Sub-aggregates redundant with NumServices
    if feat in ["ProtectionServices", "StreamingServices"] and max_new_corr > 0.85:
        print(
            f"  REMOVE  {feat:25s} | r={max_new_corr:.3f} with {redundant_new_with} (redundant aggregate)"
        )
        features_to_drop.append(feat)
        continue

    # TenureGroup loses information vs Tenure Months
    if feat == "TenureGroup":
        tm_mi = mi_df.loc[
            mi_df["Feature"] == "Tenure Months", "MutualInformation"
        ].values[0]
        if mi_val < tm_mi * 0.5:
            print(
                f"  REMOVE  {feat:25s} | MI={mi_val:.4f} vs Tenure Months MI={tm_mi:.4f} (binning lost >50% MI)"
            )
            features_to_drop.append(feat)
            continue

    print(
        f"  KEEP    {feat:25s} | MI={mi_val:.4f} | max_r_orig={max_corr:.3f} | useful signal"
    )

print(f"\nDropping {len(features_to_drop)} redundant features: {features_to_drop}")
df_work.drop(columns=features_to_drop, inplace=True)
print(f"Shape after feature removal: {df_work.shape}")

# %% [markdown]
# ## 5. Encoding

# %% [markdown]
# ### 5.1 Variable classification

# %%
categorical_cols, binary_cols, nominal_cols = classify_columns(df_work)
print(f"Categorical columns: {categorical_cols}")
print(f"Binary columns ({len(binary_cols)}): {binary_cols}")
print(f"Nominal columns ({len(nominal_cols)}): {nominal_cols}")

ordinal_map = dict(ORDINAL_ENCODINGS)

for col in ["TenureGroup"]:
    if col in nominal_cols:
        nominal_cols.remove(col)
        ordinal_map[col] = sorted(df_work[col].cat.categories.tolist())
        print(f"Ordinal column: {col} -> {ordinal_map[col]}")

if "Contract" in nominal_cols:
    nominal_cols.remove("Contract")
    print(f"Ordinal column: Contract -> {ordinal_map['Contract']}")

# %% [markdown]
# ### 5.2 Binary encoding

# %%
df_work = apply_binary_encoding(df_work, binary_cols)

print("Binary encoding applied:")
for col in binary_cols:
    print(f"  {col}: {sorted(df_work[col].unique())}")

# %% [markdown]
# ### 5.3 Ordinal encoding

# %%
df_work = apply_ordinal_encoding(df_work, ordinal_map)

for col, order in ordinal_map.items():
    if col in df_work.columns:
        mapping = {label: idx for idx, label in enumerate(order)}
        print(f"Ordinal encoding for '{col}': {mapping}")

# %% [markdown]
# ### 5.4 One Hot Encoding (nominal variables)

# %%
print(f"Applying One Hot Encoding to: {nominal_cols}")
df_work = pd.get_dummies(df_work, columns=nominal_cols, drop_first=True, dtype=int)
print(f"Shape after OHE: {df_work.shape}")

# %% [markdown]
# ### 5.5 Redundant OHE column removal

# %%
ohe_cols = [c for c in df_work.columns if c.startswith("Internet Service_")]
if "HasInternetService" in df_work.columns:
    for col in ohe_cols:
        if col in df_work.columns and df_work[col].nunique() == 2:
            corr_val = df_work[col].corr(df_work["HasInternetService"])
            if abs(corr_val) > 0.99:
                df_work.drop(columns=[col], inplace=True)
                print(f"Dropped '{col}' (r={corr_val:.2f} with HasInternetService)")

print(f"Shape after redundancy removal: {df_work.shape}")

# %% [markdown]
# ## 6. Scaling

# %% [markdown]
# ### 6.1 Identify numerical columns to scale

# %%
numerical_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [c for c in numerical_cols if c != TARGET]

candidates = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "AvgMonthlySpend",
    "NumServices",
]
scale_cols = [c for c in candidates if c in df_work.columns]

print(f"Columns to scale ({len(scale_cols)}): {scale_cols}")
print(
    f"Columns NOT scaled ({len(numerical_cols) - len(scale_cols)}): "
    f"{[c for c in numerical_cols if c not in scale_cols and c != TARGET]}"
)

# %% [markdown]
# ### 6.2 Create unscaled dataset

# %%
df_ml = df_work.copy()
print(f"Unscaled dataset shape: {df_ml.shape}")

# %% [markdown]
# ### 6.3 Create scaled dataset

# %%
df_ml_scaled = df_work.copy()
scaler = StandardScaler()

if scale_cols:
    scaler.fit(df_ml_scaled[scale_cols])
    df_ml_scaled[scale_cols] = scaler.transform(df_ml_scaled[scale_cols])
    print("Scaling applied with StandardScaler.")

print(f"Scaled dataset shape: {df_ml_scaled.shape}")

# %% [markdown]
# ## 7. Export

# %%
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df_ml.to_parquet(ML_PATH, index=False)
df_ml_scaled.to_parquet(ML_SCALED_PATH, index=False)

print(
    f"Unscaled dataset saved: {ML_PATH.resolve()} ({df_ml.shape[0]} rows, {df_ml.shape[1]} cols)"
)
print(
    f"Scaled dataset saved:   {ML_SCALED_PATH.resolve()} ({df_ml_scaled.shape[0]} rows, {df_ml_scaled.shape[1]} cols)"
)

# %%
print("Feature engineering complete. Final column list:")
print(df_ml.columns.tolist())
