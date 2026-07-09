# %%
# ruff: noqa: E402
# %% [markdown]
# # Random Forest

# %% [markdown]
# ## 1. Load processed dataset

# %%
import sys
from pathlib import Path

for p in [Path.cwd()] + list(Path.cwd().parents):
    if (p / "pyproject.toml").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from joblib import dump, load

from src.config import MODELS_DIR, ML_PATH, TARGET, RANDOM_STATE
from src.evaluation import evaluate_model, compare_models

sns.set_theme(style="whitegrid")

# %%
df = pd.read_parquet(ML_PATH)
print(f"Loaded dataset: {df.shape}")

# %%
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# %% [markdown]
# ## 2. Train / test split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(
    f"Train size: {X_train.shape[0]:,} (churn={y_train.sum():,}, {y_train.mean():.1%})"
)
print(f"Test size:  {X_test.shape[0]:,} (churn={y_test.sum():,}, {y_test.mean():.1%})")

# %% [markdown]
# ## 3. Random Forest with hyperparameter tuning

# %%
rf_base = RandomForestClassifier(
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

rf_search = RandomizedSearchCV(
    rf_base,
    param_dist,
    n_iter=10,
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)
rf = rf_search.best_estimator_

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print(f"Random Forest best CV ROC-AUC: {rf_search.best_score_:.4f}")
print(f"Best params: {rf_search.best_params_}")

# %% [markdown]
# ## 4. Evaluation

# %% [markdown]
# ### 4.1 Load reference models for comparison

# %%
baseline = load(MODELS_DIR / "baseline.pkl")
lr_pipeline = load(MODELS_DIR / "logistic_regression.pkl")

print("Loaded baseline and Logistic Regression from disk.")

# %% [markdown]
# ### 4.2 Test set metrics

# %%
results = compare_models(
    [
        evaluate_model(
            "Baseline (most frequent)",
            y_test,
            baseline.predict(X_test),
            baseline.predict_proba(X_test)[:, 1],
        ),
        evaluate_model(
            "Logistic Regression",
            y_test,
            lr_pipeline.predict(X_test),
            lr_pipeline.predict_proba(X_test)[:, 1],
        ),
        evaluate_model("Random Forest", y_test, y_pred_rf, y_proba_rf),
    ]
)

print("=== Model Comparison ===")
print(results.to_string())

# %%
print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf, target_names=["No Churn", "Churn"]))

# %% [markdown]
# ### 4.3 Random Forest cross-validation

# %%
cv_rf = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc",
)
print(f"Random Forest CV ROC-AUC: {cv_rf.mean():.4f} +/- {cv_rf.std():.4f}")

# %% [markdown]
# ### 4.4 Confusion matrix

# %%
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)
ax.set_title("Confusion Matrix — Random Forest")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.5 ROC Curve

# %%
fig, ax = plt.subplots(figsize=(8, 6))

for name, proba, auc_val in [
    (
        "Baseline",
        baseline.predict_proba(X_test)[:, 1],
        results.loc["Baseline (most frequent)", "ROC-AUC"],
    ),
    (
        "Logistic Regression",
        lr_pipeline.predict_proba(X_test)[:, 1],
        results.loc["Logistic Regression", "ROC-AUC"],
    ),
    ("Random Forest", y_proba_rf, results.loc["Random Forest", "ROC-AUC"]),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)

ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison")
ax.legend(loc="lower right")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Feature importance

# %%
feature_importances = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": rf.feature_importances_,
    }
).sort_values("Importance", ascending=False)

print("Top 15 features — Random Forest:")
print(feature_importances.head(15).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(
    data=feature_importances.head(15),
    y="Feature",
    x="Importance",
    palette="viridis",
    ax=ax,
)
ax.set_title("Top 15 Feature Importances — Random Forest")
ax.set_xlabel("Importance")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.1 Engineered feature evaluation

# %%
engineered_features = ["NumServices", "HasInternetService"]
print("=== Engineered Feature Importance ===\n")
for feat in engineered_features:
    if feat in feature_importances["Feature"].values:
        row = feature_importances[feature_importances["Feature"] == feat].iloc[0]
        imp = row["Importance"]
        rank = feature_importances.index.get_loc(row.name) + 1
        pct_of_max = (imp / feature_importances["Importance"].max()) * 100
        print(
            f"  {feat:25s} | Importance={imp:.5f} | Rank={rank}/{len(feature_importances)} | {pct_of_max:.1f}% of top importance"
        )
    else:
        print(f"  {feat:25s} | NOT PRESENT in final feature set")

low_imp_features = feature_importances[feature_importances["Importance"] < 0.005]
if len(low_imp_features) > 0:
    print("\nFeatures with near-zero importance (< 0.005):")
    for _, row in low_imp_features.iterrows():
        print(f"  {row['Feature']:30s} | Importance={row['Importance']:.5f}")
else:
    print("\nNo features with near-zero importance (< 0.005).")

# %% [markdown]
# ## 6. Export model

# %%
MODELS_DIR.mkdir(parents=True, exist_ok=True)

dump(rf, MODELS_DIR / "random_forest.pkl")

print(f"Model saved to {MODELS_DIR.resolve()}")
print("  - random_forest.pkl")
