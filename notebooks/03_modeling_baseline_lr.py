# %%
# ruff: noqa: E402
# %% [markdown]
# # Baseline & Logistic Regression

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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, classification_report

from joblib import dump

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
# ## 3. Model pipelines

# %% [markdown]
# ### 3.1 Preprocessor (for linear models)

# %%
numeric_features = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "AvgMonthlySpend",
]
numeric_features = [c for c in numeric_features if c in X_train.columns]

preprocessor = ColumnTransformer(
    transformers=[("scaler", StandardScaler(), numeric_features)],
    remainder="passthrough",
)

print(
    f"Preprocessor: {len(numeric_features)} scaled, {X_train.shape[1] - len(numeric_features)} passthrough"
)

# %% [markdown]
# ### 3.2 Baseline

# %%
baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

print("Baseline trained (always predicts majority class).")

# %% [markdown]
# ### 3.3 Logistic Regression

# %%
lr_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "lr",
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
        ),
    ]
)

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("Logistic Regression trained with Pipeline (scaling inside CV).")

# %% [markdown]
# ## 4. Evaluation

# %% [markdown]
# ### 4.1 Test set metrics

# %%
results = compare_models(
    [
        evaluate_model(
            "Baseline (most frequent)",
            y_test,
            y_pred_baseline,
            baseline.predict_proba(X_test)[:, 1],
        ),
        evaluate_model("Logistic Regression", y_test, y_pred_lr, y_proba_lr),
    ]
)

print("=== Model Comparison ===")
print(results.to_string())

# %%
print("\nLogistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=["No Churn", "Churn"]))

# %% [markdown]
# ### 4.2 Cross-validation (ROC-AUC)

# %%
cv_lr = cross_val_score(
    lr_pipeline,
    X_train,
    y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc",
)
print(f"Logistic Regression CV ROC-AUC: {cv_lr.mean():.4f} +/- {cv_lr.std():.4f}")

# %% [markdown]
# ### 4.3 Confusion matrix

# %%
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)
ax.set_title("Confusion Matrix — Logistic Regression")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 ROC Curve

# %%
fig, ax = plt.subplots(figsize=(8, 6))

fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
ax.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {results.loc['Logistic Regression', 'ROC-AUC']:.4f})",
    linewidth=2,
)

ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Logistic Regression")
ax.legend(loc="lower right")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Feature importance (Logistic Regression |coefficients|)

# %%
scaled_names = numeric_features
passthrough_names = [c for c in X.columns if c not in numeric_features]
ordered_feature_names = scaled_names + passthrough_names

lr_coefficients = pd.DataFrame(
    {
        "Feature": ordered_feature_names,
        "Coef": lr_pipeline[-1].coef_[0],
        "Abs_Coef": np.abs(lr_pipeline[-1].coef_[0]),
    }
).sort_values("Abs_Coef", ascending=False)

print("Top 15 features — Logistic Regression (by |coefficient|):")
print(lr_coefficients.head(15).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=lr_coefficients.head(15), y="Feature", x="Abs_Coef", palette="RdBu_r", ax=ax
)
ax.set_title("Logistic Regression — Top 15 |Coefficients|")
ax.set_xlabel("|Coefficient|")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Export models

# %%
MODELS_DIR.mkdir(parents=True, exist_ok=True)

dump(baseline, MODELS_DIR / "baseline.pkl")
dump(lr_pipeline, MODELS_DIR / "logistic_regression.pkl")

print(f"Models saved to {MODELS_DIR.resolve()}")
print("  - baseline.pkl")
print("  - logistic_regression.pkl")
