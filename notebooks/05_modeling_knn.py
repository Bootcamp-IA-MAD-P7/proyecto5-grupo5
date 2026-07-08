# %%
# ruff: noqa: E402
# %% [markdown]
# # K-Nearest Neighbors (KNN)

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

matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from joblib import dump, load

import optuna
from optuna.samplers import TPESampler

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
# ## 3. Preprocessor (scaling for KNN)

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
# ## 4. KNN hyperparameter optimization with Optuna


# %%
def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""

    # Define hyperparameter search space
    n_neighbors = trial.suggest_int("n_neighbors", 3, 50)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    algorithm = trial.suggest_categorical(
        "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
    )
    metric = trial.suggest_categorical(
        "metric", ["euclidean", "manhattan", "chebyshev"]
    )

    # Create KNN pipeline
    knn_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm=algorithm,
                    metric=metric,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Cross-validation score (ROC-AUC)
    cv_scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
    )

    return cv_scores.mean()


# Run Optuna optimization
print("Starting Optuna hyperparameter optimization...")
sampler = TPESampler(seed=RANDOM_STATE)
study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
)
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\nOptimization completed!")
print(f"Best trial value (ROC-AUC): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# %% [markdown]
# ## 5. Train final KNN model with best hyperparameters

# %%
best_params = study.best_params

knn_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=best_params["n_neighbors"],
                weights=best_params["weights"],
                algorithm=best_params["algorithm"],
                metric=best_params["metric"],
                n_jobs=-1,
            ),
        ),
    ]
)

knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)
y_proba_knn = knn_pipeline.predict_proba(X_test)[:, 1]

print("KNN trained with best hyperparameters.")

# %% [markdown]
# ## 6. Evaluation

# %% [markdown]
# ### 6.1 Load reference models for comparison

# %%
baseline = load(MODELS_DIR / "baseline.pkl")
lr_pipeline = load(MODELS_DIR / "logistic_regression.pkl")
rf = load(MODELS_DIR / "random_forest.pkl")

print("Loaded baseline, Logistic Regression, and Random Forest from disk.")

# %% [markdown]
# ### 6.2 Test set metrics

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
        evaluate_model(
            "Random Forest",
            y_test,
            rf.predict(X_test),
            rf.predict_proba(X_test)[:, 1],
        ),
        evaluate_model("KNN", y_test, y_pred_knn, y_proba_knn),
    ]
)

print("=== Model Comparison ===")
print(results.to_string())

# %%
print("\nKNN:")
print(classification_report(y_test, y_pred_knn, target_names=["No Churn", "Churn"]))

# %% [markdown]
# ### 6.3 KNN cross-validation

# %%
cv_knn = cross_val_score(
    knn_pipeline,
    X_train,
    y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc",
)
print(f"KNN CV ROC-AUC: {cv_knn.mean():.4f} +/- {cv_knn.std():.4f}")

# %% [markdown]
# ### 6.4 Confusion matrix

# %%
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)
ax.set_title("Confusion Matrix — KNN")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.5 ROC Curve (All Models)

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
    (
        "Random Forest",
        rf.predict_proba(X_test)[:, 1],
        results.loc["Random Forest", "ROC-AUC"],
    ),
    ("KNN", y_proba_knn, results.loc["KNN", "ROC-AUC"]),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)

ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison (Including KNN)")
ax.legend(loc="lower right")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Optuna trial history

# %%
trials_df = study.trials_dataframe()
print(f"\nOptuna Trials Summary ({len(trials_df)} trials):")
print(
    trials_df[
        [
            "number",
            "value",
            "params_n_neighbors",
            "params_weights",
            "params_algorithm",
            "params_metric",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trials_df["number"], trials_df["value"], marker="o", linestyle="-", alpha=0.7)
ax.axhline(
    y=study.best_value,
    color="r",
    linestyle="--",
    label=f"Best (ROC-AUC={study.best_value:.4f})",
)
ax.set_xlabel("Trial Number")
ax.set_ylabel("ROC-AUC Score")
ax.set_title("Optuna Optimization History — KNN")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Export model and optimization results

# %%
MODELS_DIR.mkdir(parents=True, exist_ok=True)

dump(knn_pipeline, MODELS_DIR / "knn.pkl")

print(f"Model saved to {MODELS_DIR.resolve()}")
print("  - knn.pkl")

# Save best parameters to reference
best_params_dict = {
    "n_neighbors": best_params["n_neighbors"],
    "weights": best_params["weights"],
    "algorithm": best_params["algorithm"],
    "metric": best_params["metric"],
}
print("\nBest KNN Hyperparameters:")
for key, val in best_params_dict.items():
    print(f"  {key}: {val}")
