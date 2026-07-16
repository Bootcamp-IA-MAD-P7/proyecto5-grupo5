# %%
# %% [markdown]
# # K-Nearest Neighbors (KNN)

# %% [markdown]
# ## 1. Load clean dataset

# %%

import sys
from src.training import make_train_test_split, evaluate_models, run_optuna_study

from pathlib import Path

root = Path.cwd()
if not (root / "src").exists():
    root = root.parent

sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.config import OPTUNA_TRIALS
from src.data import load_clean_data
from src.pipelines import build_knn_pipeline
from src.utils import save_pipeline, load_pipeline

sns.set_theme(style="whitegrid")

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

def build_pipeline_knn(
    n_neighbors=15,
    weights="distance",
    k_neighbors_smote=5,
    random_state=42,
    sampling_strategy="auto"
):
    smote = SMOTE(
        k_neighbors=k_neighbors_smote,
        random_state=random_state,
        sampling_strategy=sampling_strategy
    )

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights
    )

    return Pipeline([
        ("smote", smote),
        ("model", model),
    ])


# %%
df = load_clean_data()
print(f"Loaded dataset: {df.shape}")

# %%
X_train, X_test, y_train, y_test = make_train_test_split(df)

# %% [markdown]
# ## 2. KNN hyperparameter optimization with Optuna


# %%
def suggest_knn_params(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "chebyshev"]),
        "k_neighbors_smote": trial.suggest_int("k_neighbors_smote", 3, 10),
    }

# %% [markdown]
# ## 3. Evaluate final KNN model

# %%
y_pred_knn = knn_pipeline.predict(X_test)
y_proba_knn = knn_pipeline.predict_proba(X_test)[:, 1]

print("KNN trained with best hyperparameters (from Optuna study).")

# %% [markdown]
# ## 4. Evaluation

# %% [markdown]
# ### 4.1 Load reference models for comparison

# %%
lr_pipeline = load_pipeline("logistic_pipeline.joblib")
rf = load_pipeline("random_forest_pipeline.joblib")

print("Loaded Logistic Regression and Random Forest.")

# %% [markdown]
# ### 4.2 Test set metrics

# %%
results = evaluate_models(
    [
        ("Logistic Regression", lr_pipeline),
        ("Random Forest", rf),
        ("KNN", knn_pipeline),
    ],
    X_test,
    y_test,
)

# %%
print("\nKNN:")
print(classification_report(y_test, y_pred_knn, target_names=["No Churn", "Churn"]))

# %% [markdown]
# ### 4.3 Confusion matrix

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
# ### 4.5 ROC Curve (All Models)

# %%
fig, ax = plt.subplots(figsize=(8, 6))

for name, proba in [
    ("Logistic Regression", lr_pipeline.predict_proba(X_test)[:, 1]),
    ("Random Forest", rf.predict_proba(X_test)[:, 1]),
    ("KNN", y_proba_knn),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = results.loc[name, "ROC-AUC"]
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
# ## 5. Optuna trial history

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
# ## 6. Export model

# %%
save_pipeline(knn_pipeline, "knn_pipeline.joblib")

print("Complete pipeline saved.")
