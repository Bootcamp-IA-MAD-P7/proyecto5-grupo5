# %%
# %% [markdown]
# # Random Forest

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
from src.pipelines import build_random_forest_pipeline
from src.utils import save_pipeline, load_pipeline

sns.set_theme(style="whitegrid")

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_pipeline_random_forest(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    k_neighbors_smote=5,
    random_state=42,
    sampling_strategy="auto"
):
    smote = SMOTE(
        k_neighbors=k_neighbors_smote,
        random_state=random_state,
        sampling_strategy=sampling_strategy
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
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
# ## 2. Random Forest with Optuna


# %%
def suggest_rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "k_neighbors_smote": trial.suggest_int("k_neighbors_smote", 3, 10),
    }


# %% [markdown]
# ## 3. Evaluation

# %% [markdown]
# ### 3.1 Load reference models for comparison

# %%
lr_pipeline = load_pipeline("logistic_pipeline.joblib")

print("Loaded Logistic Regression.")

# %% [markdown]
# ### 3.2 Test set metrics

# %%
results = evaluate_models(
    [
        ("Logistic Regression", lr_pipeline),
        ("Random Forest", rf_best),
    ],
    X_test,
    y_test,
)

# %%
print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf, target_names=["No Churn", "Churn"]))

# %% [markdown]
# ### 3.3 Confusion matrix

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
# ### 3.5 ROC Curve

# %%
fig, ax = plt.subplots(figsize=(8, 6))

for name, proba in [
    ("Logistic Regression", lr_pipeline.predict_proba(X_test)[:, 1]),
    ("Random Forest", y_proba_rf),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = results.loc[name, "ROC-AUC"]
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
# ## 4. Export model

# %%
save_pipeline(rf_best, "random_forest_pipeline.joblib")
print("Complete pipeline saved.")
