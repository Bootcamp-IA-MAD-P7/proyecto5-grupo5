
# %%

# %% [markdown]
# # Logistic Regression

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
from src.pipelines import build_logistic_pipeline
from src.utils import save_pipeline

sns.set_theme(style="whitegrid")

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_pipeline_logreg(C=1.0, k_neighbors=5, random_state=42):
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        random_state=random_state
    )

    return Pipeline([
        ("smote", smote),
        ("model", model),
    ])


# %%
df = load_clean_data()
print(f"Train shape: {df.shape}")

# %%
X_train, X_test, y_train, y_test = make_train_test_split(df)

# %% [markdown]
# ## 2. Logistic Regression with Optuna


# %%
def suggest_lr_params(trial):
    solver_penalty = trial.suggest_categorical(
        "solver_penalty",
        [("liblinear", "l1"), ("liblinear", "l2"), ("lbfgs", "l2")],
    )
    return {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "solver": solver_penalty[0],
        "penalty": solver_penalty[1],
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "k_neighbors_smote": trial.suggest_int("k_neighbors_smote", 3, 10),
    }

# %% [markdown]
# ## 3. Evaluation

# %%
results_df = evaluate_models(
    [("Logistic Regression", lr_pipeline)],
    X_test,
    y_test,
)

# %%
print("\nLogistic Regression:")
print(
    classification_report(
        y_test, lr_pipeline.predict(X_test), target_names=["No Churn", "Churn"]
    )
)

# %% [markdown]
# ### 3.1 Confusion matrix

# %%
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, lr_pipeline.predict(X_test))
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
# ### 3.2 ROC Curve

# %%
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, lr_pipeline.predict_proba(X_test)[:, 1])
ax.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {results_df.loc['Logistic Regression', 'ROC-AUC']:.4f})",
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
# ## 4. Export pipeline

# %%
save_pipeline(lr_pipeline, "logistic_pipeline.joblib")

print("Pipeline saved (features + encoding + scaling + model).")
