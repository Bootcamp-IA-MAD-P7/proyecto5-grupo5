# %% [markdown]
# # XGBoost

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

from src.data import load_clean_data
from src.pipelines import build_xgboost_pipeline
from src.utils import save_pipeline, load_pipeline

import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def build_pipeline_xgboost(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    k_neighbors_smote=5,
    random_state=42,
    sampling_strategy="auto"
):
    smote = SMOTE(
        k_neighbors=k_neighbors_smote,
        random_state=random_state,
        sampling_strategy=sampling_strategy
    )

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        eval_metric="logloss",
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
# ## 2. XGBoost with Optuna

# %%
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {pos_weight:.2f}")


# %%
def suggest_xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": pos_weight,
        "k_neighbors_smote": trial.suggest_int("k_neighbors_smote", 3, 10),
    }


# %% [markdown]
# ## 3. Evaluation

# %% [markdown]
# ### 3.1 Load reference models for comparison

# %%
lr_pipeline = load_pipeline("logistic_pipeline.joblib")
rf = load_pipeline("random_forest_pipeline.joblib")

print("Loaded Logistic Regression and Random Forest.")

# %% [markdown]
# ### 3.2 Test set metrics

# %%
results = evaluate_models(
    [
        ("Logistic Regression", lr_pipeline),
        ("Random Forest", rf),
        ("XGBoost", xgb_pipeline),
    ],
    X_test,
    y_test,
)

# %%
print("\nXGBoost:")
print(classification_report(y_test, y_pred_xgb, target_names=["No Churn", "Churn"]))

# %% [markdown]
# ### 3.3 Confusion matrix

# %%
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)
ax.set_title("Confusion Matrix — XGBoost")
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
    ("Random Forest", rf.predict_proba(X_test)[:, 1]),
    ("XGBoost", y_proba_xgb),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = results.loc[name, "ROC-AUC"]
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)

ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison (Including XGBoost)")
ax.legend(loc="lower right")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Export model

# %%
save_pipeline(xgb_pipeline, "xgboost_pipeline.joblib")
print("Complete pipeline saved.")
