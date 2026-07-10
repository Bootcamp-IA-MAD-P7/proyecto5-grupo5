"""
train.py — Entrenamiento XGBoost sobre datos ya preprocesados (Issues #18 + #19).
El equipo ya preparó los datos en data/processed/, así que aquí NO se preprocesa:
  - telco_ml_scaled.parquet -> baseline Regresión Logística (necesita escalado)
  - telco_ml.parquet        -> XGBoost (los árboles no necesitan escalado)
"""
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
TARGET = "Churn Value"                 # nombre real del target en los parquet
DATA_DIR = "data/processed"
PATH_XGB = f"{DATA_DIR}/telco_ml.parquet"
PATH_BASE = f"{DATA_DIR}/telco_ml_scaled.parquet"


def load_xy(path: str):
    """Carga un parquet ya preprocesado y separa features/target."""
    df = pd.read_parquet(path)
    if TARGET not in df.columns:
        raise ValueError(
            f"No encuentro '{TARGET}' en {path}. Columnas: {list(df.columns)}")
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def evaluate(model, X_tr, y_tr, X_te, y_te):
    """Métricas en test + gap de overfitting sobre ROC-AUC (train vs test)."""
    model.fit(X_tr, y_tr)
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    yhat = (p_te >= 0.5).astype(int)
    auc_tr, auc_te = roc_auc_score(y_tr, p_tr), roc_auc_score(y_te, p_te)
    return {
        "accuracy": accuracy_score(y_te, yhat),
        "precision": precision_score(y_te, yhat),
        "recall": recall_score(y_te, yhat),
        "f1": f1_score(y_te, yhat),
        "roc_auc": auc_te,
        "auc_train": auc_tr,
        "overfit_gap": auc_tr - auc_te,
    }


def main():
    # ---------- BASELINE: LogReg sobre datos escalados ----------
    Xb, yb = load_xy(PATH_BASE)
    pos_weight = (yb == 0).sum() / (yb == 1).sum()   # ~2.76 (desbalance 73/27)
    print(f"Baseline data: {Xb.shape} | scale_pos_weight = {pos_weight:.2f}\n")

    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        Xb, yb, test_size=0.2, stratify=yb, random_state=RANDOM_STATE)
    base = LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=RANDOM_STATE)
    m_base = evaluate(base, Xb_tr, yb_tr, Xb_te, yb_te)
    print("BASELINE — Regresión Logística")
    for k, v in m_base.items():
        print(f"  {k:<12} {v:.4f}")

    # ---------- XGBoost sobre datos SIN escalar ----------
    X, y = load_xy(PATH_XGB)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
        aucs, gaps = [], []
        for tr_idx, va_idx in cv.split(X_tr, y_tr):
            Xa, Xv = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            ya, yv = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            clf = XGBClassifier(**params, scale_pos_weight=pos_weight,
                                eval_metric="logloss", random_state=RANDOM_STATE)
            clf.fit(Xa, ya)
            aucs.append(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
            gaps.append(roc_auc_score(ya, clf.predict_proba(Xa)[:, 1]) - aucs[-1])
        penalty = max(0, np.mean(gaps) - 0.05) * 2    # penaliza gap > 5%
        return np.mean(aucs) - penalty

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    best = study.best_params
    final = XGBClassifier(**best, scale_pos_weight=pos_weight,
                          eval_metric="logloss", random_state=RANDOM_STATE)
    m_xgb = evaluate(final, X_tr, y_tr, X_te, y_te)
    print("\nXGBOOST (tuneado con Optuna)")
    for k, v in m_xgb.items():
        print(f"  {k:<12} {v:.4f}")

    print(f"\nOverfitting gap: {m_xgb['overfit_gap']*100:.2f}% "
          f"({'CUMPLE <5%' if m_xgb['overfit_gap'] < 0.05 else 'REVISAR'})")

    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump(final, "models/churn_model.joblib")
    joblib.dump(list(X.columns), "models/feature_names.joblib")  # orden de columnas
    print("Guardado -> models/churn_model.joblib + feature_names.joblib")


if __name__ == "__main__":
    main()