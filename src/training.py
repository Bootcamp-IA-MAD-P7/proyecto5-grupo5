import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, TARGET
from src.evaluation import compare_models, evaluate_model


def make_train_test_split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(
        f"Train size: {X_train.shape[0]:,} (churn={y_train.sum():,}, {y_train.mean():.1%})"
    )
    print(
        f"Test size:  {X_test.shape[0]:,} (churn={y_test.sum():,}, {y_test.mean():.1%})"
    )

    return X_train, X_test, y_train, y_test


def evaluate_models(models, X_test, y_test):
    results = compare_models(
        [
            evaluate_model(
                name,
                y_test,
                pipeline.predict(X_test),
                pipeline.predict_proba(X_test)[:, 1],
            )
            for name, pipeline in models
        ]
    )
    print("=== Model Comparison ===")
    print(results.to_string())
    return results


def run_optuna_study(
    build_pipeline_fn,
    suggest_params_fn,
    X_train,
    y_train,
    n_trials=30,
    direction="maximize",
    overfit_threshold=0.05,
    n_splits=5,
    show_progress_bar=True,
):
    import optuna
    from optuna.samplers import TPESampler

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial):
        params = suggest_params_fn(trial)
        pipeline = build_pipeline_fn(**params)
        aucs, gaps = [], []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            Xa, Xv = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ya, yv = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            pipeline.fit(Xa, ya)
            val_auc = roc_auc_score(yv, pipeline.predict_proba(Xv)[:, 1])
            train_auc = roc_auc_score(ya, pipeline.predict_proba(Xa)[:, 1])
            aucs.append(val_auc)
            gaps.append(train_auc - val_auc)
        penalty = max(0, np.mean(gaps) - overfit_threshold) * 2
        return np.mean(aucs) - penalty

    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    from optuna.trial import FixedTrial

    best_trial_params = study.best_params
    mapped_params = suggest_params_fn(FixedTrial(best_trial_params))
    raw_aucs = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        Xa, Xv = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ya, yv = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        pipe = build_pipeline_fn(**mapped_params)
        pipe.fit(Xa, ya)
        raw_aucs.append(roc_auc_score(yv, pipe.predict_proba(Xv)[:, 1]))
    raw_cv_auc = np.mean(raw_aucs)

    print(f"\nBest trial value (penalized): {study.best_value:.4f}")
    print(f"Best trial raw CV ROC-AUC:     {raw_cv_auc:.4f}")
    print(f"Best params (mapped): {mapped_params}")

    best_pipeline = build_pipeline_fn(**mapped_params)
    best_pipeline.fit(X_train, y_train)

    return study, best_pipeline
