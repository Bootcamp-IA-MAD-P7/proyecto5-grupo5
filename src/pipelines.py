from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    BINARY_COLS,
    BINARY_ENCODING_MAP,
    NOMINAL_COLS,
    NUMERIC_SCALE_COLS,
    ORDINAL_ENCODINGS,
    RANDOM_STATE,
)
from src.features import create_selected_feature_transformer as create_feature_transformer


def _encode_binary(df):
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            unique_vals = set(df[col].unique())
            mapping = {
                k: v for k, v in BINARY_ENCODING_MAP.items() if k in unique_vals
            }
            df[col] = df[col].map(mapping).astype(int)
    return df


def _make_binary_encoder() -> FunctionTransformer:
    return FunctionTransformer(_encode_binary, validate=False)


def _build_all_transforms() -> ColumnTransformer:
    ordinal_cols = list(ORDINAL_ENCODINGS.keys())
    ordinal_categories = list(ORDINAL_ENCODINGS.values())

    transformers = []
    if ordinal_cols:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(categories=ordinal_categories, dtype=int),
                ordinal_cols,
            )
        )
    if NOMINAL_COLS:
        transformers.append(
            (
                "nominal",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False, dtype=int
                ),
                NOMINAL_COLS,
            )
        )
    if NUMERIC_SCALE_COLS:
        transformers.append(
            ("scaler", StandardScaler(), [c for c in NUMERIC_SCALE_COLS])
        )

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def _build_model_pipeline(model_step) -> Pipeline:
    return Pipeline(
        [
            ("features", create_feature_transformer()),
            ("binary", _make_binary_encoder()),
            ("transform", _build_all_transforms()),
            ("clf", model_step),
        ]
    )


def build_logistic_pipeline(**kwargs) -> Pipeline:
    params = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        **kwargs,
    }
    return _build_model_pipeline(LogisticRegression(**params))


def build_knn_pipeline(**kwargs) -> Pipeline:
    params = {"n_jobs": -1, **kwargs}
    return _build_model_pipeline(KNeighborsClassifier(**params))


def build_random_forest_pipeline(**kwargs) -> Pipeline:
    params = {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        **kwargs,
    }
    return _build_model_pipeline(RandomForestClassifier(**params))


def build_xgboost_pipeline(**kwargs) -> Pipeline:
    params = {
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        **kwargs,
    }
    return _build_model_pipeline(XGBClassifier(**params))


