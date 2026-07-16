from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=int,
                ),
                NOMINAL_COLS,
            )
        )
    if NUMERIC_SCALE_COLS:
        transformers.append(
            ("scaler", StandardScaler(), [c for c in NUMERIC_SCALE_COLS])
        )

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def _build_preprocess_pipeline() -> Pipeline:
    """
    Preprocesamiento SOLO (sin SMOTE).
    ImbPipeline se encargará de SMOTE después.
    """
    return Pipeline(
        [
            ("features", create_feature_transformer()),
            ("binary", _make_binary_encoder()),
            ("transform", _build_all_transforms()),
        ]
    )


def _build_model_pipeline_with_smote(model_step, random_state=RANDOM_STATE) -> ImbPipeline:
    """
    Pipeline final: preprocess -> SMOTE (en el train) -> clasificador.
    Esto evita leakage y hace que el test quede sin tocar.
    """
    return ImbPipeline(
        steps=[
            ("preprocess", _build_preprocess_pipeline()),
            ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
            ("clf", model_step),
        ]
    )


def build_logistic_pipeline(**kwargs) -> ImbPipeline:
    # Con SMOTE normalmente NO necesitas class_weight="balanced"
    params = {
        "class_weight": None,
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        **kwargs,
    }
    model = LogisticRegression(**params)
    return _build_model_pipeline_with_smote(model, random_state=RANDOM_STATE)


def build_knn_pipeline(**kwargs) -> ImbPipeline:
    params = {"n_jobs": -1, **kwargs}
    model = KNeighborsClassifier(**params)
    return _build_model_pipeline_with_smote(model, random_state=RANDOM_STATE)


def build_random_forest_pipeline(**kwargs) -> ImbPipeline:
    params = {
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        **kwargs,
    }
    model = RandomForestClassifier(**params)
    return _build_model_pipeline_with_smote(model, random_state=RANDOM_STATE)


def build_xgboost_pipeline(**kwargs) -> ImbPipeline:
    params = {
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        **kwargs,
    }
    model = XGBClassifier(**params)
    return _build_model_pipeline_with_smote(model, random_state=RANDOM_STATE)
