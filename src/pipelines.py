from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
            mapping = {k: v for k, v in BINARY_ENCODING_MAP.items() if k in unique_vals}
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
            ("ordinal", OrdinalEncoder(categories=ordinal_categories, dtype=int), ordinal_cols)
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
        transformers.append(("scaler", StandardScaler(), [c for c in NUMERIC_SCALE_COLS]))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def _has_steps(obj) -> bool:
    # cubre sklearn Pipeline / imblearn Pipeline y cualquier wrapper similar
    return hasattr(obj, "steps") and isinstance(getattr(obj, "steps"), list)


def _flatten_into_steps(transformer, prefix: str):
    """
    Devuelve una lista de (nombre, paso) para insertar directo en el ImbPipeline final.
    Si 'transformer' es un Pipeline (sklearn o imblearn), lo expande recursivamente.
    """
    if _has_steps(transformer):
        out = []
        for name, step in transformer.steps:
            new_prefix = f"{prefix}__{name}"
            out.extend(_flatten_into_steps(step, new_prefix))
        return out

    # No es pipeline: lo insertamos como un solo paso
    return [(prefix, transformer)]


def _build_model_pipeline_with_smote(model_step, random_state=RANDOM_STATE, k_neighbors_smote=5) -> ImbPipeline:
    ft = create_feature_transformer()
    ft_steps = _flatten_into_steps(ft, prefix="features")

    steps = []
    steps.extend(ft_steps)
    steps.append(("binary", _make_binary_encoder()))
    steps.append(("transform", _build_all_transforms()))
    steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=k_neighbors_smote)))
    steps.append(("clf", model_step))

    return ImbPipeline(steps=steps)


def build_logistic_pipeline(**kwargs) -> ImbPipeline:
    k_neighbors_smote = kwargs.pop("k_neighbors_smote", 5)

    params = {
        "class_weight": None,
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        **kwargs,
    }
    model = LogisticRegression(**params)

    return _build_model_pipeline_with_smote(
        model,
        random_state=RANDOM_STATE,
        k_neighbors_smote=k_neighbors_smote
    )



def build_knn_pipeline(**kwargs) -> ImbPipeline:
    params = {"n_jobs": -1, **kwargs}
    model = KNeighborsClassifier(**params)
    return _build_model_pipeline_with_smote(model, random_state=RANDOM_STATE)


def build_random_forest_pipeline(**kwargs) -> ImbPipeline:
    k_neighbors_smote = kwargs.pop("k_neighbors_smote", 5)

    params = {
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        **kwargs,
    }
    model = RandomForestClassifier(**params)

    return _build_model_pipeline_with_smote(
        model,
        random_state=RANDOM_STATE,
        k_neighbors_smote=k_neighbors_smote,
    )



def build_xgboost_pipeline(**kwargs) -> ImbPipeline:
    k_neighbors_smote = kwargs.pop("k_neighbors_smote", 5)

    params = {
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        **kwargs,
    }
    model = XGBClassifier(**params)

    return _build_model_pipeline_with_smote(
        model,
        random_state=RANDOM_STATE,
        k_neighbors_smote=k_neighbors_smote,
    )

