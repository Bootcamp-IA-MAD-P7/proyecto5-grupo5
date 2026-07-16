from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import TARGET
from src.utils import load_pipeline
from src.shap_config import SHAP_OUTPUTS_DIR, SHAP_BACKGROUND_MAX, SHAP_SAMPLE_MAX, SHAP_RANDOM_STATE


def load_X_y_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Devuelve X e y para SHAP/explicación.
    OJO: aquí X son las columnas del dataset limpio (SIN el target).
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_background_and_sample(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selecciona background y sample con límites para que KernelExplainer no tarde infinito.
    """
    n = len(X)
    if n == 0:
        raise ValueError("X está vacío. No se puede ejecutar SHAP.")

    rng = np.random.default_rng(SHAP_RANDOM_STATE)

    # background
    bg_n = min(SHAP_BACKGROUND_MAX, n)
    bg_idx = rng.choice(n, size=bg_n, replace=False)
    background = X.iloc[bg_idx].copy()

    # sample para explicar
    sample_n = min(SHAP_SAMPLE_MAX, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    X_sample = X.iloc[sample_idx].copy()

    return background, X_sample


def build_predict_proba_fn(pipeline):
    feature_names = getattr(pipeline, "feature_names_in_", None)

    def predict_proba_class1(x_df):
        # shap suele pasar numpy en algunos casos: lo convertimos
        if isinstance(x_df, np.ndarray):
            if feature_names is None:
                x_df = pd.DataFrame(x_df)
            else:
                x_df = pd.DataFrame(x_df, columns=feature_names)

        # Si ya es DataFrame, úsalo tal cual
        return pipeline.predict_proba(x_df)[:, 1]

    return predict_proba_class1



def ensure_shap_outputs_dir():
    SHAP_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_matplotlib_fig(fig, out_path: Path):
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def transform_X_for_model(pipeline, X: pd.DataFrame):
    """
    Aplica TODO el pipeline excepto el último paso ('clf'),
    para producir la representación que espera el modelo.
    """
    preproc = pipeline[:-1]  # quita el último paso 'clf'
    return preproc.transform(X)


import numpy as np
import shap

def compute_tree_shap_values(pipeline, X_background: pd.DataFrame, X_to_explain: pd.DataFrame):
    """
    TreeExplainer para XGBoost (rápido).
    Genera SHAP en "raw" (log-odds / margin interno). 
    Si quieres probabilidad clase 1, la derivamos con sigmoide.
    """
    ensure_shap_outputs_dir()

    X_bg_trans = transform_X_for_model(pipeline, X_background)
    X_explain_trans = transform_X_for_model(pipeline, X_to_explain)

    model = pipeline.named_steps["clf"]

    # IMPORTANTE: tree_path_dependent => model_output debe ser "raw"
    explainer = shap.TreeExplainer(model, model_output="raw")

    shap_values = explainer.shap_values(X_explain_trans)

    # Si luego necesitas probabilidad clase 1:
    # SHAP(raw) se suma al expected_value para dar el raw_output.
    # raw_output = expected_value + sum(shap_values)
    # probability = sigmoid(raw_output)
    expected = explainer.expected_value
    # shap_values puede venir como array (n, features) o lista por clase (depende del modelo)
    return explainer, shap_values, X_bg_trans, X_explain_trans, expected

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
