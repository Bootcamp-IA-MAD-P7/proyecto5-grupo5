from pathlib import Path
import numpy as np
from src.config import TARGET

# carpeta raíz (asumiendo que src/ está dentro de la raíz del repo)
ROOT = Path(__file__).resolve().parents[1]

SHAP_DIR = ROOT / "shap"
SHAP_OUTPUTS_DIR = SHAP_DIR / "outputs"

# Control de tiempo
SHAP_BACKGROUND_MAX = 25     # background para KernelExplainer (reduce tiempo)
SHAP_SAMPLE_MAX = 200        # cuántas filas para el explainer (reduce tiempo)
SHAP_LOCAL_INDEX = 0          # qué fila usar para la explicación local
SHAP_RANDOM_STATE = 42

# Outputs
SHAP_PLOTS_DPI = 150
