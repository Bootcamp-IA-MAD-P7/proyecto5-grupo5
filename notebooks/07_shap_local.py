# %%
import sys
from pathlib import Path

root = Path.cwd()
if not (root / "src").exists():
    root = root.parent

sys.path.insert(0, str(root))

import shap

print("sys.executable:", sys.executable, flush=True)
print("shap file:", shap.__file__, flush=True)
print("shap version:", getattr(shap, "__version__", "no __version__"), flush=True)
print("has KernelExplainer:", hasattr(shap, "KernelExplainer"), flush=True)

from src.shap_explainer import shap_local_for_model


# %%
results = shap_local_for_model(
    model_filename="xgboost_pipeline.joblib",
    row_index=None,
)

print(results)

# %%
print("Revisa la carpeta: shap/shap/outputs/ para ver el gráfico local.")
