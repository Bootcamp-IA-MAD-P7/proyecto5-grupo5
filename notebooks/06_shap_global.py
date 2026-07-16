# %%
import sys
import inspect

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

from src.shap_explainer import shap_global_for_model

# Ver qué parámetros acepta tu función (así no rompemos si cambiaste la firma)
sig = inspect.signature(shap_global_for_model)
print("shap_global_for_model signature:", sig, flush=True)

# Ajustes para que SHAP global no tarde horas:
# - Reducimos la cantidad de samples a explicar
# - Reducimos el "background" (referencia) si tu función lo usa
desired_kwargs = {
    # posibles nombres típicos que podrían existir en tu función
    "n_samples": 2000,
    "max_samples": 2000,
    "background_size": 100,
    "bg_size": 100,
    "background_n": 100,
    "seed": 42,
    "random_state": 42,
}

# Filtra solo las kwargs que tu función realmente soporta
call_kwargs = {k: v for k, v in desired_kwargs.items() if k in sig.parameters}

print("Calling shap_global_for_model with kwargs:", call_kwargs, flush=True)

# %%
results = shap_global_for_model(
    model_filename="xgboost_pipeline.joblib",
    **call_kwargs,
)

print(results, flush=True)

# %%
print("Revisa la carpeta: shap/shap/outputs/ para ver los gráficos global.")
