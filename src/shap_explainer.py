from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.data import load_clean_data
from src.utils import load_pipeline
from src.config import TARGET
from src.shap_config import SHAP_OUTPUTS_DIR, SHAP_LOCAL_INDEX
from src.shap_utils import (
    load_X_y_clean,
    get_background_and_sample,
    compute_tree_shap_values,
    ensure_shap_outputs_dir,
    save_matplotlib_fig,
)


def shap_global_for_model(
    model_filename: str,
    max_display: int = 20,
):
    """
    Genera gráficas globales (summary plot) y las guarda.
    """
    ensure_shap_outputs_dir()

    df = load_clean_data()
    X, _ = load_X_y_clean(df)

    pipeline = load_pipeline(model_filename)

    X_background, X_sample = get_background_and_sample(X)

    _, shap_values, _, X_sample_trans, expected = compute_tree_shap_values(
        pipeline=pipeline,
        X_background=X_background,
        X_to_explain=X_sample,
    )
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[-1]
    else:
        shap_values_to_plot = shap_values




    # SHAP para clasificación binaria con KernelExplainer suele dar:
    # - o bien lista de arrays (por clase)
    # - o bien array único
    # Como explicamos directamente probabilidad clase 1,ç normalmente será array 2D (n, features).
    if isinstance(shap_values, list):
        # tomamos clase 1
        shap_values_to_plot = shap_values[-1]
    else:
        shap_values_to_plot = shap_values

    # Summary plot (beeswarm)
    # SHAP crea figuras internamente con matplotlib si "show=False" no existe para Kernel
    plt.figure()
    shap.summary_plot(
    shap_values_to_plot,
    X_sample_trans,
    max_display=max_display,
    show=False,
    )

    out1 = SHAP_OUTPUTS_DIR / "shap_summary_global.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (importancia media)
    plt.figure()
    shap.summary_plot(
    shap_values_to_plot,
    X_sample_trans,
    plot_type="bar",
    max_display=max_display,
    show=False,
)

    out2 = SHAP_OUTPUTS_DIR / "shap_summary_global_bar.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "out_summary": str(out1),
        "out_bar": str(out2),
        "n_explained": len(X_sample),
        "n_background": len(X_background),
    }


def shap_local_for_model(
    model_filename: str,
    row_index: int | None = None,
):
    """
    Genera explicación local para una fila (waterfall usando base_values = expected_value del explainer).
    """
    ensure_shap_outputs_dir()

    df = load_clean_data()
    X, _ = load_X_y_clean(df)

    pipeline = load_pipeline(model_filename)

    X_background, X_sample = get_background_and_sample(X)

    if row_index is None:
        row_index = SHAP_LOCAL_INDEX
    if row_index >= len(X_sample):
        row_index = 0

    X_one = X_sample.iloc[[row_index]].copy()

    _, shap_values, _, _, expected = compute_tree_shap_values(
        pipeline=pipeline,
        X_background=X_background,
        X_to_explain=X_one,
    )

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[-1]
        # expected también puede venir como lista por clase (típico en clasificación binaria)
        base = np.array(expected[-1])
    else:
        shap_values_to_plot = shap_values
        base = np.array(expected)

    shap_single = shap.Explanation(
        values=shap_values_to_plot[0],   # (n_features,)
        base_values=base,                # escalar o (1,)
        data=X_one.iloc[0].values,      # (n_features,)
        feature_names=list(X_one.columns),
    )

    plt.figure(figsize=(12, 7))
    shap.plots.waterfall(shap_single, show=False)
    out = SHAP_OUTPUTS_DIR / f"shap_waterfall_local_idx_{row_index}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "out_waterfall": str(out),
        "row_index_used": row_index,
    }
