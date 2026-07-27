from pathlib import Path

import shap
import pandas as pd

from src.interpretability import apply_preprocessing


def compute_shap_values(
    model,
    transformers,
    X
):
    """
    Compute SHAP values after preprocessing.
    """

    X_processed = apply_preprocessing(
        X,
        transformers
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_processed
    )

    return shap_values, X_processed


def save_shap_values(
    shap_values,
    filename="shap_values.joblib"
):
    """
    Save SHAP values.
    """

    import joblib

    output = Path(
        "results/interpretability"
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        shap_values,
        output / filename
    )

def compute_shap_explainer(
    model,
    transformers,
    X
):
    """
    Return SHAP explainer and SHAP values.
    """

    X_processed = apply_preprocessing(
        X,
        transformers
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_processed
    )

    return explainer, shap_values, X_processed



import matplotlib.pyplot as plt

from pathlib import Path


