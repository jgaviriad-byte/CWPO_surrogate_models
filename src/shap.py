from pathlib import Path

import joblib
import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

from src.interpretability import apply_preprocessing


# ============================================================
# CONFIGURATION
# ============================================================

SHAP_BACKGROUND_SIZE = 50
SHAP_RANDOM_STATE = 42


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_shap_data(
    X,
    transformers
):
    """
    Apply the same preprocessing used during model training.
    """

    X_processed = apply_preprocessing(
        X,
        transformers
    )

    # --------------------------------------------------------
    # Preserve feature names
    # --------------------------------------------------------

    if not isinstance(
        X_processed,
        pd.DataFrame
    ):

        X_processed = pd.DataFrame(
            X_processed,
            columns=X.columns
        )

    return X_processed


# ============================================================
# CREATE BACKGROUND DATASET
# ============================================================

def create_background_dataset(
    X_processed
):
    """
    Create the background dataset used by KernelExplainer.
    """

    return shap.sample(
        X_processed,
        min(
            SHAP_BACKGROUND_SIZE,
            len(X_processed)
        ),
        random_state=SHAP_RANDOM_STATE
    )


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

def create_shap_explainer(
    model,
    X_processed
):
    """
    Select the appropriate SHAP explainer according
    to the model type.

    Random Forest
        -> TreeExplainer

    Artificial Neural Network
        -> KernelExplainer

    Gaussian Process Regression
        -> KernelExplainer
    """

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    if isinstance(
        model,
        RandomForestRegressor
    ):

        print(
            "Using TreeExplainer for Random Forest..."
        )

        explainer = shap.TreeExplainer(
            model
        )

        return explainer


    # ========================================================
    # ARTIFICIAL NEURAL NETWORK
    # ========================================================

    elif isinstance(
        model,
        MLPRegressor
    ):

        print(
            "Using KernelExplainer for ANN..."
        )

        background = create_background_dataset(
            X_processed
        )

        def predict_fn(X_input):

            return model.predict(
                X_input
            )

        explainer = shap.KernelExplainer(
            predict_fn,
            background
        )

        return explainer


    # ========================================================
    # GAUSSIAN PROCESS REGRESSION
    # ========================================================

    elif isinstance(
        model,
        GaussianProcessRegressor
    ):

        print(
            "Using KernelExplainer for GPR..."
        )

        background = create_background_dataset(
            X_processed
        )

        def predict_fn(X_input):

            return model.predict(
                X_input
            )

        explainer = shap.KernelExplainer(
            predict_fn,
            background
        )

        return explainer


    # ========================================================
    # UNSUPPORTED MODEL
    # ========================================================

    else:

        raise TypeError(
            "Unsupported model type: "
            f"{type(model).__name__}"
        )


# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

def compute_shap_values(
    model,
    transformers,
    X
):
    """
    Compute SHAP values for a trained model.

    The appropriate SHAP explainer is selected automatically
    according to the model type.
    """

    # --------------------------------------------------------
    # Apply training preprocessing
    # --------------------------------------------------------

    X_processed = prepare_shap_data(
        X,
        transformers
    )

    # --------------------------------------------------------
    # Create appropriate explainer
    # --------------------------------------------------------

    explainer = create_shap_explainer(
        model,
        X_processed
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X_processed
    )

    return (
        shap_values,
        X_processed
    )


# ============================================================
# COMPUTE SHAP EXPLAINER
# ============================================================

def compute_shap_explainer(
    model,
    transformers,
    X
):
    """
    Return:

        explainer
        shap_values
        X_processed
    """

    # --------------------------------------------------------
    # Apply preprocessing
    # --------------------------------------------------------

    X_processed = prepare_shap_data(
        X,
        transformers
    )

    # --------------------------------------------------------
    # Create explainer
    # --------------------------------------------------------

    explainer = create_shap_explainer(
        model,
        X_processed
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        X_processed
    )

    return (
        explainer,
        shap_values,
        X_processed
    )


# ============================================================
# SAVE SHAP VALUES
# ============================================================

def save_shap_values(
    shap_values,
    filename="shap_values.joblib"
):
    """
    Save SHAP values.
    """

    output_directory = Path(
        "results/interpretability"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        shap_values,
        output_directory / filename
    )

    print(
        f"✓ SHAP values saved: "
        f"{output_directory / filename}"
    )

