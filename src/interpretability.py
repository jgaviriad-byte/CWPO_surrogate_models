from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance

from src.feature_engineering import (
    get_features_target
)


def load_trained_model(model_name):
    """
    Load trained model and preprocessing objects.
    """

    model_path = (
        Path("results")
        / "models"
        / f"{model_name}.joblib"
    )

    package = joblib.load(model_path)

    return package["model"], package["transformers"]

def apply_preprocessing(
    X,
    transformers
):
    """
    Apply exactly the same preprocessing used
    during training.
    """

    if "poly" in transformers:

        X = transformers["poly"].transform(X)

    if "scaler_X" in transformers:

        X = transformers["scaler_X"].transform(X)

    return X


def compute_feature_importance(
    model
):
    """
    Feature importance for tree models.
    """

    FEATURES, _ = get_features_target()

    importance = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    return importance


def compute_permutation_importance(
    model,
    transformers,
    X,
    y
):
    """
    Permutation importance.
    Works for every sklearn regressor.
    """

    FEATURES, _ = get_features_target()

    X = apply_preprocessing(
    X,
    transformers
)

    result = permutation_importance(

        model,

        X,

        y,

        n_repeats=20,

        random_state=42,

        scoring="r2"

    )

    importance = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": result.importances_mean,

        "Std": result.importances_std

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    return importance

def save_importance(
    importance,
    filename
):
    """
    Save importance table.
    """

    output = Path(
        "results/interpretability"
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    importance.to_csv(

        output / filename,

        index=False

    )

