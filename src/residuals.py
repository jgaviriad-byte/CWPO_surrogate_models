import numpy as np
import pandas as pd


import pandas as pd
import numpy as np

from src.config import FEATURES, TARGET


def compute_residuals(
    model,
    transformers,
    df,
    model_config
):
    """
    Compute residuals using a trained model.
    """

    X = df[FEATURES].copy()

    y = df[TARGET].values

    # ----------------------------------------
    # Polynomial features
    # ----------------------------------------

    if model_config["use_polynomial"]:

        poly = transformers["poly"]

        X = poly.transform(X)

    # ----------------------------------------
    # Feature scaling
    # ----------------------------------------

    if model_config["use_scaling"]:

        scaler_X = transformers["scaler_X"]

        X = scaler_X.transform(X)

    # ----------------------------------------
    # Predict
    # ----------------------------------------

    y_pred = model.predict(X)

    # ----------------------------------------
    # Undo target scaling
    # ----------------------------------------

    if model_config["use_target_scaling"]:

        scaler_y = transformers["scaler_y"]

        y_pred = scaler_y.inverse_transform(
            y_pred.reshape(-1,1)
        ).ravel()

    # ----------------------------------------
    # Undo log transform
    # ----------------------------------------

    if model_config["use_log_target"]:

        y_pred = np.exp(y_pred)

    # ----------------------------------------
    # Build dataframe
    # ----------------------------------------

    residuals = pd.DataFrame({

        "Experiment": df["Ensayo"],

        "Time": df["Tiempo"],

        "Experimental": y,

        "Prediction": y_pred

    })

    residuals["Residual"] = (

        residuals["Experimental"]

        -

        residuals["Prediction"]

    )

    return residuals