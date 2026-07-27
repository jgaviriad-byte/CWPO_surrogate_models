import numpy as np
import pandas as pd


def compute_residuals(results):
    """
    Compute residuals for every prediction.

    Residual = Experimental - Prediction
    """

    residuals = []

    for result in results:

        df = pd.DataFrame({

            "Experiment": result["experiment"],

            "Time": result["time"],

            "Experimental": result["y_true"],

            "Prediction": result["y_pred"]

        })

        df["Residual"] = (

            df["Experimental"]
            -
            df["Prediction"]

        )

        residuals.append(df)

    residuals = pd.concat(
        residuals,
        ignore_index=True
    )

    return residuals