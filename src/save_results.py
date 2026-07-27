from pathlib import Path

import pandas as pd

from src.utils import create_directory


def save_predictions(results, model_name):
    """
    Save predictions for every experiment.
    """

    output_dir = Path(
        "results/predictions"
    ) / model_name

    create_directory(output_dir)

    for result in results:

        df = pd.DataFrame({

        "Time": result["time"],

        "Experimental": result["y_true"],

        "Prediction": result["y_pred"],

        "Residual": result["y_true"] - result["y_pred"],

        "Absolute Error": abs(result["y_true"] - result["y_pred"]),

        "Squared Error": (result["y_true"] - result["y_pred"])**2

})


        filename = (
            output_dir /
            f"experiment_{result['experiment']}.csv"
        )

        df.to_csv(
            filename,
            index=False
        )


def save_metrics(metrics, model_name):
    """
    Save metrics for every validation fold.
    """

    output_dir = Path(
        "results/metrics"
    )

    create_directory(output_dir)

    df = pd.DataFrame(metrics)

    df.insert(
        0,
        "Fold",
        range(1, len(df)+1)
    )

    df.to_csv(

        output_dir /
        f"{model_name}.csv",

        index=False

    )


def save_summary(summary):
    """
    Save overall model comparison.
    """

    output_dir = Path(
        "results/metrics"
    )

    create_directory(output_dir)

    df = pd.DataFrame(summary).T

    df.to_csv(

        output_dir /
        "summary.csv"

    )