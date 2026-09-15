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


def save_summary(

    summary,

    model_name

):

    os.makedirs(

        "results/summaries",

        exist_ok=True

    )

    df = pd.DataFrame(

        [summary]

    )

    filename = model_name.replace(" ", "_")

    df.to_csv(

        f"results/summaries/{filename}.csv",

        index=False

    )
    
import os
import pandas as pd


def save_models_comparison(summaries):

    comparison = []

    for model_name, summary in summaries.items():

        comparison.append({

            "Model": model_name,

            "Mean R2": summary["Mean R2"],

            "Std R2": summary["Std R2"],

            "Mean RMSE": summary["Mean RMSE"],

            "Std RMSE": summary["Std RMSE"],

            "Mean MAE": summary["Mean MAE"],

            "Std MAE": summary["Std MAE"]

        })

    comparison = pd.DataFrame(comparison)

    comparison = comparison.sort_values(

        by="Mean R2",

        ascending=False

    )

    os.makedirs(

        "results/metrics",

        exist_ok=True

    )

    comparison.to_csv(

        "results/metrics/Models_Comparison.csv",

        index=False

    )

    print("\nModel comparison saved successfully.")

    print(comparison)

    return comparison