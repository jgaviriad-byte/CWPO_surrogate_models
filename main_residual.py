import pandas as pd

from src.models import MODEL_CONFIG, get_models

from src.config import PROCESSED_DATA_PATH

from src.training import train_all_models

from src.validation import generate_leave_one_experiment_out

from src.residuals import compute_residuals

from src.plots import (

    plot_prediction_vs_experimental,

    plot_residuals,

    plot_residual_distribution

)


def main():

    df = pd.read_csv(PROCESSED_DATA_PATH)

    folds = generate_leave_one_experiment_out(df)

    models = get_models()

    results = train_all_models(

        models,

        folds

    )

    rf_results = results["Random Forest"]

    residuals = compute_residuals(

        rf_results

    )

    plot_prediction_vs_experimental(

        residuals,

        "Prediction_vs_Experimental"

    )

    plot_residuals(

        residuals,

        "ResidualPlot"

    )

    plot_residual_distribution(

        residuals,

        "ResidualHistogram"

    )


if __name__ == "__main__":

    main()