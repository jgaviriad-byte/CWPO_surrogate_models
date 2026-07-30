import pandas as pd
import numpy as np
from src.evaluation import evaluate_all_folds
from src.evaluation import evaluate_all_folds
from src.models import MODEL_CONFIG, get_models

from src.config import EXCLUDED_EXPERIMENTS, PROCESSED_DATA_PATH

from src.training import train_all_models, train_one_model

from src.validation import generate_leave_one_experiment_out

from src.residuals import compute_residuals

from src.plots import (

    plot_prediction_vs_experimental,

    plot_residuals,

    plot_residual_distribution

)
from src.preprocessing import load_dataset


def main():
    results = {}

    df = load_dataset()
    df = df[
    ~df["Ensayo"].isin(EXCLUDED_EXPERIMENTS)
    ].copy()
    folds = generate_leave_one_experiment_out(df)
    MODELS = get_models()
    model_name = "Random Forest"
    model_config = MODELS["Random Forest"]
    results = train_one_model(
        model_name,
        model_config,
        folds
        )
    all_metrics = evaluate_all_folds(results)

    rf_results = results
    
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