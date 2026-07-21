import pandas as pd

from src.config import PROCESSED_DATA_PATH
from src.models import get_models
from src.validation import generate_leave_one_experiment_out
from src.training import train_all_models
from src.evaluation import (
    evaluate_all_models,
    print_all_models_summary
)
from src.eda import experiment_conditions
from src.plots import plot_experiment


def main():

    df = pd.read_csv(PROCESSED_DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    folds = generate_leave_one_experiment_out(df)

    print(f"Validation folds: {len(folds)}")

    experiment_conditions(df)

    models = get_models()

    print(f"\nModels available: {list(models.keys())}")

    all_results = train_all_models(
        models,
        folds
    )

    summaries = evaluate_all_models(
        all_results
    )

    print_all_models_summary(
        summaries
    )

    plot_experiment(
        all_results["Polynomial Regression"],
        13
    )

    plot_experiment(
        all_results["Polynomial Regression"],
        20
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()