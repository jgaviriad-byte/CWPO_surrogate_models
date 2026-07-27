import pandas as pd

from src.config import (PROCESSED_DATA_PATH,
                        FEATURES_NO_TIME,
                        FEATURES
    )
from src.models import get_models
from src.validation import generate_leave_one_experiment_out
from src.training import (train_all_models,
    train_and_save_all_models
)
from src.evaluation import (
    evaluate_all_models,
    print_all_models_summary,
    evaluate_all_folds,
    calculate_metrics,
    print_metrics
    
)
from src.eda import experiment_conditions, correlation_matrix
from src.plots import (plot_experiment,
                       plot_mon_vs_time,
                    plot_h2o2_vs_time,
                    plot_MONnorm_vs_time)
from src.outliers import print_q_test
from src.save_results import (

    save_metrics,
    save_predictions,
    save_summary
)
import pandas as pd

from src.predict import (predict_dataframe)

def main():

    MODEL_PATH = (
        "results/models/"
        "Gaussian Process Regression.joblib"
    )

    print("=" * 60)
    print("TESTING TRAINED MODEL")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df[FEATURES].copy()

    predictions = predict_dataframe(
        MODEL_PATH,
        X
    )

    print("\nFirst 10 predictions:\n")

    print(predictions[:10])

    print("\nPrediction shape:")

    print(predictions.shape)

    metrics = calculate_metrics(
    df["MON_norm"],
    predictions
)

    print_metrics(metrics)

    print("\nPrediction finished successfully.")

if __name__ == "__main__":
    main()