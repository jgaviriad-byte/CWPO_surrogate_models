import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestRegressor

from scipy.stats import randint
from src.config import (
    FEATURES,
    TARGET,
    EXPERIMENT_COLUMN,
    RANDOM_STATE,
    DATA_PATH, 
    EXCLUDED_EXPERIMENTS
)
from src.config import FEATURES
from src.preprocessing import (
    load_dataset,
      validate_dataset,
        save_processed_dataset
)
from src.plots import (
    plot_experiment,
    feature_summary,
    plot_prediction
)
from src.config import DATA_PATH, EXCLUDED_EXPERIMENTS
from src.eda import (
    dataset_summary,
    experiment_summary,
    correlation_matrix,
    feature_variability_by_experiment,
    plot_feature_distributions,
    experiment_conditions
)
from src.outliers import print_q_test
from src.validation import (
    generate_leave_one_experiment_out,
    summarize_validation,
    verify_validation
)
from src.training import (
    train_one_model,
    train_final_model,
    save_trained_model
)
from src.evaluation import (
    evaluate_all_folds,
    summarize_metrics,
    print_summary
)
from src.models import get_models
from src.evaluation import (
    evaluate_all_models,
    print_all_models_summary
)
from src.save_results import save_predictions
"""
for feature in FEATURES:
    
            values = (
            df
            .groupby("Ensayo")[feature]
            .first()
        )
    
            print_q_test(
            values,
            feature
        )   
"""
def main():
    df = load_dataset()
    experiment_conditions(df)
    correlation_matrix(df)
    df = df[
    ~df["Ensayo"].isin(EXCLUDED_EXPERIMENTS)
    ].copy()
    folds = generate_leave_one_experiment_out(df)
    MODELS = get_models()
    model_name = "Random Forest"
    model_config = MODELS["Random Forest"]
    model, transformers = train_final_model(
        model_config,
        df
    )
    save_trained_model(
        model,
        transformers,
        model_name
    )
    plot_experiment(df)
    
if __name__ == "__main__":
    main()