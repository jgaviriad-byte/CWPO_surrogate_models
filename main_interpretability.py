import pandas as pd

from src.config import (
    PROCESSED_DATA_PATH
)

from src.feature_engineering import (
    get_features_target
)

from src.interpretability import (

    load_trained_model,

    compute_feature_importance,

    compute_permutation_importance,

    save_importance

)

from src.plots import (

    plot_feature_importance,

    plot_permutation_importance

)


def main():

    MODEL = "Random Forest"

    print("Loading dataset...")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    FEATURES, TARGET = get_features_target()

    X = df[FEATURES]

    y = df[TARGET]

    print("Loading trained model...")

    model, transformers = load_trained_model(MODEL)
    
    print("Computing Feature Importance...")

    importance = compute_feature_importance(model)

    print(importance)

    plot_feature_importance(

        importance,

        "RandomForest_FeatureImportance"

    )

    print("Computing Permutation Importance...")

    permutation = compute_permutation_importance(

    model,

    transformers,

    X,

    y

)

    print(permutation)

    plot_permutation_importance(

        permutation,

        "RandomForest_PermutationImportance"

    )
    save_importance(

    importance,

    "feature_importance.csv"

)

    save_importance(

    permutation,

    "permutation_importance.csv"

)
    importance.insert(

    0,

    "Rank",

    range(1, len(importance)+1)

)


if __name__ == "__main__":

    main()