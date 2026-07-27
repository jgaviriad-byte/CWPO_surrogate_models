import pandas as pd

from src.config import PROCESSED_DATA_PATH

from src.feature_engineering import (
    get_features_target
)

from src.interpretability import (
    load_trained_model
)

from src.shap import (

    compute_shap_values,

    save_shap_values

)

from src.plots import (

    plot_shap_summary,

    plot_shap_bar,

    plot_shap_dependence,

    plot_partial_dependence

)


MODEL = "Random Forest"


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    FEATURES, TARGET = get_features_target()

    X = df[FEATURES]

    model, transformers = load_trained_model(
        MODEL
    )

    print("Computing SHAP values...")

    shap_values, X_processed = compute_shap_values(

        model,

        transformers,

        X

    )

    save_shap_values(
        shap_values
    )

    plot_shap_summary(

        shap_values,

        X_processed,

        "RandomForest_SHAP_Summary"

    )

    FEATURES, _ = get_features_target()

    for feature in FEATURES:

        plot_shap_dependence(

            feature,

            shap_values,

            X_processed,

            f"SHAP_{feature}")

    plot_shap_bar(

        shap_values,

        X_processed,

        "RandomForest_SHAP_Bar"

    )

    print("Generating SHAP dependence plots...")

    for feature in FEATURES:

        plot_shap_dependence(

            feature,

            shap_values,

            X_processed,

            f"SHAP_{feature}"

    )

    print("SHAP analysis completed.")

    for feature in FEATURES:

        plot_partial_dependence(

            model,

            X_processed,

            feature,

            f"PDP_{feature}"

    )


if __name__ == "__main__":

    main()