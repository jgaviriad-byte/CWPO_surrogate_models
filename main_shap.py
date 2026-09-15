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
    plot_shap_bar
)

from src.models import (
    get_models
)


# ============================================================
# SHAP ANALYSIS
# ============================================================

def analyze_model(
    model_name,
    X
):
    """
    Perform SHAP analysis for one trained model.

    Generates:
        - SHAP summary plot
        - SHAP bar plot

    Also saves the computed SHAP values.
    """

    print("\n" + "=" * 70)
    print(f"SHAP ANALYSIS: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("Loading trained model...")

    model, transformers = load_trained_model(
        model_name
    )

    # --------------------------------------------------------
    # Compute SHAP values
    # --------------------------------------------------------

    print("Computing SHAP values...")

    shap_values, X_processed = compute_shap_values(
        model,
        transformers,
        X
    )

    print("✓ SHAP values computed.")

    # --------------------------------------------------------
    # Safe model name for output files
    # --------------------------------------------------------

    safe_model_name = model_name.replace(
        " ",
        "_"
    )

    # --------------------------------------------------------
    # Save SHAP values
    # --------------------------------------------------------

    save_shap_values(
        shap_values,
        f"{safe_model_name}_SHAP"
    )

    # --------------------------------------------------------
    # SHAP SUMMARY
    # --------------------------------------------------------

    print("Generating SHAP summary plot...")

    plot_shap_summary(
        shap_values,
        X_processed,
        f"{safe_model_name}_SHAP_Summary"
    )

    # --------------------------------------------------------
    # SHAP BAR
    # --------------------------------------------------------

    print("Generating SHAP bar plot...")

    plot_shap_bar(
        shap_values,
        X_processed,
        f"{safe_model_name}_SHAP_Bar"
    )

    print(
        f"✓ SHAP analysis completed for {model_name}."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    # --------------------------------------------------------
    # Get features
    # --------------------------------------------------------

    FEATURES, _ = get_features_target()

    X = df[FEATURES]

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Number of features: {len(FEATURES)}"
    )

    print(
        f"Features: {FEATURES}"
    )

    # --------------------------------------------------------
    # Get model configurations
    # --------------------------------------------------------

    model_configs = get_models()

    print(
        f"\nModels to analyze: "
        f"{list(model_configs.keys())}"
    )

    # --------------------------------------------------------
    # SHAP analysis for every model
    # --------------------------------------------------------

    for model_name in model_configs:

        try:

            analyze_model(
                model_name,
                X
            )

        except Exception as error:

            print(
                f"\n❌ ERROR in {model_name}"
            )

            print(
                f"{type(error).__name__}: {error}"
            )

            print(
                "Continuing with next model..."
            )

            continue

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

