from wsgiref import validate

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from src.preprocessing import load_dataset, validate_dataset
from src.config import (
    EXCLUDED_EXPERIMENTS
)
from src.models import get_models
from src.validation import (
    generate_leave_one_experiment_out
)
from src.training import train_one_model, save_trained_model
from src.residuals import compute_residuals
from src.plots import (
    plot_parity,
    plot_rmse_comparison
)


def main():

    # =====================================================
    # 1. Load dataset
    # =====================================================

    df = load_dataset()

    df = df[
        ~df["Ensayo"].isin(EXCLUDED_EXPERIMENTS)
    ].copy()

    # =====================================================
    # 2. Generate LOEO folds
    # =====================================================

    folds = generate_leave_one_experiment_out(df)

    # =====================================================
    # 3. Load model configurations
    # =====================================================

    models = get_models()

    rmse_results = {}

    # =====================================================
    # 4. Train and evaluate every model
    # =====================================================

    for model_name, model_config in models.items():

        print("\n" + "=" * 70)
        print(f"Training {model_name}")
        print("=" * 70)

        try:

            # -------------------------------------------------
            # Train using LOEO
            # -------------------------------------------------

            results = train_one_model(
                model_name,
                model_config,
                folds
            )

            print(
                f"✓ {model_name} trained."
            )

            trained_model = (
    results[0]["model"]
    if isinstance(results, list) and results and "model" in results[0]
    else None
)

            if trained_model is not None:
                save_trained_model(
                model=trained_model,
                transformers={},
                model_name=model_name,
                model_config=model_config
        )

            residuals_list = []

            for result in results:

                temp = pd.DataFrame({

                    "Experiment":
                        result["experiment"],

                    "Time":
                        result["time"],

                    "Experimental":
                        result["y_true"],

                    "Prediction":
                        result["y_pred"]

                })

                temp["Residual"] = (
                    temp["Experimental"]
                    -
                    temp["Prediction"]
                )

                residuals_list.append(temp)

            residuals = pd.concat(
                residuals_list,
                ignore_index=True
            )

            # -------------------------------------------------
            # RMSE
            # -------------------------------------------------

            rmse = np.sqrt(
                mean_squared_error(
                    residuals["Experimental"],
                    residuals["Prediction"]
                )
            )

            rmse_results[model_name] = rmse

            print(
                f"RMSE = {rmse:.6f}"
            )

            # -------------------------------------------------
            # Parity plot
            # -------------------------------------------------

            r2 = plot_parity(
                residuals,
                model_name
            )

            print(
                f"Parity R² = {r2:.4f}"
            )

            print(
                f"✓ {model_name} completed."
            )

        except Exception as e:

            print(
                f"\n❌ Error in {model_name}"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            continue

    # =====================================================
    # 5. RMSE comparison
    # =====================================================

    print("\n" + "=" * 70)
    print("RMSE COMPARISON")
    print("=" * 70)

    rmse_results = dict(
        sorted(
            rmse_results.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    print(
        pd.DataFrame(
            list(
                rmse_results.items()
            ),
            columns=[
                "Model",
                "RMSE"
            ]
        )
    )

    plot_rmse_comparison(
        rmse_results,
        save_path=(
            "results/figures/"
            "H2O2_flow_RMSE_comparison.png"
        )
    )

    print(
        "\n✓ RMSE comparison saved."
    )

    print("\n" + "=" * 70)
    print("H2O2 FLOW MODELING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()