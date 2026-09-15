import pandas as pd
from src import residuals
from src.preprocessing import load_dataset

from src.config import EXCLUDED_EXPERIMENTS

from src.models import get_models

from src.predict import load_trained_model

from src.residuals import compute_residuals

from src.plots import (
        plot_prediction_vs_experimental,
        plot_residuals,
        plot_residual_distribution,
        plot_parity
    )


def main():

        df = load_dataset()

        df = df[
            ~df["Ensayo"].isin(EXCLUDED_EXPERIMENTS)
        ].copy()


        
        MODELS = get_models()

        for model_name in MODELS.keys():

            print("\n" + "="*70)
            print(f"Evaluating {model_name}")
            print("="*70)

            try:

                model, transformers = load_trained_model(
                    model_name
                )

                residuals = compute_residuals(

                    model,

                    transformers,

                    df,

                    MODELS[model_name]

                )

                plot_prediction_vs_experimental(

                    residuals,

                    model_name,

                    "Prediction_vs_Experimental"

                )

                plot_parity(

                    residuals,

                    model_name,

                    "ParityPlot"

)

                plot_residuals(

                    residuals,

                    model_name,

                    "ResidualPlot"

                )

                plot_residual_distribution(

                    residuals,

                    model_name,

                    "ResidualHistogram"

                )

                print(f"✓ {model_name} finished successfully.")

            except Exception as e:

                print(f"\n❌ ERROR in {model_name}")
                print(type(e).__name__)
                print(e)

                continue


if __name__ == "__main__":

        main()