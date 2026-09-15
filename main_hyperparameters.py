
import pandas as pd

from src.config import (
    EXCLUDED_EXPERIMENTS,
    TARGET,
    FEATURES,
    EXPERIMENT_COLUMN,
)

from src.preprocessing import load_dataset

from src.validation import (
    generate_leave_one_experiment_out
)

from src.hyperparameter_search import (
    random_search_random_forest,
    random_search_ann,
    random_search_gpr
)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    df = load_dataset()

    print("\nDataset loaded successfully.")

    print(
        f"Shape: {df.shape}"
    )

    print(
        f"Target: {TARGET}"
    )

    print(
        f"Number of experiments: "
        f"{df[EXPERIMENT_COLUMN].nunique()}"
    )

    print("\nFeatures:")

    for feature in FEATURES:
        print(
            f"  - {feature}"
        )


    # --------------------------------------------------------
    # 2. REMOVE EXCLUDED EXPERIMENTS
    # --------------------------------------------------------

    if EXCLUDED_EXPERIMENTS:

        print("\n" + "-" * 70)
        print("EXCLUDING EXPERIMENTS")
        print("-" * 70)

        print(
            f"Excluded experiments: "
            f"{EXCLUDED_EXPERIMENTS}"
        )

        df = df[
            ~df[EXPERIMENT_COLUMN].isin(
                EXCLUDED_EXPERIMENTS
            )
        ].copy()

    else:

        print(
            "\nNo experiments excluded."
        )


    print(
        f"\nDataset after exclusion: "
        f"{df.shape}"
    )

    print(
        f"Remaining experiments: "
        f"{df[EXPERIMENT_COLUMN].nunique()}"
    )


    # --------------------------------------------------------
    # 3. BASIC TARGET DIAGNOSTICS
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TARGET DIAGNOSTICS")
    print("-" * 70)

    print(
        f"Target: {TARGET}"
    )

    print(
        f"Unique target values: "
        f"{df[TARGET].nunique()}"
    )

    print(
        f"Target mean: "
        f"{df[TARGET].mean():.6f}"
    )

    print(
        f"Target std: "
        f"{df[TARGET].std():.6f}"
    )


    # --------------------------------------------------------
    # 4. GENERATE LOEO FOLDS
    # --------------------------------------------------------
    #
    # These folds are currently required by the GPR search
    # because search_gpr_hyperparameters() uses the project's
    # existing training/evaluation architecture.
    #
    # RF and ANN do NOT use these folds directly.
    # Their RandomizedSearchCV performs GroupKFold internally.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING LOEO FOLDS FOR GPR")
    print("=" * 70)

    folds = (
        generate_leave_one_experiment_out(
            df
        )
    )

    print(
        f"\nGenerated "
        f"{len(folds)} experiment-level folds."
    )


    # ========================================================
    # 5. RANDOM FOREST
    # ========================================================

    print("\n\n" + "=" * 70)
    print("1/3 — RANDOM FOREST")
    print("=" * 70)

    search_rf, rf_results = (
        random_search_random_forest(
            df,
            n_iter=100
        )
    )


    # ========================================================
    # 6. ARTIFICIAL NEURAL NETWORK
    # ========================================================

    print("\n\n" + "=" * 70)
    print("2/3 — ARTIFICIAL NEURAL NETWORK")
    print("=" * 70)

    search_ann, ann_results = (
        random_search_ann(
            df,
            n_iter=100
        )
    )


    # ========================================================
    # 7. GAUSSIAN PROCESS REGRESSION
    # ========================================================

    print("\n\n" + "=" * 70)
    print("3/3 — GAUSSIAN PROCESS REGRESSION")
    print("=" * 70)

    results_gpr = (
        random_search_gpr(
            folds,
            n_iter=100
        )
    )


    # ========================================================
    # 8. FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("=" * 70)


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("RANDOM FOREST")
    print("-" * 70)

    print(
        "\nBest parameters:"
    )

    print(
        search_rf.best_params_
    )

    print(
        "\nBest CV RMSE:"
    )

    print(
        f"{-search_rf.best_score_:.6f}"
    )


    # --------------------------------------------------------
    # ANN
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ARTIFICIAL NEURAL NETWORK")
    print("-" * 70)

    print(
        "\nBest parameters:"
    )

    print(
        search_ann.best_params_
    )

    print(
        "\nBest CV RMSE:"
    )

    print(
        f"{-search_ann.best_score_:.6f}"
    )


    # --------------------------------------------------------
    # GPR
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("GAUSSIAN PROCESS REGRESSION")
    print("-" * 70)

    if (
        results_gpr is not None
        and not results_gpr.empty
    ):

        print(
            "\nTop 10 configurations:"
        )

        print(
            results_gpr.head(10)
        )

        best_gpr = (
            results_gpr.iloc[0]
        )

        print(
            "\nBest GPR configuration:"
        )

        print(
            f"length_scale   : "
            f"{best_gpr['length_scale']}"
        )

        print(
            f"constant_value : "
            f"{best_gpr['constant_value']}"
        )

        print(
            f"noise_level    : "
            f"{best_gpr['noise_level']}"
        )

        print(
            f"nu             : "
            f"{best_gpr['nu']}"
        )

        print(
            f"Mean RMSE      : "
            f"{best_gpr['Mean_RMSE']:.6f}"
        )

    else:

        print(
            "\n❌ No valid GPR configurations."
        )


    # ========================================================
    # 9. FINISH
    # ========================================================

    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("=" * 70)

    print(
        "\nResults saved in the results/ directory."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

