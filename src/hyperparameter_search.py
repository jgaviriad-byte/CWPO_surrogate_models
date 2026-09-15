
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from scipy.stats import randint, uniform, loguniform

from src.config import (
    FEATURES,
    TARGET,
    EXPERIMENT_COLUMN,
    RANDOM_STATE,
)


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

N_SPLITS = 5

CV = GroupKFold(
    n_splits=N_SPLITS
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    """
    Extract features, target and experiment groups.
    """

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    groups = df[EXPERIMENT_COLUMN].copy()

    print("\nDATA FOR HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"Samples       : {len(X)}")
    print(f"Features      : {len(FEATURES)}")
    print(f"Experiments   : {groups.nunique()}")
    print(f"Target        : {TARGET}")

    return X, y, groups


# ============================================================
# RANDOM FOREST
# ============================================================

def random_search_random_forest(df, n_iter=100):

    X, y, groups = prepare_data(df)

    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    param_dist = {

        # Number of trees
        "n_estimators": randint(
            200,
            1000
        ),

        # Tree depth
        "max_depth": [
            None,
            5,
            10,
            15,
            20,
            30,
            40
        ],

        # Minimum samples required to split a node
        "min_samples_split": randint(
            2,
            15
        ),

        # Minimum samples in a leaf
        "min_samples_leaf": randint(
            1,
            8
        ),

        # Number of features considered per split
        "max_features": [
            1.0,
            "sqrt",
            "log2",
            0.5,
            0.75
        ],

        # Bootstrap sampling
        "bootstrap": [
            True,
            False
        ]
    }

    print("\n" + "=" * 70)
    print("RANDOM FOREST HYPERPARAMETER SEARCH")
    print("=" * 70)

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_dist,

        n_iter=n_iter,

        cv=CV,

        scoring="neg_root_mean_squared_error",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        verbose=2,

        return_train_score=True,

        refit=True
    )

    search.fit(
        X,
        y,
        groups=groups
    )

    print("\n" + "=" * 70)
    print("BEST RANDOM FOREST")
    print("=" * 70)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest CV RMSE:")
    print(-search.best_score_)

    results = pd.DataFrame(
        search.cv_results_
    )

    results["CV_RMSE"] = (
        -results["mean_test_score"]
    )

    results = results.sort_values(
        "CV_RMSE",
        ascending=True
    )

    results.to_csv(
        RESULTS_DIR / "random_forest_hyperparameter_search.csv",
        index=False
    )

    return search, results


# ============================================================
# ARTIFICIAL NEURAL NETWORK
# ============================================================

def random_search_ann(df, n_iter=100):

    X, y, groups = prepare_data(df)

    model = MLPRegressor(
        random_state=RANDOM_STATE
    )

    param_dist = {

        # Architecture
        "hidden_layer_sizes": [
            (16,),
            (32,),
            (64,),
            (128,),
            (256,),

            (32, 16),
            (64, 32),
            (128, 64),
            (128, 128),
            (256, 128),
            (256, 128, 64)
        ],

        # Activation
        "activation": [
            "relu",
            "tanh"
        ],

        # Optimization algorithm
        "solver": [
            "adam"
        ],

        # L2 regularization
        "alpha": loguniform(
            1e-5,
            1e-1
        ),

        # Learning rate
        "learning_rate_init": loguniform(
            1e-4,
            1e-2
        ),

        # Batch size
        "batch_size": [
            "auto",
            16,
            32,
            64
        ],

        # Maximum iterations
        "max_iter": [
            500,
            1000,
            1500,
            2000
        ],

        # Early stopping
        "early_stopping": [
            True,
            False
        ]
    }

    print("\n" + "=" * 70)
    print("ANN HYPERPARAMETER SEARCH")
    print("=" * 70)

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_dist,

        n_iter=n_iter,

        cv=CV,

        scoring="neg_root_mean_squared_error",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        verbose=2,

        return_train_score=True,

        refit=True
    )

    search.fit(
        X,
        y,
        groups=groups
    )

    print("\n" + "=" * 70)
    print("BEST ANN")
    print("=" * 70)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest CV RMSE:")
    print(-search.best_score_)

    results = pd.DataFrame(
        search.cv_results_
    )

    results["CV_RMSE"] = (
        -results["mean_test_score"]
    )

    results = results.sort_values(
        "CV_RMSE",
        ascending=True
    )

    results.to_csv(
        RESULTS_DIR / "ann_hyperparameter_search.csv",
        index=False
    )

    return search, results


# ============================================================
# GPR
# ============================================================

def random_search_gpr(
    folds=None,
    n_iter=100
):
    """
    GPR search.

    This function assumes that the existing project architecture
    already provides the GPR builder through src.models.get_models().
    """

    from src.models import get_models
    from src.training import train_one_model
    from src.evaluation import (
        evaluate_all_folds,
        summarize_metrics
    )

    # --------------------------------------------------------
    # Search distributions
    # --------------------------------------------------------

    length_scales = loguniform(
        0.01,
        20
    )

    constant_values = loguniform(
        0.01,
        100
    )

    noise_levels = loguniform(
        1e-8,
        1e-1
    )

    nus = [
        0.5,
        1.5,
        2.5
    ]

    # --------------------------------------------------------
    # Random combinations
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    results = []

    print("\n" + "=" * 70)
    print("GPR HYPERPARAMETER SEARCH")
    print("=" * 70)

    for iteration in range(1, n_iter + 1):

        length_scale = rng.uniform(
            np.log10(0.01),
            np.log10(20)
        )

        length_scale = 10 ** length_scale

        constant_value = rng.uniform(
            np.log10(0.01),
            np.log10(100)
        )

        constant_value = 10 ** constant_value

        noise_level = rng.uniform(
            np.log10(1e-8),
            np.log10(1e-1)
        )

        noise_level = 10 ** noise_level

        nu = rng.choice(
            nus
        )

        print("\n" + "-" * 70)
        print(
            f"GPR combination "
            f"{iteration}/{n_iter}"
        )

        print(
            f"length_scale = {length_scale:.6g} | "
            f"constant_value = {constant_value:.6g} | "
            f"noise_level = {noise_level:.6g} | "
            f"nu = {nu}"
        )

        try:

            MODELS = get_models(
                length_scale=length_scale,
                constant_value=constant_value,
                noise_level=noise_level,
                nu=nu
            )

            model_name = (
                "Gaussian Process Regression"
            )

            model_config = MODELS[
                model_name
            ]

            fold_results = train_one_model(
                model_name,
                model_config,
                folds
            )

            metrics = evaluate_all_folds(
                fold_results
            )

            summary = summarize_metrics(
                metrics
            )

            results.append({

                "length_scale":
                    length_scale,

                "constant_value":
                    constant_value,

                "noise_level":
                    noise_level,

                "nu":
                    nu,

                "Mean_R2":
                    summary["Mean R2"],

                "Std_R2":
                    summary["Std R2"],

                "Mean_RMSE":
                    summary["Mean RMSE"],

                "Mean_MAE":
                    summary["Mean MAE"]
            })

        except Exception as error:

            print(
                f"❌ Failed combination: "
                f"{type(error).__name__}: {error}"
            )

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        print(
            "\n❌ No valid GPR configurations."
        )

        return results_df

    # --------------------------------------------------------
    # Sort primarily by RMSE
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        "Mean_RMSE",
        ascending=True
    )

    results_df.to_csv(
        RESULTS_DIR / "gpr_hyperparameter_search.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("BEST GPR CONFIGURATIONS")
    print("=" * 70)

    print(
        results_df.head(10)
    )

    return results_df


# ============================================================
# SUMMARY
# ============================================================

def print_best_models(
    rf_search=None,
    ann_search=None,
    gpr_results=None
):

    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("=" * 70)

    if rf_search is not None:

        print("\nRANDOM FOREST")
        print("-" * 40)
        print(
            rf_search.best_params_
        )
        print(
            f"CV RMSE: "
            f"{-rf_search.best_score_:.6f}"
        )

    if ann_search is not None:

        print("\nARTIFICIAL NEURAL NETWORK")
        print("-" * 40)
        print(
            ann_search.best_params_
        )
        print(
            f"CV RMSE: "
            f"{-ann_search.best_score_:.6f}"
        )

    if (
        gpr_results is not None
        and not gpr_results.empty
    ):

        print("\nGAUSSIAN PROCESS REGRESSION")
        print("-" * 40)

        best = (
            gpr_results
            .iloc[0]
        )

        print(
            f"length_scale   : "
            f"{best['length_scale']}"
        )

        print(
            f"constant_value : "
            f"{best['constant_value']}"
        )

        print(
            f"noise_level    : "
            f"{best['noise_level']}"
        )

        print(
            f"nu             : "
            f"{best['nu']}"
        )

        print(
            f"Mean RMSE      : "
            f"{best['Mean_RMSE']}"
        )

