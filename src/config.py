import os
from pathlib import Path

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
EXCLUDED_EXPERIMENTS = [2, 4, 6, 7, 8, 10, 18, 13, 14, 20]

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = BASE_DIR / "results"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "Doe_processed.csv"
MODEL_DIR = BASE_DIR / "models"

MODELS_PATH = os.path.join(
    RESULTS_PATH,
    "models"
)

DATA_PATH = BASE_DIR / "data" / "raw" / "doe_data.csv"

RANDOM_STATE = 42

EXPERIMENT_COLUMN = "Ensayo"

TARGET = "H2O2_concentration"

TIME_COLUMN = "Tiempo"

TEMPERATURE_COLUMN = "Temp_K"

PH_COLUMN = "pH"

CATALYST_COLUMN = "Fe_activo"

H2O2_COLUMN = "H2O2_flujo"

H2O2_CONCENTRATION_COLUMN = "H2O2_concentration"

MON_INITIAL_COLUMN = "MON_0"

MON_COLUMN = "MON"

MON_RELATIVE = "C/C0"

FEATURES = [TEMPERATURE_COLUMN,
            CATALYST_COLUMN,
            PH_COLUMN,
            H2O2_COLUMN,
            TIME_COLUMN]

FIGURES_PATH = "results/figures"
PREDICTIONS_PATH = "results/figures/predictions"

FEATURE_LABELS = {
    "Temp_K": "Temperature (K)",
    "Fe_activo": "Active Fe (mg/L)",
    "pH": "pH (-)",
    "H2O2_flujo": "H₂O₂ flow (mg/min)",
    "Tiempo": "Time (min)",
    "C/C0": "C/C₀ (-)",
    "H2O2_concentration": "H₂O₂ concentration (mg/L)",
}

MODEL_CONFIG = {

    "Random Forest": {

        "use_polynomial": False,

        "use_scaling": False,

        "use_target_scaling": False,

        "use_log_target": False

    },

    "Artificial Neural Network": {

        "use_polynomial": False,

        "use_scaling": True,

        "use_target_scaling": True,

        "use_log_target": False

    },

    "Gaussian Process Regression": {

        "use_polynomial": False,

        "use_scaling": True,

        "use_target_scaling": False,

        "use_log_target": False
    }

}

RF_PARAMS = {

    "n_estimators": 841,

    "max_depth": None,

    "min_samples_leaf": 1,

    "min_samples_split": 3,

    "random_state": 42,

    "n_jobs": -1,

    "max_features": "log2",

    "bootstrap": True

}

ANN_PARAMS = {

    "hidden_layer_sizes": (64, 32),

    "activation": "relu",

    "solver": "adam",

    "alpha":0.00298,

    "max_iter":100,

    "random_state":42

}

POLY_DEGREE = 2

# ============================================================
# GAUSSIAN PROCESS REGRESSION
# ============================================================

GPR_PARAMS = {
    "normalize_y": True,
    "random_state": RANDOM_STATE,
    "n_restarts_optimizer": 3,
}


# Initial kernel hyperparameters
GPR_KERNEL_PARAMS = {
    "length_scale": 2.049255,
    "constant_value": 0.1776,
    "noise_level": 6.69604e-03,
    "nu": 1.5,
}


# Optimization bounds for the kernel
GPR_KERNEL_BOUNDS = {
    "length_scale_bounds": (1e-3, 1e3),
    "constant_value_bounds": (1e-3, 1e3),
    "noise_level_bounds": (1e-8, 1.0),
}

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

def KERNEL_BUILD(
    length_scale=None,
    constant_value=None,
    noise_level=None,
    nu=None
):

    if length_scale is None:
        length_scale = GPR_KERNEL_PARAMS["length_scale"]

    if constant_value is None:
        constant_value = GPR_KERNEL_PARAMS["constant_value"]

    if noise_level is None:
        noise_level = GPR_KERNEL_PARAMS["noise_level"]

    if nu is None:
        nu = GPR_KERNEL_PARAMS["nu"]

    return (
        ConstantKernel(
            constant_value=constant_value,
            constant_value_bounds=
                GPR_KERNEL_BOUNDS[
                    "constant_value_bounds"
                ]
        )

        * Matern(
            length_scale=length_scale,
            length_scale_bounds=
                GPR_KERNEL_BOUNDS[
                    "length_scale_bounds"
                ],
            nu=nu
        )

        + WhiteKernel(
            noise_level=noise_level,
            noise_level_bounds=
                GPR_KERNEL_BOUNDS[
                    "noise_level_bounds"
                ]
        )
    )