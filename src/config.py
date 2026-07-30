from pathlib import Path

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
EXCLUDED_EXPERIMENTS = [4, 7, 13, 16, 20]

DATA_PATH = "data/raw/doe_data.csv"

PROCESSED_DATA_PATH = "data/processed/doe_processed.csv"

RANDOM_STATE = 42

EXPERIMENT_COLUMN = "Ensayo"

TARGET = "C/C0"

TIME_COLUMN = "Tiempo"

TEMPERATURE_COLUMN = "Temp_K"

PH_COLUMN = "pH"

CATALYST_COLUMN = "Fe_activo"

H2O2_COLUMN = "H2O2_flujo"

MON_INITIAL_COLUMN = "MON_0"

MON_COLUMN = "MON"

FEATURES = [TEMPERATURE_COLUMN,
            PH_COLUMN,
            CATALYST_COLUMN,
            H2O2_COLUMN,
            TIME_COLUMN,
            TARGET]

FIGURES_PATH = "results/figures"
PREDICTIONS_PATH = f"{FIGURES_PATH}/predictions"

MODEL_CONFIG = {

    "Polynomial Regression": {

        "use_polynomial": True,

        "use_scaling": False,

        "use_target_scaling": False,

        "use_log_target": False

    },

    "Random Forest": {

        "use_polynomial": False,

        "use_scaling": False,

        "use_target_scaling": False,

        "use_log_target": True

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

        "use_target_scaling": True,

        "use_log_scaling": False
    }

}

RF_PARAMS = {

    "n_estimators": 748,

    "max_depth": 40,

    "min_samples_leaf": 4,

    "min_samples_split": 2,

    "random_state": 42,

    "n_jobs": -1,

    "max_features": None,

}

ANN_PARAMS = {

    "hidden_layer_sizes": (128,64),

    "activation": "tanh",

    "solver": "adam",

    "alpha":0.0293,

    "max_iter":1000,

    "random_state":42,

    "bootstrap":True

}

POLY_DEGREE = 2

GPR_PARAMS = {

    "normalize_y":True,

    "random_state":42

}

KERNEL_PARAMS = {
    "constant_value": 1.0,

    "length_scale": 1.0,

    "noise_level": 1e-5,

    "nu": 2.5
}

def KERNEL_BUILD():   
    return(

        ConstantKernel(

            constant_value=KERNEL_PARAMS["constant_value"]

        )

        *

        Matern(

            length_scale=KERNEL_PARAMS["length_scale"],

            nu=KERNEL_PARAMS["nu"]

        )

        +

        WhiteKernel(

            noise_level=KERNEL_PARAMS["noise_level"]

        )

    )
