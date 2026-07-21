from pathlib import Path

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
DATA_PATH = "data/raw/Doe_Completos.csv"

PROCESSED_DATA_PATH = "data/processed/Doe_processed.csv"

RANDOM_STATE = 42

EXPERIMENT_COLUMN = "Ensayo"

TARGET = "[MON]"

TARGET_NORMALIZED = "MON_norm"

FEATURES_MON = [
    "tiempo (min)",
    "[CAT]",
    "Temp(K?)",
    "Flujo_M"
]

FEATURES_H2O2 = [
    "tiempo (min)",
    "[CAT]",
    "Temp(K?)",
    "Flujo_M"
]

MODEL_CONFIG = {

    "Polynomial Regression": {

        "use_polynomial": True,

        "use_scaling": False,

        "use_target_scaling": False

    },

    "Random Forest": {

        "use_polynomial": False,

        "use_scaling": False,

        "use_target_scaling": False

    },

    "Artificial Neural Network": {

        "use_polynomial": False,

        "use_scaling": True,

        "use_target_scaling": True

    },

    "Gaussian Process Regression": {

        "use_polynomial": False,

        "use_scaling": True,

        "use_target_scaling": True

    }

}

RF_PARAMS = {

    "n_estimators": 200,

    "max_depth": 15,

    "min_samples_leaf": 5,

    "random_state": 42,

    "n_jobs": -1

}

ANN_PARAMS = {

    "hidden_layer_sizes": (128,64,32),

    "activation": "relu",

    "solver": "adam",

    "alpha":0.0293,

    "max_iter":1000,

    "random_state":42

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
