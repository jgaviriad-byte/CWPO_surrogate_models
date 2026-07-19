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