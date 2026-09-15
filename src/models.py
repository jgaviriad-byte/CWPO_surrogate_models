from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from src.config import (
    RF_PARAMS,
    ANN_PARAMS,
    GPR_PARAMS,
    GPR_KERNEL_PARAMS,
    MODEL_CONFIG,
    KERNEL_BUILD,
)


def random_forest():
    return RandomForestRegressor(**RF_PARAMS)


def neural_network():
    return MLPRegressor(**ANN_PARAMS)


def build_gaussian_process(
    length_scale=None,
    constant_value=None,
    noise_level=None,
    nu=None,
):
    if length_scale is None:
        length_scale = GPR_KERNEL_PARAMS["length_scale"]

    if constant_value is None:
        constant_value = GPR_KERNEL_PARAMS["constant_value"]

    if noise_level is None:
        noise_level = GPR_KERNEL_PARAMS["noise_level"]

    if nu is None:
        nu = GPR_KERNEL_PARAMS["nu"]

    kernel = KERNEL_BUILD(
        length_scale=length_scale,
        constant_value=constant_value,
        noise_level=noise_level,
        nu=nu,
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        **GPR_PARAMS
    )


def get_models():
    return {
        "Random Forest": {
            "builder": random_forest,
            **MODEL_CONFIG["Random Forest"]
        },
        "Artificial Neural Network": {
            "builder": neural_network,
            **MODEL_CONFIG["Artificial Neural Network"]
        },
        "Gaussian Process Regression": {
            "builder": build_gaussian_process,
            **MODEL_CONFIG["Gaussian Process Regression"]
        }
    }