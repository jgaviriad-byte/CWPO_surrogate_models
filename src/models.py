from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from src.config import KERNEL_BUILD

from src.config import (

    RF_PARAMS,

    ANN_PARAMS,

    GPR_PARAMS,

    MODEL_CONFIG,

    KERNEL_BUILD

)

def polynomial_regression():
    return LinearRegression()


def random_forest():
    return RandomForestRegressor(

        **RF_PARAMS

    )

def neural_network():
    return MLPRegressor(

        **ANN_PARAMS

    )



def gaussian_process():
    return GaussianProcessRegressor(
        kernel=KERNEL_BUILD(),     

        **GPR_PARAMS

    )


def get_models():

    return {

        "Polynomial Regression": {

            "builder": polynomial_regression,

            **MODEL_CONFIG["Polynomial Regression"]

        },

        "Random Forest": {

            "builder": random_forest,

            **MODEL_CONFIG["Random Forest"]

        },

        "Artificial Neural Network": {

            "builder": neural_network,

            **MODEL_CONFIG["Artificial Neural Network"]

        },

        "Gaussian Process Regression": {

            "builder": gaussian_process,

            **MODEL_CONFIG["Gaussian Process Regression"]

        }

    }