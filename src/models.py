from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel
)

def polynomial_regression():

    model = LinearRegression()

    return model

def random_forest():

    model = RandomForestRegressor(

        n_estimators=200,

        max_depth=15,

        min_samples_leaf=5,

        random_state=42,

        n_jobs=-1

    )

    return model

def neural_network():

    model = MLPRegressor(

        hidden_layer_sizes=(128,64,32),

        activation="relu",

        solver="adam",

        alpha=0.0293,

        max_iter=1000,

        random_state=42

    )

    return model

def gaussian_process():

    kernel = (

        ConstantKernel()

        *

        Matern(

            nu=2.5

        )

        +

        WhiteKernel()

    )

    model = GaussianProcessRegressor(

        kernel=kernel,

        normalize_y=True,

        random_state=42

    )

    return model

def get_models():

    return {

        "Polynomial Regression": {

            "model": polynomial_regression(),

            "use_polynomial": True,

            "use_scaling": False,

            "use_target_scaling": False

        },

        "Random Forest": {

            "model": random_forest(),

            "use_polynomial": False,

            "use_scaling": False,

            "use_target_scaling": False

        },

        "Artificial Neural Network": {

            "model": neural_network(),

            "use_polynomial": False,

            "use_scaling": True,

            "use_target_scaling": True

        },

        "Gaussian Process Regression": {

            "model": gaussian_process(),

            "use_polynomial": False,

            "use_scaling": True,

            "use_target_scaling": True

        }

    }
