import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestRegressor

from scipy.stats import randint


from src.config import (
    FEATURES,
    TARGET,
    EXPERIMENT_COLUMN,
    RANDOM_STATE
)


def random_search_random_forest(df):

    X = df[FEATURES]

    y = df[TARGET]

    groups = df[EXPERIMENT_COLUMN]

    model = RandomForestRegressor(
        random_state=RANDOM_STATE
    )

    param_dist = {

        "n_estimators": randint(10,1000),

        "max_depth":[
            None,
            5,
            10,
            20,
            30,
            40,
            50
        ],

        "min_samples_split":[
            2,
            4,
            6,
            8,
            10
        ],

        "min_samples_leaf":[
            1,
            2,
            4,
            6
        ],

        "max_features":[
            "sqrt",
            "log2",
            None
        ],

        "bootstrap":[
            True,
            False
        ]

    }

    cv = GroupKFold(
        n_splits=12
    )

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_dist,

        n_iter=50,

        cv=cv,

        scoring="r2",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        verbose=2,

        return_train_score=True

    )

    search.fit(

        X,

        y,

        groups=groups

    )

    print()

    print("="*70)

    print("BEST PARAMETERS")

    print("="*70)

    print(search.best_params_)

    print()

    print("Best CV R2")

    print(search.best_score_)

    return search
