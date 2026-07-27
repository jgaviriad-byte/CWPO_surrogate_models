from pathlib import Path

import joblib

from src.feature_engineering import (
    get_features_target,
    split_features_target,
    scale_features,
    scale_target,
    create_polynomial_features
)

from src.utils import create_directory

def train_single_fold(
    model_config,
    train_df,
    test_df
):
    features, target = get_features_target()

    X_train, y_train = split_features_target(
        train_df,
        features,
        target
    )

    X_test, y_test = split_features_target(
        test_df,
        features,
        target
    )

    if model_config["use_polynomial"]:

        X_train, X_test, _ = create_polynomial_features(
            X_train,
            X_test
        )

    if model_config["use_scaling"]:

        X_train, X_test, scaler_X = scale_features(
            X_train,
            X_test
        )

    if model_config["use_target_scaling"]:

        y_train, y_test, scaler_y = scale_target(
            y_train,
            y_test
        )
    
    model = model_config["builder"]()

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    if model_config["use_target_scaling"]:

        y_pred = scaler_y.inverse_transform(
            y_pred.reshape(-1, 1)
        ).ravel()

        y_test = scaler_y.inverse_transform(
            y_test.reshape(-1, 1)
        ).ravel()

    return {

        "experiment": test_df["Ensayo"].iloc[0],

        "time": test_df["tiempo (min)"].values,

        "y_true": y_test,

        "y_pred": y_pred,

        "model": model,
    
   
}

def train_one_model(
    model_name,
    model_config,
    folds
):
    results = []

    print(f"\nTraining {model_name}")

    print("-" * 40)

    for fold_number, fold in enumerate(folds, start=1):

        train_df, test_df, experiment = fold

        result = train_single_fold(
            model_config,
            train_df,
            test_df
        )

        result["fold"] = fold_number

        results.append(result)

        print(
            f"Fold {fold_number:2d} | "
            f"Experiment {experiment:2d} completed."
        )

    return results


def train_all_models(models, folds):
  
    all_results = {}

    for model_name, model_config in models.items():

        print("\n" + "=" * 60)
        print(f"MODEL: {model_name}")
        print("=" * 60)

        model_results = train_one_model(
            model_name,
            model_config,
            folds
        )

        all_results[model_name] = model_results

    return all_results

def train_final_model(
    model_config,
    df
):
    """
    Train one model using the complete dataset.
    """

    FEATURES, TARGET = get_features_target()

    X, y = split_features_target(
        df,
        FEATURES,
        TARGET
    )

    transformers = {}

    if model_config["use_polynomial"]:

        X, _, poly = create_polynomial_features(
            X,
            X
        )

        transformers["poly"] = poly

    if model_config["use_scaling"]:

        X, _, scaler_X = scale_features(
            X,
            X
        )

        transformers["scaler_X"] = scaler_X

    if model_config["use_target_scaling"]:

        y, _, scaler_y = scale_target(
            y,
            y
        )

        transformers["scaler_y"] = scaler_y

    model = model_config["builder"]()

    model.fit(
        X,
        y
    )

    return model, transformers


def save_trained_model(
    model,
    transformers,
    model_name
):
    """
    Save trained model and preprocessing objects.
    """

    output_dir = Path(
        "results/models"
    )

    create_directory(output_dir)

    package = {

        "model": model,

        "transformers": transformers

    }

    filename = output_dir / f"{model_name}.joblib"

    joblib.dump(
        package,
        filename
    )

    print(f"Saved: {filename}")


def train_and_save_all_models(
    models,
    df
):
    """
    Train every model using the complete dataset
    and save it to disk.
    """

    print("\nTRAINING FINAL MODELS")
    print("=" * 60)

    for model_name, model_config in models.items():

        print(f"\nTraining final {model_name}")

        model, transformers = train_final_model(
            model_config,
            df
        )

        save_trained_model(
            model,
            transformers,
            model_name
        )

    print("\nAll trained models saved successfully.")