from src.feature_engineering import (
    get_features_target,
    split_features_target,
    scale_features,
    scale_target,
    create_polynomial_features
)

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

        "y_pred": y_pred
    
   
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