import joblib

import numpy as np

import pandas as pd

from src.feature_engineering import get_features_target

from src.config import (
        PROCESSED_DATA_PATH,
        FEATURES
    )


def load_trained_model(model_path):
    """
    Load a trained model and its preprocessing objects.
    """

    package = joblib.load(model_path)

    return (

        package["model"],

        package["transformers"]

    )


def preprocess_input(
    X,
    transformers
):
    """
    Apply the same preprocessing used during training.
    """

    if "poly" in transformers:

        X = transformers["poly"].transform(X)

    if "scaler_X" in transformers:

        X = transformers["scaler_X"].transform(X)

    return X


def predict(
    model,
    transformers,
    X
):
    """
    Generate predictions.
    """

    X = preprocess_input(

        X,

        transformers

    )

    y_pred = model.predict(X)

    if "scaler_y" in transformers:

        y_pred = (

            transformers["scaler_y"]

            .inverse_transform(

                y_pred.reshape(-1,1)

            )

            .ravel()

        )

    return y_pred


def predict_dataframe(
    model_path,
    dataframe
):
    """
    Predict directly from a dataframe.
    """

    model, transformers = load_trained_model(
        model_path
    )

    FEATURES, _ = get_features_target()

    X = dataframe[FEATURES].copy()

    prediction = predict(
        model,
        transformers,
        X
    )

    return prediction

