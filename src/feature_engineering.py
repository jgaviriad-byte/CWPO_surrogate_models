from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.config import FEATURES, TARGET, POLY_DEGREE

def get_features_target():
    
    return FEATURES, TARGET

def split_features_target(df, features, target):

    X = df[features].copy()

    y = df[target].copy()

    return X, y

def scale_features(
        X_train,
        X_test
):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return X_train_scaled, X_test_scaled, scaler

def scale_target(
        y_train,
        y_test
):

    scaler = StandardScaler()

    y_train_scaled = scaler.fit_transform(
        y_train.values.reshape(-1,1)
    ).ravel()

    y_test_scaled = scaler.transform(
        y_test.values.reshape(-1,1)
    ).ravel()

    return (
        y_train_scaled,
        y_test_scaled,
        scaler
    )

def create_polynomial_features(
    X_train,
    X_test,
    degree=POLY_DEGREE
):
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    X_train_poly = poly.fit_transform(X_train)

    X_test_poly = poly.transform(X_test)

    return X_train_poly, X_test_poly, poly

def log_transform_target(
        y_train,
        y_test
):
    log = np.log

    y_train_log = log(y_train)

    y_test_log = log(y_test)

    return y_train_log, y_test_log, log