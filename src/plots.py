import os
import matplotlib.pyplot as plt
from src.config import (PREDICTIONS_PATH,
                        FEATURES
                    )
from pathlib import Path
import shap
from sklearn.inspection import PartialDependenceDisplay
from src.utils import sanitize_filename

def plot_prediction(result, save=True, show=True):
    experiment = result["experiment"]

    time = result["time"]

    y_true = result["y_true"]

    y_pred = result["y_pred"]

    plt.figure(figsize=(8,5))

    plt.plot(
        time,
        y_true,
        color="black",
        linewidth=2,
        label="Experimental"
    )

    plt.plot(
        time,
        y_pred,
        "--",
        linewidth=2,
        color="royalblue",
        label="Prediction"
    )

    plt.xlabel("Time (min)")
    plt.ylabel("MON normalized")
    plt.title(f"Experiment {experiment}")

    plt.grid(
    True,
    alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    if save:

        os.makedirs(
            PREDICTIONS_PATH,
            exist_ok=True
        )

        plt.savefig(
            f"{PREDICTIONS_PATH}/experiment_{experiment}.png",
            dpi=300
        )

    if show:
        plt.show()
    else:
        plt.close()

def plot_experiment(results, experiment):

    for result in results:

        if result["experiment"] == experiment:

            plot_prediction(result)

            return

    raise ValueError(
    f"Experiment {experiment} not found."
    )

def plot_mon_vs_time(df):
    plt.figure(figsize=(10,6))

    for ensayo, group in df.groupby("Ensayo"):

        plt.plot(
            group["tiempo (min)"],
            group["[MON]"],
            linewidth=2,
            label=f"Exp {ensayo}"
        )

    plt.title("MON concentration evolution")

    plt.xlabel("Time (min)")

    plt.ylabel("MON concentration")

    plt.grid(alpha=0.3)

    plt.legend(
        ncol=2,
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        "results/figures/MON_vs_time.png",
        dpi=300
    )

    plt.show()

def plot_MONnorm_vs_time(df):
  
    Path("results/figures").mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10,6))

    for ensayo, group in df.groupby("Ensayo"):

        plt.plot(
            group["tiempo (min)"],
            group["MON_norm"],
            linewidth=2,
            label=f"Exp {ensayo}"
        )

    plt.title(
        "Normalized MON concentration evolution"
    )

    plt.xlabel("Time (min)")

    plt.ylabel(r"$MON/MON_0$")

    plt.grid(alpha=0.3)

    plt.legend(
        ncol=2,
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        "results/figures/MONnorm_vs_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_h2o2_vs_time(df):
    Path("results/figures").mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10,6))

    for ensayo, group in df.groupby("Ensayo"):

        plt.plot(
            group["tiempo (min)"],
            group["[H2O2]"],
            linewidth=2,
            label=f"Exp {ensayo}"
        )

    plt.title(
        "Hydrogen peroxide concentration evolution"
    )

    plt.xlabel("Time (min)")

    plt.ylabel(r"$H_2O_2$ concentration")

    plt.grid(alpha=0.3)

    plt.legend(
        fontsize=8,
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        "results/figures/H2O2_vs_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def feature_summary(df):
    features = FEATURES

    print("FEATURE SUMMARY")
    for feature in features:

        print(f"\n{feature}")

        print(f"Minimum : {df[feature].min():.4f}")

        print(f"Maximum : {df[feature].max():.4f}")

        print(f"Mean    : {df[feature].mean():.4f}")

        print(f"Std     : {df[feature].std():.4f}")

        print(f"Unique values : {df[feature].nunique()}")

def plot_feature_importance(
    importance,
    title
):

    Path(

        "results/figures/interpretability"

    ).mkdir(

        parents=True,

        exist_ok=True

    )

    plt.figure(figsize=(8,5))

    plt.barh(

        importance["Feature"],

        importance["Importance"]

    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(

        f"results/figures/interpretability/{title}.png",

        dpi=300

    )

    plt.show()

def plot_permutation_importance(
    importance,
    title
):

    Path(

        "results/figures/interpretability"

    ).mkdir(

        parents=True,

        exist_ok=True

    )

    plt.figure(figsize=(8,5))

    plt.barh(

        importance["Feature"],

        importance["Importance"],

        xerr=importance["Std"]

    )

    plt.gca().invert_yaxis()

    plt.xlabel("Permutation Importance")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(

        f"results/figures/interpretability/{title}.png",

        dpi=300

    )

    plt.show()

def plot_shap_summary(
    shap_values,
    X,
    save_name="SHAP_Summary"
):
    """
    SHAP beeswarm summary.
    """

    shap.summary_plot(

        shap_values,

        X,

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        f"results/figures/{save_name}.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

def plot_shap_bar(
    shap_values,
    X,
    save_name="SHAP_Bar"
):
    """
    SHAP global importance.
    """

    shap.summary_plot(

        shap_values,

        X,

        plot_type="bar",

        show=False

    )

    plt.tight_layout()


    plt.savefig(

        f"results/figures/{save_name}.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

def plot_shap_dependence(
    feature,
    shap_values,
    X,
    filename
):
    """
    Plot SHAP dependence plot.
    """

    shap.dependence_plot(

        feature,

        shap_values,

        X,

        show=False

    )

    safe_filename = sanitize_filename(filename)

    plt.tight_layout()

    plt.savefig(

        f"results/figures/{safe_filename}.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

def plot_partial_dependence(

    model,

    X,

    feature,

    filename

):

    Path(

        "results/figures"

    ).mkdir(

        parents=True,

        exist_ok=True

    )

    X = X.astype(float)

    PartialDependenceDisplay.from_estimator(

        model,

        X,

        [feature]

    )

    plt.tight_layout()

    safe_filename = sanitize_filename(filename)

    plt.savefig(

        f"results/figures/{safe_filename}.png",

        dpi=300

    )

    plt.close()

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path


def plot_residuals(
    residuals,
    filename
):

    plt.figure(figsize=(7,5))

    plt.scatter(

        residuals["Prediction"],

        residuals["Residual"],

        alpha=0.6,

        s=20

    )

    plt.axhline(

        0,

        color="red",

        linestyle="--"

    )

    plt.xlabel("Prediction")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        f"results/figures/{filename}.png",

        dpi=300

    )

    plt.close()

def plot_prediction_vs_experimental(
    residuals,
    filename
):

    Path("results/figures").mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(6,6))

    plt.scatter(

        residuals["Experimental"],

        residuals["Prediction"],

        alpha=0.6,

        s=20

    )

    minimum = min(

        residuals["Experimental"].min(),

        residuals["Prediction"].min()

    )

    maximum = max(

        residuals["Experimental"].max(),

        residuals["Prediction"].max()

    )

    plt.plot(

        [minimum, maximum],

        [minimum, maximum],

        "r--",

        linewidth=2

    )

    plt.xlabel("Experimental")

    plt.ylabel("Prediction")

    plt.title("Prediction vs Experimental")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        f"results/figures/{filename}.png",

        dpi=300

    )

    plt.close()

def plot_residual_distribution(
    residuals,
    filename
):

    plt.figure(figsize=(7,5))

    plt.hist(

        residuals["Residual"],

        bins=30

    )

    plt.axvline(

        0,

        color="red",

        linestyle="--"

    )

    plt.xlabel("Residual")

    plt.ylabel("Frequency")

    plt.title("Residual Distribution")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        f"results/figures/{filename}.png",

        dpi=300

    )

    plt.close()