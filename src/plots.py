import os
import matplotlib.pyplot as plt
from src.config import (PREDICTIONS_PATH,
                        FEATURES,
                        TIME_COLUMN
                    )
from pathlib import Path
import shap
from sklearn.inspection import PartialDependenceDisplay
from src.utils import sanitize_filename
from sklearn.metrics import r2_score

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

    plt.xlabel("Tiempo")
    plt.ylabel("C/C0")
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

    r2_val = r2_score(residuals["Experimental"], residuals["Prediction"])

    plt.gca().text(
        0.05,
        0.92,
        f"$R^2 = {r2_val:.4f}$",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85),
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

def plot_gpr_prediction(
    result,
    model_name,
    save=True,
    show=False
):

    experiment = result["experiment"]

    time = result["time"]

    y_true = result["y_true"]

    y_pred = result["y_pred"]

    y_std = result["y_std"]

    plt.figure(figsize=(8,5))

    ####################################
    # Experimental
    ####################################

    plt.scatter(
        time,
        y_true,
        color="black",
        s=45,
        label="Experimental",
        zorder=3
    )

    ####################################
    # Prediction
    ####################################

    plt.plot(
        time,
        y_pred,
        color="royalblue",
        linewidth=2,
        label="GPR prediction"
    )

    ####################################
    # Confidence interval
    ####################################

    if y_std is not None:

        lower = y_pred - 1.96*y_std

        upper = y_pred + 1.96*y_std

        plt.fill_between(

            time,

            lower,

            upper,

            color="royalblue",

            alpha=0.25,

            label="95% confidence interval"

        )

    plt.xlabel("Time (min)")

    plt.ylabel("H2O2")

    plt.title(f"Experiment {experiment}")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    if save:

        folder = os.path.join(
            PREDICTIONS_PATH,
            model_name
        )

        os.makedirs(
            folder,
            exist_ok=True
        )

        plt.savefig(

            os.path.join(

                folder,

                f"GPR_experiment_{experiment}.png"

            ),

            dpi=300

        )

    if show:

        plt.show()

    else:

        plt.close()

import os
import matplotlib.pyplot as plt

from src.config import PREDICTIONS_PATH


def plot_model_comparison(

    experiment,

    models_results,

    save=True,
    show=True

):
    """
    Parameters
    ----------
    experiment : int

    models_results : dict

    {

        "Random Forest": rf_results,

        "Gaussian Process": gpr_results,

        ...

    }

    """

    plt.figure(figsize=(8,5))

    experimental_drawn = False

    for model_name, results in models_results.items():

        for result in results:

            if result["experiment"] != experiment:
                continue

            if not experimental_drawn:

                plt.plot(

                    result["time"],

                    result["y_true"],

                    color="black",

                    linewidth=3,

                    label="Experimental"

                )

                experimental_drawn = True

            plt.plot(

                result["time"],

                result["y_pred"],

                linewidth=2,

                label=model_name

            )

    plt.xlabel("Time (min)")

    plt.ylabel("H2O2")

    plt.title(f"Experiment {experiment}")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    if save:

        folder = os.path.join(

            PREDICTIONS_PATH,

            "Comparison"

        )

        os.makedirs(folder, exist_ok=True)

        plt.savefig(

            os.path.join(

                folder,

                f"comparison_exp_{experiment}.png"

            ),

            dpi=300

        )

    if show:

        plt.show()

    else:

        plt.close()
        
def plot_rmse_comparison(
    rmse_values,
    save_path="results/figures/rmse_comparison.png"
):

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    display_names = {
    "Random Forest": "RF",
    "Artificial Neural Network": "ANN",
    "Gaussian Process Regression": "GPR",
    "Lumped Kinetic Model": "Kinetic model"
}

    names = [
    display_names.get(name, name)
    for name in rmse_values.keys()
]
    values = list(rmse_values.values())

    plt.figure(figsize=(9,5))

    bars = plt.bar(
    names,
    values
)

    plt.xticks(
    rotation=20,
    ha="right"
)

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()+bar.get_width()/2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.ylabel("Average RMSE")
    plt.title("Comparison of prediction error between models")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()
    
def plot_parity(
    residuals,
    model_name,
    save_path="results/parity_plots"
):
    """
    Generate a parity plot (Experimental vs Predicted)
    and return the R² value.

    Parameters
    ----------
    residuals : pandas.DataFrame
        Must contain:
        - Experimental
        - Prediction

    model_name : str
        Name of the model.

    save_path : str
        Directory where the figure will be saved.
    """

    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    # =====================================================
    # Create output directory
    # =====================================================

    os.makedirs(
        save_path,
        exist_ok=True
    )

    # =====================================================
    # Data
    # =====================================================

    experimental = residuals["Experimental"]
    prediction = residuals["Prediction"]

    # =====================================================
    # R²
    # =====================================================

    r2 = r2_score(
        experimental,
        prediction
    )

    # =====================================================
    # Figure
    # =====================================================

    plt.figure(
        figsize=(6, 6)
    )

    # Experimental vs prediction points
    plt.scatter(
        experimental,
        prediction,
        s=45,
        color="royalblue",
        edgecolor="black",
        alpha=0.8,
        label="Predictions"
    )

    # =====================================================
    # Ideal prediction line
    # =====================================================

    minimum = min(
        experimental.min(),
        prediction.min()
    )

    maximum = max(
        experimental.max(),
        prediction.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        linewidth=2,
        label="Ideal prediction"
    )

    # =====================================================
    # R² text
    # =====================================================

    plt.text(
        0.05,
        0.92,
        f"$R^2$ = {r2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(
            facecolor="white",
            alpha=0.85
        )
    )

    # =====================================================
    # Labels
    # =====================================================

    plt.xlabel(
        r"Experimental $H2O2$"
    )

    plt.ylabel(
        r"Predicted $H2O2$"
    )

    plt.title(
        f"Parity Plot - {model_name}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    # =====================================================
    # Filename
    # =====================================================

    safe_model_name = (
        model_name
        .replace(" ", "_")
    )

    filename = os.path.join(
        save_path,
        f"{safe_model_name}_Parity.png"
    )

    # =====================================================
    # Save
    # =====================================================

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Parity plot saved: {filename}"
    )

    return r2

