import os
import matplotlib.pyplot as plt
from src.config import PREDICTIONS_PATH

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