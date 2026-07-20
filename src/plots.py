import os
import matplotlib.pyplot as plt

def plot_prediction(result, save=True):
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

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    if save:

        os.makedirs(
            "results/figures/predictions",
            exist_ok=True
        )

        plt.savefig(
            f"results/figures/predictions/experiment_{experiment}.png",
            dpi=300
        )

    plt.show()

def plot_experiment(results, experiment):

    for result in results:

        if result["experiment"] == experiment:

            plot_prediction(result)

            return

    print(f"Experiment {experiment} not found.")