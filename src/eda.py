import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
def dataset_summary(df):

    print("DATASET SUMMARY")

    print(f"Number of observations : {len(df)}")
    print(f"Number of variables    : {len(df.columns)}")
    print(f"Number of experiments  : {df['Ensayo'].nunique()}")

    print("\nVariables:")
    for col in df.columns:
        print(f"  • {col}")

    print("\nNumerical summary:")
    print(df.drop(columns="Ensayo").describe().round(4))

    print("\nTime interval:")
    print(
        f"  {df['tiempo (min)'].min()} "
        f"to "
        f"{df['tiempo (min)'].max()} min"
    )
    print("\nSamples per experiment:")

    print(df.groupby("Ensayo").size())
    print("\nProcessed dataset successfully summarized.\n")


def experiment_summary(df):
    summary = []

    for ensayo, group in df.groupby("Ensayo"):

        summary.append({

            "MON_degradation(%)":(1-group["[MON]"].iloc[-1]/group["[MON]"].iloc[0])*100,

            "Ensayo": ensayo,

            "Samples": len(group),

            "t_initial": group["tiempo (min)"].min(),

            "t_final": group["tiempo (min)"].max(),

            "Temperature": group["Temp(K?)"].iloc[0],

            "Catalyst": group["[CAT]"].iloc[0],

            "Mass Flow": group["Flujo_M"].iloc[0],

            "MON_initial": group["[MON]"].iloc[0],

            "MON_final": group["[MON]"].iloc[-1],

            "H2O2_initial": group["[H2O2]"].iloc[0]
            
            

        })

    summary_df = pd.DataFrame(summary)

    print("EXPERIMENT SUMMARY")

    print(summary_df)

    return summary_df


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
    summary = (
    df.groupby("Ensayo")["[H2O2]"]
      .agg(["first", "last", "max"])
)

    print(summary.sort_values("max", ascending=False))
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
    features = [
        "tiempo (min)",
        "Temp(K?)",
        "Flujo_M",
        "[CAT]"
    ]

    print("FEATURE SUMMARY")
    for feature in features:

        print(f"\n{feature}")

        print(f"Minimum : {df[feature].min():.4f}")

        print(f"Maximum : {df[feature].max():.4f}")

        print(f"Mean    : {df[feature].mean():.4f}")

        print(f"Std     : {df[feature].std():.4f}")

        print(f"Unique values : {df[feature].nunique()}")

import matplotlib.pyplot as plt
from pathlib import Path


def correlation_matrix(df):
    Path("results/figures").mkdir(
        parents=True,
        exist_ok=True
    )

    variables = [
        "[MON]",
        "MON_norm",
        "[H2O2]",
        "[CAT]",
        "Temp(K?)",
        "Flujo_M",
        "tiempo (min)"
    ]

    corr = df[variables].corr(method="pearson")
    print("PEARSON CORRELATION MATRIX")
    print(corr.round(3))

    plt.figure(figsize=(8,7))

    im = plt.imshow(
        corr,
        interpolation="nearest",
        aspect="auto"
    )

    plt.xticks(
        range(len(variables)),
        variables,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(variables)),
        variables
    )

    plt.colorbar(im)

    plt.title("Pearson Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        "results/figures/correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def feature_variability_by_experiment(df):
    variables = [
        "Temp(K?)",
        "[CAT]",
        "Flujo_M"
    ]

    print("\n" + "="*70)
    print("OPERATING CONDITIONS BY EXPERIMENT")
    print("="*70)

    for variable in variables:

        print(f"\n{variable}")

        summary = (
            df.groupby("Ensayo")[variable]
              .agg(["min","max","std","nunique"])
        )

        print(summary)

def plot_feature_distributions(df):

    Path("results/figures").mkdir(
        parents=True,
        exist_ok=True
    )

    variables = [
        "[MON]",
        "MON_norm",
        "[H2O2]",
        "[CAT]",
        "Temp(K?)",
        "Flujo_M"
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13,8)
    )

    axes = axes.flatten()

    for ax, variable in zip(axes, variables):

        ax.hist(
            df[variable],
            bins=20
        )

        ax.set_title(variable)

        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "results/figures/feature_distributions.png",
        dpi=300
    )

    plt.show()

def experiment_conditions(df):

    print("\nEXPERIMENT CONDITIONS")
    print("-"*70)

    cols = [
        "Temp(K?)",
        "[CAT]",
        "Flujo_M",
        "[H2O2]"
    ]

    summary = (
        df.groupby("Ensayo")[cols]
          .first()
    )

    print(summary.round(4))