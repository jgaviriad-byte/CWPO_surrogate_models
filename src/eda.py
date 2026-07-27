import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import (
    ALL_FEATURES,
    FEATURES_NO_TIME,
    FEATURES
)
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



import matplotlib.pyplot as plt
from pathlib import Path


def correlation_matrix(df):
    Path("results/figures").mkdir(
        parents=True,
        exist_ok=True
    )

    variables = ALL_FEATURES

    corr = df[variables].corr(method="pearson")
    print("PEARSON CORRELATION MATRIX")
    print(corr.round(3))

    plt.figure(figsize=(8,7))

    im = plt.imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
    

    for i in range(len(variables)):
        for j in range(len(variables)):
            value = corr.iloc[i, j]

            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.5 else "black",
                fontsize=9
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

    cbar = plt.colorbar(im)

    cbar.set_label("Pearson correlation coefficient")

    plt.title("Pearson Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        "results/figures/correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def feature_variability_by_experiment(df):
    variables = []

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

    variables = ALL_FEATURES

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

    cols = FEATURES_NO_TIME

    summary = (
        df.groupby("Ensayo")[cols]
          .first()
    )

    print(summary.round(4))

