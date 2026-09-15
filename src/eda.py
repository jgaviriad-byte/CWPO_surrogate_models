import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import FEATURES, FEATURE_LABELS

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
        f"  {df['Tiempo'].min()} "
        f"to "
        f"{df['Tiempo'].max()} min"
    )
    print("\nSamples per experiment:")

    print(df.groupby("Ensayo").size())
    print("\nProcessed dataset successfully summarized.\n")

from src.config import MON_COLUMN, TEMPERATURE_COLUMN

def experiment_summary(df):
    summary = []

    for ensayo, group in df.groupby("Ensayo"):

        summary.append({

            "MON_degradation(%)":(1-group[MON_COLUMN].iloc[-1]/group[MON_COLUMN].iloc[0])*100,

            "Ensayo": ensayo,

            "Samples": len(group),

            "t_initial": group["Tiempo"].min(),

            "t_final": group["Tiempo"].max(),

            "Temperature": group["Temp_K"].iloc[0],

            "Catalyst": group["Fe_activo"].iloc[0],

            "Mass Flow": group["H2O2_flujo"].iloc[0],

            "MON_initial": group[MON_COLUMN].iloc[0],

            "MON_final": group[MON_COLUMN].iloc[-1],

            "H2O2_initial": group["H2O2_flujo"].iloc[0]
            
            

        })

    summary_df = pd.DataFrame(summary)

    print("EXPERIMENT SUMMARY")

    print(summary_df)

    return summary_df



import matplotlib.pyplot as plt
from pathlib import Path


def correlation_matrix(df):

    output_dir = Path(
        "results/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # Variables originales del dataset
    # =====================================================

    variables = FEATURES

    corr = df[variables].corr(
        method="pearson"
    )

    print(
        "\nPEARSON CORRELATION MATRIX"
    )

    print(
        corr.round(3)
    )

    # =====================================================
    # Nombres que SOLO aparecerán en la gráfica
    # =====================================================

    labels = FEATURE_LABELS
    
    plot_labels = [
        labels.get(
            variable,
            variable
        )
        for variable in variables
    ]

    # =====================================================
    # Plot
    # =====================================================

    plt.figure(
        figsize=(9, 8)
    )

    im = plt.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    # =====================================================
    # Correlation values
    # =====================================================

    for i in range(len(variables)):

        for j in range(len(variables)):

            value = corr.iloc[i, j]

            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if abs(value) > 0.5
                    else "black"
                ),
                fontsize=9
            )

    # =====================================================
    # Axis labels
    # =====================================================

    plt.xticks(
        range(len(variables)),
        plot_labels,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(variables)),
        plot_labels
    )

    # =====================================================
    # Colorbar
    # =====================================================

    cbar = plt.colorbar(
        im
    )

    cbar.set_label(
        "Pearson correlation coefficient"
    )

    plt.title(
        "Pearson Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

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

    variables = FEATURES

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

    cols = FEATURES

    summary = (
        df.groupby("Ensayo")[cols]
          .first()
    )

    print(summary.round(4))

