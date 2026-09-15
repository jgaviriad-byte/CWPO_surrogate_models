import pandas as pd
from src.config import DATA_PATH

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("DATASET LOADED")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def validate_dataset(df):
    print("DATASET VALIDATION")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicated rows:")
    print(df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nExperiments:")
    print(df["Ensayo"].nunique())

    print("\nTime range:")
    print(
        f"{df['Tiempo'].min()} "
        f"→ {df['Tiempo'].max()} min"
    )

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")

    if df.duplicated().sum() > 0:
        raise ValueError("Dataset contains duplicated rows.")

    print("\nDataset validation passed.")


def create_mon_fraction(df):
    df = df.copy()

    # Obtain the initial MON concentration of every experiment
    mon_initial = (
        df[df["Tiempo"] == 0]
        .set_index("Ensayo")["[MON]"]
    )

    # Verify every experiment has an initial condition
    missing = set(df["Ensayo"].unique()) - set(mon_initial.index)

    if len(missing) > 0:
        raise ValueError(
            f"The following experiments do not contain t = 0: {missing}"
        )

    # Compute degradation fraction
    df["MON_norm"] = df.apply(
        lambda row: row["[MON]"] / mon_initial[row["Ensayo"]],
        axis=1
    )

    print("\nDegradation fraction (MON_norm) successfully created.")

    return df

from pathlib import Path


def save_processed_dataset(df):
    output_folder = Path("data/processed")

    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "Doe_processed.csv"

    df.to_csv(output_file, index=False)

    print(f"\nProcessed dataset saved in:")
    print(output_file.resolve())
