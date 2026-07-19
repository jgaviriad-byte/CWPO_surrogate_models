from src.preprocessing import (
    load_dataset,
    validate_dataset,
    create_mon_fraction,
    save_processed_dataset
)
from src.eda import (
    dataset_summary,
    experiment_summary,
    plot_feature_distributions
)
def main():

    df = load_dataset()

    validate_dataset(df)

    df = create_mon_fraction(df)

    save_processed_dataset(df)

    plot_feature_distributions(df)

if __name__ == "__main__":
    main()