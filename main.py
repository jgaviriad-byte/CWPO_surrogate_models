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
from src.validation import (
    generate_leave_one_experiment_out,
    summarize_validation,
    verify_validation
)
def main():

    df = load_dataset()

    validate_dataset(df)

    df = create_mon_fraction(df)

    save_processed_dataset(df)

    folds = generate_leave_one_experiment_out(df)

    summarize_validation(folds)

    verify_validation(folds)
if __name__ == "__main__":
    main()