import pandas as pd

def generate_leave_one_experiment_out(df):
    folds = []

    experiments = sorted(df["Ensayo"].unique())

    for experiment in experiments:

        train_df = df[df["Ensayo"] != experiment].copy()

        test_df = df[df["Ensayo"] == experiment].copy()

        folds.append(
            (
                train_df,
                test_df,
                experiment
            )
        )

    return folds

def summarize_validation(folds):
    print("LEAVE-ONE-EXPERIMENT-OUT VALIDATION")

    for i, (train, test, experiment) in enumerate(folds):

        print(f"\nFold {i+1}")

        print(f"Test experiment      : {experiment}")

        print(f"Training experiments : {train['Ensayo'].nunique()}")

        print(f"Testing experiments  : {test['Ensayo'].nunique()}")

        print(f"Training samples     : {len(train)}")

        print(f"Testing samples      : {len(test)}")

def verify_validation(folds):
    print("VALIDATION CHECK")

    valid = True

    for train, test, experiment in folds:

        train_ids = set(train["Ensayo"].unique())

        test_ids = set(test["Ensayo"].unique())

        overlap = train_ids.intersection(test_ids)

        if overlap:

            valid = False

            print(f"Overlap detected in experiment {experiment}")

    if valid:

        print("Validation successful.")

        print("No experiment appears simultaneously")

        print("in training and testing.")