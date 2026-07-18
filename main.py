from src.preprocessing import (
    load_dataset,
    validate_dataset,
    create_mon_fraction,
    save_processed_dataset
)


def main():

    df = load_dataset()

    validate_dataset(df)

    df = create_mon_fraction(df)

    save_processed_dataset(df)


if __name__ == "__main__":
    main()