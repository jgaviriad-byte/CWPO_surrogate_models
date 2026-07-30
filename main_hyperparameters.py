import pandas as pd

from src.config import PROCESSED_DATA_PATH

from src.hyperparameter_search import random_search_random_forest


def main():

    df = pd.read_csv(PROCESSED_DATA_PATH)

    search = random_search_random_forest(df)


if __name__ == "__main__":

    main()