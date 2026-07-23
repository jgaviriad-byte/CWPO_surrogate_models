import numpy as np


Q_CRITICAL = {

    3: 0.970,
    4: 0.829,
    5: 0.710,
    6: 0.625,
    7: 0.568,
    8: 0.526,
    9: 0.493,
    10: 0.466,
    11: 0.444,
    12: 0.426,
    13: 0.410,
    14: 0.396,
    15: 0.384,
    16: 0.374,
    17: 0.365,
    18: 0.356,
    19: 0.349,
    20: 0.342

}


def dixon_q_test(values):

    """
    Dixon Q test.

    Parameters
    ----------
    values : pandas.Series

        Index = experiment
        Values = variable

    Returns
    -------
    dict
    """

    sorted_values = values.sort_values()

    n = len(sorted_values)

    if n < 3 or n > 20:

        raise ValueError(
            "Dixon Q Test supports only 3 <= n <= 20."
        )

    data_range = (

        sorted_values.iloc[-1]

        -

        sorted_values.iloc[0]

    )

    if data_range == 0:

        return None

    q_low = (

        sorted_values.iloc[1]

        -

        sorted_values.iloc[0]

    ) / data_range

    q_high = (

        sorted_values.iloc[-1]

        -

        sorted_values.iloc[-2]

    ) / data_range

    q_critical = Q_CRITICAL[n]

    return {

        "Q_low": q_low,

        "Q_high": q_high,

        "Q_critical": q_critical,

        "low_outlier": q_low > q_critical,

        "high_outlier": q_high > q_critical,

        "low_value": sorted_values.iloc[0],

        "high_value": sorted_values.iloc[-1],

        "low_experiment": sorted_values.index[0],

        "high_experiment": sorted_values.index[-1]

    }


def print_q_test(values, variable):

    result = dixon_q_test(values)

    print()

    print("=" * 60)

    print(f"DIXON Q TEST - {variable}")

    print("=" * 60)

    print()

    print("Lowest value")

    print(f"Experiment : {result['low_experiment']}")

    print(f"Value      : {result['low_value']:.6f}")

    print()

    print("Highest value")

    print(f"Experiment : {result['high_experiment']}")

    print(f"Value      : {result['high_value']:.6f}")

    print()

    print(f"Q lower    : {result['Q_low']:.4f}")

    print(f"Q upper    : {result['Q_high']:.4f}")

    print(f"Q critical : {result['Q_critical']:.4f}")

    print()

    if result["low_outlier"]:

        print(
            f"Lower outlier detected "
            f"(Experiment {result['low_experiment']})"
        )

    else:

        print("No lower outlier detected.")

    if result["high_outlier"]:

        print(
            f"Upper outlier detected "
            f"(Experiment {result['high_experiment']})"
        )

    else:

        print("No upper outlier detected.")