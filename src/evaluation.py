import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

def calculate_metrics(
    y_true,
    y_pred
):

    r2 = r2_score(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return {

        "R2": r2,

        "RMSE": rmse,

        "MAE": mae

    }

def print_metrics(
    metrics
):
    """
    Print regression metrics.
    """

    print("\nMODEL PERFORMANCE")

    print("-"*35)

    print(f"R²   : {metrics['R2']:.4f}")

    print(f"RMSE : {metrics['RMSE']:.6f}")

    print(f"MAE  : {metrics['MAE']:.6f}")

def evaluate_all_folds(results):

    all_metrics = []

    for result in results:

        metrics = calculate_metrics(
            result["y_true"],
            result["y_pred"]
        )

        all_metrics.append(metrics)

    return all_metrics

def summarize_metrics(all_metrics):

    r2 = []
    rmse = []
    mae = []

    for metrics in all_metrics:

        r2.append(metrics["R2"])
        rmse.append(metrics["RMSE"])
        mae.append(metrics["MAE"])

    summary = {

        "Mean R2": np.mean(r2),
        "Std R2": np.std(r2),

        "Mean RMSE": np.mean(rmse),
        "Std RMSE": np.std(rmse),

        "Mean MAE": np.mean(mae),
        "Std MAE": np.std(mae)

    }

    return summary

def print_summary(summary):
    print("\nAVERAGE PERFORMANCE")
    print("-" * 40)

    print(f"Mean R²   : {summary['Mean R2']:.4f}")
    print(f"Std R²    : {summary['Std R2']:.4f}")

    print()

    print(f"Mean RMSE : {summary['Mean RMSE']:.6f}")
    print(f"Std RMSE  : {summary['Std RMSE']:.6f}")

    print()

    print(f"Mean MAE  : {summary['Mean MAE']:.6f}")
    print(f"Std MAE   : {summary['Std MAE']:.6f}")

def print_fold_metrics(all_metrics):
    """
    Print metrics for every validation fold.
    """

    print("\nPERFORMANCE BY FOLD")
    print("-" * 55)

    print(
        f"{'Fold':<6}"
        f"{'R²':>10}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
    )

    for i, metrics in enumerate(all_metrics, start=1):

        print(
            f"{i:<6}"
            f"{metrics['R2']:>10.4f}"
            f"{metrics['RMSE']:>12.5f}"
            f"{metrics['MAE']:>12.5f}"
        )

def evaluate_all_models(all_results):
 
    summaries = {}

    for model_name, results in all_results.items():

        metrics = evaluate_all_folds(results)

        summary = summarize_metrics(metrics)

        summaries[model_name] = summary

    return summaries

def print_all_models_summary(summaries):
    """
    Print summary metrics for every model.
    """

    print("\nMODEL COMPARISON")
    print("=" * 70)

    for model_name, summary in summaries.items():

        print(f"\n{model_name}")

        print("-" * 40)

        print(f"Mean R²   : {summary['Mean R2']:.4f}")
        print(f"Mean RMSE : {summary['Mean RMSE']:.6f}")
        print(f"Mean MAE  : {summary['Mean MAE']:.6f}")