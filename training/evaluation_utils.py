import os
import csv
import pandas as pd
import torch
from evaluation.metrics import evaluate_inverse
from evaluation.plot import plot_pred_vs_true, plot_loss_curve

def write_metrics(path, tag, t0, t3, train_res, test_res, write_header):
    """
    Append evaluation metrics for train and test splits to a CSV file.

    Args:
        path (str): Path to the metrics CSV file.
        tag (str): Identifier for the current run (e.g., hyperparameter setting).
        t0 (Timestamp): Start date of the backtest window.
        t3 (Timestamp): End date of the backtest window.
        train_res (dict): Evaluation results on training set.
        test_res (dict): Evaluation results on test set.
        write_header (bool): Whether to write the header row (if file is new).
    """
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Tag", "WindowStart", "WindowEnd", "Split", 
                            "MSE", "SignAcc", "Pearson", "IC", "IR", "CR", "Prec@K"])
        for split_name, res in [("Train", train_res), ("Test", test_res)]:
            writer.writerow([tag, t0.date(), t3.date(), split_name,
                            res["mse"], res["sign_accuracy"], res["pearson"],
                            res["ic"], res["ir"], res["cr"], res["prec@k"]])
                
def save_test_predictions(y_true, y_pred, idx_test, all_dates, stock_keys, window_start, window_end, window_size, path, write_header):
    """
    Save model predictions and ground truths for each stock and time step to CSV.

    Args:
        y_true (Tensor): Ground truth values, shape (num_samples, num_stocks).
        y_pred (Tensor): Predicted values, same shape as y_true.
        idx_test (Tensor): Indices of the test time range.
        all_dates (array-like): Array of all available timestamps.
        stock_keys (list): List of stock identifiers (e.g., tickers).
        window_start (date): Start date of the prediction window.
        window_end (date): End date of the prediction window.
        window_size (int): Number of past steps used per sample (to offset future prediction time).
        path (str): File path to write predictions.
        write_header (bool): Whether to write the header row.
    """
    rows = []
    num_samples, num_stocks = y_true.shape
    test_dates = all_dates.values[idx_test.numpy()]

    for i in range(num_samples):
        date = pd.Timestamp(test_dates[i + window_size]).date()
        for j in range(num_stocks):
            rows.append({
                "WindowStart": window_start,
                "WindowEnd": window_end,
                "Date": date,
                "Stock": stock_keys[j],
                "y_true": y_true[i, j],
                "y_pred": y_pred[i, j]
            })

    df_preds = pd.DataFrame(rows)
    df_preds.to_csv(path, mode="a", header=write_header, index=False)