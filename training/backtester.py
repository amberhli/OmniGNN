import os
import torch
import logging
import pandas as pd
from torch import nn
from torch.optim import Adam

from training.data_utils import (
    normalize_tensor_per_node,
    scale_labels_tensor_per_node,
    normalize_edges_minmax,
    prepare_data
)
from training.train_utils import build_model, train_one_epoch
from training.evaluation_utils import (
    write_metrics,
    save_test_predictions
)
from evaluation.metrics import evaluate_inverse
from evaluation.plot import plot_pred_vs_true, plot_loss_curve

from training.logger import setup_logging
setup_logging()

class Backtester:
    """
    Backtester for training and evaluating a temporal GNN model (OmniGNN) using a rolling-window approach.

    This class automates the workflow of:
    - Normalizing and preparing graph-based time series data.
    - Training the model with early stopping.
    - Evaluating on training and testing periods.
    - Saving metrics, predictions, and model checkpoints.

    Args:
        features_time (Tensor): Time-series tensor of node features (T, N, F).
        labels_time (Tensor): Time-series tensor of target labels (T, N).
        adj_time (dict): Dictionary of adjacency matrices per metapath (keyed by metapath).
        edge_time (dict): Dictionary of edge attributes per metapath (keyed by metapath).
        meta_paths (list): List of metapath keys used to index adj_time and edge_time.
        all_dates (list or pd.Series): Timestamps corresponding to the data.
        window_size (int): Lookback window size for temporal modeling.
        device (torch.device): Torch device to run the model on (CPU or CUDA).
        stock_names (list): List of stock tickers.
        hidden_dim (int): Hidden dimension for the model layers.
        train_months (int): Length of training window (months).
        val_months (int): Length of validation window (months).
        test_months (int): Length of test window (months).
        batch_size (int): Batch size for training.
    """
    def __init__(self, features_time, labels_time, adj_time, edge_time, meta_paths,
                 all_dates, device, stock_names, hyperparams, train_months=6, 
                 val_months=2, test_months=2, batch_size=16):
        self.features_time = features_time
        self.labels_time = labels_time
        self.adj_time = adj_time
        self.edge_time = edge_time
        self.meta_paths = meta_paths
        self.all_dates = pd.to_datetime(all_dates)
        self.device = device
        self.hyperparams = hyperparams
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.batch_size = batch_size
        self.stock_keys = stock_names
        self.n_nodes = features_time.shape[1]
        self.feature_dim = features_time.shape[2]
        self.edge_dim = edge_time[meta_paths[0]].shape[-1]
        self.n_stocks = len(stock_names)

    def run(self, epochs=10, lr=0.001, tag=None):
        """
        Executes the rolling-window backtest, training and evaluating the model on each window.

        This method handles:
        - Data normalization based on training window.
        - Data slicing into rolling train/val/test splits.
        - Model training with early stopping based on validation loss.
        - Evaluation on both training and test sets.
        - Logging, metric saving, prediction output, and model checkpointing.

        Args:
            epochs (int): Maximum number of training epochs per rolling window.
            lr (float): Learning rate for the optimizer.
            tag (str): Unique identifier used to name result folders and logs.

        Returns:
            None
        """
        results_dir = os.path.join("results", tag)
        os.makedirs(results_dir, exist_ok=True)
        predictions_path = os.path.join(results_dir, "predictions.csv")
        metrics_path = os.path.join("results/metrics_log.csv")
        write_pred_header = not os.path.exists(predictions_path)
        write_metrics_header = not os.path.exists(metrics_path)

        t0 = self.all_dates[0]
        while True:
            t1 = t0 + pd.DateOffset(months=self.train_months)
            t2 = t1 + pd.DateOffset(months=self.val_months)
            t3 = t2 + pd.DateOffset(months=self.test_months)
            if t3 > self.all_dates[-1]: break

            logging.info(f"Rolling Window: Train {t0.date()} to {t1.date()} | Val Test {t1.date()} to {t2.date()} | {t2.date()} to {t3.date()}")

            mask = lambda s, e: (self.all_dates >= s) & (self.all_dates <= e)
            train_mask = mask(t0, t1)
            val_mask = mask(t1, t2)
            test_mask = mask(t2, t3)

            feats = self.features_time
            labels = self.labels_time

            idx_train = torch.where(torch.tensor(train_mask))[0]
            idx_val = torch.where(torch.tensor(val_mask))[0]
            idx_test = torch.where(torch.tensor(test_mask))[0]

            feats_train, mu_f, sigma_f = normalize_tensor_per_node(feats[idx_train])
            labels_train, mu_y, sigma_y = scale_labels_tensor_per_node(labels[idx_train])
            feats_val = (feats[idx_val] - mu_f.unsqueeze(0)) / sigma_f.unsqueeze(0)
            labels_val = (labels[idx_val] - mu_y.unsqueeze(0)) / sigma_y.unsqueeze(0)
            feats_test = (feats[idx_test] - mu_f.unsqueeze(0)) / sigma_f.unsqueeze(0)
            labels_test = (labels[idx_test] - mu_y.unsqueeze(0)) / sigma_y.unsqueeze(0)

            edge_norm = normalize_edges_minmax(self.edge_time, self.meta_paths, idx_train)

            # begin hyperparameter tuning
            best_val_loss = float("inf")
            best_hyperparam = None
            best_model_state = None

            criterion = nn.MSELoss()
            
            for hyperparam in self.hyperparams:
                window_size = hyperparam[0]
                hidden_dim = hyperparam[1]

                model = build_model(self.feature_dim, self.edge_dim, self.meta_paths, window_size, self.n_nodes, self.device, hidden_dim)
                optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                train_X, train_y, adj, edge = prepare_data(feats_train, labels_train, 0, feats_train.shape[0], self.adj_time, edge_norm, self.meta_paths, window_size)
                val_X, val_y, val_adj, val_edge = prepare_data(feats_val, labels_val, 0, feats_val.shape[0], self.adj_time, self.edge_time, self.meta_paths, window_size)

                # loss_history, val_loss_history = [], []
                wait = 0
                patience = 25
                for epoch in range(epochs):
                    train_loss = train_one_epoch(model, optimizer, criterion, train_X, train_y, adj, edge,
                                             self.device, self.batch_size, self.meta_paths, self.n_stocks)
                    model.eval()
                    with torch.no_grad():
                        val_preds = model(val_X.to(self.device),
                                        {mp: val_adj[mp].to(self.device) for mp in self.meta_paths},
                                        {mp: val_edge[mp].to(self.device) for mp in self.meta_paths})
                        val_loss = criterion(val_preds[:, :self.n_stocks], val_y[:, :self.n_stocks].to(self.device)).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()
                        best_hyperparam = hyperparam # still storing a tuple
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break
            
            # retrain model on best hyperparams
            window_size = best_hyperparam[0]
            hidden_dim = best_hyperparam[1]

            logging.info(f"Best Hyperparams: {best_hyperparam} | Val Loss: {best_val_loss:.4f}")

            model = build_model(self.feature_dim, self.edge_dim, self.meta_paths, window_size, self.n_nodes, self.device, hidden_dim).to(self.device)
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

            X_all = torch.cat([feats_train, feats_val], dim=0) # as to not lose val data
            y_all = torch.cat([labels_train, labels_val], dim=0)
            X_combined, y_combined, adj_all, edge_all = prepare_data(X_all, y_all, 0, X_all.shape[0],
                                                         self.adj_time, self.edge_time,
                                                         self.meta_paths, window_size)
            
            for epoch in range(epochs):
                train_loss = train_one_epoch(model, optimizer, criterion, X_combined, y_combined,
                                             adj_all, edge_all, self.device, self.batch_size,
                                             self.meta_paths, self.n_stocks)

            model.load_state_dict(best_model_state)
            test_X, test_y, test_adj, test_edge = prepare_data(feats_test, labels_test, 0, feats_test.shape[0],
                                                               self.adj_time, self.edge_time, self.meta_paths, window_size)

            # ~~~ Save model checkpoint ~~~
            model_save_path = os.path.join(results_dir, "model_checkpoints")
            os.makedirs(model_save_path, exist_ok=True)
            checkpoint_path = os.path.join(model_save_path, f"model_{t2.date()}_to_{t3.date()}.pt")
            torch.save({
                'model_state_dict': best_model_state,
                'mu_y': mu_y,
                'sigma_y': sigma_y,
                'best_hyperparam': best_hyperparam
            }, checkpoint_path)
            logging.info(f"Saved model to {checkpoint_path}")

            train_res, _, _ = evaluate_inverse(model, X_combined, y_combined, adj_all, edge_all, mu_y, sigma_y, self.n_stocks, self.meta_paths, self.device, self.batch_size)
            test_res, y_true_te, y_pred_te = evaluate_inverse(model, test_X, test_y, test_adj, test_edge, mu_y, sigma_y, self.n_stocks, self.meta_paths, self.device, self.batch_size)

            logging.info(f"Train MSE: {train_res['mse']:.4f} | Test MSE: {test_res['mse']:.4f}")
            plot_path = os.path.join(results_dir, f"pred_vs_true_{t2.date()}_to_{t3.date()}_{tag}.png")
            plot_pred_vs_true(y_true_te, y_pred_te, self.stock_keys, save_path=plot_path)

            save_test_predictions(y_true_te, y_pred_te, idx_test, self.all_dates, self.stock_keys,
                                  t2.date(), t3.date(), window_size,
                                  predictions_path, write_pred_header)
            write_pred_header = False

            write_metrics(metrics_path, tag, t0, t3, train_res, test_res, write_metrics_header)
            write_metrics_header = False

            t0 = t0 + pd.DateOffset(months=self.test_months)