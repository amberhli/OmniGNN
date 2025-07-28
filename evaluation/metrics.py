import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr

def get_topk_indices(predictions, top_k):
    """
    Generate a boolean mask indicating the top-K% scoring stocks per day.

    Args:
        predictions (np.ndarray): Array of shape (T, N) with model-predicted values.
        top_k (float): Proportion (e.g., 0.3 for top 30%) of stocks to select.

    Returns:
        np.ndarray: Boolean mask of shape (T, N) with True for top-K entries per row.
    """
    n_days, n_stocks = predictions.shape
    k = int(n_stocks * top_k)
    topk_mask = np.zeros_like(predictions, dtype=bool)
    partition_indices = np.argpartition(predictions, -k, axis=1)[:, -k:]
    for i in range(n_days):
        topk_mask[i, partition_indices[i]] = True
    return topk_mask

def compute_metrics(y_true, y_pred, top_k=0.3):
    """
    Compute evaluation metrics for predicted vs. true excess returns.

    Args:
        y_true (np.ndarray): Ground truth returns (T, N).
        y_pred (np.ndarray): Predicted returns (T, N).
        top_k (float): Top-K proportion for precision, IR, and CR metrics.

    Returns:
        dict: Dictionary containing MSE, sign accuracy, Pearson/Spearman correlations,
              information ratio (IR), cumulative return (CR), and precision@K.
    """
    topk_mask = get_topk_indices(y_pred, top_k)

    topk_returns = (y_true * topk_mask).sum(axis=1) / topk_mask.sum(axis=1)

    ir = topk_returns.mean() / (topk_returns.std() + 1e-8)

    cr = topk_returns.sum()

    precision_per_day = ((y_true > 0) & topk_mask).sum(axis=1) / topk_mask.sum(axis=1)
    prec_at_k = precision_per_day.mean()

    sign_match = (np.sign(y_true) == np.sign(y_pred))
    sign_accuracy = sign_match[y_true != 0].mean()

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    return {
        "mse": mean_squared_error(y_true_flat, y_pred_flat),
        "sign_accuracy": sign_accuracy,
        "pearson": pearsonr(y_true_flat, y_pred_flat)[0],
        "ic": spearmanr(y_pred_flat, y_true_flat).correlation,
        "ir": ir,
        "cr": cr,
        "prec@k": prec_at_k
    }

def evaluate_inverse(model, X, y_scaled, adj, edge, mu, sigma, n_stocks, meta_paths, device, batch_size):
    """
    Evaluate a model's predictions on scaled data, inverse-transforming the outputs.

    Args:
        model (nn.Module): Trained PyTorch model.
        X (torch.Tensor): Input node features (T, N, F).
        y_scaled (torch.Tensor): Scaled ground-truth labels.
        adj (dict): Adjacency matrices by metapath.
        edge (dict): Edge features by metapath.
        mu (np.ndarray): Mean of training labels (for inverse transform).
        sigma (np.ndarray): Std of training labels (for inverse transform).
        n_stocks (int): Number of stocks (subset of all nodes).
        meta_paths (list): List of metapath names used.
        device (torch.device): Computation device.

    Returns:
        tuple: (metrics dict, true values (np.ndarray), predicted values (np.ndarray))
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            x_b = X[i : i + batch_size].to(device)
            y_b = y_scaled[i : i + batch_size].to(device)
            adj_b = {mp: adj[mp][i : i + batch_size].to(device) for mp in meta_paths}
            edge_b = {mp: edge[mp][i : i + batch_size].to(device) for mp in meta_paths}

            out = model(x_b, adj_b, edge_b)[:, :n_stocks].cpu()
            y_b = y_b[:, :n_stocks].cpu()

            preds.append(out * sigma[:n_stocks] + mu[:n_stocks])
            trues.append(y_b * sigma[:n_stocks] + mu[:n_stocks])

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()

    metrics = compute_metrics(y_true, y_pred, top_k=0.3)
    return metrics, y_true, y_pred