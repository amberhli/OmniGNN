import torch
import torch.nn as nn
from torch.optim import Adam
from omnignn.model import OmniGNN

def build_model(feature_dim, edge_dim, meta_paths, window_size, n_nodes, device, hidden_dim):
    """
    Construct and return an OmniGNN model instance.

    Args:
        feature_dim (int): Dimension of input node features.
        edge_dim (int): Dimension of edge attributes.
        meta_paths (List[str]): List of metapaths used for multi-relational graph input.
        window_size (int): Number of past time steps used as input to Transformer.
        n_nodes (int): Number of nodes in the graph.
        device (torch.device): Target device (CPU or CUDA).
        hidden_dim (int): Hidden dimension size in the GNN layers.

    Returns:
        OmniGNN: Initialized model instance on the specified device.
    """
    return OmniGNN(
        in_features=feature_dim,
        hidden_dim=hidden_dim,
        edge_attr_dim=edge_dim,
        meta_paths=meta_paths,
        window_size=window_size,
        n_nodes=n_nodes
    ).to(device)

def train_one_epoch(model, optimizer, criterion, X, y, adj, edge, device, batch_size, meta_paths, n_stocks):
    """
    Trains the model for one epoch over the input batch data.

    Args:
        model (nn.Module): The GNN model to train.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        criterion (nn.Module): Loss function to minimize (e.g., MSELoss).
        X (Tensor): Input node features, shape (N_samples, window_size, N_nodes, F).
        y (Tensor): Ground truth labels, shape (N_samples, N_nodes).
        adj (Dict[str, Tensor]): Dictionary of adjacency tensors per metapath.
        edge (Dict[str, Tensor]): Dictionary of edge feature tensors per metapath.
        device (torch.device): Computation device.
        batch_size (int): Mini-batch size for training.
        meta_paths (List[str]): List of metapath keys for adj/edge dictionaries.
        n_stocks (int): Number of stock nodes to compute loss on.

    Returns:
        float: Average training loss over all batches.
    """
    model.train()
    total_loss = 0
    perm = torch.randperm(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        idx = perm[i:i + batch_size]
        x_batch = X[idx].to(device)
        y_batch = y[idx].to(device)
        adj_batch  = {mp: adj[mp][idx].to(device) for mp in meta_paths}
        edge_batch = {mp: edge[mp][idx].to(device) for mp in meta_paths}

        optimizer.zero_grad()
        outputs = model(x_batch, adj_batch, edge_batch)
        loss = criterion(outputs[:, :n_stocks], y_batch[:, :n_stocks])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    total_batches = max(1, (X.shape[0] + batch_size - 1) // batch_size)
    return total_loss / total_batches