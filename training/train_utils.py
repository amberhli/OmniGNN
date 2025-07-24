import torch
import torch.nn as nn
from torch.optim import Adam
from omnignn.model import OmniGNN

def build_model(feature_dim, edge_dim, meta_paths, window_size, n_nodes, device, hidden_dim):
    return OmniGNN(
        in_features=feature_dim,
        hidden_dim=hidden_dim,
        edge_attr_dim=edge_dim,
        meta_paths=meta_paths,
        window_size=window_size,
        n_nodes=n_nodes
    ).to(device)

def train_one_epoch(model, optimizer, criterion, X, y, adj, edge, device, batch_size, meta_paths, n_stocks):
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