"""
construct_graph.py

This script constructs and caches dynamic graph data structures for financial GNN modeling.
It loads edge-level time series data and per-node features, processes them into time-indexed 
adjacency and edge attribute tensors, computes meta-path-based connectivity (e.g., SIS), and 
saves the resulting tensors for training temporal GNN models.

Main Components:
- Loads raw edge metadata from CSV and organizes data by timestamp
- Builds dynamic adjacency and edge feature tensors for "SS", "SI", and "SIS" relations
- Derives 2-hop meta-paths (e.g., SIS = SI @ SI^T) via matrix multiplication
- Loads PCA-compressed per-node features and excess return labels
- Caches the processed graph tensors

Output Files (saved to `data/processed/`):
- adj_time.pt        : Dictionary of adjacency matrices over time for each path type
- edge_time.pt       : Dictionary of edge feature tensors over time
- features_time.pt   : Node-level input features (T x N x F)
- labels_time.pt     : Node-level labels (T x N)
- all_dates.csv      : Chronological dates aligned with tensor index
- meta_paths.txt     : List of meta-paths used for modeling
- meta_info.json     : Summary metadata for debugging and downstream use

Note:
- The script assumes raw data exists in `data/raw/`, with one CSV per node and a master edge file.
- Only stocks and industries listed in `select_stocks` and `select_industries` are included.
"""

import os
import torch
import pandas as pd
import numpy as np
import json

# ~~~ Configuration ~~~
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
edge_fn    = os.path.join(BASE_DIR, "raw", "edges.csv")
cache_dir  = os.path.join(BASE_DIR, "processed")
start_date = pd.Timestamp("2019-01-04")

select_stocks     = ["MSFT","IBM","CRM","INTU","NOW","ACN","TXN","ADBE","MU","PANW"]
select_industries = ["IT"]
node_list         = select_stocks + select_industries

n_stocks        = len(select_stocks)
n_industries    = len(select_industries)
n_nodes         = len(node_list)   

industry_idx    = n_stocks 

edge_types = ["SS", "SI"]                
meta_paths = ["SS", "SIS"]
all_paths  = list(set(edge_types).union(meta_paths))

# ~~~ Load Master Edge CSV --> basic metadata ~~~

df = pd.read_csv(edge_fn, index_col='Date')

feature_cols = [c for c in df.columns if c not in ("SRC","DST","RELATION")]
E = len(feature_cols)

dates    = (sorted(df.index.unique()))
date_idx = {date:t for t,date in enumerate(dates)}
T        = len(dates)

node_idx = {n:i for i,n in enumerate(node_list)}

# ~~~ Allocate space for tensors, 0-init ~~~
adj_time  = {p: torch.zeros((T,n_nodes,n_nodes),dtype=torch.float32)
             for p in all_paths}
edge_time = {p: torch.zeros((T,n_nodes,n_nodes,E),dtype=torch.float32)
             for p in all_paths} 

# ~~~ Fill SS, SI from CSV ~~~
for idx, row in df.iterrows():
    rel = row["RELATION"]
    if row["RELATION"] not in edge_types:
        continue

    t = date_idx[idx]
    i = node_idx[row["SRC"]]
    j = node_idx[row["DST"]]

    if row["E1"] != 0 and row["E2"] != 0:
        adj_time[rel][t,i,j] = 1.0 
        edge_time[rel][t,i,j] = torch.tensor([row["E1"], row["E2"]], dtype=torch.float32)

# ~~~ Derive MPs with matmul ~~~
'''
Matrix multiplication of adjacency matrices is the canonical way to implement meta-paths in GNNs.
To encode the SIS metapath, we compute: A_sis = A_si @ A_si.T  - self loops.
Since SIS is a 2-hop metapath with stock endpoints, we can aggregate the edge weights along that 
path with a simple average of the two 1-hop SI edge features. Then, the meta-path is encoded holistically 
via its final subgraph (adjacency + edge attributes), which will serve as an input to a dedicated GAT layer.
'''

for t in range(T):
    A_si = adj_time["SI"][t]
    E_si = edge_time["SI"][t]

    # === SIS ===
    A_sis = torch.matmul(A_si, A_si.T)
    adj_time["SIS"][t] = (A_sis > 0).float()
    for i in range(n_stocks):
        for j in range(n_stocks):
            if i == j or A_sis[i, j] == 0:
                continue
            e_i = E_si[i, industry_idx]
            e_j = E_si[j, industry_idx]
            edge_time["SIS"][t, i, j] = (e_i + e_j) / 2
            edge_time["SIS"][t, j, i] = (e_i + e_j) / 2

# ~~~ Feature construction ~~~
dfs = {node: pd.read_csv(f"data/raw/{node}.csv", parse_dates=["Date"]) for node in node_list}
all_dates = None
feature_list = lambda node: [f"{node}_PC1",
                             f"{node}_PC2",
                             f"{node}_PC3",
                             f"{node}_PC4",
                             f"{node}_PC5",
                             f"{node}_PC6",
                             f"{node}_PC7",
                             f"{node}_PC8",
                             f"{node}_PC9",
                             f"{node}_PC10",
                             f"{node}_PC11",
                             f"{node}_PC12",
                             f"{node}_PC13",
                             f"{node}_PC14",
                             f"{node}_PC15",
                             f"{node}_PC16"]
label_fn = lambda node: f"{node}_EXCESS_RETURN"

features_tensor = []
labels_tensor = []

for node in node_list:
    df = dfs[node]
    df = df[df["Date"] >= start_date].reset_index(drop=True)

    features_tensor.append(df[feature_list(node)].values)
    labels_tensor.append(df[label_fn(node)].values)

    if all_dates is None:
        all_dates = df["Date"].values
    else:
        assert np.array_equal(all_dates, df["Date"].values)

features_tensor = torch.tensor(np.stack(features_tensor, axis=1))
labels_tensor = torch.tensor(np.stack(labels_tensor, axis=1))

# ~~~ Cache Graph ~~~
torch.save(adj_time, os.path.join(cache_dir, "adj_time.pt"))
torch.save(edge_time, os.path.join(cache_dir, "edge_time.pt"))
torch.save(features_tensor, os.path.join(cache_dir, "features_time.pt"))
torch.save(labels_tensor, os.path.join(cache_dir, "labels_time.pt"))
pd.Series(pd.to_datetime(all_dates), name="Date").to_csv(os.path.join(cache_dir, "all_dates.csv"), index=False)

with open(os.path.join(cache_dir, "meta_paths.txt"), "w") as f:
    for mp in meta_paths:
        f.write(mp + "\n")

meta_info = {
    "T": T,
    "N": n_nodes,
    "E": E,
    "F": features_tensor.shape[2],
    "n_stocks": n_stocks,
    "n_industries": n_industries,
    "paths": list(adj_time.keys()),
    "feature_nodes": node_list,
    "start_date": str(start_date.date()),
}

with open(os.path.join(cache_dir, "meta_info.json"), "w") as f:
    json.dump(meta_info, f, indent=4)

print(f"[✓] Dynamic graphs cached to:  {cache_dir}")
print(f"    dates   : {T}")
print(f"    nodes   : {n_nodes} : (stocks {n_stocks} + industries {n_industries}")
print(f"    features: {E} per edge")