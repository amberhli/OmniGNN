import os
import torch
import pandas as pd
import numpy as np
import random

from training.backtester import Backtester
from training.logger import setup_logging

setup_logging()

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)

# ~~~ Configuration ~~~
select_stocks = ["MSFT", "IBM", "CRM", "INTU", "NOW", "ACN", "TXN", "ADBE", "MU", "PANW"]
window_size = 12
start_date = pd.Timestamp("2019-01-04")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ~~~ Load graph cache (adj_dict, edge_attr_dict, meta_paths) ~~~
cache_dir = "data/processed"
adj_time = torch.load(os.path.join(cache_dir, "adj_time.pt"))
edge_time = torch.load(os.path.join(cache_dir, "edge_time.pt"))
features_time = torch.load(os.path.join(cache_dir, "features_time.pt"))
labels_time = torch.load(os.path.join(cache_dir, "labels_time.pt"))
all_dates = pd.read_csv(os.path.join(cache_dir, "all_dates.csv"), parse_dates=["Date"]).squeeze().values
with open(os.path.join(cache_dir, "meta_paths.txt"), "r") as f:
    meta_paths = [line.strip() for line in f.readlines()]

# ~~~ Run Backtest ~~~
backtester = Backtester(
    features_time=features_time,
    labels_time=labels_time,
    adj_time=adj_time,               
    edge_time=edge_time,            
    meta_paths=meta_paths,
    all_dates=all_dates,
    window_size=window_size, # HP
    device=device,
    stock_names=select_stocks,
    hidden_dim=32, # HP
    train_months=6,
    val_months=2,
    test_months=2,
)

backtester.run()