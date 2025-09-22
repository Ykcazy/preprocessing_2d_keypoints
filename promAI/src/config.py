# src/config.py
"""
Global configuration dictionary for the promAI project.
Adjust values here as needed.
"""

import torch
import os

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "data": {
        "root_dir": os.path.expanduser("~/Downloads/Thesis_Dataset"),
        "dataset_version": "dataset_v1",   # change when you want to switch dataset_v2
        "processed_root": "promAI/data/processed",
        "seq_len": 200,
        "csv_dir": "promAI/data/processed/dataset_v1",  # keep for backward compatibility if needed
        "splits_dir": "promAI/data/processed/dataset_v1"
    },
    "model": {
        "input_dim": None,       # will be set dynamically after loading data
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "training": {
        "batch_size": 16,
        "lr": 1e-3,          # change from "learning_rate" to "lr"
        "epochs": 20,        # change from "num_epochs" to "epochs"
        "save_every": 5,
    },
    "output": {
        "model_dir": "promAI/checkpoints",
        "results_dir": "promAI/results",
    }
}

def set_dataset_version(version):
    """
    Helper to update dataset version and related paths in CONFIG.
    Usage: set_dataset_version("dataset_v2")
    """
    CONFIG["data"]["dataset_version"] = version
    CONFIG["data"]["csv_dir"] = f"data/processed/{version}"
    CONFIG["data"]["splits_dir"] = f"data/processed/{version}"
