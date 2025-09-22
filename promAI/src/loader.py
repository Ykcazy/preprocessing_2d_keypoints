import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .config import CONFIG

# 🔑 Label mapping: binary classification
# correct -> 0, incorrect & subtle_incorrect -> 1
LABEL_MAP = {
    "correct": 0,
    "incorrect": 1,
    "subtle_incorrect": 1
}

# This function is not used by create_dataloader but is kept for completeness.
def load_split(split="train", splits_dir=None, label_map=LABEL_MAP):
    if splits_dir is None:
        splits_dir = CONFIG["data"]["splits_dir"]

    split_file = os.path.join(splits_dir, f"{split}_manifest.csv")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    df = pd.read_csv(split_file)

    X, y = [], []
    for _, row in df.iterrows():
        filename = row["filename"]
        label_str = row["label"]
        # Assuming manifest contains string labels that need mapping
        if isinstance(label_str, str) and label_str not in label_map:
             raise ValueError(f"Unknown label '{label_str}' in manifest.")
        
        label = label_map[label_str] if isinstance(label_str, str) else label_str

        filepath = os.path.join(os.path.dirname(split_file), split, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File listed in manifest not found: {filepath}")

        sequence = pd.read_csv(filepath).values
        X.append(sequence)
        y.append(label)

    return np.array(X, dtype=object), np.array(y, dtype=int)


class MotionDataset(Dataset):
    """
    Loads motion CSV files into tensors.
    Now also handles passing along the original filename.
    """
    # <<< MODIFIED: Now accepts filenames to pass along
    def __init__(self, file_paths, labels, filenames):
        self.file_paths = file_paths
        self.labels = labels
        self.filenames = filenames # <<< NEW

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        df = pd.read_csv(file_path)

        # Convert to tensor
        data = torch.tensor(df.values, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        filename = self.filenames[idx] # <<< NEW

        # <<< MODIFIED: Returns filename along with data and label
        return data, label, filename


def create_dataloader(split="train", splits_dir=None,
                      batch_size=32, shuffle=True, label_map=LABEL_MAP):
    """
    Create a PyTorch DataLoader from a split manifest.
    Now yields batches of (features, labels, filenames).
    """
    if splits_dir is None:
        splits_dir = CONFIG["data"]["splits_dir"]

    split_file = os.path.join(splits_dir, f"{split}_manifest.csv")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    df = pd.read_csv(split_file)
    split_subdir = os.path.join(os.path.dirname(split_file), split)
    
    file_paths = [os.path.join(split_subdir, fname) for fname in df["filename"].tolist()]
    
    # Handle both string and integer labels from the manifest
    raw_labels = df["label"].tolist()
    labels = []
    for l in raw_labels:
        if isinstance(l, str):
            if l not in label_map:
                raise ValueError(f"Unknown label string '{l}' in manifest {split_file}")
            labels.append(label_map[l])
        else:
            labels.append(int(l))

    filenames = df["filename"].tolist() # <<< NEW: Get a list of filenames

    if not file_paths:
        raise ValueError(f"No files listed in {split_file}")

    # <<< MODIFIED: Pass filenames to the dataset
    dataset = MotionDataset(file_paths, labels, filenames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Peek at one file to determine dimensions
    sample_df = pd.read_csv(file_paths[0])
    seq_len, input_dim = sample_df.shape

    return dataloader, input_dim, seq_len