import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

def collect_files(dataset_dir, exercise):
    """Recursively collect all CSV files for the given exercise."""
    files = list((dataset_dir / exercise).rglob("*.csv"))
    data = []
    for f in files:
        # Infer label from parent folder name
        if f.parent.name == "correct":
            label = 0
        elif f.parent.name in ["incorrect", "subtle_incorrect"]:
            label = 1
        else:
            continue  # skip files not in expected folders
        
        data.append({
            "filename": f.name,
            "label": label,
            "exercise": exercise,
            "src_path": str(f)
        })
    return pd.DataFrame(data)

def prepare_split(df, seed=42):
    """Splits the dataframe into train, validation, and test sets."""
    # 10% for test, 20% for val, 70% for train
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df["label"], random_state=seed)

    # Balance the training set WITHOUT duplication
    correct = train_df[train_df['label'] == 0]
    subtle_incorrect = train_df[train_df['src_path'].str.contains("subtle_incorrect")]
    incorrect = train_df[(train_df['label'] == 1) & (~train_df['src_path'].str.contains("subtle_incorrect"))]

    n_correct = len(correct)
    n_subtle = len(subtle_incorrect)
    n_needed = n_correct - n_subtle

    # Only use unique files
    if n_needed > 0:
        incorrect = incorrect.drop(subtle_incorrect.index, errors='ignore')
        if len(incorrect) >= n_needed:
            incorrect_sample = incorrect.sample(n=n_needed, random_state=seed)
            balanced_incorrect = pd.concat([subtle_incorrect, incorrect_sample])
        else:
            balanced_incorrect = pd.concat([subtle_incorrect, incorrect])
    else:
        balanced_incorrect = subtle_incorrect.sample(n=n_correct, random_state=seed)

    # Remove any duplicates and shuffle
    balanced_incorrect = balanced_incorrect.drop_duplicates(subset=["filename"])
    balanced_train = pd.concat([correct, balanced_incorrect]).drop_duplicates(subset=["filename"]).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return balanced_train, val_df, test_df

def copy_splits_to_processed(train_df, val_df, test_df, dataset_version, exercise, seed):
    """Copies files to processed directory and creates manifests."""
    base_out = Path("data/processed") / dataset_version / exercise

    # Ensure split directories exist
    for split_name in ["train", "val", "test"]:
        split_dir = base_out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        # Clear split directory before copying
        for f in split_dir.glob("*.csv"):
            f.unlink()

    for split_name, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        split_dir = base_out / split_name
        manifest_path = base_out / f"{split_name}_manifest.csv"
        # Save all columns to the manifest for completeness
        split_df.to_csv(manifest_path, index=False)
        
        # Copy files using the full src_path
        for _, row in split_df.iterrows():
            src = Path(row["src_path"])
            if not src.exists():
                print(f"❌ Source file not found: {src}")
                continue
            dst = split_dir / row["filename"]
            shutil.copy(src, dst)
            
    # Save split config
    split_info = {
        "timestamp": datetime.now().isoformat(),
        "dataset_version": dataset_version,
        "exercise": exercise,
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "class_counts": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict()
        },
        "seed": seed
    }
    config_path = base_out / "split_config.json"
    with open(config_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"✅ Split config saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset, generate manifests, and copy files.")
    parser.add_argument("--dataset_version", type=str, required=True, help="Dataset version (e.g., dataset_v1)")
    parser.add_argument("--exercise", type=str, required=True, help="Exercise name (e.g., elbow_extension)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for splitting. If not provided, a random seed will be generated.")
    args = parser.parse_args()

    # Dynamic Seed Generation
    seed = args.seed
    if seed is None:
        # Generate a seed from the current timestamp if none is provided
        seed = int(datetime.now().timestamp())
        print(f"🌱 No seed provided. Using a random seed: {seed}")
    else:
        print(f"🌱 Using provided seed: {seed}")

    raw_root = os.path.expanduser("~/Downloads/Thesis_Dataset")
    dataset_dir = Path(raw_root) / args.dataset_version
    print(f"📂 Using raw dataset directory: {dataset_dir}")

    # Collect files
    df = collect_files(dataset_dir, args.exercise)
    print(f"🔍 Collected {len(df)} files for '{args.exercise}' in '{args.dataset_version}'")

    if df.empty:
        print(f"⚠️ No files found for '{args.exercise}' in '{args.dataset_version}'. Exiting.")
        return

    # Split using the determined seed
    train_df, val_df, test_df = prepare_split(df, seed=seed)
    print("\n📊 Split Summary:")
    print(f"  - Train: {len(train_df)} (Correct: {len(train_df[train_df.label==0])}, Incorrect: {len(train_df[train_df.label==1])})")
    print(f"  - Val:   {len(val_df)} (Correct: {len(val_df[val_df.label==0])}, Incorrect: {len(val_df[val_df.label==1])})")
    print(f"  - Test:  {len(test_df)} (Correct: {len(test_df[test_df.label==0])}, Incorrect: {len(test_df[test_df.label==1])})")
    print(f"  - Total: {len(train_df) + len(val_df) + len(test_df)}")

    # Copy and manifest, passing the seed to be saved in the config
    copy_splits_to_processed(train_df, val_df, test_df, args.dataset_version, args.exercise, seed=seed)
    print("\n✨ Splitting and copying complete.")

if __name__ == "__main__":
    main()

