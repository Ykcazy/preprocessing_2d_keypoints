import argparse
import os
import pandas as pd
from pathlib import Path
import numpy as np

def check_split(split_name, split_dir):
    split_dir = Path(split_dir)
    all_files = list(split_dir.glob("*.csv"))
    total_files = len(all_files)

    if total_files == 0:
        print(f"⚠️ No files found in {split_dir}")
        return

    shapes = set()
    total_samples = 0
    nan_files = []
    feature_names = None
    all_data = []

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Failed to read {file_path}: {e}")
            continue

        shapes.add(df.shape)
        total_samples += len(df)

        if df.isna().any().any():
            nan_files.append(str(file_path.resolve()))

        if feature_names is None:
            feature_names = df.columns
        all_data.append(df.values)

    # Stack for variance check
    all_data = np.vstack(all_data)
    variances = np.nanvar(all_data, axis=0)
    constant_features = [
        feature_names[i] for i, var in enumerate(variances) if var == 0
    ]

    print(f"--- Checking {split_name} split ---")
    print(f"✅ {split_name} check complete. Shape: {list(shapes)[0] if len(shapes)==1 else shapes}, Total samples: {total_samples}\n")
    print("📊 Summary:")
    print(f"Total files: {total_files}")
    print(f"Files with NaNs: {len(nan_files)}")
    for f in nan_files:
        print(f"  - {f}")
    print(f"Constant features across all files: {len(constant_features)}")
    if constant_features:
        print("  - " + ", ".join(constant_features))
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_version", type=str, required=True, help="Dataset version (e.g., dataset_v1, dataset_v2)")
    parser.add_argument("--exercise", type=str, choices=["elbow_extension", "shoulder_flexion"], help="Check only one exercise.")
    args = parser.parse_args()

    base = Path("data/processed") / args.dataset_version
    splits = ["train", "val", "test"]

    if args.exercise:
        exercises = [args.exercise]
    else:
        exercises = [d.name for d in base.iterdir() if d.is_dir()]

    for exercise in exercises:
        print(f"\n=== Checking splits for {exercise} ===")
        for split in splits:
            split_dir = base / exercise / split
            check_split(split, split_dir)

if __name__ == "__main__":
    main()