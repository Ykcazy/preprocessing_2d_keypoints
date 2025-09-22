import pandas as pd
from pathlib import Path

# Paths to manifest CSVs for both exercises
base = Path("data/processed/dataset_v1")
exercises = ["elbow_extension", "shoulder_flexion"]
splits = ["train", "val", "test"]

# Load datasets for each exercise and split
datasets = {}
split_files = {exercise: {} for exercise in exercises}
for exercise in exercises:
    for split in splits:
        manifest_path = base / exercise / f"{split}_manifest.csv"
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            datasets[f"{exercise.capitalize()} {split.capitalize()}"] = df
            split_files[exercise][split] = set(df['filename'].tolist())
        else:
            print(f"Warning: {manifest_path} not found.")

# Check for duplicate files across splits for each exercise
for exercise in exercises:
    all_files = []
    for split in splits:
        all_files.extend(split_files[exercise].get(split, []))
    duplicates = set([f for f in all_files if all_files.count(f) > 1])
    if duplicates:
        print(f"\n❌ Duplicate files found across splits for {exercise}:")
        for dup in duplicates:
            print(f"  - {dup}")
    else:
        print(f"\n✅ No duplicate files across splits for {exercise}.")

TARGET_LABELS = [0, 1]  # 0: correct, 1: incorrect/subtle_incorrect

def dataset_summary(df, name):
    print(f"\n--- {name} Dataset Report ---")
    
    if 'label' not in df.columns:
        print("Error: 'label' column not found!")
        return
    
    # Count per label
    counts = df['label'].value_counts().reindex(TARGET_LABELS, fill_value=0)
    percentages = (counts / counts.sum() * 100).round(2)
    
    label_names = {0: "correct", 1: "incorrect/subtle_incorrect"}
    report_df = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })
    report_df.index = [label_names.get(idx, idx) for idx in report_df.index]
    
    print(report_df)
    
    # Training set checks
    if "Train" in name:
        min_count = counts.min()
        max_count = counts.max()
        if max_count - min_count > 0:
            print("\n⚠️ Training set is imbalanced.")
        else:
            print("\n✅ Training set is balanced between classes.")
    # Validation/Test check
    if "Val" in name or "Test" in name:
        print("\nInfo: Validation/Test sets maintain stratification, not necessarily balanced.")

# Generate report
for name, df in datasets.items():
    dataset_summary(df, name)
