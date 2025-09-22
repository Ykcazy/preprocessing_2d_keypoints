import os
import shutil
from pathlib import Path

def sort_exercise_folder(exercise_dir: Path):
    """
    Sorts all CSV files in an exercise folder into subfolders:
    - correct/
    - incorrect/
    - subtle_incorrect/
    """
    # Create subfolders if not exist
    subfolders = ["correct", "incorrect", "subtle_incorrect"]
    for sub in subfolders:
        (exercise_dir / sub).mkdir(exist_ok=True)

    # Move files based on filename patterns
    for f in exercise_dir.glob("*.csv"):
        fname = f.name.lower()
        if "subtle incorrect" in fname:
            target = exercise_dir / "subtle_incorrect" / f.name
        elif "incorrect" in fname:
            target = exercise_dir / "incorrect" / f.name
        elif "correct" in fname:
            target = exercise_dir / "correct" / f.name
        else:
            print(f"⚠️ Skipping {f.name} (no label in filename)")
            continue

        if not target.exists():
            shutil.move(str(f), str(target))
            print(f"Moved {f.name} → {target.parent.name}/")
        else:
            print(f"⚠️ Skipping {f.name} (already exists in {target.parent.name}/)")

def sort_dataset(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    for exercise_folder in dataset_path.iterdir():
        if exercise_folder.is_dir():
            print(f"\n📂 Sorting {exercise_folder.name}...")
            sort_exercise_folder(exercise_folder)

if __name__ == "__main__":
    # Change this path to your dataset root
    DATASET_ROOT = r"promAI/Thesis_Dataset/dataset_v1"
    sort_dataset(DATASET_ROOT)
    print("\n✅ Sorting complete.")
