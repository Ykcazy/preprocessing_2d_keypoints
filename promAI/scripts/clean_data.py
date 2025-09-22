import os
import argparse

def clean_dataset(dataset_version):
    base_dir = os.path.expanduser("~/Downloads/Thesis_Dataset")
    dataset_dir = os.path.join(base_dir, dataset_version)

    # Step 1: Rename exercise folders (replace spaces with underscores)
    folder_renames = {
        "elbow extension": "elbow_extension",
        "shoulder flexion": "shoulder_flexion"
    }

    for old_name, new_name in folder_renames.items():
        old_path = os.path.join(dataset_dir, old_name)
        new_path = os.path.join(dataset_dir, new_name)
        if os.path.exists(old_path):
            print(f"[RENAME] {old_path} -> {new_path}")
            os.rename(old_path, new_path)

    # Step 2: Recursively rename files in subfolders (only remove whitespaces)
    for exercise_folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, exercise_folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for filename in os.listdir(subfolder_path):
                old_file = os.path.join(subfolder_path, filename)
                # Only replace spaces with underscores, keep numbers and other parts unchanged
                new_filename = filename.replace(" ", "_")
                new_file = os.path.join(subfolder_path, new_filename)
                if old_file != new_file:
                    print(f"[RENAME] {old_file} -> {new_file}")
                    os.rename(old_file, new_file)

    print(f"[DONE] Dataset cleanup complete for {dataset_version}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and standardize dataset folder/file names (remove whitespaces only).")
    parser.add_argument("--dataset_version", type=str, required=True, help="Dataset version (e.g., dataset_v1, dataset_v2)")
    args = parser.parse_args()
    clean_dataset(args.dataset_version)
