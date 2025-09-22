import argparse
from pathlib import Path
import pandas as pd

# Import the necessary functions from your existing scripts
from data_extractor import extract_data_for_gru
from preprocess_data_csv import preprocess_single_csv

def process_video_pair(cam0_path_str: str, cam1_path_str: str, exercise_name: str, correctness_label: str, numeric_label: int):
    """
    Contains the core processing logic for a single video pair.
    """
    cam0_path = Path(cam0_path_str)
    pair_id = cam0_path.stem.replace('_cam0_', '_')

    # Setup output paths
    base_processed_folder = Path("./processed_files_from_manifest")
    exercise_specific_folder = base_processed_folder / exercise_name
    exercise_specific_folder.mkdir(parents=True, exist_ok=True)
    final_csv_path = exercise_specific_folder / f"processed_{pair_id}.csv"

    # --- FEATURE 1: Check if the output file already exists ---
    if final_csv_path.exists():
        print(f"    ⏭️  Skipping pair: {pair_id} (Output already exists)")
        # Still return the info so the manifest is complete
        return {
            "video_id": pair_id,
            "exercise": exercise_name,
            "correctness_label": correctness_label,
            "numeric_label": numeric_label,
            "processed_file_path": str(final_csv_path)
        }

    temp_folder = Path("./temp_data")
    temp_folder.mkdir(exist_ok=True)
    raw_csv_path = temp_folder / f"raw_{pair_id}.csv"
    
    print(f"    ▶️  Processing pair: {pair_id} (Label: {correctness_label})")
    
    # Run data extraction and preprocessing
    try:
        extract_data_for_gru(
            video_paths=[cam0_path_str, cam1_path_str], 
            output_csv_path=str(raw_csv_path),
            exercise_label=exercise_name, 
            correctness_label=correctness_label
        )
        preprocess_single_csv(
            raw_csv_path=str(raw_csv_path), 
            exercise_name=exercise_name,
            output_path=str(final_csv_path)
        )
    except Exception as e:
        print(f"      ❌ ERROR during processing for {pair_id}: {e}")
        return None

    # Return data for the new manifest
    return {
        "video_id": pair_id,
        "exercise": exercise_name,
        "correctness_label": correctness_label,
        "numeric_label": numeric_label,
        "processed_file_path": str(final_csv_path)
    }

def main():
    parser = argparse.ArgumentParser(description="Process video pairs from a manifest, skipping existing ones.")
    parser.add_argument("--manifest", required=True, type=str, help="Path to the manifest.csv file.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"❌ Error: Manifest file not found at '{manifest_path}'")
        return

    print(f"🚀 Starting processing based on manifest: {manifest_path.name}")
    manifest_df = pd.read_csv(manifest_path)

    manifest_df['pair_id'] = manifest_df['video_id'].str.replace('_cam0_', '_').str.replace('_cam1_', '_')
    grouped = manifest_df.groupby('pair_id')
    print(f"Found {len(grouped)} pairs to process in the manifest.")

    new_manifest_data = []

    for pair_id, group in grouped:
        if len(group) != 2:
            print(f"  ⚠️  Warning: Skipping pair '{pair_id}' (expected 2 files, found {len(group)}).")
            continue
        
        cam0_row = group[group['video_id'].str.contains('cam0')]
        cam1_row = group[group['video_id'].str.contains('cam1')]

        if cam0_row.empty or cam1_row.empty:
            print(f"  ⚠️  Warning: Skipping pair '{pair_id}' (missing cam0 or cam1 entry).")
            continue
            
        cam0_info = cam0_row.iloc[0]
        result = process_video_pair(
            cam0_path_str=cam0_info['file_path'],
            cam1_path_str=cam1_row.iloc[0]['file_path'],
            exercise_name=cam0_info['exercise'],
            correctness_label=cam0_info['correctness_label'],
            numeric_label=int(cam0_info['numeric_label'])
        )
        if result:
            new_manifest_data.append(result)

    if not new_manifest_data:
        print("\n🤷 No data to report. No new manifests created.")
    else:
        # --- FEATURE 2: Create a manifest for each exercise ---
        print("\n✍️  Creating new manifest(s) for the processed files...")
        final_manifest_df = pd.DataFrame(new_manifest_data)
        
        for exercise_name, exercise_df in final_manifest_df.groupby('exercise'):
            manifest_path = f"{exercise_name}_processed_manifest.csv"
            exercise_df.to_csv(manifest_path, index=False)
            print(f"✅ Final manifest for '{exercise_name}' with {len(exercise_df)} entries created at: {manifest_path}")

    print("\n--- Pipeline finished. ---")


if __name__ == '__main__':
    main()