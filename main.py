# FILE: automated_extractor.py

import os
from pathlib import Path
from data_extractor import extract_data_for_gru

def run_automated_extraction():
    """
    Finds video pairs in a structured directory, extracts labels from the
    folder names, and saves the results to a MIRRORED output directory.
    """
    # --- 1. SETUP FOLDER PATHS ---
    base_dataset_folder = Path("dataset") # This points to your input videos
    output_folder = Path("extracted_raw_data") # This is the main output folder
    output_folder.mkdir(exist_ok=True)

    print(f"Searching for exercise data in: {base_dataset_folder}")
    print(f"Output will be saved to: {output_folder}")

    # --- 2. FIND VIDEO PAIRS AND EXTRACT LABELS ---
    video_tasks = []
    for cam0_folder in base_dataset_folder.rglob('cam0'):
        correctness_label = cam0_folder.parent.name
        exercise_label = cam0_folder.parent.parent.name
        
        cam1_folder = cam0_folder.parent / 'cam1'
        if not cam1_folder.exists(): continue

        for cam0_file in sorted(cam0_folder.glob("*.mp4")):
            identifier = cam0_file.stem.split('_')[-1]
            cam1_files = list(cam1_folder.glob(f"*_{identifier}.mp4"))
            
            if cam1_files:
                video_tasks.append({
                    "pair_id": identifier,
                    "exercise": exercise_label,
                    "correctness": correctness_label,
                    "paths": [str(cam0_file), str(cam1_files[0])]
                })

    if not video_tasks:
        print("Error: No video pairs found."); return

    print(f"\nFound {len(video_tasks)} video pairs to process.")

    # --- 3. PROCESS EACH PAIR ---
    for i, task in enumerate(video_tasks):
        print(f"\n--- Processing Task {i+1}/{len(video_tasks)} (ID: {task['pair_id']}) ---")
        print(f"   Exercise: {task['exercise']}, Correctness: {task['correctness']}")
        
        # THE FIX: Create a structured output path that mirrors the input path
        task_output_folder = output_folder / task['exercise'] / task['correctness']
        task_output_folder.mkdir(parents=True, exist_ok=True)
        output_csv_path = task_output_folder / f"raw_data_id_{task['pair_id']}.csv"
        
        if output_csv_path.exists():
            print(f"Output file {output_csv_path.name} already exists. Skipping."); continue
        try:
            extract_data_for_gru(
                video_paths=task['paths'], output_csv_path=str(output_csv_path),
                exercise_label=task['exercise'], correctness_label=task['correctness']
            )
        except Exception as e:
            print(f"!!!!!! An error occurred while processing task {task['pair_id']}: {e}")

    print("\n\n✅ Automated data extraction finished.")

if __name__ == '__main__':
    run_automated_extraction()

