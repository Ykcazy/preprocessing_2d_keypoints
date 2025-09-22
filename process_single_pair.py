# FILE: process_single_pair.py

import argparse
from pathlib import Path
import os

# Import the necessary functions from your existing scripts
from data_extractor import extract_data_for_gru
from preprocess_data_csv import preprocess_single_csv # We will create this function next
from promAI.src.predict_single import predict_from_csv

def main():
    """
    A script to extract and preprocess data for a single pair of videos.
    """
    parser = argparse.ArgumentParser(description="Process a single pair of exercise videos.")
    parser.add_argument("--cam0_video", required=True, type=str, help="Path to the video from camera 0.")
    parser.add_argument("--cam1_video", required=True, type=str, help="Path to the video from camera 1.")
    parser.add_argument("--exercise", required=True, type=str, help="Name of the exercise (e.g., 'shoulder flexion').")
    args = parser.parse_args()

    # --- 1. SETUP PATHS ---
    video_paths = [args.cam0_video, args.cam1_video]
    exercise_name = args.exercise
    
    # Create output directories
    temp_folder = Path("./temp_data")
    temp_folder.mkdir(exist_ok=True)
    processed_folder = Path("./processed_single_files")
    processed_folder.mkdir(exist_ok=True)

    # Define intermediate and final file paths
    video_id = Path(args.cam0_video).stem.replace('cam0_', '')
    raw_csv_path = temp_folder / f"raw_data_{exercise_name}_{video_id}.csv"
    final_csv_path = processed_folder / f"processed_{exercise_name}_{video_id}.csv"

    print(f"🎥 Starting processing for exercise: {exercise_name}")
    print(f"   - Input videos: {video_paths}")
    print(f"   - Final output will be: {final_csv_path}")

    # --- 2. RUN DATA EXTRACTION ---
    try:
        print("\n[STEP 1/2] Extracting raw 2D keypoint data...")
        extract_data_for_gru(
            video_paths=video_paths,
            output_csv_path=str(raw_csv_path),
            exercise_label=exercise_name,
            correctness_label="unlabeled" # Using a placeholder for correctness
        )
    except Exception as e:
        print(f"❌ An error occurred during data extraction: {e}")
        return

    if not raw_csv_path.exists():
        print("❌ Data extraction failed to produce an output file. Aborting.")
        return
        
    # --- 3. RUN DATA PREPROCESSING ---
    try:
        print("\n[STEP 2/2] Preprocessing and normalizing data...")
        preprocess_single_csv(
            raw_csv_path=str(raw_csv_path),
            exercise_name=exercise_name,
            output_path=str(final_csv_path)
        )
    except Exception as e:
        print(f"❌ An error occurred during data preprocessing: {e}")
        return
    
    # --- 4. CLEANUP AND FINISH ---
    # os.remove(raw_csv_path) # Optional: uncomment to delete the raw intermediate file
    # print(f"\n✅ Successfully processed video pair. Cleaned up intermediate file.")
    print(f"\n✅ Successfully processed video pair. Final data saved to {final_csv_path}")
    # --- 5. RUN PREDICTION ---
    print("\n[STEP 3/3] Running prediction on processed data...")
    predict_from_csv(csv_path=str(final_csv_path), exercise=exercise_name)


if __name__ == '__main__':
    main()