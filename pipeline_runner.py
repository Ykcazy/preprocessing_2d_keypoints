# FILE: pipeline_runner.py (Updated to manage progress bar)

import tempfile
from pathlib import Path
import os
import datetime

from video_processor import process_videos
from preprocess_data_csv import preprocess_single_csv
from promAI.src.predict_single import predict_from_csv 

# Add st_progress_bar and st_status_text to accept Streamlit UI elements
def run_full_pipeline(cam0_video_path: str, cam1_video_path: str, exercise_name: str, st_progress_bar=None, st_status_text=None):
    base_output_dir = Path("analysis_outputs")
    base_output_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_id = Path(cam0_video_path).stem.replace(" ", "_")
    session_dir = base_output_dir / f"{timestamp}_{exercise_name}_{video_id}"
    session_dir.mkdir(exist_ok=True)
    
    raw_csv_path = session_dir / "raw_keypoints.csv"
    final_csv_path = session_dir / "processed_keypoints.csv"
    output_video_path = session_dir / "visualization.mp4"

    print(f"🎥 Starting processing for exercise: {exercise_name}")
    print(f"   - Saving all outputs to: {session_dir}")

    # --- 1. RUN VIDEO PROCESSING ---
    try:
        if st_status_text: st_status_text.text("Step 1/3: Extracting keypoints and generating video...")
        process_videos(
            video_paths=[cam0_video_path, cam1_video_path],
            exercise_label=exercise_name,
            correctness_label="unlabeled",
            output_csv_path=str(raw_csv_path),
            output_video_path=str(output_video_path),
            # Pass the UI elements down to the processor
            st_progress_bar=st_progress_bar,
            st_status_text=st_status_text
        )
    except Exception as e:
        print(f"❌ An error occurred during video processing: {e}")
        return {"error": f"Video processing failed: {e}"}

    if not raw_csv_path.exists() or raw_csv_path.stat().st_size == 0:
        return {"error": "Processing failed to produce a valid data file (raw_keypoints.csv)."}
        
    # --- 2. RUN DATA PREPROCESSING ---
    try:
        if st_status_text: st_status_text.text("Step 2/3: Preprocessing and normalizing data...")
        if st_progress_bar: st_progress_bar.progress(0.85, text="Step 2/3: Preprocessing data...")
        preprocess_single_csv(
            raw_csv_path=str(raw_csv_path),
            exercise_name=exercise_name,
            output_path=str(final_csv_path)
        )
    except Exception as e:
        print(f"❌ An error occurred during data preprocessing: {e}")
        return {"error": f"Data preprocessing failed: {e}"}
    
    # --- 4. RUN PREDICTION & PREPARE RESULTS ---
    try:
        if st_status_text: st_status_text.text("Step 3/3: Running prediction model...")
        if st_progress_bar: st_progress_bar.progress(0.95, text="Step 3/3: Running prediction...")
        prediction_result = predict_from_csv(
            csv_path=str(final_csv_path), 
            exercise=exercise_name
        )
        if st_progress_bar: st_progress_bar.progress(1.0, text="Analysis Complete!")
        if st_status_text: st_status_text.text("Analysis Complete!")

        final_video_path = str(output_video_path) if output_video_path.exists() and output_video_path.stat().st_size > 100 else None
        final_csv_path_str = str(final_csv_path) if final_csv_path.exists() else None

        print(f"\n✅ Pipeline complete. Result: {prediction_result}")
        return {
            "prediction": prediction_result, 
            "video_path": final_video_path,
            "csv_path": final_csv_path_str,
            "output_folder": str(session_dir)
        }
    except Exception as e:
        print(f"❌ An error occurred during prediction: {e}")
        return {"error": f"Prediction failed: {e}"}

