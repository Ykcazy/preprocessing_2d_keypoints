# FILE: App.py (Updated with Progress Bar UI)

import streamlit as st
import tempfile
from pathlib import Path
import os
import warnings

# Suppress the specific NumPy version warning from SciPy
warnings.filterwarnings(action='ignore', message='A NumPy version')

from pipeline_runner import run_full_pipeline

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Exercise Analysis", layout="wide")


def main():
    """Main function to run the Streamlit App."""
    st.title("Physical Therapy Exercise Evaluation")

    exercise_options = {
        "Elbow Extension": "elbow_extension",
        "Shoulder Flexion": "shoulder_flexion"
    }
    selected_exercise_display = st.selectbox(
        "Select Exercise:",
        options=list(exercise_options.keys())
    )
    exercise_name = exercise_options[selected_exercise_display]

    st.subheader("Upload Video Pair")
    col1, col2 = st.columns(2)
    with col1:
        cam0_video = st.file_uploader("Upload Camera 0 Video (Therapist View)", type=["mp4", "mov", "avi"])
    with col2:
        cam1_video = st.file_uploader("Upload Camera 1 Video (Patient View)", type=["mp4", "mov", "avi"])

    if st.button("Analyze Exercise"):
        if cam0_video and cam1_video:
            # --- Create UI elements for progress reporting ---
            status_text = st.empty()
            progress_bar = st.progress(0, text="Starting analysis...")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                cam0_path = temp_path / "cam0_upload.mp4"
                cam1_path = temp_path / "cam1_upload.mp4"

                cam0_path.write_bytes(cam0_video.getbuffer())
                cam1_path.write_bytes(cam1_video.getbuffer())

                # --- Call the pipeline, passing the UI elements ---
                result = run_full_pipeline(
                    cam0_video_path=str(cam0_path),
                    cam1_video_path=str(cam1_path),
                    exercise_name=exercise_name,
                    st_progress_bar=progress_bar,
                    st_status_text=status_text
                )

            # --- Display the results ---
            # Clear the progress bar and status text after completion
            status_text.empty()
            progress_bar.empty()

            if "error" in result:
                st.error(f"An error occurred: {result['error']}")
            else:
                st.success("✅ Analysis complete!")
                
                prediction_data = result.get("prediction", {})
                video_path = result.get("video_path")
                csv_path = result.get("csv_path")
                output_folder = result.get("output_folder")

                st.info(f"All output files have been saved to: `{output_folder}`", icon="📁")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.subheader("Prediction Results")
                    prediction = prediction_data.get("prediction", "N/A").capitalize()
                    confidence = prediction_data.get("confidence", 0)
                    st.metric(label="Predicted Execution", value=prediction)
                    st.metric(label="Model Confidence", value=f"{confidence*100:.2f}%")
                    if csv_path:
                        with open(csv_path, "rb") as f:
                            st.download_button(
                                label="Download Processed Data (CSV)",
                                data=f,
                                file_name=Path(csv_path).name,
                                mime="text/csv"
                            )
                with res_col2:
                    st.subheader("Analysis Visualization")
                    if video_path:
                        video_file = open(video_path, 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                        st.download_button(
                            label="Download Visualization Video",
                            data=video_bytes,
                            file_name=Path(video_path).name,
                            mime="video/mp4"
                        )
                    else:
                        st.warning("Could not generate visualization video. The input video format might be incompatible.")

                st.info("This result is based on a machine learning model and should be used for informational purposes only.", icon="ℹ️")

        else:
            st.warning("⚠️ Please upload both videos before processing.")

if __name__ == "__main__":
    main()

