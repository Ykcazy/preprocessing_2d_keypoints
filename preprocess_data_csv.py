# FILE: preprocess_data_csv.py

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# --- CONFIGURATION ---
RAW_DATA_FOLDER = Path("./extracted_raw_data")
PROCESSED_DATA_FOLDER = Path("./processed_gru_data")
SCALER_PATH = Path("./feature_scaler.pkl")
TARGET_LENGTHS_PATH = Path("./normalization_params.json")
TARGET_LENGTH_PERCENTILE = 95

# --- NEW: FEATURE SELECTION (COCO BASED) ---
# Keypoints available from the 2D detector
PT_KEYPOINTS_TO_KEEP = [
    'mid_hip', 'right_hip', 'left_hip', 'mid_shoulder',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist'
]
PATIENT_KEYPOINTS_TO_KEEP = [
    'mid_hip', 'right_hip', 'left_hip', 'mid_shoulder',
]
# Angles now use keypoints that can be derived from COCO
ANGLE_DEFINITIONS = {
    'left_shoulder_angle': ('mid_shoulder', 'left_shoulder', 'left_elbow'),
    'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist'),
    'right_shoulder_angle': ('mid_shoulder', 'right_shoulder', 'right_elbow'),
    'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
}

# --- UTILITY FUNCTIONS (UPDATED FOR 2D) ---
def calculate_2d_angle(p1, p2, p3):
    """Calculates angle from 2D coordinates."""
    v1, v2 = p1 - p2, p3 - p2
    dot_product = np.dot(v1, v2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def get_coco_kpt_index(name):
    from config import kpt_name_to_id
    return kpt_name_to_id[name]

def create_feature_sequence(df: pd.DataFrame):
    # Load raw 2D keypoints (17 keypoints, 2 coords)
    pt_kpts_raw = df[[f"pt_kpt{i}_{axis}" for i in range(17) for axis in 'xy']].values.reshape(-1, 17, 2)
    patient_kpts_raw = df[[f"patient_kpt{i}_{axis}" for i in range(17) for axis in 'xy']].values.reshape(-1, 17, 2)

    # Derive midpoints for a stable reference frame
    pt_mid_hip = (pt_kpts_raw[:, get_coco_kpt_index('left_hip')] + pt_kpts_raw[:, get_coco_kpt_index('right_hip')]) / 2
    pt_mid_shoulder = (pt_kpts_raw[:, get_coco_kpt_index('left_shoulder')] + pt_kpts_raw[:, get_coco_kpt_index('right_shoulder')]) / 2
    patient_mid_hip = (patient_kpts_raw[:, get_coco_kpt_index('left_hip')] + patient_kpts_raw[:, get_coco_kpt_index('right_hip')]) / 2
    patient_mid_shoulder = (patient_kpts_raw[:, get_coco_kpt_index('left_shoulder')] + patient_kpts_raw[:, get_coco_kpt_index('right_shoulder')]) / 2
    
    # Normalize all keypoints by subtracting the hip center
    pt_kpts_normalized = pt_kpts_raw - pt_mid_hip[:, np.newaxis, :]
    patient_kpts_normalized = patient_kpts_raw - patient_mid_hip[:, np.newaxis, :]
    
    # Create dictionaries for easy lookup of selected keypoints, including derived ones
    pt_kpt_map = {name: pt_kpts_normalized[:, get_coco_kpt_index(name)] for name in ['right_hip', 'left_hip', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']}
    pt_kpt_map['mid_hip'] = pt_mid_hip - pt_mid_hip # Should be [0,0] after normalization
    pt_kpt_map['mid_shoulder'] = pt_mid_shoulder - pt_mid_hip

    pt_kpts_selected = np.stack([pt_kpt_map[k] for k in PT_KEYPOINTS_TO_KEEP], axis=1)

    # Determine active arm and select patient keypoints
    dist_to_left = np.nanmean(df.filter(regex=r'left_elbow|left_wrist').values)
    dist_to_right = np.nanmean(df.filter(regex=r'right_elbow|right_wrist').values)
    active_arm_side = 'left' if dist_to_left < dist_to_right else 'right'
    proximity_cols = [col for col in df.columns if f'dist_' in col and f'_{active_arm_side}_' in col]
    proximity_data = df[proximity_cols].values
    
    patient_active_arm_kpts = [f'{active_arm_side}_shoulder', f'{active_arm_side}_elbow', f'{active_arm_side}_wrist']
    patient_final_kpts_to_keep = PATIENT_KEYPOINTS_TO_KEEP + patient_active_arm_kpts
    
    patient_kpt_map = {name: patient_kpts_normalized[:, get_coco_kpt_index(name)] for name in ['right_hip', 'left_hip', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']}
    patient_kpt_map['mid_hip'] = patient_mid_hip - patient_mid_hip
    patient_kpt_map['mid_shoulder'] = patient_mid_shoulder - patient_mid_hip
    patient_kpts_selected = np.stack([patient_kpt_map[k] for k in patient_final_kpts_to_keep], axis=1)

    # Calculate Velocities
    pt_velocities = np.diff(pt_kpts_selected, axis=0, prepend=np.zeros((1, *pt_kpts_selected.shape[1:])))
    patient_velocities = np.diff(patient_kpts_selected, axis=0, prepend=np.zeros((1, *patient_kpts_selected.shape[1:])))
    
    # Calculate 2D Angles
    pt_angles, patient_angles = [], []
    pt_kpt_name_to_idx = {name: i for i, name in enumerate(PT_KEYPOINTS_TO_KEEP)}
    patient_kpt_name_to_idx = {name: i for i, name in enumerate(patient_final_kpts_to_keep)}

    pt_angle_names = [name for name, parts in ANGLE_DEFINITIONS.items() if all(k in PT_KEYPOINTS_TO_KEEP for k in parts)]
    patient_angle_names = [name for name, parts in ANGLE_DEFINITIONS.items() if all(k in patient_final_kpts_to_keep for k in parts)]
    
    for frame_idx in range(len(df)):
        pt_frame_angles = {name: calculate_2d_angle(pt_kpts_selected[frame_idx, pt_kpt_name_to_idx[p1]], pt_kpts_selected[frame_idx, pt_kpt_name_to_idx[p2]], pt_kpts_selected[frame_idx, pt_kpt_name_to_idx[p3]]) for name, (p1, p2, p3) in ANGLE_DEFINITIONS.items() if name in pt_angle_names}
        patient_frame_angles = {name: calculate_2d_angle(patient_kpts_selected[frame_idx, patient_kpt_name_to_idx[p1]], patient_kpts_selected[frame_idx, patient_kpt_name_to_idx[p2]], patient_kpts_selected[frame_idx, patient_kpt_name_to_idx[p3]]) for name, (p1, p2, p3) in ANGLE_DEFINITIONS.items() if name in patient_angle_names}
        pt_angles.append(list(pt_frame_angles.values()))
        patient_angles.append(list(patient_frame_angles.values()))

    # Combine all features into a single sequence
    full_sequence = np.hstack([
        proximity_data, pt_kpts_selected.reshape(len(df), -1), patient_kpts_selected.reshape(len(df), -1),
        pt_velocities.reshape(len(df), -1), patient_velocities.reshape(len(df), -1),
        np.array(pt_angles), np.array(patient_angles)
    ])
    
    # Create header for the final DataFrame
    header = proximity_cols
    header += [f"pt_{kpt}_{ax}" for kpt in PT_KEYPOINTS_TO_KEEP for ax in 'xy']
    header += [f"patient_{kpt}_{ax}" for kpt in patient_final_kpts_to_keep for ax in 'xy']
    header += [f"pt_{kpt}_{ax}_vel" for kpt in PT_KEYPOINTS_TO_KEEP for ax in 'xy']
    header += [f"patient_{kpt}_{ax}_vel" for kpt in patient_final_kpts_to_keep for ax in 'xy']
    header += [f"pt_{name}" for name in pt_angle_names]
    header += [f"patient_{name}" for name in patient_angle_names]
    
    return full_sequence, header

# =================================================================================
# MAIN PREPROCESSING FUNCTION (Logic remains the same, but operates on new features)
# =================================================================================
def preprocess_all_data_to_csvs():
    raw_csv_files = sorted(list(RAW_DATA_FOLDER.rglob("*.csv")))
    if not raw_csv_files:
        print(f"Error: No raw CSV files found in '{RAW_DATA_FOLDER}'. Run 'main.py' first.")
        return

    PROCESSED_DATA_FOLDER.mkdir(exist_ok=True)
    
    # Step 1: Fit a GLOBAL scaler on all 'correct' data
    print("\nIdentifying 'correct' data to fit a global feature scaler...")
    training_files = [f for f in raw_csv_files if 'correct' in f.parent.name]
    if not training_files:
        print("Error: No 'correct' sequences found to fit the scaler. Cannot proceed."); return
        
    training_sequences_unscaled = []
    for csv_file in training_files:
        df = pd.read_csv(csv_file)
        full_sequence, _ = create_feature_sequence(df)
        if full_sequence.shape[0] > 1:
            training_sequences_unscaled.append(full_sequence)
    
    concatenated_training_data = np.vstack(training_sequences_unscaled)
    scaler = StandardScaler()
    scaler.fit(concatenated_training_data)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Feature scaler has been fitted on all 'correct' data and saved to '{SCALER_PATH}'.")

    # Step 2: Group files and determine per-exercise target lengths
    exercise_files = defaultdict(list)
    for file in raw_csv_files:
        exercise_files[file.parent.parent.name].append(file)
    
    exercise_target_lengths = {}
    for exercise_name, files_in_group in exercise_files.items():
        sequence_lengths = [len(pd.read_csv(file)) for file in files_in_group]
        target_length = int(np.percentile(sequence_lengths, TARGET_LENGTH_PERCENTILE))
        exercise_target_lengths[exercise_name] = target_length
        print(f"Optimal target length for '{exercise_name}' set to {target_length} frames.")

    with open(TARGET_LENGTHS_PATH, 'w') as f:
        json.dump(exercise_target_lengths, f, indent=4)
    print(f"Optimal target lengths saved to '{TARGET_LENGTHS_PATH}'.")

    # Step 3: Process each file using the correct parameters
    for exercise_name, files_in_group in exercise_files.items():
        TARGET_SEQ_LENGTH = exercise_target_lengths[exercise_name]
        for i, csv_file in enumerate(files_in_group):
            print(f"--- Processing file {i+1}/{len(files_in_group)} for '{exercise_name}': {csv_file.name} ---")
            df = pd.read_csv(csv_file)
            exercise, correctness = df['exercise'].iloc[0], df['correctness'].iloc[0]
            
            full_sequence, header = create_feature_sequence(df)
            if full_sequence.shape[0] < 2: continue
            
            # Replace NaNs or infs which might occur from bad frames
            full_sequence[~np.isfinite(full_sequence)] = 0.0

            scaled_sequence = scaler.transform(full_sequence)
            
            original_len = scaled_sequence.shape[0]
            resampler = interp1d(np.linspace(0, 1, original_len), scaled_sequence, axis=0)
            resampled_sequence = resampler(np.linspace(0, 1, TARGET_SEQ_LENGTH))
            
            try:
                pair_id = csv_file.stem.split('_')[-1]
                formatted_id = f"{int(pair_id):03d}"
            except (IndexError, ValueError): formatted_id = "unknown"
                
            new_filename = f"{exercise}_{correctness}_{formatted_id}.csv"
            output_folder = PROCESSED_DATA_FOLDER / exercise
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / new_filename

            processed_df = pd.DataFrame(resampled_sequence, columns=header)
            processed_df.to_csv(output_path, index=False)
            print(f"   - Saved processed file to {output_path}")

    print("\n\n" + "="*50 + "\n✅ PREPROCESSING COMPLETE\n" +
          f"   - Processed {len(raw_csv_files)} files.\n" +
          f"   - Output saved to individual CSVs in: '{PROCESSED_DATA_FOLDER}'\n" + "="*50)

if __name__ == '__main__':
    preprocess_all_data_to_csvs()