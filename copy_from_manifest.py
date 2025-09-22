import argparse
from pathlib import Path
import pandas as pd
import shutil

def parse_guide_filename(filename: str):
    """
    Parses a guide filename to extract exercise, label, and the unique ID.
    Example: "elbow_extension_subtle_incorrect_034.csv" -> ("elbow_extension", "subtle_incorrect", "34")
    """
    base_name = Path(filename).stem
    parts = base_name.split('_')
    
    if len(parts) < 3:
        return None, None, None, None

    try:
        id_with_zeros = parts[-1]
        unique_id = str(int(id_with_zeros))
    except ValueError:
        unique_id = parts[-1]
    
    core_name = '_'.join(parts[:-1])
    
    if 'subtle_incorrect' in core_name:
        label = 'subtle_incorrect'
    elif 'incorrect' in core_name:
        label = 'incorrect'
    elif 'correct' in core_name:
        label = 'correct'
    else:
        return None, None, None, None

    exercise = core_name.replace(f'_{label}', '')
    numeric_label = 0 if label == 'correct' else 1
    
    return exercise, label, unique_id, numeric_label

def find_and_copy_from_manifest(manifest_path: Path, source_dir: Path, dest_dir: Path):
    """
    Reads a manifest of guide filenames, finds video pairs by ID, copies them,
    and creates a new manifest for the copied dataset.
    """
    if not manifest_path.exists():
        print(f"❌ Error: Source manifest file not found at '{manifest_path}'")
        return
    if not source_dir.is_dir():
        print(f"❌ Error: Source directory not found at '{source_dir}'")
        return

    manifest_df = pd.read_csv(manifest_path)
    if 'filename' not in manifest_df.columns:
        print("❌ Error: Manifest must have a column named 'filename'.")
        return
        
    print(f"🚀 Starting video search and copy process...")
    print(f"    Source Manifest: {manifest_path.name}")
    print(f"    Source Dataset:  {source_dir}")
    print(f"    Destination:     {dest_dir}")

    copied_count = 0
    not_found_count = 0
    new_manifest_entries = []

    for guide_filename in manifest_df['filename']:
        exercise, label, unique_id, numeric_label = parse_guide_filename(guide_filename)
        
        if not exercise:
            print(f"  ⚠️  Warning: Could not parse guide filename '{guide_filename}'. Skipping.")
            continue
            
        cam0_folder = source_dir / exercise / label / 'cam0'
        cam1_folder = source_dir / exercise / label / 'cam1'
        
        cam0_videos = list(cam0_folder.glob(f"*_{unique_id}.*"))
        cam1_videos = list(cam1_folder.glob(f"*_{unique_id}.*"))
        
        if len(cam0_videos) == 1 and len(cam1_videos) == 1:
            source_cam0_path = cam0_videos[0]
            source_cam1_path = cam1_videos[0]
            
            dest_cam0_folder = dest_dir / exercise / label / 'cam0'
            dest_cam0_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_cam0_path, dest_cam0_folder)
            
            dest_cam1_folder = dest_dir / exercise / label / 'cam1'
            dest_cam1_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_cam1_path, dest_cam1_folder)
            
            print(f"  ✅ Found and copied pair for ID: {unique_id} in {exercise}/{label}")
            copied_count += 2
            
            new_manifest_entries.append({
                'video_id': source_cam0_path.stem, 'exercise': exercise, 'correctness_label': label,
                'numeric_label': numeric_label, 'file_path': str(dest_cam0_folder / source_cam0_path.name)
            })
            new_manifest_entries.append({
                'video_id': source_cam1_path.stem, 'exercise': exercise, 'correctness_label': label,
                'numeric_label': numeric_label, 'file_path': str(dest_cam1_folder / source_cam1_path.name)
            })
        else:
            print(f"  ⚠️  Warning: Could not find a unique pair for ID {unique_id} in {exercise}/{label}. (Found {len(cam0_videos)} cam0, {len(cam1_videos)} cam1)")
            not_found_count += 2

    print("\n" + "="*50)
    print("✅ File copying complete!")
    print(f"Total videos copied: {copied_count}")
    print(f"Pairs not found or not unique: {not_found_count // 2}")
    print("="*50)

    # --- MODIFIED SECTION: Create a manifest for each exercise ---
    if new_manifest_entries:
        print("\n✍️  Creating new manifest(s) for the copied dataset...")
        new_manifest_df = pd.DataFrame(new_manifest_entries)
        
        # Group by exercise and save a file for each group
        for exercise_name, exercise_df in new_manifest_df.groupby('exercise'):
            new_manifest_path = dest_dir / f"{exercise_name}_manifest.csv"
            exercise_df.to_csv(new_manifest_path, index=False)
            print(f"✅ New manifest with {len(exercise_df)} entries created at: {new_manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and copy video files based on a guide manifest.")
    parser.add_argument("--manifest", required=True, type=str, help="Path to the source manifest CSV file.")
    parser.add_argument("--source", required=True, type=str, help="Path to the source dataset directory.")
    parser.add_argument("--dest", required=True, type=str, help="Path to the destination directory.")
    
    args = parser.parse_args()
    
    find_and_copy_from_manifest(
        manifest_path=Path(args.manifest),
        source_dir=Path(args.source),
        dest_dir=Path(args.dest)
    )