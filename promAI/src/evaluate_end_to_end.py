import os
import argparse
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path # <-- FIX: Add this import
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Assuming these imports are correct for your project structure
from .config import CONFIG
from .models.gru_classifier import GRUClassifier
from .utils.checkpoint import load_checkpoint

# --- Helper function for naming result files ---
def get_exercise_code(exercise):
    """Converts exercise name to a simple code for file naming."""
    return "1" if exercise == "elbow_extension" else "2"

# --- 1. Custom Dataset for loading from your Manifest ---
class ManifestDataset(Dataset):
    """PyTorch Dataset that reads a manifest to load processed CSV files."""
    def __init__(self, manifest_path: str, exercise: str):
        super().__init__()
        manifest_df = pd.read_csv(manifest_path)
        self.exercise_df = manifest_df[manifest_df['exercise'] == exercise].reset_index(drop=True)
        
        if self.exercise_df.empty:
            raise FileNotFoundError(f"No data found for exercise '{exercise}' in manifest.")

        first_csv_path = self.exercise_df.iloc[0]['processed_file_path']
        self.input_dim = pd.read_csv(first_csv_path).shape[1]

    def __len__(self):
        return len(self.exercise_df)

    def __getitem__(self, index):
        row = self.exercise_df.iloc[index]
        csv_path = row['processed_file_path']
        features = torch.tensor(pd.read_csv(csv_path).values, dtype=torch.float32)
        label = torch.tensor(int(row['numeric_label']), dtype=torch.long)
        return features, label

# --- 2. Collate function to handle padding ---
def pad_collate_fn(batch):
    """Pads sequences in a batch to the same length."""
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    features_padded = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels_tensor = torch.stack(labels_list)
    return features_padded, labels_tensor

def evaluate_from_manifest(manifest_path: str, exercise: str, checkpoint_name: str = None):
    """
    Evaluates a model using a manifest file and an efficient DataLoader.
    """
    device = CONFIG["device"]
    
    # --- 3. Create Dataset and DataLoader ---
    try:
        eval_dataset = ManifestDataset(manifest_path=manifest_path, exercise=exercise)
    except (FileNotFoundError, IndexError) as e:
        print(f"❌ Error: Could not create dataset. {e}")
        return
        
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        collate_fn=pad_collate_fn
    )
    
    print(f"Found {len(eval_dataset)} samples for '{exercise}' to evaluate.")

    # --- 4. Load Model Checkpoint ---
    checkpoint_dir = os.path.join(CONFIG["output"]["model_dir"], exercise)
    if checkpoint_name is None:
        try:
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith("_best.pth")]
            checkpoint_name = sorted(files)[-1]
        except (FileNotFoundError, IndexError):
            print(f"❌ Error: No '_best.pth' checkpoint found in '{checkpoint_dir}'")
            return
            
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Loading model checkpoint: {ckpt_path}")

    # --- 5. Initialize Model and Run Evaluation ---
    model = GRUClassifier(
        input_dim=eval_dataset.input_dim,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_classes=2,
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"]
    ).to(device)

    model, _, _, _ = load_checkpoint(model, None, ckpt_path, device)
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in eval_dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # --- 6. Display and Save Metrics ---
    print(f"\n=== Evaluation on [{exercise}] from manifest ===")
    
    report_dict = classification_report(all_labels, all_preds, labels=[0, 1], target_names=["correct", "incorrect"], output_dict=True, zero_division=0)
    print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=["correct", "incorrect"], zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)
    
    print("\n" + "-"*20)
    print("✍️  Saving evaluation results to files...")
    
    results_dir = Path("results") / exercise
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    ex_code = get_exercise_code(exercise)

    cm_df = pd.DataFrame(cm, index=["true_correct", "true_incorrect"], columns=["pred_correct", "pred_incorrect"])
    cm_file = results_dir / f"{timestamp}_{ex_code}_cmatrix.csv"
    cm_df.to_csv(cm_file)
    print(f"✅ Confusion matrix saved to: {cm_file}")

    report_df = pd.DataFrame(report_dict).transpose()
    report_file = results_dir / f"{timestamp}_{ex_code}_classification.csv"
    report_df.to_csv(report_file)
    print(f"✅ Classification report saved to: {report_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using a processed data manifest.")
    parser.add_argument("--manifest", required=True, type=str, help="Path to the processed_manifest.csv file.")
    parser.add_argument("--exercise", required=True, type=str, choices=["elbow_extension", "shoulder_flexion"], help="The exercise to evaluate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint filename to use (default: latest *_best.pth)")
    args = parser.parse_args()

    evaluate_from_manifest(
        manifest_path=args.manifest,
        exercise=args.exercise,
        checkpoint_name=args.checkpoint
    )