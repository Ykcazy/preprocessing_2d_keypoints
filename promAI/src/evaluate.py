import os
import argparse
import torch
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# Assuming these imports are correctly pointing to your project structure
from .config import CONFIG, set_dataset_version
from .loader import create_dataloader 
from .models.gru_classifier import GRUClassifier
from .utils import load_checkpoint


def get_exercise_code(exercise):
    """Returns a numeric code for the exercise for file naming."""
    return "1" if exercise == "elbow_extension" else "2"


def evaluate_model(exercise="elbow_extension", split="test", checkpoint_name=None):
    device = CONFIG["device"]
    splits_dir = os.path.join(CONFIG["data"]["splits_dir"], exercise)
    checkpoint_dir = os.path.join(CONFIG["output"]["model_dir"], exercise)

    # Default to the latest best checkpoint if not specified
    if checkpoint_name is None:
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith("_best.pth")]
        if not files:
            raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")
        checkpoint_name = sorted(files)[-1] # Get the latest one
    
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Load dataset
    # This now returns filenames as the 3rd item in each batch
    dataloader, input_dim, seq_len = create_dataloader(
        split=split,
        splits_dir=splits_dir,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False
    )

    # Initialize model
    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_classes=2, # correct (0), incorrect (1)
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"]
    ).to(device)

    # Load the specified checkpoint
    model, _, epoch, loss = load_checkpoint(model, None, ckpt_path, device)
    print(f"🔄 Checkpoint loaded from {ckpt_path} (epoch {epoch}, loss {loss:.4f})\n")

    # --- Evaluation ---
    model.eval()
    all_labels, all_preds, all_filenames = [], [], [] # <<< MODIFIED: Added all_filenames

    with torch.no_grad():
        # <<< MODIFIED: Unpack filenames from dataloader
        for features, labels, filenames in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_filenames.extend(filenames) # <<< NEW: Collect filenames

    # --- Metrics Reporting ---
    print(f"=== Evaluation on [{exercise}] {split} split ===")
    report_dict = classification_report(
        all_labels, all_preds,
        labels=[0, 1],
        target_names=["correct", "incorrect"],
        output_dict=True,
        zero_division=0
    )
    print(classification_report(
        all_labels, all_preds,
        labels=[0, 1],
        target_names=["correct", "incorrect"],
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)
    
    # <<< NEW SECTION: Identify and display misclassified files >>>
    false_positives = []
    false_negatives = []
    misclassified_records = []

    for i in range(len(all_labels)):
        true_label = all_labels[i]
        pred_label = all_preds[i]
        filename = all_filenames[i]
        
        # False Positive: Predicted 'incorrect' (1) but was actually 'correct' (0)
        if pred_label == 1 and true_label == 0:
            false_positives.append(filename)
            misclassified_records.append({
                "filename": filename,
                "true_label": "correct",
                "predicted_label": "incorrect",
                "error_type": "False Positive"
            })
            
        # False Negative: Predicted 'correct' (0) but was actually 'incorrect' (1)
        elif pred_label == 0 and true_label == 1:
            false_negatives.append(filename)
            misclassified_records.append({
                "filename": filename,
                "true_label": "incorrect",
                "predicted_label": "correct",
                "error_type": "False Negative"
            })

    print("\n--- Misclassification Report ---")
    print(f"Total False Positives: {len(false_positives)}")
    for fp in false_positives:
        print(f"  - {fp}")

    print(f"\nTotal False Negatives: {len(false_negatives)}")
    for fn in false_negatives:
        print(f"  - {fn}")
    print("---------------------------------\n")
    # <<< END OF NEW SECTION >>>

    # --- Save Results to Files ---
    print("✍️  Saving evaluation results to files...")
    cm_df = pd.DataFrame(
        cm,
        index=["true_correct", "true_incorrect"],
        columns=["pred_correct", "pred_incorrect"]
    )

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    ex_code = get_exercise_code(exercise)
    results_dir = os.path.join("results", exercise)
    os.makedirs(results_dir, exist_ok=True)

    cm_file = os.path.join(results_dir, f"{timestamp}_{ex_code}_cmatrix.csv")
    cm_df.to_csv(cm_file)
    print(f"✅ Confusion matrix saved to: {cm_file}")

    report_df = pd.DataFrame(report_dict).transpose()
    report_file = os.path.join(results_dir, f"{timestamp}_{ex_code}_classification.csv")
    report_df.to_csv(report_file)
    print(f"✅ Classification report saved to: {report_file}")

    # <<< NEW: Save misclassified files to a CSV >>>
    if misclassified_records:
        misclassified_df = pd.DataFrame(misclassified_records)
        misclassified_file = os.path.join(results_dir, f"{timestamp}_{ex_code}_misclassified.csv")
        misclassified_df.to_csv(misclassified_file, index=False)
        print(f"✅ Misclassified files report saved to: {misclassified_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained GRU model for exercise classification.")
    parser.add_argument("--exercise", type=str, required=True, choices=["elbow_extension", "shoulder_flexion"], help="The exercise to evaluate.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="The data split to evaluate on.")
    parser.add_argument("--dataset_version", type=str, default="dataset_v1", help="The version of the dataset to use.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint filename to use (e.g., '250919_0920_1_best.pth'). Defaults to the latest best checkpoint.")
    args = parser.parse_args()

    set_dataset_version(args.dataset_version)
    evaluate_model(exercise=args.exercise, split=args.split, checkpoint_name=args.checkpoint)