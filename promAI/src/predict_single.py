# FILE: src/predict_single.py

import os
import argparse
import torch
import pandas as pd

# Assuming this script is in the 'src' folder, we can import from siblings
from .config import CONFIG
from .models.gru_classifier import GRUClassifier

def load_checkpoint(model, optimizer, path, device):
    """Utility function to load a model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    print(f"🔄 Checkpoint loaded from {os.path.basename(path)} (epoch {epoch+1}, loss {loss:.4f})")
    return model, optimizer, epoch, loss


def predict_from_csv(csv_path: str, exercise: str):
    """
    Loads a model, predicts the class for a single processed CSV file,
    and returns the result as a dictionary.
    """
    # --- 1. SETUP ---
    device = CONFIG["device"]
    checkpoint_dir = os.path.join(CONFIG["output"]["model_dir"], exercise)
    
    try:
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith("_best.pth")]
        if not files:
            raise FileNotFoundError(f"No '_best.pth' checkpoint found in '{checkpoint_dir}'")
        checkpoint_name = sorted(files)[-1]
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

    # --- 2. LOAD DATA FROM CSV ---
    if not os.path.exists(csv_path):
        error_msg = f"Input file not found at '{csv_path}'"
        print(f"❌ Error: {error_msg}")
        return {"error": error_msg}
        
    df = pd.read_csv(csv_path)
    features_tensor = torch.tensor(df.values, dtype=torch.float32)
    features_tensor = features_tensor.unsqueeze(0).to(device)
    _, seq_len, input_dim = features_tensor.shape

    # --- 3. INITIALIZE AND LOAD MODEL ---
    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_classes=2,  # 0 for correct, 1 for incorrect
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"]
    ).to(device)

    model, _, _, _ = load_checkpoint(model, None, ckpt_path, device)
    model.eval()

    # --- 4. PREDICT ---
    with torch.no_grad():
        outputs = model(features_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(outputs, dim=1)
        # Get the prediction with the highest probability
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        prediction_int = predicted_idx.item()
        confidence_score = confidence.item()

    # --- 5. FORMAT AND RETURN RESULT ---
    # This section is changed to return a dictionary instead of printing
    class_map = {0: "correct", 1: "incorrect"}
    predicted_label = class_map.get(prediction_int, "unknown")

    result = {
        "prediction": predicted_label,
        "confidence": float(f"{confidence_score:.4f}") # Format to 4 decimal places
    }

    # This function now returns a value for the app to use
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict exercise correctness from a single processed CSV file.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the single processed CSV file.")
    parser.add_argument("--exercise", type=str, required=True, choices=["elbow_extension", "shoulder_flexion"], help="The exercise type to select the correct model.")
    args = parser.parse_args()

    # The function is called, and its returned dictionary is printed
    prediction_output = predict_from_csv(csv_path=args.csv_file, exercise=args.exercise)
    
    print("\n" + "="*40)
    print("🔎 ANALYSIS COMPLETE")
    print(f"   - File: {os.path.basename(args.csv_file)}")
    print(f"   - Result Dictionary: {prediction_output}")
    print("="*40 + "\n")