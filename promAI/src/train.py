# src/train.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from src.config import CONFIG, set_dataset_version
from src.loader import create_dataloader
from src.models.gru_classifier import GRUClassifier
from src.utils import save_checkpoint, load_checkpoint


def get_exercise_code(exercise):
    return "1" if exercise == "elbow_extension" else "2"


def train_model(exercise="elbow_extension"):
    device = CONFIG["device"]

    # Exercise-specific directories
    splits_dir = os.path.join(CONFIG["data"]["splits_dir"], exercise)
    checkpoint_dir = os.path.join(CONFIG["output"]["model_dir"], exercise)
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    ex_code = get_exercise_code(exercise)

    # Load training and validation splits
    train_loader, input_dim, seq_len = create_dataloader(
        split="train",
        splits_dir=splits_dir,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True
    )
    val_loader, _, _ = create_dataloader(
        split="val",
        splits_dir=splits_dir,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False
    )

    # Update config for reference
    CONFIG["model"]["input_dim"] = input_dim

    # Initialize model, loss, optimizer
    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_classes=2,
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["training"]["lr"])

    # Resume from last checkpoint if available
    last_ckpt = os.path.join(checkpoint_dir, f"{exercise}_last.pth")
    start_epoch = 0
    if os.path.exists(last_ckpt):
        model, optimizer, start_epoch, last_loss = load_checkpoint(
            model, optimizer, last_ckpt, device
        )
        print(f"Resumed from {last_ckpt}, epoch {start_epoch+1}, loss {last_loss:.4f}")

    best_val_acc = 0.0  # Track best validation accuracy

    # Training loop
    for epoch in range(start_epoch, CONFIG["training"]["epochs"]):
        model.train()
        total_loss = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"[{exercise}] Epoch [{epoch+1}/{CONFIG['training']['epochs']}], "
            f"Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save "last" checkpoint every epoch
        last_ckpt_name = f"{timestamp}_{ex_code}_last.pth"
        save_checkpoint(
            model, optimizer, epoch, avg_loss,
            os.path.join(checkpoint_dir, last_ckpt_name)
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_name = f"{timestamp}_{ex_code}_best.pth"
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(checkpoint_dir, best_ckpt_name)
            )
            print(f"🌟 New best model saved at epoch {epoch+1} with val_acc {val_acc:.4f}")

        # Save periodic checkpoints
        if (epoch + 1) % CONFIG["training"]["save_every"] == 0:
            periodic_ckpt_name = f"{timestamp}_{ex_code}_{epoch+1}.pth"
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(checkpoint_dir, periodic_ckpt_name)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", type=str, required=True, choices=["elbow_extension", "shoulder_flexion"])
    parser.add_argument("--dataset_version", type=str, default="dataset_v1", help="Dataset version to use.")
    args = parser.parse_args()

    set_dataset_version(args.dataset_version)
    train_model(exercise=args.exercise)
