import torch
import os


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"✅ Checkpoint saved at {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # FIX: Only try to load the optimizer if it exists
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"🔄 Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss
