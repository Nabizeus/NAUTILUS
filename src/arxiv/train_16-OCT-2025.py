import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import logging

from src.data import WindowedOceanDataset
from src.model import OceanPredictor
from src.utils import load_config

from src.utils import clean_tensor, mask_invalid

def save_checkpoint(model, optimizer, epoch, best_val_loss, path="checkpoints/model.pt"):
    """Save full training state (model + optimizer + metadata)."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(model, optimizer, path="checkpoints/model.pt", device="cpu"):
    """Load checkpoint if available."""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"🔁 Resumed from checkpoint at epoch {start_epoch+1}")
        return start_epoch + 1, best_val_loss
    return 0, float("inf")


def main():
    cfg = load_config("config/default.yaml")

    os.makedirs("checkpoints", exist_ok=True)
    logging.basicConfig(filename="ome_cgpt.log", level=logging.INFO)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    print(f"🌊 Starting training on device: {device}")
    logging.info(f"🌊 Training started on {device}")

    # -------------------------------
    # Data setup
    # -------------------------------
    full_dataset = WindowedOceanDataset(cfg, split="train")
    val_dataset = WindowedOceanDataset(cfg, split="val")

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -------------------------------
    # Model, optimizer, loss
    # -------------------------------
    model = OceanPredictor(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.MSELoss()

    # Resume if checkpoint exists
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, "checkpoints/model.pt", device=device)

    # -------------------------------
    # Training loop (with OOM + Resume)
    # -------------------------------
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        running_loss = 0.0
        oom_retry = True

        while oom_retry:
            try:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]", leave=False)
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    preds = model(inputs)
                     
                    # Clean invalid predictions and targets
                    preds = clean_tensor(preds)
                    targets = clean_tensor(targets)

                    # Mask land (NaN areas)
                    valid_mask = mask_invalid(targets)

                    if valid_mask.sum() == 0:
                        print("⚠️ Warning: No valid ocean points in batch — skipping loss.")
                        loss = torch.tensor(0.0, requires_grad=True)
                    else:
                        # Compute loss only over valid (ocean) points
                        loss = loss_fn(preds[valid_mask], targets[valid_mask])

                    #mask = ~torch.isnan(targets)
                    #masked_loss = loss_fn(preds[mask], targets[mask])
                    #loss = masked_loss


                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

                oom_retry = False

            except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logging.warning(f"⚠️ OOM - reducing batch size to {batch_size} and retrying epoch {epoch+1}")
                    print(f"⚠️ OOM - reducing batch size to {batch_size} and retrying epoch {epoch+1}")
                    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    time.sleep(3)
                    continue
                else:
                    raise e

        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"✅ Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}")
        print(f"✅ Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}")

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Val]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                val_loss += loss_fn(preds, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"📉 Validation Loss = {avg_val_loss:.6f}")
        print(f"📉 Validation Loss = {avg_val_loss:.6f}")

        # -------------------------------
        # Checkpoint saving
        # -------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss)
            logging.info(f"💾 Saved best model at epoch {epoch+1} (val_loss={best_val_loss:.6f})")
            print(f"💾 Saved best model at epoch {epoch+1} (val_loss={best_val_loss:.6f})")
        else:
            # also save periodic checkpoint
            save_checkpoint(model, optimizer, epoch, best_val_loss)

    print("🏁 Training complete.")


if __name__ == "__main__":
    main()

