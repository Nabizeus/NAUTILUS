import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import WindowedOceanDataset
from src.model import OceanPredictor
from src.utils import save_checkpoint,  set_seed

# ==============================
# ✅ NaN-Safe Loss Function
# ==============================
def masked_mse_loss(pred, target, mask):
    """
    Compute mean squared error on valid (non-land) ocean points.
    """
    # Create mask if not provided
    
    masked_pred = preds * mask
    masked_target = targets * mask
    return torch.mean((masked_pred - masked_target) ** 2)

def main():
    # ==============================
    # 🧩 Load config
    # ==============================
    cfg = load_yaml("config/default.yaml")
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    set_seed(cfg.training.seed)

    # ==============================
    # 📦 Dataset & Dataloader
    # ==============================
    print("🔧 Loading datasets...")
    train_ds = WindowedOceanDataset(cfg, split="train")
    val_ds = WindowedOceanDataset(cfg, split="val")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2)

    # ==============================
    # 🧠 Model Setup
    # ==============================
    print("🧱 Building model...")
    model = OceanPredictor(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float("inf")

    # ==============================
    # 🚀 Training Loop
    # ==============================
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]")

        for inputs, targets in pbar:
            #inputs, targets = inputs.to(device), targets.to(device)

            for inputs, targets in train_loader:
                # --- Depth-averaged 2D slicing ---
                if inputs.ndim == 6:
                    inputs = inputs[..., :10, :, :].mean(dim=3) # top 3 depth levels
                    print(f"🪣 Train depth-averaged input: {inputs.shape}")
                    inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)


            # Ensure NaN-safe tensors
            inputs = torch.nan_to_num(inputs)
            targets = torch.nan_to_num(targets)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # --- NaN/Inf safety in input batch ---
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print("⚠️ [Train Loop] NaN/Inf detected in inputs! Fixing batch.")
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e3, neginf=-1e3)

                preds = model(inputs)

                # --- NaN/Inf safety in output ---
                if torch.isnan(preds).any() or torch.isinf(preds).any():
                    print("⚠️ [Train Loop] NaN/Inf detected in predictions! Fixing.")
                    preds = torch.nan_to_num(preds, nan=0.0, posinf=1e3, neginf=-1e3)
                
                loss = masked_mse_loss(preds, targets)

            if torch.isnan(loss):
                print("⚠️ NaN detected in loss — skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})

        train_loss /= max(1, len(train_loader))

        # ==============================
        # 🧪 Validation
        # ==============================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets, mask =  inputs.to(device), targets.to(device), mask.to(device)
                if inputs.ndim == 6:
                    inputs_2d = inputs[..., :3, :, :].mean(dim=3)
                else:
                    inputs_2d = inputs
                preds = model(inputs_2d)
                loss = masked_mse_loss(preds, targets, mask)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        print(f"✅ Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # ==============================
        # 💾 Checkpoint
        # ==============================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, f"checkpoints/best_model.pt")
            print(f"💾 Saved new best model (val_loss={val_loss:.6f})")

    print("🎉 Training complete.")


if __name__ == "__main__":
    main()

