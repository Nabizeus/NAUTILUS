import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from src.data import WindowedOceanDataset
from src.model import OceanPredictor
from src.utils import load_config, load_or_compute_norm_stats
import os
import logging

def main():
    cfg = load_config("config/default.yaml")

    logging.basicConfig(filename="ome_cgpt.log", level=logging.INFO)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    logging.info(f"🌊 Starting training on device: {device}")
    print(f"🌊 Starting training on device: {device}")

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
    # Model setup
    # -------------------------------
    model = OceanPredictor(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")

    # -------------------------------
    # Training Loop with OOM Protection
    # -------------------------------
    for epoch in range(cfg.training.epochs):
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
                    loss = loss_fn(preds, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

                oom_retry = False  # completed successfully

            except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logging.warning(f"⚠️ OOM encountered. Reducing batch size to {batch_size} and retrying epoch {epoch+1}.")
                    print(f"⚠️ OOM encountered. Reducing batch size to {batch_size} and retrying epoch {epoch+1}.")
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
        # Save best model
        # -------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/model.pt")
            logging.info(f"💾 Saved best model at epoch {epoch+1} (val_loss={best_val_loss:.6f})")
            print(f"💾 Saved best model at epoch {epoch+1} (val_loss={best_val_loss:.6f})")

    print("🏁 Training complete.")


if __name__ == "__main__":
    main()

