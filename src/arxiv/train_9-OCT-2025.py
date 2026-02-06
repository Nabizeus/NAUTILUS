import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data import WindowedOceanDataset
from src.model import OceanPredictor
from src.utils import (
    load_config,
    set_seed,
    Logger,
    save_checkpoint,
)

def main():
    # ===========================================================
    # 1️⃣ LOAD CONFIG AND INITIALIZE
    # ===========================================================
    cfg = load_config("config/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🌊 Starting training on device: {device}")
    set_seed(cfg.training.seed)

    # ===========================================================
    # 2️⃣ LOAD DATASETS
    # ===========================================================
    print("📦 Loading datasets...")
    full_dataset = WindowedOceanDataset(cfg, split="train")

    val_fraction = cfg.training.val_fraction
    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    print(f"✅ Train samples: {train_size}, Validation samples: {val_size}")

    # ===========================================================
    # 3️⃣ MODEL + OPTIMIZER + LOSS
    # ===========================================================
    model = OceanPredictor(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.MSELoss()

    logger = Logger(cfg.training.log_dir, cfg.training.experiment_name)

    # Optionally resume training
    start_epoch = 0
    ckpt_path = cfg.training.resume_from
    if ckpt_path and os.path.exists(ckpt_path):
        from utils import load_checkpoint
        start_epoch = load_checkpoint(model, optimizer, ckpt_path, device=device)
        print(f"⏩ Resumed training from epoch {start_epoch}")

    # ===========================================================
    # 4️⃣ TRAINING LOOP
    # ===========================================================
    best_val_loss = float("inf")
    num_epochs = cfg.training.epochs

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # =======================================================
        # VALIDATION
        # =======================================================
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                loss = criterion(preds, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # =======================================================
        # LOG + CHECKPOINT
        # =======================================================
        logger.log(epoch, avg_train_loss, avg_val_loss)
        logger.flush()

        print(f"📉 Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_save_path = os.path.join(cfg.training.checkpoint_dir, f"best_model.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_save_path)

    # ===========================================================
    # 5️⃣ CLEANUP
    # ===========================================================
    logger.close()
    print("✅ Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

