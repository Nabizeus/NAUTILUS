from __future__ import annotations
import os
import csv
import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import imageio

from omegaconf import DictConfig

# imports from your package
# make sure src is on PYTHONPATH (python -m src.train ensures that)
from src.data import WindowedOceanDataset
from src.model import OceanPredictor
from src.utils import load_config, set_seed


def log_training_time(region_name, elapsed_minutes, best_val_loss, epochs, batch_size, csv_path= f"logs/training_times_{time.strftime('%Y%m%d_%H%M')}.csv"):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        # If file is new, write the header
        if not file_exists:
            writer.writerow(["region", "elapsed_minutes", "best_val_loss", "epochs", "batch_size", "time_stamp"])
        # Write values
        #writer.writerow([region_name, round(elapsed_minutes, 2), round(best_val_loss, 4)])
        writer.writerow([
            region_name,
            round(elapsed_minutes, 2),
            round(best_val_loss, 2),
            epochs,
            batch_size,
            time.strftime('%Y-%m-%d %H:%M')
        ])




# Optional helper functions (you might have equivalents in src.utils)
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(path: str,model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, epoch: Optional[int]=None):
    """Save model state dict with optional optimizer and epoch (simple portable format)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    torch.save(ckpt, path)

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """Compute MSE, applying mask if provided. mask shape broadcastable to pred/target (0=ignore,1=keep)."""
    if mask is None:
        return F.mse_loss(pred, target)
    # ensure floats
    mask = mask.to(pred.dtype)
    denom = mask.sum()
    if denom == 0:
        # nothing to compare -> return zero loss (or large number); choose zero to avoid noisy grads
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / denom

# Visualization helpers
def debug_visualize_sample(inputs: torch.Tensor, region_name: str, number: int, out_dir="artifacts/debug_inputs"):
    """
    Save a grid of input frames for one sample.
    inputs shape expected: (B, T, C, H, W)  or (T, C, H, W) etc. We handle common cases.
    Returns: fullpath to saved PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    # take first sample if batch
    if inputs.dim() == 5:
        sample = inputs[0].detach().cpu().numpy()  # (T, C, H, W)
    elif inputs.dim() == 4:
        sample = inputs.detach().cpu().numpy()  # (T, C, H, W) maybe
    else:
        # fallback: convert to 4D
        sample = inputs.detach().cpu().numpy()
    # normalize shape to (T, C, H, W)
    if sample.shape[0] != sample.shape[0] and sample.ndim == 3:
        # nothing
        pass
    T, C, H, W = sample.shape
    vars_names = ["Tracer", "U", "V", "W"][:C]
    fig, axes = plt.subplots(C, T, figsize=(T * 1.6, C * 1.6), squeeze=False)
    for c in range(C):
        for t in range(T):
            ax = axes[c][t]
            im = ax.imshow(sample[t, c], origin="lower")
            ax.axis("off")
            if c == 0:
                ax.set_title(f"t={t}")
    plt.suptitle(f"{region_name} sample {number}")
    out = os.path.join(out_dir, f"sample_{region_name}_{number:04d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out

def animate_frames(image_paths: List[str], save_gif: str, fps: int = 2):
    """Create GIF from pngs using imageio."""
    if not image_paths:
        print("No frames to animate.")
        return
    frames = []
    for p in image_paths:
        frames.append(imageio.imread(p))
    imageio.mimsave(save_gif, frames, fps=fps)
    print(f"Saved animation to {save_gif}")

def plot_loss_curve(train_losses: List[float], val_losses: List[float], out_path: str, region_name: str):
    plt.figure(figsize=(6, 3))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curve - {region_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def write_log_csv(csv_path: str, region_name: str, epoch: int, train_loss: float, val_loss: float):
    header = ["region", "epoch", "train_loss", "val_loss"]
    newrow = [region_name, int(epoch), float(round(train_loss,2)), float(round(val_loss,2))]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(newrow)

# ----------------------
# Training flow
# ----------------------


def main():



    print("🧠 Using pure YAML config —")

    # 1️⃣ Load configuration
    cfg = load_config("config/default.yaml")
    print("🔧 Starting training script (merged debug/vis).")
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(cfg.training, "use_cuda", False) else "cpu")
    print(f"🧠 Using device: {device}")
    set_seed(int(getattr(cfg.training, "seed", 42)))

    # Loop over regions defined in config.data.regions
    regions = cfg.data.get("regions", None)
    if not regions:
        raise ValueError("No regions defined in config.data.regions - please define per-region boxes.")

    # global artifacts/checkpoints folder
    ckpt_dir = Path(getattr(cfg.training, "checkpoint_dir", "checkpoints"))
    artifacts_dir = Path(getattr(cfg, "artifacts_dir", "artifacts") if "artifacts_dir" in cfg else "artifacts")
    logs_dir = Path(getattr(cfg, "log_dir", "logs") if "log_dir" in cfg else "logs")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Options (you can put in YAML)
    debug_save_every_n_batches = int(getattr(cfg.debug, "save_every_n_batches", 10)) if "debug" in cfg else 10
    debug_save_first_k_batches = int(getattr(cfg.debug, "save_first_k_batches", 200)) if "debug" in cfg else 200

    for region_cfg in regions:
        start_time = time.time() # <-- Start timer
        region_name = region_cfg["name"]
        print(f"\n🌍 Starting training for region: {region_name}")

        # build per-region datasets
        train_ds = WindowedOceanDataset(cfg, split="train", region_name=region_name)
        val_ds   = WindowedOceanDataset(cfg, split="val", region_name=region_name)

        train_loader = DataLoader(train_ds, batch_size=int(cfg.training.batch_size), shuffle=True, num_workers=int(getattr(cfg.training, "num_workers", 0)))
        val_loader = DataLoader(val_ds, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=int(getattr(cfg.training, "num_workers", 0)))

        # Create model for this region (pass H/W if dataset made them available)
        H = getattr(train_ds, "H", None)
        W = getattr(train_ds, "W", None)
        model = OceanPredictor(cfg, H=H, W=W).to(device)

        optimizer = optim.Adam(model.parameters(), lr=float(cfg.training.lr))
        best_val_loss = float("inf")

        # training bookkeeping
        train_losses, val_losses = [], []
        csv_log_path = logs_dir / f"training_log_{region_name}.csv"

        # per-epoch debug frames aggregator (for gif)
        for epoch in range(1, int(cfg.training.epochs) + 1):
            model.train()
            running_loss = 0.0
            batch_count = 0
            debug_frames = []  # store saved debug png paths for this epoch
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{cfg.training.epochs} [Train]")

            for batch_idx, batch in pbar:
                # batch may be (x,y) or (x,y,mask)
                if len(batch) == 2:
                    inputs, targets = batch
                    mask = None
                elif len(batch) == 3:
                    inputs, targets, mask = batch
                else:
                    # unexpected
                    raise ValueError("Dataset __getitem__ must return (x,y) or (x,y,mask)")

                # Move to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                mask = mask.to(device) if mask is not None else None

                # Collapse depth into one 2D slice if config requests it (keep ability to use 3D convLSTM later)
                if inputs.ndim == 6 and cfg.data.get("use_2d_slice", True):
                    # if dataset provides z_index use it, else take middle
                    z_idx = getattr(train_ds, "z_index", inputs.shape[3] // 2)
                    # inputs shape: (B, T, C, Z, H, W) -> choose depth slice
                    inputs_2d = inputs[..., z_idx, :, :]
                else:
                    inputs_2d = inputs  # already 2D or model expects full volume

                # safety
                inputs_2d = torch.nan_to_num(inputs_2d, nan=0.0, posinf=1e6, neginf=-1e6)
                targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)

                optimizer.zero_grad()
                preds = model(inputs_2d)
                loss = masked_mse_loss(preds, targets, mask) if mask is not None else F.mse_loss(preds, targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ Skipping batch due to NaN/Inf loss.")
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += float(loss.detach().cpu().item())
                batch_count += 1
                pbar.set_postfix({"train_loss": f"{running_loss / batch_count:.6f}"})

                # Debug visualization save
                if (batch_idx % debug_save_every_n_batches) == 0 and (batch_idx < debug_save_first_k_batches):
                    saved_png = debug_visualize_sample(inputs, region_name, epoch * 10000 + batch_idx)
                    debug_frames.append(saved_png)

            train_epoch_loss = running_loss / max(1, batch_count)
            train_losses.append(train_epoch_loss)

            # --- Validation ---
            model.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 2:
                        v_inputs, v_targets = batch
                        v_mask = None
                    else:
                        v_inputs, v_targets, v_mask = batch
                    if v_inputs.ndim == 6 and cfg.data.get("use_2d_slice", True):
                        z_idx = getattr(val_ds, "z_index", v_inputs.shape[3] // 2)
                        v_inputs_2d = v_inputs[..., z_idx, :, :]
                    else:
                        v_inputs_2d = v_inputs
                    v_inputs_2d = torch.nan_to_num(v_inputs_2d, nan=0.0, posinf=1e6, neginf=-1e6)
                    v_targets = torch.nan_to_num(v_targets, nan=0.0, posinf=1e6, neginf=-1e6)
                    v_preds = model(v_inputs_2d.to(device))
                    v_loss = masked_mse_loss(v_preds, v_targets.to(device), v_mask.to(device) if v_mask is not None else None)
                    val_running += float(v_loss.detach().cpu().item())
                    val_count += 1
            val_epoch_loss = val_running / max(1, val_count)
            val_losses.append(val_epoch_loss)

            print(f"✅ Epoch [{epoch}/{cfg.training.epochs}] Train Loss: {train_epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")

            # Save gif for debug_frames if any
            if debug_frames:
                gif_path = artifacts_dir / f"debug_inputs_{region_name}_ep{epoch}.gif"
                animate_frames(sorted(debug_frames), str(gif_path), fps=2)

            # Save loss curve and CSV
            plot_loss_curve(train_losses, val_losses, str(artifacts_dir / f"loss_curve_{region_name}.png"), region_name)
            write_log_csv(str(csv_log_path), region_name, epoch, train_epoch_loss, val_epoch_loss)

            # Save checkpoint if improved
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                ckpt_path = ckpt_dir / f"{region_name}_best_model.pt"
                save_checkpoint(str(ckpt_path), model, optimizer=optimizer, epoch=epoch)
                print(f"🔁 Saved best model for {region_name} -> {ckpt_path}")

        print(f"🎯 Finished region: {region_name}. Best val loss: {best_val_loss:.6f}")
        end_time = time.time()     # <-- End timer
        elapsed = (end_time - start_time) / 60  # minutes
        log_training_time(region_name, elapsed, best_val_loss, cfg.training.epochs, cfg.training.batch_size)
        print(f"�~O� Training time for {region_name}: {elapsed:.2f} minutes\n")
    print("�~_~N~I Training complete.")

if __name__ == "__main__":
    main()

