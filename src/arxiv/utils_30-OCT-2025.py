import os
import json
import yaml
import random
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from omegaconf import OmegaConf
import csv
import pandas as pd
import torch
import numpy as np




# ===============================================================
# �~Z~Y�CLEANING
# ===============================================================





def clean_tensor(x: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """
    Replace NaN or Inf in a tensor with a fill value (default 0.0).
    Useful for ocean data with land masks as NaN.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    # Replace NaN and Inf
    x = torch.nan_to_num(x, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return x


def mask_invalid(targets: torch.Tensor) -> torch.Tensor:
    """
    Returns a boolean mask (True where data is valid).
    NaN or Inf are considered invalid.
    """
    return torch.isfinite(targets)











# ===============================================================
# ⚙️ CONFIGURATION
# ===============================================================

def load_config(path: str):
    """Load YAML configuration file using OmegaConf."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


# ===============================================================
# 🔒 REPRODUCIBILITY
# ===============================================================

def set_seed(seed: int = 42):
    """
    Ensure full reproducibility across NumPy, PyTorch, and Python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔁 Reproducibility set with seed={seed}")


# ===============================================================
# 📊 NORMALIZATION UTILS
# ===============================================================

def load_or_compute_norm_stats(config, tracer_datasets, neighbor_datasets, out_path):
    """
    Compute or load normalization statistics for tracer and velocity data.
    """
    if os.path.exists(out_path):
        print(f"✅ Loading normalization stats from {out_path}")
        with open(out_path, "r") as f:
            return json.load(f)

    print("📊 Computing normalization stats from datasets...")
    stats = {}

    # --- Tracers ---
    tracer_vars = config.data.tracers.variables
    for ds in tracer_datasets:
        for varname in tracer_vars:
            if varname not in ds:
                raise KeyError(f"Tracer var '{varname}' not found. Available: {list(ds.data_vars)}")
            arr = ds[varname].values.astype(np.float32)
            stats[f"{varname}_mean"] = float(np.nanmean(arr))
            stats[f"{varname}_std"] = float(np.nanstd(arr) + 1e-6)

    # --- Velocities ---
    vel_paths = config.data.neighbors.velocity_paths
    u_var, v_var, w_var = (
        config.data.neighbors.u_var,
        config.data.neighbors.v_var,
        config.data.neighbors.w_var,
    )

    # Case 1: Separate U/V/W files
    if len(vel_paths) == 3:
        print("🔹 Using separate velocity files (U, V, W)")
        for path, varname in zip(vel_paths, [u_var, v_var, w_var]):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Velocity file not found: {path}")
            ds = xr.open_dataset(path)
            arr = ds[varname].values.astype(np.float32)
            stats[f"{varname}_mean"] = float(np.nanmean(arr))
            stats[f"{varname}_std"] = float(np.nanstd(arr) + 1e-6)
            ds.close()

    # Case 2: Combined neighbor datasets
    else:
        print("🔹 Using neighbor datasets (combined velocities)")
        for i, ds in enumerate(neighbor_datasets):
            for varname in [u_var, v_var, w_var]:
                arr = ds[varname].values.astype(np.float32)
                stats[f"{varname}_mean_{i}"] = float(np.nanmean(arr))
                stats[f"{varname}_std_{i}"] = float(np.nanstd(arr) + 1e-6)

    # --- Save ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved normalization stats → {out_path}")
    return stats


def normalize_field(field, mean, std):
    """Normalize a NumPy array given mean and std."""
    return (field - mean) / (std + 1e-8)


def denormalize_field(field, mean, std):
    """Reverse normalization."""
    return field * (std + 1e-8) + mean


# ===============================================================
# 📈 LOGGING UTILS
# ===============================================================
# ===============================================================
# 📈 LOGGING UTILS (HPC-SAFE)
# ===============================================================
import csv
from datetime import datetime

class Logger:
    """
    Unified logger for CSV (+ optional TensorBoard if available).
    Works even when TensorBoard isn't installed (HPC-safe).
    """

    def __init__(self, log_dir, experiment_name="default"):
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, experiment_name + "_" + self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

        # Try importing TensorBoard — skip if unavailable
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            self.use_tb = True
            print(f"🧠 TensorBoard logging enabled at {self.log_dir}")
        except Exception:
            self.tb_writer = None
            self.use_tb = False
            print(f"⚠️ TensorBoard not available — using CSV only (HPC-safe).")

        # CSV logger setup
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "notes"])

    def log(self, epoch, train_loss, val_loss=None, notes=""):
        """Log metrics to CSV (+ TensorBoard if available)."""
        if self.use_tb and self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
            if val_loss is not None:
                self.tb_writer.add_scalar("Loss/val", val_loss, epoch)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss if val_loss else "", notes])

    def flush(self):
        if self.use_tb and self.tb_writer is not None:
            self.tb_writer.flush()

    def close(self):
        if self.use_tb and self.tb_writer is not None:
            self.tb_writer.close()

# ===============================================================
# 🎨 VISUALIZATION
# ===============================================================

def visualize_field(field, title="Field", cmap="viridis"):
    plt.figure(figsize=(6, 5))
    plt.imshow(field, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def visualize_forecast(target, pred, title_prefix="", save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Ensure arrays are 2D ---
    target = np.squeeze(target)
    pred = np.squeeze(pred)

    # If they became 1D (e.g., shape (30,)), convert to 2D (1, 30)
    if target.ndim == 1:
        target = target[np.newaxis, :]
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]

    # Final check
    print(f"[visualize_forecast] target shape: {target.shape}, pred shape: {pred.shape}")

    # --- Create side-by-side comparison ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axs[0].imshow(target, cmap='viridis')
    axs[0].set_title(f"{title_prefix} - Target")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred, cmap='viridis')
    axs[1].set_title(f"{title_prefix} - Prediction")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # --- Difference plot ---
    diff = pred - target
    im2 = axs[2].imshow(diff, cmap='coolwarm')
    axs[2].set_title(f"{title_prefix} - Difference")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved forecast visualization to {save_path}")
    else:
        plt.show()

    plt.close()



# ===============================================================
# 💾 CHECKPOINTS
# ===============================================================

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"💾 Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"✅ Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]
# ===============================================================
# 📉 LOSS PLOTTER (HPC-SAFE)
# ===============================================================


def plot_training_curves(log_dir, save_path=None):
    """
    Plot training & validation loss curves from CSV logs.

    Args:
        log_dir (str): Path to directory containing metrics.csv
        save_path (str): Optional path to save PNG (default: same folder)
    """
    csv_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print(f"❌ No metrics.csv found in {log_dir}")
        return

    df = pd.read_csv(csv_path)
    if "epoch" not in df or "train_loss" not in df:
        print("❌ CSV missing required columns: 'epoch', 'train_loss'")
        print("Columns found:", df.columns.tolist())
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in df and not df["val_loss"].isnull().all():
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(log_dir, "loss_curve.png")

    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✅ Saved loss plot → {save_path}")

