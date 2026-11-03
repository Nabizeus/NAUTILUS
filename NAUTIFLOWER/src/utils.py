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
import math
from scipy.ndimage import gaussian_filter
import time

import time
import os



class SimpleProfiler:
    def __init__(self, save_path="profiler/profiler.log", region=None, mode=None):
        self.times = {}
        self.save_path = save_path
        self.region = region
        self.mode = mode

        # ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


    def start(self, key):
        self.times.setdefault(key, [0.0, 0])
        self._t0 = time.time()
        self._key = key

    def stop(self):
        dt = time.time() - self._t0
        self.times[self._key][0] += dt
        self.times[self._key][1] += 1

    def _format_time(self, seconds):
        minutes = seconds / 60.0
        hours = minutes / 60.0

        if hours >= 1:
            return f"{hours:.2f}h ({minutes:.1f} min)"
        else:
            return f"{minutes:.1f} min"

    def report(self, write_to_file=True):
        header = f"\n===== PROFILING | region={self.region} | mode={self.mode} ====="

        print(header)
        lines = [header]

        for k, (total, count) in self.times.items():
            avg = total / count if count > 0 else 0.0

            line = (
                f"{k:20s}: "
                f"{total:.3f}s total ({self._format_time(total)}) | "
                f"{avg:.5f}s avg"
            )

            print(line)
            lines.append(line)

        lines.append("========================\n")

        # write to file
        if write_to_file:
            with open(self.save_path, "a") as f:
                f.write("\n".join(lines) + "\n")


    

    def plot(self, color, show=False, save=True):
        if len(self.times) == 0:
            print("â� ïž� No profiling data to plot.")
            return

        # --- prepare data ---
        items = []
        for k, (total, count) in self.times.items():
            items.append((k, total, count))

        # sort by total time (descending)
        items.sort(key=lambda x: x[1], reverse=True)

        keys   = [k for k, _, _ in items]
        totals = [t / 60.0 for _, t, _ in items]  # convert to minutes
        avgs   = [(t / c) if c > 0 else 0 for _, t, c in items]

        # --- plot ---
        plt.figure(figsize=(8, 4 + 0.3 * len(keys)))

        y_pos = range(len(keys))
        plt.barh(y_pos, totals, color=color)

        plt.yticks(y_pos, keys)
        plt.xlabel("Time (minutes)")
        plt.title(f"Profiler | {self.region} | {self.mode}")

        plt.gca().invert_yaxis()  # biggest on top

        # annotate bars
        for i, (t, avg) in enumerate(zip(totals, avgs)):
            plt.text(t, i, f" {t:.2f} min | {avg:.3f}s avg", va="center")

        plt.tight_layout()

        # --- save ---
        if save:
            plot_path = self.save_path.replace(".log", ".png")
            plt.savefig(plot_path, dpi=150)
            print(f"ð��� Saved profiler plot â�� {plot_path}")

        if show:
            plt.show()

        plt.close()



def compute_mass_time_series(
    nc_path: str,
    tracer_var: str = "mytrc",
    area_var: str = "area",
    z_idxs=None,
    t_start: int = 0,
    t_end: int = None,
    outdir: str = "artifacts/mass_diagnostics",
    region: str = "unknown"
):
    """
    Compute and plot total tracer mass over time and mass drift.

    mytrc is assumed to already include thickness (MYTRC * e3t).
    """
    os.makedirs(outdir, exist_ok=True)

    print(f"@UTILS.PY: Z_IDXS {z_idxs}")

    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        trc = ds[tracer_var]
        area = ds[area_var].values.astype(np.float64)

        if t_end is None:
            t_end = trc.sizes["time_counter"]

        masses = []

        for t in range(t_start, t_end):
            slice_t = trc.isel(time_counter=t)

            if z_idxs is not None:
                slice_t = slice_t.isel(olevel=z_idxs)

            trc_np = slice_t.values.astype(np.float64)
            

            # Broadcast area to (Z, Y, X)
            area_np = np.broadcast_to(area[None, :, :], trc_np.shape)

            valid = np.isfinite(trc_np) & np.isfinite(area_np)

            total_mass = np.sum(trc_np[valid] * area_np[valid])
            
            
            
            
            masses.append(total_mass)
        print("Mass calc shapes:", "trc_np:", trc_np.shape, "area_np:", area_np.shape)
        masses = np.array(masses)
        drift = (masses - masses[0]) / masses[0]

        # === Plot absolute mass ===
        plt.figure(figsize=(10, 4))
        plt.plot(masses, lw=2)
        plt.xlabel("Time index")
        plt.ylabel("Total tracer mass [mol]")
        plt.title(f"{region} – Total tracer mass over time")
        plt.grid(alpha=0.3)
        out_mass = os.path.join(outdir, f"{region}_mass_timeseries.png")
        plt.tight_layout()
        plt.savefig(out_mass, dpi=150)
        plt.close()

        # === Plot relative drift ===
        plt.figure(figsize=(10, 4))
        plt.plot(drift * 100, lw=2)
        plt.axhline(0, color="k", ls="--", lw=2)
        plt.xlabel("Time index")
        plt.ylabel("Mass drift [%]")
        plt.title(f"{region} – Relative tracer mass drift")
        plt.grid(alpha=0.3)
        out_drift = os.path.join(outdir, f"{region}_mass_drift.png")
        plt.tight_layout()
        plt.savefig(out_drift, dpi=150)
        plt.close()

        print(f"📈 Saved mass time series → {out_mass}")
        print(f"📉 Saved mass drift plot  → {out_drift}")

    return masses, drift


def compute_ssim(pred, target):
    """
    Compute a simplified SSIM between two PyTorch tensors or NumPy arrays.
    Works even if scikit-image is unavailable.
    """
    # Convert tensors to NumPy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure float32 dtype
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    # Normalize
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)

    # Apply Gaussian blur
    pred_blur = gaussian_filter(pred, sigma=1)
    target_blur = gaussian_filter(target, sigma=1)

    # Compute local mean and variance
    mu_pred = pred_blur.mean()
    mu_target = target_blur.mean()
    sigma_pred = pred_blur.var()
    sigma_target = target_blur.var()
    sigma_cross = ((pred_blur - mu_pred) * (target_blur - mu_target)).mean()

    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    return float(ssim)






def compute_basic_metrics(y_true, y_pred):
    """Compute RMSE, MAE, and R² metrics manually without torchmetrics."""
    y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)

    # Flatten for simplicity
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Filter out NaN or inf
    valid_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]

    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"RMSE": rmse, "MAE": mae, "R2": r2}



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




def masked_mse_loss(pred, target, mask=None):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        mask = torch.nan_to_num(mask, nan=0.0)
        diff = (pred - target) ** 2 * mask
        denom = torch.sum(mask) + 1e-8  # prevent divide-by-zero
        return torch.sum(diff) / denom
    else:
        diff = (pred - target) ** 2
        return torch.mean(diff)







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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
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
            ds = xr.open_dataset(path, engine="netcdf4")
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


#def save_checkpoint(path, model, optimizer=None, epoch=None):
#    os.makedirs(os.path.dirname(path), exist_ok=True)
#    state = {"model_state_dict": model.state_dict()}
#    if optimizer:
#        state["optimizer_state_dict"] = optimizer.state_dict()
#    if epoch is not None:
#        state["epoch"] = epoch
#    torch.save(state, path)

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
    """
    Save model and optimizer state with epoch metadata.
    Creates directory automatically.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"✅ Checkpoint saved to {path}")





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



# ====== METRICS ======
def compute_metrics(preds, targets, mask=None):
    """Compute RMSE, MAE, SSIM, and R² metrics."""
    with torch.no_grad():
        preds = preds.float()
        targets = targets.float()
        if mask is not None:
            mask = mask.float()
            preds = preds * mask
            targets = targets * mask

        mse = ((preds - targets) ** 2).mean().item()
        rmse = math.sqrt(mse)
        mae = torch.abs(preds - targets).mean().item()

        # SSIM: normalize inputs [0,1]
        pmin, pmax = preds.min(), preds.max()
        tmin, tmax = targets.min(), targets.max()
        if (pmax - pmin) > 1e-8 and (tmax - tmin) > 1e-8:
            ssim = ssim_fn(
                (preds - pmin) / (pmax - pmin + 1e-8),
                (targets - tmin) / (tmax - tmin + 1e-8)
            ).item()
        else:
            ssim = 0.0

        # R² = 1 - SS_res / SS_tot
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return {
            "RMSE": rmse,
            "MAE": mae,
            "SSIM": float(ssim),
            "R2": float(r2.item() if torch.is_tensor(r2) else r2),
        }

def composite_loss(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mae = torch.mean(torch.abs(pred - target))
    grad_loss = torch.mean((torch.gradient(pred)[0] - torch.gradient(target)[0]) ** 2)
    return mse + 0.5 * mae + 0.1 * grad_loss

def get_device(cfg):
    hw = cfg.hardware

    if hw.device == "cpu":
        return torch.device("cpu")
    elif hw.device == "cuda":
        return torch.device("cuda")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader_settings(cfg, device):
    hw = cfg.hardware

    if device.type == "cuda":
        return {
            "batch_size": hw.batch_size_gpu,
            "num_workers": hw.num_workers_gpu,
            "pin_memory": hw.pin_memory,
        }
    else:
        return {
            "batch_size": hw.batch_size_cpu,
            "num_workers": hw.num_workers_cpu,
            "pin_memory": False,
        }
