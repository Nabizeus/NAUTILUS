import os
import json
import yaml
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


# ===============================================================
# 🔧 CONFIGURATION HELPERS
# ===============================================================

def load_config(path: str):
    """
    Load YAML configuration file into OmegaConf.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


# ===============================================================
# 📊 NORMALIZATION UTILS
# ===============================================================

def load_or_compute_norm_stats(config, tracer_datasets, neighbor_datasets, out_path):
    """
    Compute or load normalization statistics (mean/std) for tracer and velocity data.
    Supports both separate U/V/W files and combined neighbor datasets.
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
                raise KeyError(f"Tracer var '{varname}' not in dataset. Found: {list(ds.data_vars)}")
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
        print("🔹 Using separate velocity files for U, V, W")
        for path, varname in zip(vel_paths, [u_var, v_var, w_var]):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Velocity file not found: {path}")
            ds = xr.open_dataset(path)
            if varname not in ds:
                raise KeyError(f"Variable '{varname}' not in {path}. Found: {list(ds.data_vars)}")
            arr = ds[varname].values.astype(np.float32)
            stats[f"{varname}_mean"] = float(np.nanmean(arr))
            stats[f"{varname}_std"] = float(np.nanstd(arr) + 1e-6)
            ds.close()

    # Case 2: Combined neighbor datasets
    else:
        print("🔹 Using combined neighbor datasets")
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
# 🎨 VISUALIZATION UTILITIES
# ===============================================================

def visualize_field(field, title="Field", cmap="viridis"):
    """Quick 2D plot of a field (e.g., tracer slice)."""
    plt.figure(figsize=(6, 5))
    plt.imshow(field, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def visualize_forecast(true_field, pred_field, title_prefix=""):
    """
    Visualize a comparison between ground truth and forecast.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(true_field, cmap="viridis")
    plt.title(f"{title_prefix} True")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(pred_field, cmap="viridis")
    plt.title(f"{title_prefix} Predicted")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# ===============================================================
# 💾 CHECKPOINT UTILS
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

