from __future__ import annotations
import os
import csv
import time
from pathlib import Path
from typing import Optional, List
import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm


import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from omegaconf import DictConfig

# imports from your package
# make sure src is on PYTHONPATH (python -m src.train ensures that)
from src.data import WindowedOceanDataset
from src.data_v2 import (RegionData, MaskedOceanDataset, region_cfg_by_name, stats_path,
                         model_name as v3_model_name, read_boxes, compute_stats, save_stats,
                         split_bounds, n_timesteps, read_depths)
from src.model_conv2d_asym_hybrid import OceanConv2DAsymXPU
from src.model_light import build_model
from src.utils import load_config, set_seed, compute_basic_metrics, composite_loss
from src.utils import compute_ssim as ssim
# for automatic configuration between CPU and GPU
from src.utils import get_device, get_dataloader_settings
from src.utils import SimpleProfiler

import datetime


from src.dashboard import plot_training_dashboard



def gradient_loss(preds, targets, mask=None):

    def grads(x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    # --- gradients ---
    pgx, pgy = grads(preds)
    tgx, tgy = grads(targets)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # 🔥 CRITICAL: align BEFORE slicing
        H = min(preds.shape[-2], mask.shape[-2])
        W = min(preds.shape[-1], mask.shape[-1])

        preds   = preds[..., :H, :W]
        targets = targets[..., :H, :W]
        mask    = mask[..., :H, :W]

        # recompute grads on aligned tensors
        pgx, pgy = grads(preds)
        tgx, tgy = grads(targets)

        # now safe
        # v2: a gradient counts only if BOTH neighbouring cells are ocean
        # (before, one ocean cell was enough, so gradients across the coast
        # and across the sea floor were trained)
        mx = mask[..., :, 1:] * mask[..., :, :-1]
        my = mask[..., 1:, :] * mask[..., :-1, :]

        gx_loss = ((pgx - tgx)**2 * mx).sum() / (mx.sum() + 1e-8)
        gy_loss = ((pgy - tgy)**2 * my).sum() / (my.sum() + 1e-8)

    else:
        gx_loss = F.mse_loss(pgx, tgx)
        gy_loss = F.mse_loss(pgy, tgy)

    return gx_loss + gy_loss


def combined_loss(preds, targets, mask=None, alpha_grad=0.5):
    mse = masked_mse_loss(preds, targets, mask)
    grad = gradient_loss(preds, targets, mask)
    return mse + alpha_grad * grad


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

def _checkpoint_prefix(cfg):
    """
    Prefix used in checkpoint filenames, so different configs training
    the SAME region (e.g. a depth-budget sweep: 0-2000m vs 0-4000m)
    don't silently overwrite each other's checkpoint. Set a distinct
    `training.experiment_name` per sweep config to get distinct files;
    leaving it unset/blank preserves the old `{region}_best_model.pt`
    naming for single-config usage.
    """
    experiment_tag = getattr(cfg.training, "experiment_name", None)
    return f"{experiment_tag}_" if experiment_tag else ""


def build_depth_training_mask(cfg, depths, device):
    """
    Build a depth mask according to cfg.data.depth_training.

    Returns:
        mask: shape (Z,) with 1 for depths contributing to loss,
              0 for depths ignored by loss.
    """

    depth_cfg = cfg.data.get("depth_training", None)

    # Backward-compatible default: train on all depths
    if depth_cfg is None:
        return torch.ones(
            len(depths),
            dtype=torch.float32,
            device=device
        )

    mode = depth_cfg.get("mode", "all")

    if mode == "all":
        valid = np.ones(len(depths), dtype=bool)

    elif mode == "range":
        min_depth = float(depth_cfg.get("min_depth_m", -np.inf))
        max_depth = float(depth_cfg.get("max_depth_m", np.inf))

        valid = (
            (depths >= min_depth) &
            (depths <= max_depth)
        )

    elif mode == "indices":
        indices = depth_cfg.get("indices", [])

        valid = np.zeros(len(depths), dtype=bool)

        for idx in indices:
            idx = int(idx)

            if 0 <= idx < len(depths):
                valid[idx] = True
            else:
                raise ValueError(
                    f"depth_training index {idx} is outside "
                    f"available range 0..{len(depths)-1}"
                )

    else:
        raise ValueError(
            f"Unknown depth_training mode: {mode}. "
            f"Expected 'all', 'range', or 'indices'."
        )

    mask = torch.from_numpy(valid.astype(np.float32)).to(device)

    print("\n🌊 ===== DEPTH TRAINING MASK =====")
    print(f"mode       : {mode}")
    print(f"levels     : {len(depths)}")
    print(f"active     : {int(valid.sum())}")
    print(f"ignored    : {int((~valid).sum())}")

    if valid.any():
        active_indices = np.where(valid)[0]
        print(
            f"depth range: "
            f"{depths[active_indices[0]]:.2f} - "
            f"{depths[active_indices[-1]]:.2f} m"
        )

    print("=================================\n")

    return mask




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

def debug_visualize_sample(inputs: torch.Tensor, region_name: str, number: int,
                           out_dir="artifacts/debug_inputs"):
    """
    Visualize model inputs with multi-depth support.

    Supports:
      (B,T,C,H,W)
      (B,T,C,Z,H,W)

    For Z-dim: plots Z-1 | Z | Z+1 side-by-side (if available)

    Output layout:
      rows    = variables
      columns = time * depth_slices
    """
    
    
    

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------
    # Select first batch
    # -----------------------
    if inputs.dim() == 6:
        x = inputs[0].detach().cpu().numpy()  # (T,C,Z,H,W)
        T, C, Z, H, W = x.shape
        z_center = Z // 2
        depth_mode = True


        z_indices = sorted(set([
            0,
            Z // 4,
            Z // 2,
            3 * Z // 4,
            Z - 1
        ]))


    elif inputs.dim() == 5:
        x = inputs[0].detach().cpu().numpy()  # (T,C,H,W)
        T, C, H, W = x.shape
        z_indices = [None]
        z_center = None
        depth_mode = False

    else:
        raise ValueError(f"Unsupported input shape {inputs.shape}")

    depth_count = len(z_indices)

    base_vars = ["Tracer", "U", "V", "W"]
    vars_names = base_vars[:C]
    # -----------------------
    # Create plot grid
    # -----------------------
    fig, axes = plt.subplots(C, T * depth_count,
                             figsize=(T * depth_count * 1.6, C * 1.6),
                             squeeze=False)

    for c in range(C):
        for t in range(T):
            for zi, z in enumerate(z_indices):
                col = t * depth_count + zi
                ax = axes[c][col]

                if depth_mode:
                    img = x[t, c, z]
                else:
                    img = x[t, c]

                im = ax.imshow(img, origin="lower", cmap="viridis")
                ax.axis("off")

                # Titles
                if c == 0:
                    if depth_mode:
                        ax.set_title(f"t={t}\nz={z}", fontsize=7)
                    else:
                        ax.set_title(f"t={t}", fontsize=7)

                if col == 0:
                    label = vars_names[c] if c < len(vars_names) else f"ch{c}"
                    ax.set_ylabel(label, fontsize=8)

                    
    # -----------------------
    # Global title
    # -----------------------
    if depth_mode:
        plt.suptitle(
            f"{region_name} sample {number} | depths {z_indices}",
            fontsize=10
        )
    else:
        plt.suptitle(f"{region_name} sample {number}", fontsize=10)

    # -----------------------
    # Filename
    # -----------------------
    if depth_mode:
        ztag = f"z{z_center}"
    else:
        ztag = "noZ"

    out_path = os.path.join(
        out_dir,
        f"sample_{region_name}_{number:05d}_{ztag}.png"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path






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

def plot_loss_curve(train_losses: List[float], val_losses: List[float], out_path: str, region_name: str, model_setup: str):
    plt.figure(figsize=(6, 3))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curve - {region_name} | {model_setup}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def write_log_csv(csv_path: str, region_name: str, epoch: int, train_loss: float, val_loss: float):
    header = ["region", "epoch", "train_loss", "val_loss"]
    newrow = [region_name, int(epoch), float(f"{train_loss:.6g}"), float(f"{val_loss:.6g}")]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(newrow)

def flatten_depth(inputs):
    # inputs: (B, T, C, Z, H, W)
    B, T, C, Z, H, W = inputs.shape
    return inputs.view(B, T, C * Z, H, W)





def load_training_regions(cfg, group_name, region_cfgs):
    """
    v3: RegionData for every training region of one model.
    One region: as in v2 (statistics from that region, saved under its name).
    Several regions: all boxes are read in one pass over the global fields, the
    statistics are pooled over the ocean cells of all their training periods and
    saved under the group name, and every region is normalized with them.
    """
    stats_file = stats_path(cfg, group_name)
    if len(region_cfgs) == 1:
        rc = region_cfgs[0]
        return [RegionData(cfg, rc["x_range"], rc["y_range"], rc["name"], stats_file=stats_file)]
    rows = [dict(name=rc["name"], x0=int(rc["x_range"][0]), x1=int(rc["x_range"][1]),
                 y0=int(rc["y_range"][0]), y1=int(rc["y_range"][1])) for rc in region_cfgs]
    T_total = n_timesteps(cfg)
    raws = read_boxes(cfg, rows, 0, T_total)
    T = min(a.shape[0] for r in raws.values() for a in r.values())
    train_end, _, _ = split_bounds(cfg, T)
    names = [rc["name"] for rc in region_cfgs]
    oceans = [np.isfinite(raws[n]["tracer"][0]) for n in names]
    stats = compute_stats([raws[n] for n in names], oceans, read_depths(cfg), train_end,
                          lead=int(cfg.data.get("lead_steps", 1) or 1))
    save_stats(stats_file, stats)
    print(f"[Training] pooled normalization stats over {names} -> {stats_file}")
    out = []
    for rc in region_cfgs:
        out.append(RegionData(cfg, rc["x_range"], rc["y_range"], rc["name"], stats=stats,
                              raw=raws.pop(rc["name"])))
    return out


def fixed_training_patches(region_list, cfg, n_per_region, patch_size, min_ocean_frac, seed=1234):
    """
    v3 (diagnostic): a fixed set of training windows and patch positions per
    region, so the training loss can be measured on the same samples every
    epoch with the model in evaluation mode (no dropout, BatchNorm running stats).
    """
    rng = np.random.default_rng(seed)
    items = []
    for ri, rd in enumerate(region_list):
        ds = MaskedOceanDataset(rd, "train", cfg, patch_size=patch_size, min_ocean_frac=min_ocean_frac)
        ts = rng.choice(ds.indices, size=min(n_per_region, len(ds.indices)), replace=False)
        for t in ts:
            y0, x0 = ds.corners[int(rng.integers(len(ds.corners)))]
            items.append((ri, int(t), slice(int(y0), int(y0) + patch_size),
                          slice(int(x0), int(x0) + patch_size)))
    return items


def fixed_patch_loss(model, items, region_list, cfg, device, depth_training_mask, lambda_grad,
                     batch_size=4):
    """Mean loss over the fixed patches, model in eval mode, per region and overall."""
    t_in = int(cfg.data.window.input_steps)
    was_training = model.training
    model.eval()
    per_region = {rd.name: [] for rd in region_list}
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            for ri, t, ys, xs in chunk:
                x, y, m = region_list[ri].window(t, ys, xs, t_in)
                xb = torch.from_numpy(np.ascontiguousarray(x))[None].to(device)
                yb = torch.from_numpy(np.ascontiguousarray(y))[None].to(device)
                mb = torch.from_numpy(np.ascontiguousarray(m))[None].to(device)
                with torch.cuda.amp.autocast(enabled=bool(cfg.hardware.use_amp) and device.type == "cuda"):
                    pb = model(xb)
                lm = mb.unsqueeze(1) if mb.dim() < pb.dim() else mb
                lm = lm * depth_training_mask.view(1, 1, -1, 1, 1)
                loss = combined_loss(pb.float(), yb.float(), lm, alpha_grad=lambda_grad)
                per_region[region_list[ri].name].append(float(loss))
    if was_training:
        model.train()
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in per_region.items()}
    out["all"] = float(np.mean([v for vs in per_region.values() for v in vs]))
    return out


# ----------------------
# Training flow
# ----------------------


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/conv2dlstm_s323D_fullcolumn_GPU.yaml",
        help="Path to the YAML config file for this run "
             "(lets a config sweep pick a different depth-training range "
             "per invocation without editing this script).",
    )
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")

    mode = "Train" 

    print(f"[Training] 🧠 Using pure YAML config — {args.config}")

    # 1️⃣ Load configuration
    cfg = load_config(args.config)
    print("[Training]  🔧 Starting training script (merged debug/vis).")
    
    device = get_device(cfg)
    use_amp = (device.type == "cuda") and getattr(cfg.hardware, "use_amp", False)

    print(f"[Training] 🧠 Using device: {device}")
    #device = torch.device("cuda" if torch.cuda.is_available() and getattr(cfg.training, "use_cuda", False) else "cpu")
    #print(f"[Training] 🧠 Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")    

    set_seed(int(getattr(cfg.training, "seed", 42)))

    # v2: train only on cfg.data.training_regions (names of boxes in
    # cfg.data.regions); the other boxes are for evaluation only.
    if not cfg.data.get("regions", None):
        raise ValueError("No regions defined in config.data.regions - please define per-region boxes.")
    training_names = list(cfg.data.get("training_regions", [r["name"] for r in cfg.data.regions]))
    regions = [region_cfg_by_name(cfg, n) for n in training_names]
    print(f"[Training] Training regions: {training_names}")

    # global artifacts/checkpoints folder
    ckpt_dir = Path(getattr(cfg.training, "checkpoint_dir", "checkpoints"))
    artifacts_dir = Path(getattr(cfg, "artifacts_dir", "artifacts") if "artifacts_dir" in cfg else "artifacts")
    logs_dir = Path(getattr(cfg, "log_dir", "logs") if "log_dir" in cfg else "logs")
    # v2: one sub-folder per experiment, so runs in the same directory do not
    # append to each other's CSV logs
    exp_tag = cfg.training.get("experiment_name", "") or "run"
    artifacts_dir = artifacts_dir / exp_tag
    logs_dir = logs_dir / exp_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Options (you can put in YAML)
    debug_save_every_n_batches = int(getattr(cfg.debug, "save_every_n_batches", 10)) if "debug" in cfg else 10
    debug_save_first_k_batches = int(getattr(cfg.debug, "save_first_k_batches", 200)) if "debug" in cfg else 200


    # v3: with data.training_group all training regions train ONE model, named
    # after the group; without it every training region gets its own model (v2)
    if cfg.data.get("training_group", None):
        groups = [(v3_model_name(cfg), regions)]
    else:
        groups = [(rc["name"], [rc]) for rc in regions]

    for region_name, group_cfgs in groups:
        start_time = time.time() # <-- Start timer


        prof = SimpleProfiler(
            save_path=f"profiler/Profile_{mode}_{region_name}_{ts}.log",
            region=region_name,
            mode=mode
        )


        prof.start("train ds pre-loader")
        print(f"[Training] \n🌍 Starting training for region: {region_name}")

        # build per-region datasets (v2: read once, 3-D ocean mask, ocean-only
        # train-period stats, mask channel, random patches for training)
        # v3: several regions -> pooled statistics, one dataset per region,
        # concatenated for training (every region has the same number of
        # training windows, so each gives an equal share of the patches)
        region_list = load_training_regions(cfg, region_name, group_cfgs)
        region_data = region_list[0]
        patch_size = cfg.training.get("patch_size", None)
        patch_min_frac = cfg.training.get("patch_min_ocean_frac", 0.3)
        train_parts = [MaskedOceanDataset(rd, "train", cfg, patch_size=patch_size,
                                          min_ocean_frac=patch_min_frac) for rd in region_list]
        train_ref = train_parts[0]          # depths, channels, patch size: the same in every part
        train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        val_sets = [(rd.name, MaskedOceanDataset(rd, "val", cfg)) for rd in region_list]
        val_ds = val_sets[0][1]
        test_ds  = MaskedOceanDataset(region_data, "test", cfg)
        if "artifacts" in str(artifacts_dir):
            for rd in region_list:
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                axs[0].imshow(rd.land_mask, origin="lower", cmap="gray")
                axs[0].set_title(f"{rd.name}: surface ocean (white)")
                im = axs[1].imshow(np.where(rd.land_mask, rd.depths[np.maximum(rd.n_ocean_levels - 1, 0)], np.nan),
                                   origin="lower", cmap="viridis_r")
                axs[1].set_title("depth of deepest ocean level (m)")
                plt.colorbar(im, ax=axs[1])
                fig.savefig(artifacts_dir / f"ocean_mask3d_{rd.name}.png", dpi=130)
                plt.close(fig)

        # v3 (diagnostic): fixed training patches, loss measured in eval mode each epoch
        n_fixed = int(cfg.training.get("fixed_patches_per_region", 16))
        fixed_items = (fixed_training_patches(region_list, cfg, n_fixed, int(patch_size), patch_min_frac)
                       if (n_fixed > 0 and patch_size) else [])

        #input_channels = train_ds.n_channels   # ← best practice

        

        depth_training_mask = build_depth_training_mask(
            cfg,
            train_ref.depths,
            device
        )

        # v2: increment target -> per-depth scale to rebuild the full field
        inc_scale = None
        if region_data.target_mode == "increment":
            inc_scale = torch.as_tensor(region_data.inc_scale, device=device).view(1, 1, -1, 1, 1)
        print(f"[Training] target: {region_data.target_mode}; input channels: "
              f"{region_data.channels}")


        model_name = cfg.model.name

        num_levels = cfg.data.vertical_levels
        num_epochs = cfg.training.epochs
        num_inputs = cfg.data.window.input_steps
 
        
        print(f"[Training] Train samples: {len(train_ds)}")
        print(f"[Training] Val samples: {sum(len(d) for _, d in val_sets)}")
        print(f"[Training] Test samples: {len(test_ds)}")
        print(f"[Training] Model Name", model_name)
        print(f"[Training] N Depth Levels: {num_levels}")
        print(f"[Training] N Epochs: {num_epochs}")
        print(f"[Training] N Inputs: {num_inputs}")


        # --- Debug output for splits ---
        for split, dataset in [("train", train_ref), ("val", val_ds), ("test", test_ds)]:
            print(f"[Training] 🧩 Dataset '{split}' — indices {dataset.t_start}:{dataset.t_end}, total {len(dataset)} samples per region "
                  f"({len(region_list)} region(s): {[rd.name for rd in region_list]})")
 
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.hardware.use_amp))

        dl_cfg = get_dataloader_settings(cfg, device)

        train_loader = DataLoader(
            train_ds,
            batch_size=dl_cfg["batch_size"],
            shuffle=True,
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"]
        )

        # v3: one validation loader per region (whole box, boxes differ in size)
        val_loaders = [(name, DataLoader(
            ds,
            batch_size=int(cfg.training.get("val_batch_size", dl_cfg["batch_size"])),
            shuffle=False,
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"]
        )) for name, ds in val_sets]

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"]
        )


        #train_loader = DataLoader(train_ds, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=int(getattr(cfg.training, "num_workers", 0)))
        #val_loader = DataLoader(val_ds, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=int(getattr(cfg.training, "num_workers", 0)))
        #test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=int(getattr(cfg.training, "num_workers", 0)))


        



        # Create model for this region (pass H/W if dataset made them available)
        H = getattr(train_ref, "H", None)
        W = getattr(train_ref, "W", None)
        print("[Training] Dataset channels:", train_ref.n_channels)
        print("[Training] Dataset depths:", train_ref.n_depths)

        #input_channels = train_ds.n_channels*train_ds.n_depths
       
        print("[Training] Channels per depth:", train_ref.n_channels)
        print("[Training] Depth levels:", train_ref.n_depths)
 
        



        


        model = build_model(cfg, input_channels=train_ref.n_channels).to(device)   # v6: cfg.model.arch
        print(f"[Training] model.arch {cfg.model.get('arch', 'conv2dlstm')}: "
              f"{sum(p.numel() for p in model.parameters()):,} parameters")
        


        #model = OceanConv2DAsymXPU(
        #    input_channels=train_ds.n_channels,
        #    hidden_dim=64,
        #    max_below=cfg.model.max_below,
        #    max_above=cfg.model.max_below
        #).to(device)


        optimizer = optim.Adam(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=1e-5,
            )
        
        

        # v2c: scheduler settings from training.lr_scheduler (defaults = old behaviour)
        sched_cfg = cfg.training.get("lr_scheduler", None) or {}
        min_lr = float(sched_cfg.get("min_lr", 0.0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 3)),
            min_lr=min_lr,
            verbose=False
        )
        grad_clip = float(cfg.training.get("grad_clip_norm", 1.0))
        print(f"[Training] lr {float(cfg.training.lr):.1e}, ReduceLROnPlateau factor "
              f"{float(sched_cfg.get('factor', 0.5))} patience {int(sched_cfg.get('patience', 3))} "
              f"min_lr {min_lr:.1e}, grad clip {grad_clip}")


        best_val_loss = float("inf")
        best_train_loss = float("inf")

        early_stop = cfg.training.get("early_stopping", True)
        patience = cfg.training.get("early_stopping_patience", 10)
        delta = cfg.training.get("early_stopping_delta", 1e-5)
        min_epochs = int(cfg.training.get("min_epochs", 0))   # v2: never stop before this
        # v2c: convergence check on the training loss. Early stopping needs
        # (a) `patience` epochs without a better validation loss, and
        # (b) a training loss back at its best level: mean of the last `window` epochs
        #     at most `tol` (relative) above the lowest `window`-epoch mean so far.
        # If (b) fails (training loss went up), training continues and the scheduler
        # lowers the LR; it also stops once the LR has reached min_lr.
        # 0 = old behaviour (validation only).
        conv_window = int(cfg.training.get("convergence_window", 0))
        conv_tol = float(cfg.training.get("convergence_tol", 0.01))
        stop_reason = "maximum number of epochs reached"

        epochs_without_improvement = 0
        best_epoch = 0




        # training bookkeeping
        train_losses, val_losses = [], []
        csv_log_path = logs_dir / f"training_log_{region_name}.csv"
           
        model.train()
        optimizer.zero_grad()
        

        print("[Training] Start train loop per epoch.....")

 
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epochs {cfg.training.epochs} [Train]")
                    
        prof.stop()
       

        # v3: the per-batch training RMSE/MAE/R2/SSIM are gone (full-field R2
        # and SSIM are ~1 for this target and hide the real performance);
        # validation skill against "no change" per region instead
        history = {
           "train_loss": [],
            "val_loss": [],
            "lr": [],
            "val_rmse": [],
            "val_rmse_persistence": [],
            "val_skill": [],
            "train_loss_fixed_eval": [],
        }
        for rd in region_list:
            history[f"val_skill_{rd.name}"] = []
        n_skipped_total = 0
        
        # per-epoch debug frames aggregator (for gif)
        for epoch in range(1, int(cfg.training.epochs) + 1):
            
            running_loss = 0.0
            grad_norms = []
            n_skipped = 0
            # v3 fix: back to training mode every epoch. Before, model.train() was
            # called only once before epoch 1 and validation left the model in
            # eval mode, so from epoch 2 on the model trained with dropout off and
            # BatchNorm frozen at its epoch-1 running statistics.
            model.train()


            #if epoch % 10 == 0:        

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{cfg.training.epochs} [Train]")
    

            batch_count = 0
            debug_frames = []  # store saved debug png paths for this epoch
            
            for batch_idx, batch in pbar:

                prof.start("device_transfer")


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
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)

                
                prof.stop()

                #2D LSTM
                #if inputs.dim() == 6: 
                    # inputs shape: (B, T, C*Z, H, W) Flattend Channels & Z Levels
                    #inputs_2d = flatten_depth(inputs)
                #else:
                    #inputs_2d = inputs  # already 2D or model expects full volume

                inputs_2d = inputs

                # safety
                inputs_2d = torch.nan_to_num(inputs_2d, nan=0.0, posinf=1e6, neginf=-1e6)
                targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=cfg.hardware.use_amp):

                    prof.start("forward")
                    preds = model(inputs_2d)
                    prof.stop()



                    #if preds.dim() == 3:
                    #    preds = preds.unsqueeze(1)

                    #if targets.dim() == 3:
                    #    targets = targets.unsqueeze(1)
                    

                    lambda_grad = cfg.training.get("lambda_grad", 0.5)

                    prof.start("loss_backward")


                    # --------------------------------------------------
                    # Combine spatial ocean mask with depth-training mask
                    # --------------------------------------------------

                    loss_mask = mask

                    if loss_mask is not None:

                        # Make sure mask is compatible with (B,1,Z,H,W)
                        while loss_mask.dim() < preds.dim():
                            loss_mask = loss_mask.unsqueeze(1)

                        depth_mask = depth_training_mask.view(
                            1, 1, -1, 1, 1
                        )

                        loss_mask = loss_mask.to(preds.device) * depth_mask

                    else:
                        # No land mask: use only depth-training selection
                        loss_mask = depth_training_mask.view(
                            1, 1, -1, 1, 1
                        )

                    loss = combined_loss(preds, targets, loss_mask, alpha_grad=lambda_grad)
                    
                if batch_idx == 0 and epoch == 1:
                    print("\n🧪 ===== TRAIN DEBUG =====")
                    print("inputs:", inputs_2d.shape)
                    print("preds :", preds.shape)
                    print("targets:", targets.shape)
                    if mask is not None:
                        print("mask  :", mask.shape)
                    print("==========================\n")                                
                   
                

                if batch_idx % 200 == 0 and epoch == 1:
                     
                    print("\n🧪 TRAIN SCALE DEBUG")
                    print("targets:", targets.min().item(), targets.max().item())
                    print("preds  :", preds.min().item(), preds.max().item()) 
                    print("preds :", preds.shape)
                    print("targets:", targets.shape)
                    print("mask:", None if mask is None else mask.shape) 

                
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ Skipping batch due to NaN/Inf loss.")
                    n_skipped += 1
                    prof.stop()
                    continue
                # v3 fix: use the GradScaler with AMP (before it was created but
                # never used, so small fp16 gradients could underflow to zero)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip
                )
                if torch.isfinite(gnorm):
                    grad_norms.append(float(gnorm))
                else:
                    n_skipped += 1           # scaler skips this step and lowers its scale
                scaler.step(optimizer)
                scaler.update()
                prof.stop()

                running_loss += float(loss.detach().cpu().item())

                batch_count += 1
                
                # Debug visualization save
                if (batch_idx % debug_save_every_n_batches) == 0 and (epoch % 10 == 0):
                              
                    saved_png = debug_visualize_sample(inputs, region_name, epoch * 10000 + batch_idx)
                    debug_frames.append(saved_png)

            train_epoch_loss = running_loss / max(1, batch_count)
            train_losses.append(train_epoch_loss)
            history["train_loss"].append(train_epoch_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            n_skipped_total += n_skipped

            # v3 (diagnostic): loss on fixed training patches, eval mode
            fixed = (fixed_patch_loss(model, fixed_items, region_list, cfg, device,
                                      depth_training_mask, cfg.training.get("lambda_grad", 0.5))
                     if fixed_items else {"all": float("nan")})
            history["train_loss_fixed_eval"].append(fixed["all"])
            gn = np.asarray(grad_norms) if grad_norms else np.asarray([np.nan])
            print(f"[Training] epoch {epoch:03d} training diagnostics: loss (train mode, random patches) "
                  f"{train_epoch_loss:.6f} | loss on fixed patches, eval mode {fixed['all']:.6f} "
                  + " ".join(f"{k} {v:.4f}" for k, v in fixed.items() if k != "all")
                  + f" | grad norm mean {np.nanmean(gn):.3e} max {np.nanmax(gn):.3e} | "
                  f"AMP scale {scaler.get_scale() if scaler.is_enabled() else 1.0:.1f} | skipped steps {n_skipped}",
                  flush=True)
            with open(logs_dir / f"train_diagnostics_{region_name}.csv", "a", newline="") as f:
                wd = csv.writer(f)
                names_fixed = [rd.name for rd in region_list]
                if f.tell() == 0:
                    wd.writerow(["epoch", "train_loss", "train_loss_fixed_eval"]
                                + [f"fixed_eval_{n}" for n in names_fixed]
                                + ["grad_norm_mean", "grad_norm_max", "amp_scale", "skipped_steps", "lr"])
                wd.writerow([epoch, train_epoch_loss, fixed["all"]]
                            + [fixed.get(n, float("nan")) for n in names_fixed]
                            + [float(np.nanmean(gn)), float(np.nanmax(gn)),
                               scaler.get_scale() if scaler.is_enabled() else 1.0, n_skipped,
                               optimizer.param_groups[0]["lr"]])


            #print(f"📊 Epoch {epoch} | Loss: {train_epoch_loss:.4f} | RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f} | SSIM: {avg_ssim:.4f} | R²: {avg_r2:.4f}")

            # --- Validation ---
            # v3: every training region separately (whole box); the validation
            # loss is the mean over the regions, so each region counts equally
            model.eval()
            val_per_region = {}
            v_sse_all = v_sse_p_all = v_n_all = 0.0

            with torch.no_grad():
                for vname, val_loader in val_loaders:
                    val_running = 0.0
                    val_count = 0
                    val_zero_running = 0.0                       # loss of predicting "no change"
                    v_sse = v_sse_p = v_n = 0.0                  # full-field sums over ocean cells
                    for batch in val_loader:
                        if len(batch) == 2:
                            v_inputs, v_targets = batch
                            v_mask = None
                        else:
                            v_inputs, v_targets, v_mask = batch

                        v_inputs = torch.nan_to_num(v_inputs, nan=0.0, posinf=1e6, neginf=-1e6)
                        v_targets = torch.nan_to_num(v_targets, nan=0.0, posinf=1e6, neginf=-1e6)

                        v_inputs = v_inputs.to(device, non_blocking=True)
                        v_targets = v_targets.to(device, non_blocking=True)

                        prof.start("val_forward")
                        v_preds = model(v_inputs)
                        prof.stop()

                        if epoch == 1 and val_count == 1:
                            print(f"\n🧪 ===== VAL DEBUG ({vname}) =====")
                            print("v_inputs :", v_inputs.shape)
                            print("v_preds  :", v_preds.shape)
                            print("v_targets:", v_targets.shape)
                            print("========================\n")

                        if v_preds.dim() == 3:
                            v_preds = v_preds.unsqueeze(1)
                        if v_targets.dim() == 3:
                            v_targets = v_targets.unsqueeze(1)
                        if v_mask is not None:
                            v_mask = v_mask.to(device)

                        prof.start("val_loss")
                        # validation spatial mask combined with the depth mask
                        v_loss_mask = v_mask
                        if v_loss_mask is not None:
                            while v_loss_mask.dim() < v_preds.dim():
                                v_loss_mask = v_loss_mask.unsqueeze(1)
                            v_loss_mask = v_loss_mask * depth_training_mask.view(1, 1, -1, 1, 1)
                        else:
                            v_loss_mask = depth_training_mask.view(1, 1, -1, 1, 1)

                        v_loss = combined_loss(v_preds, v_targets, v_loss_mask, alpha_grad=lambda_grad)
                        prof.stop()

                        val_running += float(v_loss.detach().cpu().item())
                        val_count += 1

                        # full-field error of model and of "no change", ocean cells
                        v_last = v_inputs[:, -1:, 0].float()               # newest input tracer
                        v_p = v_preds.float()
                        v_t = v_targets.float()
                        if inc_scale is not None:
                            v_p = v_last + v_p * inc_scale
                            v_t = v_last + v_t * inc_scale
                            if epoch == 1:
                                val_zero_running += float(combined_loss(
                                    torch.zeros_like(v_preds), v_targets, v_loss_mask,
                                    alpha_grad=lambda_grad).item())
                        vm = v_loss_mask.expand_as(v_t).float()
                        v_sse += float((((v_p - v_t) ** 2) * vm).sum())
                        v_sse_p += float((((v_last - v_t) ** 2) * vm).sum())
                        v_n += float(vm.sum())
                    r_loss = val_running / max(1, val_count)
                    r_skill = 1.0 - v_sse / v_sse_p if v_sse_p > 0 else float("nan")
                    val_per_region[vname] = (r_loss, r_skill)
                    history[f"val_skill_{vname}"].append(r_skill)
                    v_sse_all += v_sse
                    v_sse_p_all += v_sse_p
                    v_n_all += v_n
                    if epoch == 1 and inc_scale is not None:
                        print(f"[Training] {vname}: validation loss of the 'no change' forecast: "
                              f"{val_zero_running / max(1, val_count):.6f}")
            val_epoch_loss = float(np.mean([v[0] for v in val_per_region.values()]))
            v_rmse = (v_sse_all / max(v_n_all, 1)) ** 0.5
            v_rmse_p = (v_sse_p_all / max(v_n_all, 1)) ** 0.5
            v_skill = 1.0 - v_sse_all / v_sse_p_all if v_sse_p_all > 0 else float("nan")
            history["val_rmse"].append(v_rmse)
            history["val_rmse_persistence"].append(v_rmse_p)
            history["val_skill"].append(v_skill)
            print(f"[Training] epoch {epoch:03d} validation, full field, ocean cells: "
                  f"RMSE model {v_rmse:.3e} | no change {v_rmse_p:.3e} | "
                  f"skill 1-MSE/MSE_nochange {v_skill:+.4f} | per region: "
                  + ", ".join(f"{k} loss {l:.4f} skill {sk:+.4f}" for k, (l, sk) in val_per_region.items()),
                  flush=True)
            with open(logs_dir / f"val_metrics_{region_name}.csv", "a", newline="") as f:
                wv = csv.writer(f)
                rnames = list(val_per_region)
                if f.tell() == 0:
                    wv.writerow(["epoch", "val_loss", "val_rmse", "val_rmse_nochange", "val_skill"]
                                + [f"val_loss_{n}" for n in rnames] + [f"val_skill_{n}" for n in rnames])
                wv.writerow([epoch, val_epoch_loss, v_rmse, v_rmse_p, v_skill]
                            + [val_per_region[n][0] for n in rnames] + [val_per_region[n][1] for n in rnames])
            val_losses.append(val_epoch_loss)
            history["val_loss"].append(val_epoch_loss)
            # step scheduler with validation loss
            scheduler.step(val_epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']

            with open(logs_dir / f"lr_log_{region_name}.csv", "a") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["epoch", "lr"])
                writer.writerow([epoch, current_lr])
    





            #print(f"✅ Epoch [{epoch}/{cfg.training.epochs}] Train Loss: {train_epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")

            # Save gif for debug_frames if any
            #if debug_frames:
            #    gif_path = artifacts_dir / f"debug_inputs_{region_name}_ep{epoch}.gif"
            #    animate_frames(sorted(debug_frames), str(gif_path), fps=2)

            # Save loss curve and CSV
            model_setup=f"Mod{model_name}_ep{num_epochs}_lev{num_levels}_inp{num_inputs}"
            plot_loss_curve(train_losses, val_losses, str(artifacts_dir / f"loss_curve_{region_name}_{model_setup}.png"), f"{exp_tag} {region_name}", model_setup)
            write_log_csv(str(csv_log_path), region_name, epoch, train_epoch_loss, val_epoch_loss)

            # Save checkpoint if improved
            # v2: best checkpoint = best validation loss only
            if val_epoch_loss < best_val_loss - delta:

                best_val_loss = val_epoch_loss
                best_train_loss = train_epoch_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                ckpt_path = ckpt_dir / f"{_checkpoint_prefix(cfg)}{region_name}_best_model.pt"

                save_checkpoint(
                        str(ckpt_path),
                        model,
                        optimizer=optimizer,
                        epoch=epoch)


                print(
                    f"Epoch {epoch:03d} | "
                    f"Train={train_epoch_loss:.6f} | "
                    f"Val={val_epoch_loss:.6f} | "
                    f"LR={current_lr:.2e}"
                )

                print(f"🔁 Saved best model for {region_name} -> {ckpt_path}")

            else:

                epochs_without_improvement += 1

                print(
                    f"No improvement "
                    f"({epochs_without_improvement}/{patience})"
                )


            # v2c: training loss of the last conv_window epochs relative to its best
            # (lowest) conv_window-epoch mean so far; > conv_tol means it went up
            train_trend = float("nan")
            if conv_window > 0 and len(train_losses) >= conv_window:
                means = [float(np.mean(train_losses[i - conv_window:i]))
                         for i in range(conv_window, len(train_losses) + 1)]
                lowest = min(means)
                train_trend = (means[-1] - lowest) / max(abs(lowest), 1e-12)
            print(f"[Training] epoch {epoch:03d} train {train_epoch_loss:.6f} | val {val_epoch_loss:.6f} | "
                  f"best {best_val_loss:.6f} (epoch {best_epoch}) | lr {current_lr:.2e} | "
                  f"train loss vs its best {conv_window}-epoch mean {train_trend:+.4f}")

            val_converged = epochs_without_improvement >= patience and epoch >= min_epochs
            if early_stop and val_converged and conv_window > 0:
                lr_at_floor = current_lr <= min_lr * (1 + 1e-6)
                if np.isfinite(train_trend) and train_trend <= conv_tol:
                    stop_reason = (f"converged: no better validation loss for {patience} epochs, "
                                   f"training loss at its best level (+{train_trend:.4f})")
                elif lr_at_floor:
                    stop_reason = (f"learning rate at its floor {min_lr:.1e} and no better "
                                   f"validation loss for {patience} epochs")
                else:
                    print(f"[Training] no better validation loss for {epochs_without_improvement} epochs, "
                          f"but the training loss is {train_trend:+.2%} above its best level: continuing")
                    val_converged = False
            elif early_stop and val_converged:
                stop_reason = f"no better validation loss for {patience} epochs"

            if early_stop and val_converged:

                print("\n" + "="*70)
                print("🛑 EARLY STOPPING")
                print(f"Reason         : {stop_reason}")
                print(f"Best epoch     : {best_epoch}")
                print(f"Best train loss  : {best_train_loss:.6f}")
                print(f"Best val loss  : {best_val_loss:.6f}")
                
                print("="*70)

                break


        prof.report()

        prof.plot(color="grey")


        print(f"[Training] stopped after epoch {epoch}: {stop_reason}; best epoch {best_epoch}, "
              f"best val loss {best_val_loss:.6f}")
        try:
            import json
            with open(logs_dir / f"training_summary_{region_name}.json", "w") as f:
                json.dump({"epochs_run": epoch, "best_epoch": best_epoch,
                           "best_val_loss": best_val_loss, "stop_reason": stop_reason,
                           "final_lr": current_lr, "regions": [rd.name for rd in region_list],
                           "skipped_steps": n_skipped_total,
                           "minutes": (time.time() - start_time) / 60}, f, indent=2)
        except Exception as exc:
            print(f"[Training] could not write training summary: {exc}")

        print(f"🎯 Finished region: {region_name}. Best values")
        print(
            f"Epoch {epoch:03d} | "
            f"Train={train_epoch_loss:.6f} | "
            f"Val={val_epoch_loss:.6f} | "
            f"LR={current_lr:.2e}"
        )

        end_time = time.time()     # <-- End timer
        elapsed = (end_time - start_time) / 60  # minutes
        plot_training_dashboard(
            history=history,
            profiler=prof,
            region_name=f"{exp_tag}_{region_name}",
            cfg=cfg,
            best_epoch=best_epoch,
            training_minutes=elapsed
        )



        log_training_time(region_name, elapsed, best_val_loss, cfg.training.epochs, cfg.training.batch_size)
        print(f"�~O� Training time for {region_name}: {elapsed:.2f} minutes\n")
    print("�~_~N~I Training complete.")

if __name__ == "__main__":
    main()

