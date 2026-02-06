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
import imageio.v2 as imageio

from omegaconf import DictConfig

# imports from your package
# make sure src is on PYTHONPATH (python -m src.train ensures that)
from src.data import WindowedOceanDataset
from src.model import OceanPredictor
from src.utils import load_config, set_seed, compute_basic_metrics, composite_loss

def gradient_loss(preds, targets, mask=None):
    """
    Encourage similar spatial gradients (edges) between pred and target.
    Compute simple finite differences in x & y.
    """
    def grads(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]  # (B,1,H,W-1)
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]  # (B,1,H-1,W)
        return gx, gy

    pgx, pgy = grads(preds)
    tgx, tgy = grads(targets)

    if mask is not None:
        # align mask to grads shapes
        mx = mask[:, :, :, 1:]
        my = mask[:, :, 1:, :]
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
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------
    # Select first batch
    # -----------------------
    if inputs.dim() == 6:
        x = inputs[0].detach().cpu().numpy()  # (T,C,Z,H,W)
        T, C, Z, H, W = x.shape

        z_center = Z // 2
        z_indices = [z_center]

        if z_center - 1 >= 0:
            z_indices.insert(0, z_center - 1)
        if z_center + 1 < Z:
            z_indices.append(z_center + 1)

        depth_mode = True

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

    vars_names = []
    for d in range(C // len(base_vars)):
        for v in base_vars:
            if d == 0:
                vars_names.append(f"{v} (Z)")
            elif d == 1:
                vars_names.append(f"{v} (Z+1)")
            elif d == 2:
                vars_names.append(f"{v} (Z-1)")
            else:
                vars_names.append(f"{v} (Z{d})")


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

def flatten_depth(inputs):
    # inputs: (B, T, C, Z, H, W)
    B, T, C, Z, H, W = inputs.shape
    return inputs.view(B, T, C * Z, H, W)





# ----------------------
# Training flow
# ----------------------


def main():



    print("[Training] 🧠 Using pure YAML config —")

    # 1️⃣ Load configuration
    cfg = load_config("config/default_cnnlstm.yaml")
    print("[Training]  🔧 Starting training script (merged debug/vis).")
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(cfg.training, "use_cuda", False) else "cpu")
    print(f"[Training] 🧠 Using device: {device}")
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
        print(f"[Training] \n🌍 Starting training for region: {region_name}")

        # build per-region datasets
        train_ds = WindowedOceanDataset(cfg, split="train", region_name=region_name)
        val_ds   = WindowedOceanDataset(cfg, split="val", region_name=region_name)
        test_ds   = WindowedOceanDataset(cfg, split="test", region_name=region_name)

        #input_channels = train_ds.n_channels   # ← best practice

        

        
        
        print(f"[Training] Train samples: {len(train_ds)}")
        print(f"[Training] Val samples: {len(val_ds)}")
        print(f"[Training] Test samples: {len(test_ds)}")



        # --- Debug output for splits ---
        for split, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            print(f"[Training] 🧩 Dataset '{split}' — indices {dataset.t_start}:{dataset.t_end}, total {len(dataset)} samples") 
 


        train_loader = DataLoader(train_ds, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=int(getattr(cfg.training, "num_workers", 0)))
        val_loader = DataLoader(val_ds, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=int(getattr(cfg.training, "num_workers", 0)))
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=int(getattr(cfg.training, "num_workers", 0)))


        



        # Create model for this region (pass H/W if dataset made them available)
        H = getattr(train_ds, "H", None)
        W = getattr(train_ds, "W", None)
        print("[Training] Dataset channels:", train_ds.n_channels)
        print("[Training] Dataset depths:", train_ds.n_depths)

        input_channels = train_ds.n_channels*train_ds.n_depths
        
        print("[Training] Total input channels:", input_channels)



        model = OceanPredictor(cfg, input_channels = input_channels).to(device)

        optimizer = optim.Adam(model.parameters(), lr=float(cfg.training.lr))
        
        

        # 🔽 ADD THIS
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )


        best_val_loss = float("inf")

        # training bookkeeping
        train_losses, val_losses = [], []
        csv_log_path = logs_dir / f"training_log_{region_name}.csv"
           
        model.train()
        optimizer.zero_grad()
        

        print("[Training] Start train loop per epoch.....")

        # per-epoch debug frames aggregator (for gif)
        for epoch in range(1, int(cfg.training.epochs) + 1):
            
            running_loss = running_rmse = running_mae = running_ssim = running_r2 = 0.0

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

                #2D LSTM
                if inputs.dim() == 6: 
                    # inputs shape: (B, T, C*Z, H, W) Flattend Channels & Z Levels
                    inputs_2d = flatten_depth(inputs)
                else:
                    inputs_2d = inputs  # already 2D or model expects full volume

                # safety
                inputs_2d = torch.nan_to_num(inputs_2d, nan=0.0, posinf=1e6, neginf=-1e6)
                targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)

                optimizer.zero_grad()
                preds = model(inputs_2d)
                


                loss =  combined_loss(preds, targets, mask, alpha_grad=0.5)
                
                
                metrics = compute_basic_metrics(preds, targets)
                
                rmse, mae, r2 = metrics["RMSE"], metrics["MAE"], metrics['R2']
                

                


                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ Skipping batch due to NaN/Inf loss.")
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    


                running_loss += float(loss.detach().cpu().item())
                running_rmse += rmse
                running_mae += mae
                running_r2 += r2

                batch_count += 1
                pbar.set_postfix({"train_loss": f"{running_loss / batch_count:.6f}"})

                # Debug visualization save
                if (batch_idx % debug_save_every_n_batches) == 0 and (epoch % 10 == 0):
                    saved_png = debug_visualize_sample(inputs, region_name, epoch * 10000 + batch_idx)
                    debug_frames.append(saved_png)


            avg_rmse = running_rmse / len(train_loader)
            avg_mae = running_mae / len(train_loader)
            avg_ssim = running_ssim / len(train_loader)
            avg_r2 = running_r2 / len(train_loader)

            
            train_epoch_loss = running_loss / max(1, batch_count)
            train_losses.append(train_epoch_loss)
            #print(f"📊 Epoch {epoch} | Loss: {train_epoch_loss:.4f} | RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f} | SSIM: {avg_ssim:.4f} | R²: {avg_r2:.4f}")

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
                    

                    if v_inputs.ndim == 6:
                        v_inputs_2d = flatten_depth(v_inputs)
                    else:
                        v_inputs_2d = v_inputs
                    

                    v_inputs_2d = torch.nan_to_num(v_inputs_2d, nan=0.0, posinf=1e6, neginf=-1e6)
                    v_targets = torch.nan_to_num(v_targets, nan=0.0, posinf=1e6, neginf=-1e6)
                    v_inputs_2d = v_inputs_2d.to(device)
                    v_targets = v_targets.to(device)
                    v_preds = model(v_inputs_2d.to(device))
                    if v_mask is not None:
                        v_mask = v_mask.to(device)
                    
                    

                    v_loss =  combined_loss(v_preds, v_targets, v_mask) 
                    
                    
                    val_running += float(v_loss.detach().cpu().item())
                    val_count += 1
            val_epoch_loss = val_running / max(1, val_count)
            val_losses.append(val_epoch_loss)

            # step scheduler with validation loss
            scheduler.step(val_epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']

            with open(logs_dir / f"lr_log_{region_name}.csv", "a") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["epoch", "lr"])
                writer.writerow([epoch, current_lr])
    

            metrics_csv = logs_dir / f"train_metrics_{region_name}.csv"
            with open(metrics_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["epoch", "train_rmse", "train_mae", "train_r2"])
                writer.writerow([epoch, avg_rmse, avg_mae, avg_r2])




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

