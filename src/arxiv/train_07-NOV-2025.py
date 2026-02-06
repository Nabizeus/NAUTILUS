import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt 
from src.data import WindowedOceanDataset
from src.model import OceanPredictor
from src.utils import save_checkpoint, load_config, set_seed
import time
import csv
import imageio
from pathlib import Path
import numpy as  np

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
            round(best_val_loss, 4),
            epochs,
            batch_size,
            time.strftime('%Y-%m-%d %H:%M')
        ])



# ==============================
# ✅ NaN-Safe Loss Function
# ==============================
def masked_mse_loss(preds, targets, mask):
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
    cfg = load_config("config/default.yaml")
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    set_seed(cfg.training.seed)

    # ==============================
    # 📦 Dataset & Dataloader
    # ==============================
    print("🔧 Loading datasets...")

    # === Multi-region training loop ===
    for region_cfg in cfg.data.regions:
        region_name = region_cfg["name"]
        print(f"\n🌍 Starting training for region: {region_name}")

        start_time = time.time() # <-- Start timer
        # Create region-specific dataset
        
        train_ds = WindowedOceanDataset(cfg, split="train", region_name=region_name)
        val_ds   = WindowedOceanDataset(cfg, split="val", region_name=region_name)

        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)


        def debug_visualize_sample(inputs, region_name, number, out_dir="artifacts/debug_inputs"):
            os.makedirs(out_dir, exist_ok=True)
            # inputs shape: (B, T, C, H, W)
            B, T, C, H, W = inputs.shape
            sample = inputs[0].detach().cpu().numpy()

            fig, axes = plt.subplots(C, T, figsize=(T * 2.5, C * 2.5))
            vars = ["Tracer", "U", "V", "W"]

            for c in range(C):
                for t in range(T):
                    ax = axes[c, t]
                    im = ax.imshow(sample[t, c, :, :], cmap="viridis")
                    ax.set_title(f"{vars[c]} t={t}")
                    ax.axis("off")
            str_number = str(int(number))
            plt.tight_layout()
            return fig # ✅ Return instead of closing it here

            #save_path = os.path.join(out_dir, f"sample_{region_name}_{str_number}.png")
            #plt.savefig(save_path, dpi=200)
            #plt.close()
            #print(f"📸 Saved debug input grid → {save_path}")

   
        # In train.py, before initializing model:
        x1, x2 = region_cfg["x_range"]
        y1, y2 = region_cfg["y_range"]

        H = y2 - y1  # height is Δy
        W = x2 - x1  # width is Δx

 
        # Initialize model for this region
        model = OceanPredictor(cfg, H=H, W=W).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        best_val_loss = float("inf")

        
        train_losses = []
        val_losses = []
        




        z_idx = getattr(train_ds, "z_index", 0)
        print(f"@train.py - line 60 - z_idx = getattr(train_ds): {z_idx}")

        # For collecting frames per epoch
        debug_frames = []



        for epoch in range(cfg.training.epochs):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]")
            debug_count = 0
            for inputs, targets, mask in pbar:
                if debug_count % 10 == 0:
                    #print(f"�~_~T~M [Debug] Sample batch shape: {inputs.shape}")
                    #debug_visualize_sample(inputs, region_name, debug_count)
                    fig = debug_visualize_sample(inputs, region_name, debug_count)  # your existing plotting function
                    fig.canvas.draw()

                    # Convert figure to numpy frame
                    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    debug_frames.append(frame)
                    plt.close(fig)
                debug_count += 1
                for inputs, targets, mask in train_loader:
                    
                    
                    # --- Depth-averaged 2D slicing ---
                    if inputs.ndim == 6:
                        inputs = inputs[..., z_idx, :, :] #at z_idx middle of depth range
                        print(f"🪣 Train depth-middle layer input: {inputs.shape}")
                        mean_inputs = np.mean(inputs)
                        mean_targets = np.mean(targets)
                        sigma_inputs = np.stdev(inputs)
                        sigma_targets = np.stdev(targets)
                        print(f"Mean inputs {mean_inputs}")
                        print(f"Mean targets {mean_targets}")
                        print(f"Sigma inputs {stdev_inputs}")
                        print(f"Sigma targets {stdev_targets}")
                        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

                preds = model(inputs)
          
                # Ensure NaN-safe tensors
                inputs = torch.nan_to_num(inputs)
                targets = torch.nan_to_num(targets)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    # --- NaN/Inf safety in input batch ---
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                        print("⚠️ [Train Loop] NaN/Inf detected in inputs! Fixing batch.")
                        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e20, neginf=-1e20)

                    preds = model(inputs)

                    # --- NaN/Inf safety in output ---
                    if torch.isnan(preds).any() or torch.isinf(preds).any():
                        print("⚠️ [Train Loop] NaN/Inf detected in predictions! Fixing.")
                        preds = torch.nan_to_num(preds, nan=0.0, posinf=1e20, neginf=-1e20)
                
                    loss = masked_mse_loss(preds, targets, mask)

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
                for inputs, targets, mask in val_loader:
                    inputs, targets, mask =  inputs.to(device), targets.to(device), mask.to(device)
                    if inputs.ndim == 6:
                        inputs_2d = inputs[..., z_idx, :, :] #10 .mean(dim=3)
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
            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    save_checkpoint(model, optimizer, epoch, path=f"checkpoints/{region_name}_best_model.pt")
            #    print(f"💾 Saved new best model (val_loss={val_loss:.6f})")


            train_losses.append(train_loss)
            val_losses.append(val_loss)




            if debug_frames:
                gif_path = f"artifacts/debug_inputs/{region_name}_epoch_{epoch+1}.gif"
                imageio.mimsave(gif_path, debug_frames, fps=3)
                print(f"🎞️ Saved debug GIF for {region_name}, epoch {epoch+1} ➜ {gif_path}")


            # Save checkpoint per region
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"checkpoints/{region_name}_best_model.pt"
                torch.save({
                 "model_state": model.state_dict(),
                 "H": H,
                 "W": W,
                 "epoch": epoch,
                 "optimizer_state": optimizer.state_dict()
                    }, save_path)
                print(f"💾 @train.py - line 260 - Saved checkpoint model:  {save_path}")

                #torch.save({
                #    "model_state_dict": model.state_dict(),
                #    "optimizer_state_dict": optimizer.state_dict(),
                #    "epoch": epoch,
                #    "val_loss": val_loss
                #    }, save_path)                
                print(f"✅ Saved best model for {region_name} → {save_path}")
                
        print(f"🧭 [{region_name}] Epoch {epoch+1}/{cfg.training.epochs} | " f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")



        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        #if len(test_losses) > 0:
            #plt.plot(test_losses, label='Test Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve - {region_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'artifacts/loss_curve_{region_name}.png', dpi=150)
        plt.close()

        print(f"📉 Saved loss curve → artifacts/loss_curve_{region_name}.png")

        end_time = time.time()     # <-- End timer
        elapsed = (end_time - start_time) / 60  # minutes
        log_training_time(region_name, elapsed, best_val_loss, cfg.training.epochs, cfg.training.batch_size)
        
        print(f"✅ Finished training for {region_name} | Best val loss: {best_val_loss:.4f}")
        print(f"⏱ Training time for {region_name}: {elapsed:.2f} minutes\n")
        
    print("🎉 Training complete.")


if __name__ == "__main__":
    main()

