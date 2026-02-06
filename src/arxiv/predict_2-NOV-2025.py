import torch
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import numpy as np
from src.model import OceanPredictor
from src.data import WindowedOceanDataset
from src.utils import load_config


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest saved checkpoint file."""
    files = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    return files[0] if files else None


def load_model_from_checkpoint(cfg, device, region_name):
    """Load model weights from latest available checkpoint."""
    model = OceanPredictor(cfg).to(device)

    checkpoint_dir = cfg.training.checkpoint_dir
    checkpoint_path = f"{checkpoint_dir}/{region_name}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)


    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ No checkpoint found at {checkpoint_path}")

    print(f"✅ Loading checkpoint for region '{region_name}': {checkpoint_path}")

    # ✅ Try different common formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback — assume the checkpoint is just raw state_dict
        try:
            model.load_state_dict(checkpoint)
        except:
            raise KeyError("❌ Checkpoint does not contain model weights in a recognizable format!")

    model.eval()
    return model

def visualize_forecast(true, pred, title="Forecast vs True", save_path=None):
    """Visualize a single depth slice comparison."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(true, cmap="viridis")
    plt.title("True")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="viridis")
    plt.title("Predicted")
    plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def animate_forecasts(image_paths, save_gif=None):
    """Create an animated GIF from saved PNGs."""
    if not image_paths:
        print("⚠️ No forecast images found to animate.")
        return

    print(f"🎞️ Generating animation with {len(image_paths)} frames...")
    fig = plt.figure(figsize=(8, 4))
    ims = []

    for path in image_paths:
        img = plt.imread(path)
        im = plt.imshow(img)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save(save_gif, writer="pillow")
    plt.close(fig)
    print(f"✅ Animation saved to {save_gif}")


def main():
    cfg = load_config("config/default.yaml")
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    print(f"🔍 Running predictions on device: {device}")

    # Load dataset and model
    for region_cfg in cfg.data.regions:
        region_name = region_cfg["name"]
        print("region_name: ",region_name)
        print(f"\n�~_~L~M Sta PREDICTION for regio {region_name}")

        dataset = WindowedOceanDataset(cfg, split="test", region_name=region_name)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        
        region_name = cfg.data.regions[0].name if not hasattr(cfg.data, "region_name") else cfg.data.region_name
        model = load_model_from_checkpoint(cfg, device, region_name)

        

        output_dir = "predictions"
        os.makedirs(output_dir, exist_ok=True)

        loss_fn = torch.nn.MSELoss()
        total_loss = 0.0
        image_paths = []

        for i, (inputs, targets, mask) in enumerate(tqdm(loader, desc="Predicting")):
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        
            # --- Depth-averaged 2D slicing ---
            z_idx = getattr(dataset, "z_index", 0)
            print(f"@predict.py - Line 107 - z_idx: {z_idx}")

            if inputs.ndim == 6:
                inputs_2d = inputs[...,z_idx , :, :] # middle of the z-range or depth levels
                print(f"🔹 Using z-index {z_idx} for prediction")
            else:
                inputs_2d = inputs

            
            plt.imshow(targets[0,0].cpu(), cmap='viridis')
            plt.title(f"First Target Slice {region_name}")
            plt.colorbar()
            plt.savefig(f"artifacts/first_target_slice_{region_name}.png")
            plt.close()


            with torch.no_grad():
            

                preds = model(inputs_2d.to(device), debug=False)


                #preds = model(inputs)

                # Apply mask before computing loss
                masked_pred = preds * mask
                masked_target = targets * mask
                #loss = torch.mean(((preds - targets)[mask.bool()]) ** 2).item() #For large tensor multipl. 3D grids
                
                if torch.isnan(preds).any() or torch.isnan(targets).any():
                    print("⚠️ NaNs detected in preds or targets!")
                    print(f"preds NaN count: {torch.isnan(preds).sum().item()}")
                    print(f"targets NaN count: {torch.isnan(targets).sum().item()}")

                if torch.all(targets == 0):
                    print("⚠️ All targets are zero — likely empty mask region")

                loss = torch.mean((masked_pred - masked_target) ** 2).item()
                total_loss += loss


                #loss = loss_fn(preds, targets).item()
                #total_loss += loss

            # Visualization for first few samples
            if i < 20:  # save first 20 examples
                pred_np = preds[0, 0].detach().cpu().numpy()
                true_np = targets[0, 0].detach().cpu().numpy()
                img_path = f"{output_dir}/forecast_{i:03d}_{region_name}.png"
            
            
                # --- Reduce 3D data to 2D for visualization ---
                z_idx = getattr(dataset, "z_index", 0)
                if true_np.ndim == 3:
                    true_np = np.nanmean(true_np[z_idx, :, :], axis=0)  # middle of the z-range for test
                if pred_np.ndim == 3:
                    pred_np = np.nanmean(pred_np[z_idx, :, :], axis=0)



                visualize_forecast(
                    true_np,
                    pred_np,
                    title=f"Region:{region_name} | Sample {i} | Masked Loss: {loss:.6f}",
                    save_path=img_path,
                )
                image_paths.append(img_path)

        avg_loss = total_loss / len(loader)
        print(f"✅ Average test loss: {avg_loss:.6f}")

        # Generate animation from saved forecasts
        animate_forecasts(sorted(image_paths), save_gif=os.path.join(output_dir, f"forecast_movie_{region_name}.gif"))


if __name__ == "__main__":
    main()

