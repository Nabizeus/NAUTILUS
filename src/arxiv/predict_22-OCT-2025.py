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


def load_model_from_checkpoint(cfg, device):
    """Load model weights from latest available checkpoint."""
    model = OceanPredictor(cfg).to(device)

    ckpt_path = find_latest_checkpoint("checkpoints")
    if ckpt_path:
        print(f"🔁 Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Model loaded successfully from {ckpt_path}")
    else:
        print("⚠️ No checkpoint found — using untrained model.")

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


def animate_forecasts(image_paths, save_gif="forecast_movie.gif"):
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
    dataset = WindowedOceanDataset(cfg, split="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = load_model_from_checkpoint(cfg, device)

    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    image_paths = []

    for i, (inputs, targets, mask) in enumerate(tqdm(loader, desc="Predicting")):
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        
        # --- Depth-averaged 2D slicing ---
        if inputs.ndim == 6:
            inputs_2d = inputs[..., :3, :, :].mean(dim=3)  # top 3 depth levels
        else:
            inputs_2d = inputs


        with torch.no_grad():
            

            preds = model(inputs_2d.to(device))


            #preds = model(inputs)

            # Apply mask before computing loss
            masked_pred = preds * mask
            masked_target = targets * mask
            #loss = torch.mean(((preds - targets)[mask.bool()]) ** 2).item() #For large tensor multipl. 3D grids
            loss = torch.mean((masked_pred - masked_target) ** 2).item()
            total_loss += loss


            #loss = loss_fn(preds, targets).item()
            #total_loss += loss

        # Visualization for first few samples
        if i < 10:  # save first 10 examples
            pred_np = preds[0, 0].detach().cpu().numpy()
            true_np = targets[0, 0].detach().cpu().numpy()
            img_path = f"{output_dir}/forecast_{i:03d}.png"
            
            
            # --- Reduce 3D data to 2D for visualization ---
            if true_np.ndim == 3:
                true_np = np.nanmean(true_np[:10, :, :], axis=0)  # average over depth
            if pred_np.ndim == 3:
                pred_np = np.nanmean(pred_np[:10, :, :], axis=0)



            visualize_forecast(
                true_np,
                pred_np,
                title=f"Sample {i} | Masked Loss: {loss:.6f}",
                save_path=img_path,
            )
            image_paths.append(img_path)

    avg_loss = total_loss / len(loader)
    print(f"✅ Average test loss: {avg_loss:.6f}")

    # Generate animation from saved forecasts
    animate_forecasts(sorted(image_paths), save_gif=os.path.join(output_dir, "forecast_movie.gif"))


if __name__ == "__main__":
    main()

