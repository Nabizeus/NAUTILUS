import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from src.model import OceanPredictor
from src.data import WindowedOceanDataset
from src.utils import save_checkpoint


def load_model_from_checkpoint(cfg, device, region_name):
    """Load a model checkpoint robustly, handling multiple naming conventions."""
    model = OceanPredictor(cfg).to(device)

    checkpoint_dir = getattr(cfg.training, "checkpoint_dir", "./checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"{region_name}_best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ No checkpoint found at {checkpoint_path}")

    print(f"📂 Loading checkpoint for region '{region_name}' → {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- Handle multiple formats safely ---
    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded 'model_state_dict' successfully")
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            print("✅ Loaded 'model_state' successfully")
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            print("✅ Loaded 'state_dict' successfully")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded raw state_dict directly")
    except Exception as e:
        print(f"⚠️ Could not load checkpoint properly: {e}")
        raise RuntimeError("❌ Checkpoint does not contain valid model weights")

    model.eval()
    print("🧠 Model ready for inference")
    return model


def visualize_forecast(true, pred, title, save_path, vmin=None, vmax=None):
    plt.figure(figsize=(8, 4))
    im1 = plt.subplot(1, 2, 1)
    plt.imshow(true, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title("True")
    plt.colorbar()

    im2 = plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title("Predicted")
    plt.colorbar()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def animate_forecasts(image_paths, save_gif):
    import imageio
    frames = [imageio.imread(p) for p in sorted(image_paths)]
    imageio.mimsave(save_gif, frames, duration=0.4)
    print(f"🎞️ Saved animation: {save_gif}")


def main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="../config", config_name="default", version_base=None)
    def run(cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for region_cfg in cfg.data.regions:
            region_name = region_cfg["name"]
            print(f"\n🧭 Predicting for region: {region_name}")

            dataset = WindowedOceanDataset(cfg, split="test", region_name=region_name)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            # --- Depth 2D slicing ---
            z_idx = getattr(dataset, "z_index", 0)

            model = load_model_from_checkpoint(cfg, device, region_name)

            output_dir = os.path.join("predictions", region_name)
            os.makedirs(output_dir, exist_ok=True)
            loss_fn = torch.nn.MSELoss(reduction = 'sum')
            total_loss = 0.0
            image_paths = []
            imagex_paths = []
            all_true, all_pred = [], []

            for i, (inputs, targets, _) in enumerate(loader):
                if i % 5 == 0:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.no_grad():
                        preds = model(inputs)
                        loss = loss_fn(preds, targets)
                    total_loss += loss.item()

                    true_np = targets[0, 0].cpu().numpy()
                    pred_np = preds[0, 0].cpu().numpy()
                    all_true.append(true_np)
                    all_pred.append(pred_np)


                    img_path = os.path.join(output_dir, f"forecast_{i:03d}.png")
                    visualize_forecast(true_np, pred_np, f"Sample {i} | Loss={loss:.4f}", img_path)

                    image_paths.append(img_path)

           
            plt.imshow(targets[0,0].cpu(), cmap='viridis')
            plt.title(f"First Target Slice {region_name} @z_idx {z_idx}")
            plt.colorbar()
            plt.savefig(f"artifacts/first_target_slice_{region_name}.png")
            plt.close()


            plt.imshow(inputs[0,8,2].cpu(), cmap='viridis')
            plt.title(f"[0,t=4th,2nd U, 30, 30] Input Slice {region_name} @z_idx {z_idx}")
            plt.colorbar()
            plt.savefig(f"artifacts/first_input_slice_{region_name}.png")
            plt.close()

               
            t = 3
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            feature_names = ["Tracer", "U", "V", "W"]  # adapt if needed

            for c in range(4):
                ax = axes[c]
                im = ax.imshow(inputs[0, t, c].cpu(), cmap='viridis')
                ax.set_title(f"{feature_names[c]} (t={t})")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.savefig(f"artifacts/four_time3_input_slice_{region_name}.png")




            # Global color scale for visualization
            global_vmin = np.min(all_true)
            global_vmax = np.max(all_true)

            for i, p in enumerate(image_paths):
                true_np, pred_np = all_true[i], all_pred[i]
                img_path = os.path.join(output_dir, f"forecast_fixed_{i:03d}.png")
                

                visualize_forecast(true_np, pred_np, f"Fixed scale Sample {i} | Sum Loss={loss:.2f}", img_path, vmin=global_vmin, vmax=global_vmax)
                imagex_paths.append(img_path)
            
            avg_loss = total_loss / len(loader)
            print(f"📉 Average test loss for {region_name}: | Total Loss={total_loss:.2f} | {avg_loss:.2f}")

            gif_path = os.path.join(output_dir, f"forecast_movie_{region_name}.gif")
            animate_forecasts(sorted(imagex_paths), gif_path)
        
        print(f"✅ Model evaluation finished successfully 🚀")
        print("\n" + "=" * 90)
        print("🏁✅ FINAL — Model evaluation finished successfully 🚀🌊")
        print("=" * 90 + "\n")
        # 🧩 Final summary message
        print("\n" + "=" * 80)
        print(f"🌊✨ Prediction complete for region: 🌍 {region_name}")
        print(f"📊 Output directory: {output_dir}")
        print(f"🎥 Forecast animation: {output_dir}/forecast_movie_{region_name}.gif")
        print(f"📦 Checkpoint used: {checkpoint_path}")
        print(f"🧠 Device: {device}")
        print(f"✅ Model evaluation finished successfully 🚀")
        print("=" * 80 + "\n")


    run()
    

if __name__ == "__main__":
    main()

