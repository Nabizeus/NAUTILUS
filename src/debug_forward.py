import torch
import matplotlib.pyplot as plt
from src.model import OceanPredictor
from src.data import WindowedOceanDataset
import numpy as np
import os
from omegaconf import OmegaConf

def visualize_activation_grid(tensor, title, save_path):
    C, H, W = tensor.shape
    grid_cols = int(np.ceil(np.sqrt(C)))
    grid_rows = int(np.ceil(C / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < C:
            ax.imshow(tensor[i], cmap="viridis")
            ax.set_title(f"ch {i}")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def debug_conv_lstm(cfg_path="config/default.yaml", region_name="pacific", sample_idx=0, save_dir="artifacts/debug_forward"):
    # 🔹 Load YAML manually instead of Hydra
    cfg = OmegaConf.load(cfg_path)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WindowedOceanDataset(cfg, split="test", region_name=region_name)
    inputs, targets, _ = dataset[sample_idx]
    inputs = inputs.unsqueeze(0).to(device)

    H, W = inputs.shape[-2:]
    model = OceanPredictor(cfg, H=H, W=W).to(device)

    # Load checkpoint
    ckpt_path = f"{cfg.training.checkpoint_dir}/{region_name}_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"✅ Loaded model for {region_name}, visualizing intermediate activations...")

    activations = {}
    lstm_outputs = []

    def hook(name):
        def fn(_, __, output):
            activations[name] = output.detach().cpu()
        return fn

    # --- Register hooks ---
    model.model.encoder[0].register_forward_hook(hook("conv1"))
    model.model.encoder[2].register_forward_hook(hook("conv2"))

    # Forward pass
    with torch.no_grad():
        preds = model(inputs)

    # --- Save feature maps ---
    for key, tensor in activations.items():
        tensor = tensor.squeeze(0)
        visualize_activation_grid(
            tensor,
            f"{key} features ({region_name})",
            f"{save_dir}/{region_name}_{key}.png"
        )

    # --- Prediction vs True ---
    pred_np = preds[0, 0].cpu().numpy()
    true_np = targets[0].cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(true_np, cmap="viridis")
    plt.title("True")
    plt.subplot(1, 2, 2)
    plt.imshow(pred_np, cmap="viridis")
    plt.title("Pred")
    plt.suptitle(f"{region_name} | Prediction vs True")
    plt.savefig(f"{save_dir}/{region_name}_prediction_vs_true.png")
    plt.close()

    print(f"🎨 Visualizations saved to: {save_dir}")

