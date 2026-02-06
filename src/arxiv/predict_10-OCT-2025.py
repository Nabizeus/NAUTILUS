import torch
import matplotlib.pyplot as plt
from src.model import ConvLSTMHybrid
from src.data import WindowedOceanDataset
from omegaconf import OmegaConf


def main():
    cfg = OmegaConf.load("config/default.yaml")

    ds = WindowedOceanDataset(cfg, split="test")
    model = ConvLSTMHybrid(
        input_channels=4,
        hidden_dim=cfg.model.hidden_dim,
        kernel_size=cfg.model.kernel_size,
        depth_channels=cfg.model.depth_channels,
    ).to(cfg.training.device)

    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=cfg.training.device))
    model.eval()

    x, y_true = ds[10]
    x = x.unsqueeze(0).to(cfg.training.device)
    y_pred = model(x).detach().cpu().numpy()[0, 0]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(y_true[0, 0], cmap="viridis")
    plt.title("True")
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred[0], cmap="viridis")
    plt.title("Predicted")
    plt.show()


if __name__ == "__main__":
    main()

