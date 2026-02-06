import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from src.data import WindowedOceanDataset as OceanDataset
from src.model import OceanPredictor


def main():
    # Load config
    cfg = OmegaConf.load("config/default.yaml")

    # Force small batch and short test
    cfg.training.batch_size = 1
    cfg.training.num_workers = 0
    cfg.training.device = "cpu"  # use CPU to avoid GPU OOM during debug

    print("🔧 Loading dataset...")
    ds = OceanDataset(cfg, split="train")
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # Load model
    print("🧠 Initializing model...")
    model = OceanPredictor(cfg).to(cfg.training.device)
    model.eval()

    # One batch only
    batch = next(iter(dl))
    inputs = inputs[:, :, :, 10, :, :]   # keep surface level
    targets = targets[:, :, :, 10, :, :]   # keep surface level
    inputs, targets = inputs.to(cfg.training.device), targets.to(cfg.training.device)

    print(f"✅ Input shape: {inputs.shape}, Target shape: {targets.shape}")

    # Forward pass
    with torch.no_grad():
        preds = model(inputs)
        loss = torch.nn.functional.mse_loss(preds, targets)

    print(f"✅ Forward pass successful! Loss = {loss.item():.6f}")
    print(f"✅ Pred shape: {preds.shape}")

    # Memory test
    print(f"💾 Estimated tensor memory: {preds.numel() * 4 / 1e6:.2f} MB (float32)")


if __name__ == "__main__":
    main()

