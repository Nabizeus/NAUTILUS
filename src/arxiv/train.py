import os, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from .utils import load_yaml, ensure_dir, to_device
from .data import WindowedOceanDataset
from .models import build_model

def loss_and_metrics(pred, target):
    # pred: [B, 1, 1, H, W]; target: [B, 1, Z, H, W]
    if target.shape[2] > 1:
        target2d = target[:, :, 0]
    else:
        target2d = target[:, :, 0]
    l1 = torch.mean(torch.abs(pred[:, :, 0] - target2d))
    l2 = torch.mean((pred[:, :, 0] - target2d) ** 2)
    return l2, {"mae": l1.item(), "mse": l2.item()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    full = WindowedOceanDataset(cfg, split="train")
    n_total = len(full)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=False)

    in_channels = full.in_channels * full.tr.shape[1] if cfg["model"]["spatial_mode"] == "2d" else full.in_channels
    model = build_model(cfg, in_channels=in_channels)

    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)))
    best_val = float("inf")
    epochs_no_improve = 0
    ensure_dir(cfg["train"]["ckpt_dir"])

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        running = []
        for batch in pbar:
            batch = to_device(batch, device)
            x, y = batch["x"], batch["y"]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(x)
                loss, metrics = loss_and_metrics(pred, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            running.append(metrics["mse"])
            if len(running) > 50: running.pop(0)
            pbar.set_postfix({"mse": float(np.mean(running))})
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    pred = model(batch["x"])
                    vloss, _ = loss_and_metrics(pred, batch["y"])
                val_losses.append(vloss.item())
        val_mse = float(np.mean(val_losses))
        print(f"Epoch {epoch}: val_mse={val_mse:.6f}")
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(cfg["train"]["ckpt_dir"], "best.ckpt"))
        else:
            epochs_no_improve += 1
        torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(cfg["train"]["ckpt_dir"], "last.ckpt"))
        if epochs_no_improve >= cfg["train"]["early_stop_patience"]:
            print("Early stopping.")
            break

if __name__ == "__main__":
    main()
