import os, argparse
import numpy as np
import torch
import xarray as xr
from .utils import load_yaml, to_device
from .data import WindowedOceanDataset
from .models import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    ds = WindowedOceanDataset(cfg, split="test")
    in_channels = ds.in_channels * ds.tr.shape[1] if cfg["model"]["spatial_mode"] == "2d" else ds.in_channels
    model = build_model(cfg, in_channels=in_channels)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []
    ys = []
    with torch.no_grad():
        for i in range(len(ds)):
            batch = ds[i]
            x = batch["x"].unsqueeze(0)  # [1, T, C, Z, Y, X]
            y = batch["y"].unsqueeze(0)
            x, y = to_device(x, device), to_device(y, device)
            pred = model(x)      # [1, 1, 1, H, W]
            preds.append(pred.squeeze(0).cpu().numpy())  # [1,1,H,W]
            ys.append(y.squeeze(0).cpu().numpy())

    preds = np.stack(preds)  # [N, 1, 1, H, W]
    ys = np.stack(ys)        # [N, 1, Z, H, W]

    # De-normalize predictions back to tracer units
    mean = ds.norm_stats["tracer_mean"]
    std = ds.norm_stats["tracer_std"]
    preds_denorm = preds * std + mean

    H, W = preds.shape[-2], preds.shape[-1]
    N = preds.shape[0]
    out = xr.Dataset(
        {
            "pred_tracer": (["sample", "z", "y", "x"], preds_denorm[:, 0, ...]),
            "target_tracer": (["sample", "z", "y", "x"], ys[:, 0, ...]),
        },
        coords={
            "sample": np.arange(N, dtype=np.int32),
            "z": np.arange(1, dtype=np.int32),
            "y": np.arange(H, dtype=np.int32),
            "x": np.arange(W, dtype=np.int32),
        },
        attrs={"note": "Predictions correspond to next-step tracer following the window end."}
    )
    out_path = cfg["io"]["predict_out"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_netcdf(out_path)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
