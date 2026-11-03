"""
export_nautilus_model.py

Converts a trained NAUTILUS (OceanConv2DAsymXPU / Conv2LSTM) checkpoint into:
  1. A TorchScript file FTorch can load (torch_model_load).
  2. A JSON metadata file describing tensor shapes, channel order, and
     per-channel normalization, for the adapted Fortran interface to read.

Run this from the NAUTILUS repo root (Current_FullColumn_CPU/) so that
`src.model_conv2d_asym_hybrid` etc. are importable, e.g.:

    cd Current_FullColumn_CPU
    python export_nautilus_model.py \
        --config config/conv2dlstm_s323D_fullcolumn.yaml \
        --region North_Atlantic \
        --checkpoint checkpoints/ocean_transport_fullcolumn_North_Atlantic_best_model.pt \
        --out-dir ftorch_export

WHY A WRAPPER MODULE:
The trained model's forward signature is x: (B, T, C, Z, H, W) -- a 6D
tensor. FTorch/libtorch can handle high-rank tensors, but keeping the
rank crossing the Fortran/C++ boundary as low as possible is safer and
easier to get right. So this script wraps the real model in
`FTorchExportWrapper`, which:
  - accepts a flattened 4D input  (B, T*C*Z, H, W)
  - reshapes it back to (B, T, C, Z, H, W) internally
  - runs the real model
  - squeezes the singleton channel dim of the output, returning (B, Z, H, W)

That means the Fortran side only ever constructs/reads 4D arrays.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Make `src.*` imports work when run from the repo root.
sys.path.insert(0, str(Path.cwd()))   # src/ of the training folder the job linked here

from src.model_conv2d_asym_hybrid import OceanConv2DAsymXPU  # noqa: E402


class FTorchExportWrapper(nn.Module):
    """Wraps OceanConv2DAsymXPU to expose a fixed-rank-4 tensor interface."""

    def __init__(self, model: nn.Module, T: int, C: int, Z: int):
        super().__init__()
        self.model = model
        self.T = T
        self.C = C
        self.Z = Z

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: (B, T*C*Z, H, W)  ->  (B, T, C, Z, H, W)
        B, _, H, W = x_flat.shape
        x = x_flat.view(B, self.T, self.C, self.Z, H, W)
        out = self.model(x)          # (B, 1, Z, H, W)
        out = out.squeeze(1)         # (B, Z, H, W)
        return out


def build_input_channels_and_stats(cfg, region_name: str):
    """
    Re-derive the channel list/order the same way WindowedOceanDataset does,
    without needing to open the raw NetCDF files, and load the saved
    per-region normalization stats (norms/norm_stats_<region>.npz).
    """
    input_variables = [
        cfg.data.tracers.variables[0],
        cfg.data.forcing.u_var,
        cfg.data.forcing.v_var,
        cfg.data.forcing.w_var,
    ]
    if hasattr(cfg.data.forcing, "dif_var"):
        input_variables.append(cfg.data.forcing.dif_var)
    if hasattr(cfg.data.forcing, "mld_var"):
        input_variables.append(cfg.data.forcing.mld_var)

    stats_path = Path("norms") / f"norm_stats_{region_name}.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Could not find {stats_path}. Run predict.py or train.py at least "
            f"once for region '{region_name}' first so normalization stats exist."
        )
    stats = dict(np.load(stats_path))
    return input_variables, stats


def build_channel_norm_metadata(input_variables, stats):
    """
    Build the per-channel normalization block for the metadata JSON, matching
    data.py exactly:
      - tracer  -> per-depth min-max:      (x - min[z]) / (max[z]-min[z])
      - u/v/w/dif/mld -> scalar z-score:   (x - mean) / (std + 1e-12)
    """
    name_to_statkey = {
        "u": "u", "v": "v", "w": "w",
    }
    channels = []
    tracer_name = input_variables[0]
    forcing_names = input_variables[1:]

    # tracer channel (index 0)
    channels.append({
        "name": tracer_name,
        "role": "tracer",
        "normalization": "minmax_per_depth",
        "min": stats["tracer_min"].tolist(),   # length Z
        "max": stats["tracer_max"].tolist(),   # length Z
    })

    # forcing channels, in the same order data.py appends them: u, v, w, [dif], [mld]
    forcing_roles = ["u", "v", "w", "dif", "mld"]
    for role, varname in zip(forcing_roles, forcing_names):
        mean_key = f"{role}_mean" if role != "mld" else "mld_distance_mean"
        std_key = f"{role}_std" if role != "mld" else "mld_distance_std"
        channels.append({
            "name": varname,
            "role": role,
            "normalization": "zscore_scalar",
            "mean": float(stats[mean_key]),
            "std": float(stats[std_key]),
        })

    return channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to the training YAML config")
    ap.add_argument("--region", required=True, help="Region name, e.g. North_Atlantic")
    ap.add_argument("--checkpoint", required=True, help="Path to <region>_best_model.pt")
    ap.add_argument("--height", type=int, default=None,
                     help="Grid H for the region. Defaults to y_range span from config.")
    ap.add_argument("--width", type=int, default=None,
                     help="Grid W for the region. Defaults to x_range span from config.")
    ap.add_argument("--out-dir", default="ftorch_export")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)

    region_cfg = next(
        (r for r in cfg.data.regions if r["name"] == args.region), None
    )
    if region_cfg is None:
        raise ValueError(f"Region '{args.region}' not found in config.data.regions")

    x1, x2 = region_cfg["x_range"]
    y1, y2 = region_cfg["y_range"]
    W = args.width or (x2 - x1)
    H = args.height or (y2 - y1)

    T = int(cfg.data.window.input_steps)
    Z = int(cfg.data.vertical_levels)

    input_variables, stats = build_input_channels_and_stats(cfg, args.region)
    C = len(input_variables)

    print(f"[export] T={T} C={C} Z={Z} H={H} W={W}")
    print(f"[export] channels (in order): {input_variables}")

    # --- Build model and load trained weights (mirrors predict.py) ---
    device = torch.device("cpu")
    model = OceanConv2DAsymXPU(cfg, input_channels=C).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    wrapper = FTorchExportWrapper(model, T=T, C=C, Z=Z).eval()

    # --- Trace with a correctly-shaped dummy input ---
    example = torch.zeros(1, T * C * Z, H, W, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=True)
        # Sanity check the output shape.
        test_out = traced(example)
        assert test_out.shape == (1, Z, H, W), test_out.shape

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"nautilus_{args.region}.pt"
    traced.save(str(model_path))
    print(f"[export] Saved TorchScript model -> {model_path}")

    # --- Write metadata JSON for the Fortran side ---
    metadata = {
        "model_path": model_path.name,
        "region": args.region,
        "tensor_shapes": {
            "input_steps_T": T,
            "channels_C": C,
            "depth_levels_Z": Z,
            "height_H": H,
            "width_W": W,
            "note": (
                "Fortran must build the flattened input array of shape "
                "(H, W, T*C*Z, 1) in column-major order such that, when "
                "interpreted by FTorch's torch_tensor_from_array with the "
                "layout mapping the PyTorch dims [B, T*C*Z, H, W] onto the "
                "Fortran dims, index order along the flattened axis is "
                "t-major, then c, then z (t*C*Z + c*Z + z), matching "
                "wrapper.forward's `x_flat.view(B, T, C, Z, H, W)` reshape. "
                "Verify this mapping with the included cross-check script "
                "before production use -- see README section 'Verifying "
                "the Fortran<->PyTorch tensor layout'."
            ),
        },
        "channels": build_channel_norm_metadata(input_variables, stats),
        "depths_m": None,  # fill in below if available
        "mld_distance": {
            "formula": "mld_distance[t,z,y,x] = depth_m[z] - mlotst[t,y,x]",
            "masked_on_land": True,
            "note": (
                "This channel is DERIVED, not a raw model field. The host "
                "driver (or the Fortran module) must compute it from the "
                "raw mixed-layer-depth field and the depth vector before "
                "normalizing/pushing it into the input buffer."
            ),
        },
    }

    # Try to attach the actual depth vector if reachable from the tracer file.
    try:
        import xarray as xr
        with xr.open_dataset(cfg.data.tracers.paths[0]) as ds:
            metadata["depths_m"] = ds["olevel"].values.tolist()
    except Exception as e:
        print(f"[export] Warning: could not read depth vector ({e}); "
              f"fill metadata['depths_m'] manually.")

    meta_path = out_dir / f"nautilus_{args.region}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[export] Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
