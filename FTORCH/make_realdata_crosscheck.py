"""
make_realdata_crosscheck.py

Real-data test of the Fortran interface, Python side.

Picks ONE sample from the TEST period of the training data and writes:
  1. the RAW NEMO fields of its input time steps (what an ocean model would
     hand to push_timestep), for test_nautilus_realdata.F90 to read;
  2. the reference the Fortran side must reproduce:
       - the normalized input tensor built by the REAL training data loader
         (src.data.WindowedOceanDataset), and
       - the prediction of the exported TorchScript model on that tensor;
  3. the NEMO truth for the predicted time step.

Unlike make_dummy_crosscheck_input.py, this exercises normalization, the
MLD-distance channel and land handling, because the Fortran side starts from
raw fields.

Binary layout: raw float32, Fortran (column-major) element order, so a plain
Fortran `read(unit) array` into an array of the stated shape reproduces it.

Usage (from the folder containing src/, config/, norms/):
    python make_realdata_crosscheck.py \
        --config config/conv2dlstm_s323D_fullcolumn.yaml \
        --region North_Atlantic --export-dir ftorch_export --out-dir realdata
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path.cwd()))   # src/ of the training folder the job linked here

from src.data import WindowedOceanDataset  # noqa: E402


def log(msg):
    print(f"[realdata {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_raw(path, var, t0, t1, y_range, x_range):
    """Raw NEMO values for time steps t0..t1-1 of the region box, with
    missing values set to 0 (as data.py's safe_load_and_slice does)."""
    with xr.open_dataset(path, engine="netcdf4") as ds:
        da = ds[var]
        sel = {da.dims[0]: slice(t0, t1),
               da.dims[-2]: slice(*y_range), da.dims[-1]: slice(*x_range)}
        arr = da.isel(sel).values.astype(np.float64)
        times = [str(v) for v in ds[da.dims[0]].isel({da.dims[0]: slice(t0, t1 + 1)}).values]
    arr[~np.isfinite(arr)] = 0.0
    arr[np.abs(arr) >= 1e18] = 0.0
    return arr.astype(np.float32), times


def write_f(path, arr):
    np.ascontiguousarray(arr, dtype=np.float32).ravel(order="F").tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--export-dir", default="ftorch_export")
    ap.add_argument("--out-dir", default="realdata")
    ap.add_argument("--sample", type=int, default=None,
                    help="Index within the test split (default: the middle one)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("building the training data loader for split='test' "
        "(loads the full year for the region; this is the slow part)")
    t_start = time.time()
    ds = WindowedOceanDataset(cfg, split="test", region_name=args.region)
    log(f"data loader ready in {time.time() - t_start:.1f} s; test split has {len(ds)} samples")

    sample = len(ds) // 2 if args.sample is None else args.sample
    if not 0 <= sample < len(ds):
        raise ValueError(f"--sample must be in 0..{len(ds) - 1}")
    t_idx = int(ds.indices[sample])

    x, y, _ = ds[sample]                      # x: (T,C,Z,H,W)  y: (T_out,Z,H,W)
    x = torch.nan_to_num(x, nan=0.0)          # as train.py/predict.py feed it
    T, C, Z, H, W = x.shape
    log(f"sample {sample} of the test split -> time index {t_idx}; "
        f"inputs are steps {t_idx}..{t_idx + T - 1}, target is step {t_idx + T}")
    log(f"T={T} C={C} Z={Z} H={H} W={W}")

    # ---- reference: exported TorchScript model on the loader's tensor ----
    with open(Path(args.export_dir) / f"nautilus_{args.region}_metadata.json") as f:
        meta = json.load(f)
    shp = meta["tensor_shapes"]
    assert (T, C, Z, H, W) == (shp["input_steps_T"], shp["channels_C"], shp["depth_levels_Z"],
                               shp["height_H"], shp["width_W"]), "loader/export shape mismatch"
    model = torch.jit.load(str(Path(args.export_dir) / meta["model_path"])).eval()
    x_flat = x.reshape(1, T * C * Z, H, W).contiguous()
    with torch.no_grad():
        pred_norm = model(x_flat)[0].numpy()  # (Z,H,W)
    tmin = np.asarray(ds.stats["tracer_min"], dtype=np.float64)
    tmax = np.asarray(ds.stats["tracer_max"], dtype=np.float64)
    pred_phys = pred_norm * (tmax - tmin)[:, None, None] + tmin[:, None, None]
    log(f"Python reference prediction (normalized): min={pred_norm.min():.6f} "
        f"max={pred_norm.max():.6f} mean={pred_norm.mean():.6f}")

    # ---- raw fields for the Fortran side ----
    u_path, v_path, w_path = cfg.data.forcing.velocity_paths
    yr, xr_ = tuple(ds.y_range), tuple(ds.x_range)
    raw = {}
    sources = [("tracer", cfg.data.tracers.paths[0], cfg.data.tracers.variables[0]),
               ("u", u_path, cfg.data.forcing.u_var), ("v", v_path, cfg.data.forcing.v_var),
               ("w", w_path, cfg.data.forcing.w_var), ("dif", w_path, cfg.data.forcing.dif_var),
               ("mld", cfg.data.forcing.mld_path, cfg.data.forcing.mld_var)]
    times = []
    for name, path, var in sources:
        raw[name], times = load_raw(path, var, t_idx, t_idx + T, yr, xr_)
        log(f"raw {name:6s} ({var}): shape={raw[name].shape} "
            f"min={raw[name].min():.6g} max={raw[name].max():.6g}")
    land_mask = np.asarray(ds.land_mask, dtype=bool)   # (H,W) True = ocean
    log(f"land mask: {int(land_mask.sum())} ocean cells, {int((~land_mask).sum())} land cells")

    for name in ("tracer", "u", "v", "w", "dif"):      # (T,Z,H,W) -> (Z,H,W,T)
        write_f(out_dir / f"raw_{name}.bin", raw[name].transpose(1, 2, 3, 0))
    write_f(out_dir / "raw_mld.bin", raw["mld"].transpose(1, 2, 0))   # (H,W,T)
    land_mask.astype(np.int32).ravel(order="F").tofile(out_dir / "land_mask.bin")

    write_f(out_dir / "ref_input.bin", x_flat.numpy())                # (1,T*C*Z,H,W)
    write_f(out_dir / "ref_pred_norm.bin", pred_norm)                 # (Z,H,W)
    write_f(out_dir / "ref_pred_phys.bin", pred_phys)                 # (Z,H,W)
    write_f(out_dir / "truth_norm.bin", y[0].numpy())                 # (Z,H,W)

    dt = cfg.data.get("depth_training", {})
    info = {
        "region": args.region, "sample_in_test_split": sample, "time_index": t_idx,
        "input_times": times[:T], "target_time": times[T] if len(times) > T else None,
        "T": T, "C": C, "Z": Z, "H": H, "W": W,
        "channel_roles": [c["role"] for c in meta["channels"]],
        "depths_m": [float(d) for d in ds.depths],
        "tracer_min": tmin.tolist(), "tracer_max": tmax.tolist(),
        "train_min_depth_m": float(dt.get("min_depth_m", 0.0)),
        "train_max_depth_m": float(dt.get("max_depth_m", 1e9)),
    }
    with open(out_dir / "realdata_info.json", "w") as f:
        json.dump(info, f, indent=2)
    log(f"input times : {info['input_times']}")
    log(f"target time : {info['target_time']}")
    log(f"wrote raw fields, reference and truth to {out_dir}/")


if __name__ == "__main__":
    main()
