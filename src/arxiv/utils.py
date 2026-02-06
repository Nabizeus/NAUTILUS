import os, json, math, time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import xarray as xr
import torch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_or_compute_norm_stats(config, *args, **kwargs):
    """
    Compute (or load cached) normalization statistics (mean/std) for tracer and velocity data.
    Supports both separate velocity files and combined neighbor datasets.
    """
    # Handle flexible argument passing
    # args can be: (tracer_datasets, neighbor_datasets, out_path) or more depending on call site
    tracer_datasets = args[0] if len(args) > 0 else []
    neighbor_datasets = args[1] if len(args) > 1 else []
    out_path = args[2] if len(args) > 2 else "norm_stats.json"

    import os, json, numpy as np, xarray as xr

    if os.path.exists(out_path):
        print(f"✅ Loading normalization stats from {out_path}")
        with open(out_path, "r") as f:
            return json.load(f)

    print("📊 Computing normalization stats from datasets...")
    stats = {}

    # --- 1️⃣ Tracers ---
    tracer_vars = config.data.tracers.variables
    for ds in tracer_datasets:
        for varname in tracer_vars:
            if varname not in ds:
                raise KeyError(f"Tracer variable '{varname}' not found. Available: {list(ds.data_vars)}")
            arr = ds[varname].values.astype(np.float32)
            stats[f"{varname}_mean"] = float(np.nanmean(arr))
            stats[f"{varname}_std"] = float(np.nanstd(arr) + 1e-6)

    # --- 2️⃣ Velocities ---
    vel_paths = config.data.neighbors.velocity_paths
    u_var = config.data.neighbors.u_var
    v_var = config.data.neighbors.v_var
    w_var = config.data.neighbors.w_var

    # Case A: 3 separate velocity files (U, V, W)
    if len(vel_paths) == 3:
        print("🔹 Using separate velocity files for U, V, W")
        for path, varname in zip(vel_paths, [u_var, v_var, w_var]):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Velocity file not found: {path}")
            print(f"  Inspecting {os.path.basename(path)} for {varname} ...")
            ds = xr.open_dataset(path)
            if varname not in ds:
                raise KeyError(f"Variable '{varname}' not found in {path}. Available: {list(ds.data_vars)}")
            arr = ds[varname].values.astype(np.float32)
            stats[f"{varname}_mean"] = float(np.nanmean(arr))
            stats[f"{varname}_std"] = float(np.nanstd(arr) + 1e-6)
            ds.close()

    # Case B: Combined neighbor datasets
    else:
        print("🔹 Using neighbor datasets with all U/V/W in each")
        for i, ds in enumerate(neighbor_datasets):
            for varname in [u_var, v_var, w_var]:
                if varname not in ds:
                    raise KeyError(f"Variable '{varname}' not found in neighbor {i}. Available: {list(ds.data_vars)}")
                arr = ds[varname].values.astype(np.float32)
                stats[f"{varname}_mean_{i}"] = float(np.nanmean(arr))
                stats[f"{varname}_std_{i}"] = float(np.nanstd(arr) + 1e-6)

    # --- 3️⃣ Save ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Normalization stats saved to {out_path}")
    return stats





def get_common_time_index(dsets, time_name: str):
    """Return the intersection time index across all datasets."""
    times = [xr.IndexVariable(time_name, ds[time_name].values) for ds in dsets]
    # Intersect by values
    common = set(times[0].values)
    for t in times[1:]:
        common = common.intersection(set(t.values))
    common = np.array(sorted(list(common)))
    if common.size == 0:
        raise ValueError("No overlapping time steps across provided datasets.")
    return common

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch
