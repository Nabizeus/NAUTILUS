# src/io/nemo_restart_writer.py
"""
Write emulator predictions into a NEMO-compatible restart file.
Only predictions are written (no truths).

Assumes predictions are already denormalized and 2D (H,W)
for a single depth level z_index.
"""

import os
import xarray as xr
import numpy as np

def write_nemo_restart(pred_field, dataset, region_name,
                       varname="mytrc",
                       z_index=None,
                       out_dir="nemo_restart",
                       overwrite=True):
    """
    Write predictions into a NEMO restart-style NetCDF file.

    Args:
        pred_field : np.ndarray of shape (H, W)
        dataset    : WindowedOceanDataset used for coords & metadata
        region_name: region name string
        varname    : tracer variable name in NEMO restart (e.g., 'mytrc')
        z_index    : depth index (scalar)
        out_dir    : output directory for file
        overwrite  : if True, overwrites existing files

    Returns:
        path to written restart file
    """

    os.makedirs(out_dir, exist_ok=True)

    # Ensure shape is (H,W)
    arr = np.asarray(pred_field)
    if arr.ndim != 2:
        raise ValueError(f"Prediction field must be 2D (H,W). Got shape {arr.shape}")

    H, W = arr.shape

    # Build depth coordinate
    if z_index is None:
        z_index = getattr(dataset, "z_index", 0)

    # Build coordinates following NEMO order: (time_counter, olevel, y, x)
    coords = {
        "time_counter": np.array([0]),   # restart usually single snapshot
        "olevel": np.array([z_index]),
        "y": np.arange(H),
        "x": np.arange(W),
    }

    # Build data variable for restart
    data_vars = {
        varname: (("time_counter", "olevel", "y", "x"),
                  arr[np.newaxis, np.newaxis, :, :].astype(np.float32))
    }

    # Try to attach nav_lat/nav_lon if dataset provides them
    attrs = {}
    try:
        if hasattr(dataset, "nav_lat") and hasattr(dataset, "nav_lon"):
            nav_lat = np.asarray(dataset.nav_lat)
            nav_lon = np.asarray(dataset.nav_lon)
            if nav_lat.shape == (H, W) and nav_lon.shape == (H, W):
                data_vars["nav_lat"] = (("y", "x"), nav_lat.astype(np.float32))
                data_vars["nav_lon"] = (("y", "x"), nav_lon.astype(np.float32))
    except Exception:
        pass  # don't fail if grids are unavailable

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Output file path
    out_path = os.path.join(out_dir, f"{region_name}_{varname}_restart.nc")

    if (not overwrite) and os.path.exists(out_path):
        raise FileExistsError(out_path)

    # Compression for NEMO compatibility
    encoding = {varname: {"zlib": True, "complevel": 4}}

    ds.to_netcdf(out_path, encoding=encoding)

    print(f"✅ NEMO restart file saved → {out_path}")
    print(f"• variable: {varname}")
    print(f"• depth index: {z_index}")
    print(f"• shape written: (1, 1, {H}, {W})")

    return out_path

