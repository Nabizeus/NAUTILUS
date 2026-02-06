from typing import List, Optional, Dict, Any
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from .utils import get_common_time_index, load_or_compute_norm_stats

class WindowedOceanDataset(Dataset):
    """
    Creates input/target pairs from NetCDF files.

    Inputs:
      - velocities from 3 neighbor boxes: [T, 9 (u,v,w * 3 boxes), Z, Y, X]
      - optional current tracer from central box at same times: [T, 1, Z, Y, X]
    Target:
      - tracer at t + lead_time for the CENTRAL box: [1, Z, Y, X]
    """
    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        time_slice: Optional[slice] = None,
    ):
        self.cfg = cfg
        dcfg = cfg["data"]
        # Load datasets
        self.ds_tr = xr.open_dataset(dcfg["central"]["tracer_path"])
        n_paths = dcfg["neighbors"]["velocity_paths"]
        self.ds_neighbors = [xr.open_dataset(p) for p in n_paths]

        # Names
        self.tracer_var = dcfg["central"]["tracer_var"]
        self.u_var = dcfg["neighbors"]["u_var"]
        self.v_var = dcfg["neighbors"]["v_var"]
        self.w_var = dcfg["neighbors"]["w_var"]
        self.tn = dcfg["coords"]["time"]
        self.zn = dcfg["coords"]["depth"]
        self.yn = dcfg["coords"]["y"]
        self.xn = dcfg["coords"]["x"]

        # Align time
        common_time = get_common_time_index([self.ds_tr] + self.ds_neighbors, self.tn)
        self.ds_tr = self.ds_tr.sel({self.tn: common_time})
        self.ds_neighbors = [ds.sel({self.tn: common_time}) for ds in self.ds_neighbors]

        if time_slice is not None:
            self.ds_tr = self.ds_tr.isel({self.tn: time_slice})
            self.ds_neighbors = [ds.isel({self.tn: time_slice}) for ds in self.ds_neighbors]

        # Normalization stats
        ncfg = dcfg["norm"]
        self.norm_stats = load_or_compute_norm_stats(
            self.ds_tr, self.tracer_var, self.ds_neighbors, self.u_var, self.v_var, self.w_var, ncfg.get("save_path")
        )

        self.in_steps = int(dcfg["in_steps"])
        self.lead = int(dcfg["lead_time"])
        self.stride = int(dcfg.get("stride", 1))
        self.include_tracer = bool(dcfg.get("include_current_tracer", True))

        # Preload arrays for speed (optional; for very large data, consider lazy loading)
        self.tr = self.ds_tr[self.tracer_var].values.astype(np.float32)  # [T, Z, Y, X]
        self.vels = []
        for i, ds in enumerate(self.ds_neighbors):
            u = ds[self.u_var].values.astype(np.float32)
            v = ds[self.v_var].values.astype(np.float32)
            w = ds[self.w_var].values.astype(np.float32)
            self.vels.append(np.stack([u, v, w], axis=1))  # [T, 3, Z, Y, X]
        self.vels = np.concatenate(self.vels, axis=1)  # [T, 9, Z, Y, X]

        # Expected channels
        self.in_channels = self.vels.shape[1] + (1 if self.include_tracer else 0)

        # Build indices
        self.T = self.tr.shape[0]
        self.max_start = self.T - self.in_steps - self.lead
        if self.max_start < 0:
            raise ValueError("Not enough time steps for the chosen in_steps + lead_time.")
        self.starts = np.arange(0, self.max_start + 1, self.stride)

    def __len__(self):
        return len(self.starts)

    def _norm(self, arr, mean, std):
        return (arr - mean) / std

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        e = s + self.in_steps  # not inclusive
        target_t = e + self.lead - 1  # t of target

        # Inputs
        x_vel = self.vels[s:e]  # [T, 9, Z, Y, X]
        # Normalize velocities per-neighbor-var
        x_vel_norm = x_vel.copy()
        for i in range(3): # neighbors
            for j, var in enumerate([self.u_var, self.v_var, self.w_var]):
                mean = self.norm_stats[f"{var}_mean_{i}"]
                std = self.norm_stats[f"{var}_std_{i}"]
                ch = i*3 + j
                x_vel_norm[:, ch] = (x_vel[:, ch] - mean) / std

        chans = [x_vel_norm]

        if self.include_tracer:
            x_tr = self.tr[s:e]  # [T, Z, Y, X]
            x_tr = (x_tr - self.norm_stats["tracer_mean"]) / self.norm_stats["tracer_std"]
            x_tr = x_tr[:, None, ...]  # [T, 1, Z, Y, X]
            chans.append(x_tr)

        x = np.concatenate(chans, axis=1)  # [T, C, Z, Y, X]

        # Target
        y = self.tr[target_t]  # [Z, Y, X]
        y = (y - self.norm_stats["tracer_mean"]) / self.norm_stats["tracer_std"]
        y = y[None, ...]  # [1, Z, Y, X]

        # To torch
        x = torch.from_numpy(x)         # [T, C, Z, Y, X]
        y = torch.from_numpy(y)         # [1, Z, Y, X]
        return {"x": x.float(), "y": y.float()}
