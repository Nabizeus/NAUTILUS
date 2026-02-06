import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from src.utils import load_or_compute_norm_stats


class WindowedOceanDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.window = cfg.data.window
        self.crop = cfg.data.spatial_crop
        self.tracer_datasets = [xr.open_dataset(p) for p in cfg.data.tracers.paths]
        self.velocity_paths = cfg.data.neighbors.velocity_paths
        self.norm_stats = load_or_compute_norm_stats(
            cfg, self.tracer_datasets, [], cfg.data.norm_stats_path
        )

        # Preload velocity datasets (U/V/W)
        self.u = xr.open_dataset(self.velocity_paths[0])[cfg.data.neighbors.u_var].values
        self.v = xr.open_dataset(self.velocity_paths[1])[cfg.data.neighbors.v_var].values
        self.w = xr.open_dataset(self.velocity_paths[2])[cfg.data.neighbors.w_var].values

        

        # --- Load tracer variable ---
        tracer_name = cfg.data.tracers.variables[0]
        print("Tracer name: ",tracer_name)
        ds = self.tracer_datasets[0]
        print("ds: ", ds)
        print("Varname: ", tracer_name)

        # Extract variable array
        arr = ds[tracer_name].values.astype(np.float32)

        # --- Crop region and depth ---
        arr = self.crop_region_and_depth(ds, arr)

        # --- Clean data: replace NaN or Inf with 0 ---
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"⚠️ Cleaning invalid values{tracer_name}")
            n_nan = np.isnan(arr).sum()
            n_inf = np.isinf(arr).sum()
            print(f"⚠️ Found {n_nan} NaNs and{n_inf} Infs in {tracer_name}. Replacing with 0.")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional: Fill masked ocean/land regions with mean instead of zeros
        if np.any(arr == 0.0):
            mean_val = np.mean(arr[arr != 0.0])
            arr[arr == 0.0] = mean_val


        self.tracer = arr



        # Spatial crop
        y1, y2, x1, x2 = self.crop
        self.tracer = self.tracer[..., y1:y2, x1:x2]
        self.u = self.u[..., y1:y2, x1:x2]
        self.v = self.v[..., y1:y2, x1:x2]
        self.w = self.w[..., y1:y2, x1:x2]

        self.timesteps = self.tracer.shape[0]

    def __len__(self):
        return self.timesteps - self.window.input_steps - self.window.pred_steps

    def __getitem__(self, idx):
        t_in = self.window.input_steps
        t_out = self.window.pred_steps
        idx_out = idx + t_in

        x_tracer = self.tracer[idx:idx + t_in]
        x_u = self.u[idx:idx + t_in]
        x_v = self.v[idx:idx + t_in]
        x_w = self.w[idx:idx + t_in]

        y_true = self.tracer[idx_out:idx_out + t_out]

        x = np.stack([x_tracer, x_u, x_v, x_w], axis=1)  # (T, C, Z, Y, X)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y_true, dtype=torch.float32)
        return x, y

    def crop_region_and_depth(self, ds, arr):
        """
        Crop array by region and depth safely.
        Handles both (Z, Y, X) and (T, Z, Y, X) inputs.
        """
        # ---- Depth crop ----
        if hasattr(self, "depth_crop") and self.depth_crop is not None:
            z1, z2 = self.depth_crop
            arr = arr[..., z1:z2, :, :] if arr.ndim == 4 else arr[z1:z2, :, :]
            print(f"🧭 Depth cropped to levels {z1}:{z2}")

        # ---- Spatial crop ----
        if hasattr(self, "region_mask") and self.region_mask is not None:
            lat_mask, lon_mask = self.region_mask
            # For 3D or 4D arrays
            if arr.ndim == 4:
                arr = arr[..., lat_mask, :][..., :, lon_mask]
            elif arr.ndim == 3:
                arr = arr[:, lat_mask, :][:, :, lon_mask]
            print("🗺️ Applied regional mask to array.")

        return arr

