import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from src.utils import load_or_compute_norm_stats


class WindowedOceanDataset(Dataset):
    def __init__(self, cfg, split="train", region_name=None):
        self.cfg = cfg
        self.split = split
        self.window = cfg.data.window
        self.regions = cfg.data.get("regions", None)
        self.visualize_regions = cfg.data.get("visualize_regions", False)


        # --- Per-region mode (Option C) ---
        if region_name is None:
            raise ValueError("❌ Must specify region_name for single-region dataset.")

        if self.visualize_regions:
            self._plot_region_boxes(cfg.data.tracers.paths[0])

        # Find the region info
        region_cfg = next((r for r in self.regions if r["name"] == region_name), None)
        if region_cfg is None:
            raise ValueError(f"❌ Region '{region_name}' not found in config.data.regions.")

        x1, x2 = region_cfg["x_range"]
        y1, y2 = region_cfg["y_range"]
        zmin, zmax = region_cfg["z_range"]

        # Determine the correct depth index from NetCDF
        with xr.open_dataset(cfg.data.tracers.paths[0]) as ds_depth:
            depths = ds_depth["olevel"].values
        z_index = np.argmin(np.abs(depths - np.mean([zmin, zmax])))

        tracer_path = cfg.data.tracers.paths[0]
        u_path, v_path, w_path = cfg.data.neighbors.velocity_paths

        with xr.open_dataset(tracer_path) as dtrc, \
             xr.open_dataset(u_path) as du, \
             xr.open_dataset(v_path) as dv, \
             xr.open_dataset(w_path) as dw:

            tracer = dtrc[cfg.data.tracers.variables[0]].isel(
                olevel=z_index, y=slice(y1, y2), x=slice(x1, x2)
            ).values
            u = du[cfg.data.neighbors.u_var].isel(
                olevel=z_index, y=slice(y1, y2), x=slice(x1, x2)
            ).values
            v = dv[cfg.data.neighbors.v_var].isel(
                olevel=z_index, y=slice(y1, y2), x=slice(x1, x2)
            ).values
            w = dw[cfg.data.neighbors.w_var].isel(
                olevel=z_index, y=slice(y1, y2), x=slice(x1, x2)
            ).values

            # --- Ensure spatial consistency with tracer region ---
            u = np.nan_to_num(u[..., y1:y2, x1:x2])
            v = np.nan_to_num(v[..., y1:y2, x1:x2])
            w = np.nan_to_num(w[..., y1:y2, x1:x2])
            tracer = np.nan_to_num(tracer)

            print(f"🧩 Region {region_name}: tracer {tracer.shape}, u {u.shape}, v {v.shape}, w {w.shape}")



        # Clean invalid values
        for arr, nm in zip([tracer, u, v, w], ["tracer", "u", "v", "w"]):
            nans = np.isnan(arr).sum()
            if nans > 0:
                print(f"⚠️ {region_name}: Replacing {nans:,} NaNs in {nm}")
                arr[np.isnan(arr)] = 0.0

        self.tracer, self.u, self.v, self.w = tracer, u, v, w
        self.num_samples = self.tracer.shape[0] - self.window.input_steps - self.window.pred_steps
        print(f"✅ Region '{region_name}' loaded with {self.num_samples} samples at depth index {z_index}")


        self.velocity_paths = cfg.data.neighbors.velocity_paths
        self.norm_stats = load_or_compute_norm_stats(
            cfg, self.tracer, [], cfg.data.norm_stats_path
        )

        # Preload velocity datasets (U/V/W)
        self.u = xr.open_dataset(self.velocity_paths[0])[cfg.data.neighbors.u_var].values
        self.v = xr.open_dataset(self.velocity_paths[1])[cfg.data.neighbors.v_var].values
        self.w = xr.open_dataset(self.velocity_paths[2])[cfg.data.neighbors.w_var].values

        

        # --- Load tracer variable ---
        tracer_name = cfg.data.tracers.variables[0]
        print("Tracer name: ",tracer_name)
        ds = self.tracer[0]
        #print("ds: ", ds)
        print("Varname: ", tracer_name)

        # Extract variable array
        #print("type(ds)", type(ds))
        #print(f"ds[tracer_name] {ds[tracer_name]}")
        #arr = ds[tracer_name].values.astype(np.float32)
        # ds is already the array, just convert its data type
        #arr = ds.astype(np.float32)
        # --- Crop region and depth ---
        # ✅ Use regional crop instead of global crop
        
        
        arr = self.tracer


        

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
        self.timesteps = self.tracer.shape[0]

    def __len__(self):
        return self.timesteps - self.window.input_steps - self.window.pred_steps

    def __getitem__(self, idx):
        t_in = self.window.input_steps
        t_out = self.window.pred_steps
        idx_out = idx + t_in


        # --- Input sequences (T, Z, Y, X) ---
        x_tracer = self.tracer[idx:idx + t_in]
        x_u = self.u[idx:idx + t_in]
        x_v = self.v[idx:idx + t_in]
        x_w = self.w[idx:idx + t_in]

        print(f"[idx:idx + t_in] {[idx:idx + t_in]}"



        print(f"x_tracer.shape {x_tracer.shape} ")
        print(f"x_u.shape {x_u.shape} ")
        print(f"x_v.shape {x_v.shape} ")
        print(f"x_w.shape {x_w.shape} ")



        # --- Target ---
        y_true = self.tracer[idx_out:idx_out + t_out]

        print(f"y_true.shape {y_true.shape} ")

        # --- Stack inputs as (T, C, Z, Y, X) ---
        #x = np.stack([x_tracer, x_u, x_v, x_w], axis=1)  # C = 4 (tracer + 3 velociti)
        #x = torch.tensor(x, dtype=torch.float32)
        #y = torch.tensor(y_true, dtype=torch.float32)


        # (T, C, Z, Y, X)
        print(f"x_tracer.shape {x_tracer.shape} ")
        print(f"x_u.shape {x_u.shape} ")
        print(f"x_v.shape {x_v.shape} ")
        print(f"x_w.shape {x_w.shape} ")


        # --- Enforce consistent shapes ---
        min_shape = np.min([x_tracer.shape, x_u.shape, x_v.shape, x_w.shape], axis=0)

        x_tracer = x_tracer[..., :min_shape[-2], :min_shape[-1]]
        x_u = x_u[..., :min_shape[-2], :min_shape[-1]]
        x_v = x_v[..., :min_shape[-2], :min_shape[-1]]
        x_w = x_w[..., :min_shape[-2], :min_shape[-1]]





        x = np.stack([x_tracer, x_u, x_v, x_w], axis=1)

        # ✅ Collapse depth dimension to 2D — either by mean or by selecting first layer
        x = np.nanmean(x, axis=2)  # average over depth (Z)
        y = np.nanmean(y_true, axis=1, keepdims=True)  # match shape (1, Y, X)

        # Convert to torch
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)



        # --- Build mask for valid ocean points ---
        # 1 where tracer data is valid (non-zero after NaN cleaning)
        mask = (y_true != 0).astype(np.float32)  # 1 = ocean, 0 = land
        
        # --- Convert to tensors ---
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y_true, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return x, y, mask


    def _plot_region_boxes(self, nc_path):
         """Visualize defined regions and depth slices."""
         import matplotlib.pyplot as plt
         with xr.open_dataset(nc_path) as ds:
             lon = ds["nav_lon"].values
             lat = ds["nav_lat"].values
 
         plt.figure(figsize=(8, 4))
         plt.pcolormesh(lon, lat, np.zeros_like(lon), cmap="Greys", alpha=0.1)
 
         for reg in self.region_data:
             x1, x2, y1, y2 = reg["x1"], reg["x2"], reg["y1"], reg["y2"]
             plt.plot(
                 [lon[y1, x1], lon[y1, x2], lon[y2, x2], lon[y2, x1], lon[y1, x1]],
                 [lat[y1, x1], lat[y1, x2], lat[y2, x2], lat[y2, x1], lat[y1, x1]],
                 label=f"{reg['name']} (z={reg['z_index']})"
             )
 
         plt.legend()
         plt.title("3D Region Boxes (surface projection)")
         plt.xlabel("Longitude")
         plt.ylabel("Latitude")
         plt.tight_layout()
         plt.savefig("artifacts/region_boxes.png", dpi=150)
         print("📸 Saved region visualization: artifacts/region_boxes.png")

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



    # ==========================================================
    # 🔹 Multi-region cropping and single-layer selection
    # ==========================================================
    def crop_to_regions(self, ds, arr, depth_coord="olevel"):
        """
        Crop dataset to multiple predefined 3D regions and extract one representative layer per region.
        Also saves diagnostic plots if visualize_regions=True in config.
        """
        regions = getattr(self.cfg.data, "regions", None)
        selected_layer = getattr(self.cfg.data, "selected_layer_index", 0)
        visualize = getattr(self.cfg.data, "visualize_regions", False)

        if not regions:
            print("⚠️ No regions defined, skipping crop_to_regions()")
            return arr

        lats = ds.nav_lat.values
        lons = ds.nav_lon.values
        depths = ds[depth_coord].values

        region_crops = []
        for region in regions:
            x1, x2 = region["x_range"]
            y1, y2 = region["y_range"]
            z1, z2 = region["z_range"]

            # Convert z range to nearest indices
            z_mask = (depths >= z1) & (depths <= z2)
            z_indices = np.where(z_mask)[0]
            if len(z_indices) == 0:
                print(f"⚠️ No matching depths for region {region['name']}, skipping.")
                continue

            # Select representative layer within this range
            selected_z = z_indices[min(selected_layer, len(z_indices) - 1)]

            # Crop 3D box
            cropped = arr[:, selected_z:selected_z + 1, y1:y2, x1:x2]
            region_crops.append(cropped)

            # Optional visualization
            if visualize:
                plt.figure(figsize=(6, 5))
                data_2d = np.nan_to_num(cropped[0, 0, :, :], nan=0.0)
                plt.imshow(data_2d, origin="lower", cmap="viridis")
                plt.title(f"{region['name']} | Layer {selected_z} ({depths[selected_z]:.2f} m)")
                plt.colorbar(label="Value")
                plt.tight_layout()
                out_path = f"artifacts/region_{region['name']}_layer{selected_z}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"🗺️ Saved region plot: {out_path}")

        if len(region_crops) == 0:
            raise ValueError("❌ No valid regions found in crop_to_regions().")

        arr = np.concatenate(region_crops, axis=0)
        print(f"✅ Cropped to {len(region_crops)} regions. New shape: {arr.shape}")
        return arr

