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
       

        # compute representative depth index
        with xr.open_dataset(cfg.data.tracers.paths[0]) as _ds_depth:
            depths = _ds_depth["olevel"].values
        z_index = int(np.argmin(np.abs(depths - np.mean([zmin, zmax]))))
        self.z_index = int(z_index)

        print(f"@data.py - Line 40ff - x1 x2: {x1,x2}")
        print(f"@data.py - Line 40ff - y1 y2: {y1,y2}")
        print(f"@data.py - Line 40ff - zmin zmax: {zmin,zmax}")
        print(f"@data.py - Line 40ff - z_index: {z_index}")


        tracer_path = cfg.data.tracers.paths[0]
        u_path, v_path, w_path = cfg.data.neighbors.velocity_paths


        # load and slice
        tracer = self.safe_load_and_slice(tracer_path, cfg.data.tracers.variables[0], z_index, (y1, y2), (x1, x2))
        u = self.safe_load_and_slice(u_path, cfg.data.neighbors.u_var, z_index, (y1, y2), (x1, x2))
        v = self.safe_load_and_slice(v_path, cfg.data.neighbors.v_var, z_index, (y1, y2), (x1, x2))
        w = self.safe_load_and_slice(w_path, cfg.data.neighbors.w_var, z_index, (y1, y2), (x1, x2))

        # Print shapes for debug
        print(f"🧩 Region {region_name}: tracer {tracer.shape}, u {u.shape}, v {v.shape}, w {w.shape}")

        # If time lengths mismatch (e.g. w has extra timesteps), truncate to shortest
        t_lens = [arr.shape[0] for arr in (tracer, u, v, w) if arr.size > 0]
        if len(t_lens) == 0:
            raise RuntimeError("No valid data found for region")
        t_min = min(t_lens)
        if tracer.shape[0] != t_min:
            tracer = tracer[:t_min, ...]
        if u.shape[0] != t_min:
            u = u[:t_min, ...]
        if v.shape[0] != t_min:
            v = v[:t_min, ...]
        if w.shape[0] != t_min:
            w = w[:t_min, ...]

        # If depth dim was selected (we used isel depth), arrays are (time, y, x) or (time, 1, y, x)
        # Normalize shapes so that tracer/u/v/w all become (time, y, x)
        def squeeze_depth(arr):
            if arr.ndim == 4 and arr.shape[1] == 1:
                # Z dimension exists, squeeze it safely
                print("Array dim is 4!!")
                return arr[:, z_index, :, :] # pick the first (or representative) layer
            elif arr.ndim == 4 and arr.shape[1] > 1:
                # multiple depth levels, collapse with mean
                return np.mean(arr, axis=1)
            if arr.ndim == 3:
                return arr # already (T,H,W)
            return arr  # if already 2D spatial plus time

        tracer = squeeze_depth(tracer)
        u = squeeze_depth(u)
        v = squeeze_depth(v)
        w = squeeze_depth(w)

        #print(f"tracer.shape {tracer.shape} ")
        #print(f"u.shape {u.shape} ")
        #print(f"v.shape {v.shape} ")
        #print(f"w.shape {w.shape} ")



        # last safety: clamp spatial sizes to minimum common (Y,X)
        y_sizes = [a.shape[1] for a in (tracer, u, v, w) if a.ndim==3]
        x_sizes = [a.shape[2] for a in (tracer, u, v, w) if a.ndim==3]
        y_min = min(y_sizes)
        x_min = min(x_sizes)
        tracer = tracer[:, :y_min, :x_min]
        u = u[:, :y_min, :x_min]
        v = v[:, :y_min, :x_min]
        w = w[:, :y_min, :x_min]

        

        #tracer = np.nan_to_num(tracer)
        #u = np.nan_to_num(u)
        #v = np.nan_to_num(v)
        #w = np.nan_to_num(w)


        self.tracer, self.u, self.v, self.w = tracer.astype(np.float32), u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)

        # recompute timesteps and num_samples
        self.timesteps = self.tracer.shape[0]
        self.num_samples = self.timesteps - self.window.input_steps - self.window.pred_steps
        print(f"✅ Region '{region_name}' final shapes tracer {self.tracer.shape}, u {self.u.shape} -> samples {self.num_samples}")

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

        print(f"[idx:idx + t_in] {idx}:{idx + t_in}")



        #print(f"x_tracer.shape {x_tracer.shape} ")
        #print(f"x_u.shape {x_u.shape} ")
        #print(f"x_v.shape {x_v.shape} ")
        #print(f"x_w.shape {x_w.shape} ")



        # --- Target ---
        y_true = self.tracer[idx_out:idx_out + t_out]

        #print(f"y_true.shape {y_true.shape} ")

        # --- Stack inputs as (T, C, Z, Y, X) ---
        #x = np.stack([x_tracer, x_u, x_v, x_w], axis=1)  # C = 4 (tracer + 3 velociti)
        #x = torch.tensor(x, dtype=torch.float32)
        #y = torch.tensor(y_true, dtype=torch.float32)


        # (T, C, Z, Y, X)
        #print(f"x_tracer.shape {x_tracer.shape} ")
        #print(f"x_u.shape {x_u.shape} ")
        #print(f"x_v.shape {x_v.shape} ")
        #print(f"x_w.shape {x_w.shape} ")


        # --- Enforce consistent shapes ---
        # min_shape = np.min([x_tracer.shape, x_u.shape, x_v.shape, x_w.shape], axis=0)

        #x_tracer = x_tracer[..., :min_shape[-2], :min_shape[-1]]
        #x_u = x_u[..., :min_shape[-2], :min_shape[-1]]
        #x_v = x_v[..., :min_shape[-2], :min_shape[-1]]
        #x_w = x_w[..., :min_shape[-2], :min_shape[-1]]




        # Stack all channels into the channel dimension
        # Each has shape (T, H, W)

        x = np.stack([x_tracer, x_u, x_v, x_w], axis=1) # (TIME, CHANNEL/VARIABLE=4, HEIGHT BOX, WIDTH BOX)
                       
        
        # --- Build mask for valid ocean points ---
        # 1 where tracer data is valid (non-zero after NaN cleaning)
        mask = (y_true != 0).astype(np.float32)  # 1 = ocean, 0 = land
        
        # --- Convert to tensors ---
        x = torch.tensor(x, dtype=torch.float32)
        #print(f"@data.py - Line 184 - y_true.shape {y_true.shape} ")
        y = torch.tensor(y_true, dtype=torch.float32)
        #print(f"@data.py - Line 186 - y.shape {y.shape} ")
        mask = torch.tensor(mask, dtype=torch.float32)

        return x, y, mask


    # helper to load & safely slice a variable by dim names
    def safe_load_and_slice(self,nc_path, varname, z_idx, y_slice, x_slice):
        


        with xr.open_dataset(nc_path) as ds_local:
            var = ds_local[varname]
            dims = var.dims  # e.g. ('time_counter','olevel','y','x')
            sizes = {d: var.sizes[d] for d in dims}

            # assume last two dims are spatial (works for your CF-style files)
            y_dim = dims[-2]
            x_dim = dims[-1]

            # clamp user indices to available range
            y0 = max(0, int(y_slice[0]))
            y1c = min(int(y_slice[1]), sizes[y_dim])
            x0 = max(0, int(x_slice[0]))
            x1c = min(int(x_slice[1]), sizes[x_dim])

            if y0 >= y1c or x0 >= x1c:
                # completely out of range -> return empty array with shape (time, z?, 0, 0)
                print(f"⚠️ slice [{y0}:{y1c}, {x0}:{x1c}] out of bounds for {nc_path} var {varname} dims {dims} sizes {sizes}")
                # create minimal shaped array (time, z?, 0, 0) depending on how many dims
                tlen = var.sizes[dims[0]]
                zlen = 1 if 'olevel' in dims else 1
                return np.zeros((tlen, zlen, 0, 0), dtype=np.float32)

            # build isel dict: select z index if present, else leave z dim
            isel_dict = {}
            # if there is a depth-like dim name, find it
            depth_dim = None
            for d in dims:
                if "olevel" in d or "depth" in d or d.lower().startswith("z"):
                    depth_dim = d
                    break

            if depth_dim is not None:
                isel_dict[depth_dim] = z_idx

            isel_dict[y_dim] = slice(y0, y1c)
            isel_dict[x_dim] = slice(x0, x1c)

            arr = var.isel(isel_dict).values  # numpy array
            return arr







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

