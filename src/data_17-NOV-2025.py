import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from src.utils import load_or_compute_norm_stats
import os
import matplotlib.pyplot as plt


# ==============================================================
#  Utility: Debug plot for raw & normalized variables
# ==============================================================

def debug_plot_normalization(region, split, tracer_raw, u_raw, v_raw, w_raw,
                             tracer_norm, u_norm, v_norm, w_norm,
                             t_idx=0, outdir="artifacts/debug_norm"):

    os.makedirs(outdir, exist_ok=True)

    def plot_pair(raw, norm, name):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(raw, cmap="viridis")
        plt.colorbar()
        plt.title(f"{name} RAW (t={t_idx})")

        plt.subplot(1, 2, 2)
        plt.imshow(norm, cmap="viridis")
        plt.colorbar()
        plt.title(f"{name} NORMALIZED (t={t_idx})")

        plt.tight_layout()
        plt.savefig(f"{outdir}/{region}_{split}_{name}_t{t_idx}.png")
        plt.close()

    plot_pair(tracer_raw, tracer_norm, "tracer")
    plot_pair(u_raw, u_norm, "u")
    plot_pair(v_raw, v_norm, "v")
    plot_pair(w_raw, w_norm, "w")

    print(f"📊 Saved normalization debug plots → {outdir}/")


# ==============================================================
#  Windowed Dataset Class
# ==============================================================



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

    

        # Find the region info
        region_cfg = next((r for r in self.regions if r["name"] == region_name), None)
        if region_cfg is None:
            raise ValueError(f"❌ Region '{region_name}' not found in config.data.regions.")
    
        # Draw only this region
        if self.visualize_regions:
            self._plot_region_boxes(cfg.data.tracers.paths[0], [region_cfg])
    
        x1, x2 = region_cfg["x_range"]
        y1, y2 = region_cfg["y_range"]
        zmin, zmax = region_cfg["z_range"]
       

        # compute representative depth index
        with xr.open_dataset(cfg.data.tracers.paths[0]) as _ds_depth:
            depths = _ds_depth["olevel"].values
        z_index = int(np.argmin(np.abs(depths - np.mean([zmin, zmax]))))
        self.z_index = int(z_index)

        #print(f"@data.py - Line 40ff - x1 x2: {x1,x2}")
        #print(f"@data.py - Line 40ff - y1 y2: {y1,y2}")
        #print(f"@data.py - Line 40ff - zmin zmax: {zmin,zmax}")
        #print(f"@data.py - Line 40ff - z_index: {z_index}")
        #print(f"@data.py - Line 40ff - z_level [m]: {depths[z_index]:.2f}")
        self.H = int(x2-x1)
        self.W = int(y2-y1)
        self.depthz = depths[z_index]
        
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



        # ============================================================
        #  NORMALIZATION (PER-VARIABLE, TRAIN-ONLY)
        # ============================================================

        stats_file = f"norm_stats_{region_name}.npz"

        if split == "train":
            self.stats = {
                "tracer_mean": float(np.nanmean(self.tracer)),
                "tracer_std":  float(np.nanstd(self.tracer) + 1e-6),
                "u_mean":      float(np.nanmean(self.u)),
                "u_std":       float(np.nanstd(self.u) + 1e-6),
                "v_mean":      float(np.nanmean(self.v)),
                "v_std":       float(np.nanstd(self.v) + 1e-6),
                "w_mean":      float(np.nanmean(self.w)),
                "w_std":       float(np.nanstd(self.w) + 1e-6),
            }

            np.savez(stats_file, **self.stats)
            print(f"📊 Saved normalization stats → {stats_file}")

        else:
            self.stats = dict(np.load(stats_file))
            print(f"📥 Loaded normalization stats from {stats_file}")

        # Store raw snapshot for debug before normalization
        idx0 = 0
        raw_t = self.tracer[idx0].copy()
        raw_u = self.u[idx0].copy()
        raw_v = self.v[idx0].copy()
        raw_w = self.w[idx0].copy()

        # Apply normalization
        self.tracer = (self.tracer - self.stats["tracer_mean"]) / self.stats["tracer_std"]
        self.u      = (self.u      - self.stats["u_mean"])      / self.stats["u_std"]
        self.v      = (self.v      - self.stats["v_mean"])      / self.stats["v_std"]
        self.w      = (self.w      - self.stats["w_mean"])      / self.stats["w_std"]

        print(f"✅ Applied per-variable normalization for {split}")

        # Save debug plots
        debug_plot_normalization(
            region_name, split,
            raw_t, raw_u, raw_v, raw_w,
            self.tracer[idx0], self.u[idx0], self.v[idx0], self.w[idx0],
            t_idx=idx0
        )


        assert not np.isnan(self.tracer).any(), "NaNs after tracer normalization!"
        assert not np.isnan(self.u).any(), "NaNs after u normalization!"
        assert not np.isnan(self.v).any(), "NaNs after v normalization!"
        assert not np.isnan(self.w).any(), "NaNs after w normalization!"



        # ============================================================
        #  Build window indices
        # ============================================================

        T_in  = cfg.data.window.input_steps
        T_out = cfg.data.window.pred_steps

        self.T_in = T_in
        self.T_out = T_out

        T = self.tracer.shape[0]

        self.indices = []
        for t in range(0, T - (T_in + T_out) + 1):
            self.indices.append((t, t + T_in, t + T_in + T_out))

        print(f"📐 Total samples for {region_name}/{split}: {len(self.indices)}")



        # recompute timesteps and num_samples
        self.timesteps = self.tracer.shape[0]
        self.num_samples = self.timesteps - self.window.input_steps - self.window.pred_steps
        




        


        # ✅ Region printout (before split)
        print(f"✅ Region '{region_name}' final shapes tracer {self.tracer.shape}, u {self.u.shape} -> samples {self.num_samples}")

        # ---------------------------------------------------------
        # 🧭 Time-based Split for Train / Val / Test
        # ---------------------------------------------------------
        total_len = self.tracer.shape[0]
        input_steps = self.window.input_steps
        pred_steps = self.window.pred_steps
        usable_len = total_len - (input_steps + pred_steps)

        # Define 80/10/10 split along the time axis
        train_end = int(0.8 * usable_len)
        val_end   = int(0.9 * usable_len)

        if self.split == "train":
            self.t_start, self.t_end = 0, train_end
        elif self.split == "val":
            self.t_start, self.t_end = train_end, val_end
        else:  # test
            self.t_start, self.t_end = val_end, usable_len

        print(f"🧩 Using split='{self.split}' with time indices {self.t_start}:{self.t_end} of {usable_len}")




    def __len__(self):
        return self.t_end - self.t_start

    def __getitem__(self, idx):
        t_in = self.window.input_steps
        t_out = self.window.pred_steps
        idx_out = idx + self.t_start


        # --- Input sequences (T, Z, Y, X) ---
        x_tracer = self.tracer[idx:idx + t_in]
        x_u = self.u[idx:idx + t_in]
        x_v = self.v[idx:idx + t_in]
        x_w = self.w[idx:idx + t_in]

        #print(f"[idx:idx + t_in] {idx}:{idx + t_in}")



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
                #print(f"⚠️ slice [{y0}:{y1c}, {x0}:{x1c}] out of bounds for {nc_path} var {varname} dims {dims} sizes {sizes}")
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
            #print(f"@data.py - line 247 -def safe_load_slice() - isel_dict: {isel_dict}")
            arr = var.isel(isel_dict).values  # numpy array
            return arr

    def _plot_region_boxes(self, nc_path, regions=None):
        """Visualize region boxes with index labels and grid ticks."""
        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        if regions is None:
            if not hasattr(self, "region_data"):
                print("⚠️ No region_data found — skipping region plot.")
                return
            regions = self.region_data

        with xr.open_dataset(nc_path) as ds:
            ny, nx = ds[self.cfg.data.tracers.variables[0]].shape[-2:] 
            depths = ds["olevel"].values            
          

        plt.figure(figsize=(9, 5))
   
        plt.title("Defined Regions in Grid Index Space")
        plt.xlabel("X index")
        plt.ylabel("Y index")

        for reg in regions:
            name = reg.get("name", "region")
            # ✅ Support both YAML-style (x_range, y_range) and flattened x1/x2 format
           
            zmin, zmax = reg["z_range"]
            x1, x2 = reg["x_range"]
            y1, y2 = reg["y_range"]
            
            z_index = int(np.argmin(np.abs(depths - np.mean([zmin, zmax]))))
            self.z_index = int(z_index)
            print(f"@data.py - Line 284ff - z_level [m]: {depths[z_index]:.2f}")

    
            # Draw box using 5 points (closed loop)
            x_corners = [x1, x2, x2, x1, x1]
            y_corners = [y1, y1, y2, y2, y1]

            plt.plot(x_corners, y_corners, "-", lw=1.5, label=reg["name"])

            # Set axis range to full model grid
            plt.xlim(0, 362)   # X-direction (longitude index)
            plt.ylim(0, 292)   # Y-direction (latitude index)


            # Center label (avoid list error!)
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            plt.text(xc, yc, reg["name"]+f"@ z_lev {depths[z_index]:.2f} m", fontsize=8, color="red")

            plt.legend(loc="upper right", fontsize=7)
            plt.tight_layout()
            #plt.savefig(f"artifacts/region_boxes_{name}.png", dpi=150)
            plt.close()




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

