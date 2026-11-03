import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from src.utils import load_or_compute_norm_stats, compute_mass_time_series
import os
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from matplotlib import colors



def get_depth_indices(z_center, depth_stack, max_z):
    idxs = []
    for dz in depth_stack:
        zi = z_center + dz
        zi = max(0, min(max_z - 1, zi))
        idxs.append(zi)
    return sorted(list(set(idxs)))



# For 3D multi-Z levels

def load_multi_z(var, z_indices):
    return var.isel(olevel=z_indices).values



def plot_land_mask(mask, region_name, split, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(mask.astype(int), cmap="gray")
    plt.title(f"Land–Ocean Mask | {region_name} | {split}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="1 = Ocean, 0 = Land")
    plt.tight_layout()

    path = os.path.join(out_dir, f"land_mask_{region_name}_{split}.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"🗺️ Land–ocean mask saved → {path}")





# ---- helpers for saving/loading normalization stats ----
def save_norm_stats_npz(path, stats: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **stats)

def save_norm_stats_yaml(path, stats: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(stats, f)

def load_norm_stats_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================
#  Utility: Debug plot for raw & normalized variables
# ==============================================================

def debug_plot_normalization(region, split, tracer_raw, u_raw, v_raw, w_raw, dif_raw, mld_distance_raw,
                             tracer_norm, u_norm, v_norm, w_norm, dif_norm, mld_distance_norm,
                             t_idx=0, outdir="artifacts/debug_norm"):

    os.makedirs(outdir, exist_ok=True)

    def plot_pair(raw, norm, name):


        def pick_2d(arr):
            if arr.ndim == 4:
                return arr[0, 0]
            elif arr.ndim == 3:
                return arr[0]
            else:
                return arr


        raw2d  = pick_2d(raw)
        norm2d = pick_2d(norm)

        plt.subplot(1, 2, 1)
        plt.imshow(raw2d, cmap="viridis")
        plt.colorbar()
        plt.title(f"{name} RAW (t={t_idx})")

        plt.subplot(1, 2, 2)
        plt.imshow(norm2d, cmap="viridis")
        plt.colorbar()
        plt.title(f"{name} NORMALIZED (t={t_idx})")

        plt.tight_layout()
        plt.savefig(f"{outdir}/{region}_{split}_{name}_t{t_idx}.png")
        plt.close()

    plot_pair(tracer_raw, tracer_norm, "tracer")
    plot_pair(u_raw, u_norm, "u")
    plot_pair(v_raw, v_norm, "v")
    plot_pair(w_raw, w_norm, "w")
    plot_pair(dif_raw, dif_norm, "dif")
    plot_pair(mld_distance_raw, mld_distance_norm, "mld_distance")


    print(f"📊 Saved normalization debug plots → {outdir}/")


def plot_histograms_of_vars(region, split, raw_dict, norm_dict, outdir="artifacts/hists"):
    """Create histograms of raw and normalized variables side-by-side (multi-depth safe)."""
    os.makedirs(outdir, exist_ok=True)

    for var in raw_dict.keys():
        raw  = raw_dict[var]
        norm = norm_dict[var]

        raw_flat  = raw[np.isfinite(raw)].ravel()
        norm_flat = norm[np.isfinite(norm)].ravel()

        if raw_flat.size == 0 or norm_flat.size == 0:
            print(f"⚠️ Skipping histogram for {var}: empty after filtering")
            continue

        plt.figure(figsize=(8, 3))

        plt.subplot(1, 2, 1)
        plt.hist(raw_flat, bins=80)
        plt.title(f"{var} RAW")

        plt.subplot(1, 2, 2)
        plt.hist(norm_flat, bins=80)
        plt.title(f"{var} NORMALIZED")

        plt.suptitle(f"{region} [{split}] {var} distributions")
        plt.tight_layout()

        out = os.path.join(outdir, f"{region}_{split}_{var}_hist.png")
        plt.savefig(out, dpi=150)
        plt.close()

    print(f"📊 Saved histograms → {outdir}/")



# ==============================================================
#  Windowed Dataset Class
# ==============================================================



class WindowedOceanDataset(Dataset):
    """
    Sliding z_k dataset.

    Each sample = (time index, depth index)

    Input:
        (T, C, window, H, W)

    Target:
        (H, W)  ← central depth z_k
    """




    def __init__(self, cfg, split="train", region_name=None):
        self.cfg = cfg
        self.split = split
        self.window = cfg.data.window
        self.regions = cfg.data.get("regions", None)
        self.visualize_regions = cfg.data.get("visualize_regions", False)
    

        # compute representative depth index
        with xr.open_dataset(cfg.data.tracers.paths[0], engine="netcdf4") as _ds_depth:
            depths = _ds_depth["olevel"].values
       

        self.depths = depths #[z_index] not only one level 
        self.n_depths = len(cfg.data.depth_stack)	

        # ---- derive variable names from config ----


        self.input_variables = [
            cfg.data.tracers.variables[0],
            cfg.data.forcing.u_var,
            cfg.data.forcing.v_var,
            cfg.data.forcing.w_var,
        ]

        if hasattr(cfg.data.forcing, "dif_var"):
            self.input_variables.append(cfg.data.forcing.dif_var)

        if hasattr(cfg.data.forcing, "mld_var"):
            self.input_variables.append(cfg.data.forcing.mld_var)

        self.n_channels = len(self.input_variables)





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
        #zmin, zmax = region_cfg["z_range"]
        mixed_layer = region_cfg["mix_layer"]
       

        self.x_range = (x1, x2)
        self.y_range = (y1, y2)

        #z_index = int(np.argmin(np.abs(depths - np.mean([zmin, zmax]))))
        

        #zmin, zmax = region_cfg["z_range"]
        #self.valid_z = np.where((depths >= zmin) & (depths <= zmax))[0]


        # --- choose center depth ---
        
        z_center = region_cfg["z_index"]
        
        
        self.z_index = z_center
              
        # --- sliding window around center ---
        self.max_below = getattr(cfg.model, "max_below", 2)
        self.max_above = getattr(cfg.model, "max_above", 2)
       
        Z = len(depths)

        z_window = []
        for dz in range(-self.max_below, self.max_above + 1):
            zi = z_center + dz
            zi = max(0, min(Z - 1, zi))   # clamp
            z_window.append(zi)

        self.valid_z = np.array(z_window)

        self.local_depths = self.depths[self.valid_z]
        depth_stack = getattr(cfg.data, "depth_stack", [0])
        input_steps = getattr(cfg.data.window, "input_steps")
        pred_steps = getattr(cfg.data.window, "pred_steps")
        self.z_indices = self.valid_z #get_depth_indices(self.z_index, depth_stack, len(depths))
        self.z_to_local = {z: i for i, z in enumerate(self.valid_z)}


        
        self.n_depths = len(self.z_indices)   # or depth_window

        #print(f"Z start    : {z_start}")
        #print(f"Z end      : {z_end}")
        print(f"self.valid_z      : {self.valid_z}")
        print(f"📊 Input steps: {input_steps}")
        print(f"📊 Prediction steps: {pred_steps}")
        print(f"📊 Variables used: {self.input_variables}")
        print(f"📊 Channels per depth: {self.n_channels}")
        print(f"📊 Depth levels used: {self.n_depths}")
        print(f"📊 Total variables/input channels: {self.n_channels * self.n_depths}")
        print(f"🌊 Using depth indices: {self.z_indices} -> depths {[depths[i] for i in self.z_indices]}")
        self.mix_lay = mixed_layer

        #print(f"@data.py - Line 40ff - x1 x2: {x1,x2}")
        #print(f"@data.py - Line 40ff - y1 y2: {y1,y2}")
        #print(f"@data.py - Line 40ff - zmin zmax: {zmin,zmax}")
        #print(f"@data.py - Line 40ff - z_index: {z_index}")
        #print(f"@data.py - Line 40ff - z_level [m]: {depths[z_index]:.2f}")
        self.H = int(x2-x1)
        self.W = int(y2-y1)

        tracer_path = cfg.data.tracers.paths[0]
        u_path, v_path, w_path = cfg.data.forcing.velocity_paths


        # load and slice
        tracer = self.safe_load_and_slice(tracer_path, cfg.data.tracers.variables[0], self.z_indices, (y1, y2), (x1, x2))
        u = self.safe_load_and_slice(u_path, cfg.data.forcing.u_var, self.z_indices, (y1, y2), (x1, x2))
        v = self.safe_load_and_slice(v_path, cfg.data.forcing.v_var, self.z_indices, (y1, y2), (x1, x2))
        w = self.safe_load_and_slice(w_path, cfg.data.forcing.w_var, self.z_indices, (y1, y2), (x1, x2))

        # add vertical diffusivity
        dif = self.safe_load_and_slice(w_path, cfg.data.forcing.dif_var, self.z_indices, (y1, y2), (x1, x2))



        # ------------------------------------------------------------
        # Mixed Layer Depth
        # ------------------------------------------------------------

        mld = None

        if getattr(cfg.data.forcing, "use_mld_distance", False):

            mld = self.load_mld_and_slice(
                cfg.data.forcing.mld_path,
                cfg.data.forcing.mld_var,
                (y1, y2),
                (x1, x2),
            )



        # Print shapes for debug
        print(f"🧩 Region {region_name} - Shapes after loading...tracer {tracer.shape}, u {u.shape}, v {v.shape}, w {w.shape}")

        # If time lengths mismatch (e.g. w has extra timesteps), truncate to shortest
        t_lens = [arr.shape[0] for arr in (tracer, u, v, w, dif, mld) if arr.size > 0]
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
        if dif.shape[0] != t_min:
            dif = dif[:t_min, ...]
        if mld is not None and mld.shape[0] != t_min:
            mld = mld[:t_min]


        
        # ------------------------------------------------------------
        # Last safety: clamp spatial sizes to common (Y, X)
        #
        # 3D ocean variables: (T, Z, Y, X)
        # MLD variable:       (T, Y, X)
        # ------------------------------------------------------------
        y_sizes = [a.shape[-2] for a in (tracer, u, v, w, dif) if a.ndim >= 3]
        x_sizes = [a.shape[-1] for a in (tracer, u, v, w, dif) if a.ndim >= 3]
        
        if len(y_sizes) == 0 or len(x_sizes) == 0:
            raise RuntimeError("No valid spatial dimensions found after loading data.")

        if mld is not None:
            y_sizes.append(mld.shape[-2])
            x_sizes.append(mld.shape[-1])

        y_min = min(y_sizes)
        x_min = min(x_sizes)

        # 3D variables: (T, Z, Y, X)
        tracer = tracer[:, :, :y_min, :x_min]
        u = u[:, :, :y_min, :x_min]
        v = v[:, :, :y_min, :x_min]
        w = w[:, :, :y_min, :x_min]
        dif = dif[:, :, :y_min, :x_min]
        mld = mld[:, :y_min, :x_min]

        print(
            f"📐 Common spatial size: Y={y_min}, X={x_min}"
        )

        print(
            f"📐 Shapes after spatial safety crop: "
            f"tracer={tracer.shape}, "
            f"u={u.shape}, "
            f"v={v.shape}, "
            f"w={w.shape}, "
            f"dif={dif.shape}, "
            f"mld={None if mld is None else mld.shape}"
        )

        #tracer = np.nan_to_num(tracer)
        #u = np.nan_to_num(u)
        #v = np.nan_to_num(v)
        #w = np.nan_to_num(w)


        # ------------------------------------------------------------
        # Distance from spatially varying Mixed Layer Depth
        # ------------------------------------------------------------

        mld_distance = None

        if mld is not None:

            # Selected model depths in metres
            depth_values = np.asarray(
                [depths[i] for i in self.z_indices],
                dtype=np.float64
            )

            # depth_values:
            #     (Z,)
            #
            # mld:
            #     (T,H,W)
            #
            # Result:
            #     (T,Z,H,W)


            # Signed distance from Mixed Layer Depth
            #
            #   negative → above / shallower than MLD
            #   zero     → at MLD
            #   positive → below / deeper than MLD
            #
            # Units: metres


            mld_distance = (
                depth_values[None, :, None, None]
                - mld[:, None, :, :]
            )


            print(
                f"📏 Signed MLD distance shape: {mld_distance.shape}"
            )

            print(
                f"📏 Signed MLD distance range: "
                f"{np.nanmin(mld_distance):.2f} - "
                f"{np.nanmax(mld_distance):.2f} m"
            )





        # Land is where tracer is NaN or fill-value in raw data
        raw_tracer = self.nan_load_and_slice(tracer_path, cfg.data.tracers.variables[0], self.z_indices, (y1, y2), (x1, x2)) #tracer.astype(np.float64)

        raw_tracer[~np.isfinite(raw_tracer)] = np.nan
        raw_tracer[np.abs(raw_tracer) >= 1e18] = np.nan

        # === Build fixed land mask from raw tracer ===
        # Land = NaN in the original tracer field
        self.land_mask = ~np.isnan(raw_tracer[0, 0])    # first time step 0, first depth 0, shape (H, W) # True = ocean, False = land
        # Will be used in getitem to mask predictions & targets.
        print(f"✔ Land mask created: ocean={self.land_mask.sum()} land={self.land_mask.size - self.land_mask.sum()}")

        # --------------------------------------------------
        # Apply the existing ocean/land mask to MLD distance
        # --------------------------------------------------
      
        mld_distance = np.where(
            self.land_mask[None, None, :, :],
            mld_distance,
            np.nan
        )

        print(
            f"📏 MLD distance after land masking: "
            f"NaNs={np.isnan(mld_distance).sum()}"
        )

 
        # 🔍 Debug plot (ONLY once per dataset)
        if split in ("train", "test"):
            plot_land_mask(self.land_mask, region_name, split)







        self.tracer, self.u, self.v, self.w, self.dif, self.mld_distance  = tracer.astype(np.float32), u.astype(np.float32), v.astype(np.float32), w.astype(np.float32), dif.astype(np.float32), mld_distance.astype(np.float32)

        # ------------------ SANITIZE raw arrays (before computing stats) ------------------
        def sanitize_array(arr, name):
            arr = arr.astype(np.float64)  # ensure float for NaN handling
            # Convert explicit infinities to NaN
            arr[~np.isfinite(arr)] = np.nan
            # Convert huge fill-values (common sentinels) to NaN
            arr[np.abs(arr) >= 1e18] = 0
            return arr

        self.tracer = sanitize_array(self.tracer, "tracer")
        self.u      = sanitize_array(self.u, "u")
        self.v      = sanitize_array(self.v, "v")
        self.w      = sanitize_array(self.w, "w")
        self.dif      = sanitize_array(self.dif, "dif")
        self.mld_distance = sanitize_array(self.mld_distance, "mld_distance")



        #if self.mld_distance is not None:

        #    self.mld_distance = self.mld_distance.astype(np.float64)

        #    self.mld_distance[~np.isfinite(self.mld_distance)] = np.nan

        #    self.mld_distance[
        #        np.abs(self.mld_distance) >= 1e18
        #    ] = np.nan




        # DEBUG after sanitize
        for name, arr in (("tracer", self.tracer), ("u", self.u), ("v", self.v), ("w", self.w), ("dif", self.dif), ("mld_distance", self.mld_distance)):
            print(f"SANITIZED {name}: n_nans={int(np.isnan(arr).sum())}, n_infs={int(np.isinf(arr).sum())}, min={np.nanmin(arr) if np.isfinite(np.nanmin(arr)) else 'nan'}, max={np.nanmax(arr) if np.isfinite(np.nanmax(arr)) else 'nan'}")


# ------------------ COMPUTE / LOAD STATS ------------------




        # ============================================================
        #  NORMALIZATION (PER-VARIABLE, TRAIN-ONLY)
        # ============================================================
        os.makedirs("norms", exist_ok=True)
        stats_file = f"norms/norm_stats_{region_name}.npz"


        




        if split == "train":


            # compute stats ignoring NaNs
            tr_mean = float(np.nanmean(self.tracer))
            tr_std = float(np.nanstd(self.tracer))
            # --- Tracer: switch to Min-Max normalization (0..1) instead of z-score ---
            # compute per-region min/max ignoring NaNs
            #per depth normalization

            # (T, Z, H, W)

            tr_min = np.nanmin(self.tracer, axis=(0,2,3))  # (Z,)
            tr_max = np.nanmax(self.tracer, axis=(0,2,3))  # (Z,)

            #tr_min = float(np.nanmin(self.tracer))
            #tr_max = float(np.nanmax(self.tracer))
            # --- tracer normalization (Z-SCORE) ---


            # avoid division by zero
            tr_std = tr_std if tr_std > 1e-8 else 1.0

            #tracer = (tracer - tr_mean) / tr_std

            u_mean = float(np.nanmean(self.u))
            u_std  = float(np.nanstd(self.u))
            v_mean = float(np.nanmean(self.v))
            v_std  = float(np.nanstd(self.v))
            w_mean = float(np.nanmean(self.w))
            w_std  = float(np.nanstd(self.w))
            dif_mean = float(np.nanmean(self.dif))
            dif_std  = float(np.nanstd(self.dif))






            # Guard against non-finite or zero std (fallback)
            def guard_std(std, name):
                if not np.isfinite(std) or std < 1e-12:
                    print(f"⚠️ {name} std invalid ({std}). Replacing with 1.0 to avoid div-by-zero.")
                    return 1.0
                return std






            tr_std = guard_std(tr_std, "tracer")
            u_std  = guard_std(u_std, "u")
            v_std  = guard_std(v_std, "v")
            w_std  = guard_std(w_std, "w")
            dif_std  = guard_std(dif_std, "dif")        




            if self.mld_distance is not None:

                mld_mean = float(
                    np.nanmean(self.mld_distance)
                )

                mld_std = float(
                    np.nanstd(self.mld_distance)
                )

                mld_std = guard_std(
                    mld_std,
                    "mld_distance"
                )

            else:
                mld_mean = 0.0
                mld_std = 1.0





    
            self.stats = {
                "tracer_mean": tr_mean,
                "tracer_std":  tr_std,
                "tracer_min":  tr_min,
                "tracer_max":  tr_max,
                "u_mean":      u_mean,
                "u_std":       u_std,
                "v_mean":      v_mean,
                "v_std":       v_std,
                "w_mean":      w_mean,
                "w_std":       w_std,
                "dif_mean":      dif_mean,
                "dif_std":       dif_std,
                "mld_distance_mean": mld_mean,
                "mld_distance_std": mld_std,
            }
            # Save sanitized stats
            np.savez(stats_file, **self.stats)
            print(f"📊 Saved normalization stats → {stats_file}")

        else:
            self.stats = dict(np.load(stats_file))
            print(f"📥 Loaded normalization stats from {stats_file}")


             # validate stats
            for key in ("tracer_std","u_std","v_std","w_std"):
                val = float(self.stats.get(key, np.nan))
                if not np.isfinite(val) or val < 1e-8:
                    raise ValueError(f"Invalid std in stats file for {key}: {val}. Recompute stats by running train with split='train'.")





        # Store raw snapshot for debug before normalization
        idx0 = 880
        raw_t = self.tracer[idx0].copy()
        raw_u = self.u[idx0].copy()
        raw_v = self.v[idx0].copy()
        raw_w = self.w[idx0].copy()
        raw_dif = self.dif[idx0].copy()
        raw_mld_distance = self.mld_distance[idx0].copy()

        # Make raw copies for histograms
        self.tracer_raw = self.tracer.copy()
        self.u_raw      = self.u.copy()
        self.v_raw      = self.v.copy()
        self.w_raw      = self.w.copy()
        self.dif_raw      = self.dif.copy()
        self.mld_distance_raw      = self.mld_distance.copy()        

        self.tracer = self.tracer.astype(np.float32)
        self.u      = self.u.astype(np.float32)
        self.v      = self.v.astype(np.float32)
        self.w      = self.w.astype(np.float32)
        self.dif      = self.dif.astype(np.float32)
        self.mld_distance      = self.mld_distance.astype(np.float32)


        # --- SAFE normalization for TRACERS (depth-aware) ---
        tr_min = self.stats["tracer_min"]   # (Z,)
        tr_max = self.stats["tracer_max"]   # (Z,)

        # ✅ check all values are finite
        if not np.all(np.isfinite(tr_min)) or not np.all(np.isfinite(tr_max)):
            raise ValueError(f"Invalid tracer stats: min={tr_min}, max={tr_max}")

        # ✅ check no zero-range depths
        if np.any((tr_max - tr_min) < 1e-12):
            raise ValueError("⚠️ Some depth levels have tracer_min == tracer_max")
        
        # --- SAFE normalization for TRACERS (detailed diagnostics on failure) ---
        #tr_min = self.stats["tracer_min"]
        #tr_max = self.stats["tracer_max"]

        #if not np.isfinite(tr_min) or not np.isfinite(tr_max):
        #    raise ValueError(f"Invalid tracer stats: min={tr_min}, max={tr_max}")

        # avoid division-by-zero / tiny diff: show warning but keep behaviour deterministic
        #if abs(tr_max - tr_min) < 1e-12:
        #    raise ValueError(f"⚠️ trace rmin==max ({tr_min})!!!")
        
        
        # Normalize min-max scaling into [0,1]; keep NaNs where they were
            
        # apply min-max: (x - min)/(max-min)
        denom = (tr_max - tr_min)
        denom[denom < 1e-8] = 1.0
        # use np.divide to avoid warnings; NaNs remain NaN
            
        self.tracer = (self.tracer - tr_min[None,:,None,None]) / denom[None,:,None,None]

        #self.tracer = np.divide((tracer - tr_min), denom)

        
        #def minmax(arr):
        #    amin = np.nanmin(arr)
        #    amax = np.nanmax(arr)
        #    denom = max(amax - amin, 1e-8)
        #    return (arr - amin) / denom, amin, amax

        #self.u, u_min, u_max = minmax(self.u)
        #self.v, v_min, v_max = minmax(self.v)
        #self.w, w_min, w_max = minmax(self.w)


        #self.stats.update({
        #    "u_min": u_min, "u_max": u_max,
        #    "v_min": v_min, "v_max": v_max,
        #    "w_min": w_min, "w_max": w_max,
        #})

        

        self.u      = (self.u      - self.stats["u_mean"])      / (self.stats["u_std"] + 1e-12)
        self.v      = (self.v      - self.stats["v_mean"])      / (self.stats["v_std"] + 1e-12)
        self.w      = (self.w      - self.stats["w_mean"])      / (self.stats["w_std"] + 1e-12)
        self.dif      = (self.dif      - self.stats["dif_mean"])      / (self.stats["dif_std"] + 1e-12)

        if self.mld_distance is not None:

            self.mld_distance = (
                self.mld_distance
                - self.stats["mld_distance_mean"]
            ) / (
                self.stats["mld_distance_std"] + 1e-12
            )

        print(f"✅ Applied per-variable normalization for {split}")

        print(
            "MLD distance:",
            "shape =", self.mld_distance.shape,
            "NaNs =", np.isnan(self.mld_distance).sum(),
            "Inf =", np.isinf(self.mld_distance).sum(),
            "min =", np.nanmin(self.mld_distance),
            "max =", np.nanmax(self.mld_distance),
        )

        # Post-normalization check: identify any non-finite elements and print raw context
        def find_nonfinite_and_report(arr, name):

            print(f"DEBUG nonfinite check: {name}, shape={arr.shape}, ndim={arr.ndim}")
            mask_inf = ~np.isfinite(arr)
            total = int(mask_inf.sum())
            if total == 0:
                return 0
           
            coords = np.argwhere(mask_inf)

            sample_coords = [
                tuple(int(i) for i in c)
                for c in coords[:10]
            ]

            print(
                f"POST-NORM {name}: found {total} non-finite values. "
                f"Example positions up to 10: {sample_coords}"
            )

            if sample_coords:
                coord = sample_coords[0]

                if arr.ndim == 4:
                    t0, z0, y0, x0 = coord
                    print(
                        f"  first nonfinite location: "
                        f"t={t0}, z={z0}, y={y0}, x={x0}"
                    )

                elif arr.ndim == 3:
                    t0, y0, x0 = coord
                    print(
                        f"  first nonfinite location: "
                        f"t={t0}, y={y0}, x={x0}"
                    )

                else:
                    print(f"  first nonfinite location: {coord}")


            return total

        tot_tr = find_nonfinite_and_report(self.tracer, "tracer")
        if tot_tr > 0:
            # Distinguish NaNs (expected for land/missing) from Infs (fatal)
            n_infs = int(np.isinf(self.tracer).sum())
            n_nans = int(np.isnan(self.tracer).sum())
            print(f"POST-NORM tracer: {n_nans} NaNs, {n_infs} Infs")
            if n_infs > 0:
                # INF values indicate a numerical error (division by zero or overflow)
                raise ValueError("Found INF values in tracer after normalization!")
            else:
                # Only NaNs present — this is expected for land/missing values.
                # Continue: NaNs will be handled by the dataset mask and ignored by loss/metrics.
                print("Proceeding: NaNs detected in tracer (land/missing). They will be masked during training/eval.")
    
        



         # ------------------ REPLACEMENT PATCH END ------------------





        # -------------------------------------------------------------
        # 📊 Diagnostics & Visualization (only on TRAIN split)
        # -------------------------------------------------------------
        if split == "train":

            # Raw copies for histogram comparison
            raw_dict = {
                "tracer": self.tracer_raw,
                "u": self.u_raw,
                "v": self.v_raw,
                "w": self.w_raw,
                "dif": self.dif_raw,
                "mld_distance": self.mld_distance_raw,
            }
            norm_dict = {
                "tracer": self.tracer,
                "u": self.u,
                "v": self.v,
                "w": self.w,
                "dif": self.dif,
                "mld_distance": self.mld_distance,
            }

            # Save histograms of raw vs normalized variables
            plot_histograms_of_vars(
                region_name,
                split,
                raw_dict,
                norm_dict,
                outdir="artifacts/histograms"
            )

            




            # Save debug plots
            debug_plot_normalization(
                region_name, split,
                raw_t, raw_u, raw_v, raw_w, raw_dif, raw_mld_distance,
                self.tracer[idx0], self.u[idx0], self.v[idx0], self.w[idx0], self.dif[idx0], self.mld_distance[idx0],
                t_idx=idx0
            )

        # ------------------ POST-NORM CHECK (precise diagnostics) ------------------
        def find_nonfinite_locations(arr):
            # returns list of (t,y,x) indices where arr is non-finite; limit output to first few
            mask_inf = ~np.isfinite(arr)
            total = int(mask_inf.sum())
            coords = []
            if total > 0:
                it = np.argwhere(mask_inf)
                for idx in it[:10]:
                    coords.append(tuple(int(i) for i in idx))
            return total, coords

        for name, arr in (("tracer", self.tracer), ("u", self.u), ("v", self.v), ("w", self.w), ("dif", self.dif), ("mld_distance", self.mld_distance)):
            tot = find_nonfinite_and_report(arr, name)
            if tot > 0:
                n_infs = int(np.isinf(arr).sum())
                n_nans = int(np.isnan(arr).sum())
                print(f"POST-NORM {name}: {n_nans} NaNs, {n_infs} Infs")
                if n_infs > 0:
                    raise ValueError(f"Found INF values in {name} after normalization!")
                else:
                    print(f"Proceeding: NaNs detected in {name} (land/missing). They will be masked out.")




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

        # Define 90/5/5 split along the time axis
        train_end = int(0.80 * usable_len)
        val_end   = int(0.99 * usable_len)

        if self.split == "train":
            self.t_start, self.t_end = 0, train_end
        elif self.split == "val":
            self.t_start, self.t_end = train_end, val_end
        else:  # test
            self.t_start, self.t_end = val_end, usable_len

        print(f"🧩 Using split='{self.split}' with time indices {self.t_start}:{self.t_end} of {usable_len}")

        


        # ============================================================
        #  Build window indices
        # ============================================================

        T_in  = cfg.data.window.input_steps
        T_out = cfg.data.window.pred_steps

        self.T_in = T_in
        self.T_out = T_out

        T = self.tracer.shape[0]

        self.indices = []
        #for t in range(0, T - (T_in + T_out) + 1):
        #    for z in self.valid_z:

        #        self.indices.append((t, z))


        self.indices = []

        for t in range(self.t_start, self.t_end):
            for z in self.valid_z:
                self.indices.append((t, z))

        print(f"📐 Total samples for {region_name}/{split}: {len(self.indices)}")



        if split == "train" and getattr(cfg.debug, "mass_diagnostics", True):
            compute_mass_time_series(
                nc_path=cfg.data.tracers.paths[0],
                tracer_var=cfg.data.tracers.variables[0],
                area_var="area",
                z_idxs=self.z_indices,
                t_start=self.t_start,
                t_end=self.t_end,
                region=region_name
            )


    def load_mld_and_slice(self, nc_path, varname, y_slice, x_slice):

        with xr.open_dataset(nc_path) as ds:

            da = ds[varname]

            print(f"🌊 Loading MLD: {varname}")
            print(f"🌊 MLD dims:  {da.dims}")
            print(f"🌊 MLD shape: {da.shape}")

            # Make sure ordering is exactly (time, y, x)
            da = da.transpose("time_counter", "y", "x")

            # Spatial crop
            da = da.isel(
                y=slice(y_slice[0], y_slice[1]),
                x=slice(x_slice[0], x_slice[1])
            )

            mld = da.values.astype(np.float64)

        # Clean fill values / invalid values
        mld[~np.isfinite(mld)] = np.nan
        mld[np.abs(mld) >= 1e18] = np.nan

        print(
            f"🌊 MLD after crop: {mld.shape}, "
            f"min={np.nanmin(mld):.2f} m, "
            f"max={np.nanmax(mld):.2f} m"
        )

        return mld

    def __len__(self):
        return len(self.indices)
        #return self.t_end - self.t_start

    def __getitem__(self, idx):
        t_in = self.window.input_steps
        t_out = self.window.pred_steps
        #idx_out = idx + self.t_start
        # To have a column within a box start with z0 levels then z1 levels...
        #t_idx, z_center_global = self.indices[self.t_start + idx]
        t_idx, z_center_global = self.indices[idx]
        # convert to local index
        z_center = self.z_to_local[z_center_global]


        # Use all locally loaded depth levels.
        # self.tracer already has shape (T, Z, H, W),
        # where Z == len(self.valid_z)

        z_window = np.arange(self.n_depths)

        assert len(z_window) == self.n_depths




        # --- DEBUG (only first few samples) ---
        #if idx == 1:
        #    print("\n🌊 ===== Z-WINDOW DEBUG =====")
        #    print(f"Sample idx : {idx}")
        #    print(f"Time idx   : {t_idx}")
        #    print("z_center_glob :", z_center_global)
        #    print("valid_z       :", self.valid_z)
        #    print(f"Z center   : {z_center}")
        #    print("z_window:", z_window)
        #    print("len(z_window) :", len(z_window))
            #print(f"Z start    : {z_start}")
            #print(f"Z end      : {z_end}")
        #    print(f"Window len : {len(z_window)}")

        #   print(f"Z window   : {list(z_window)}")
        #    print(
        #        f"t_idx={t_idx}, "
        #        f"t_in={t_in}, "
        #        f"available={self.tracer.shape[0]}, "
        #        f"slice={t_idx}:{t_idx+t_in}")
             # If depths available → print real depth values
        #    try:
        #        depths_vals = self.local_depths[z_window]
        #        print("Depths [m] : [" + ", ".join(f"{d:.1f}" for d in depths_vals) + "]")
        #    except Exception:
        #        print("Depths [m] : (not available)")

        #    print("================================\n")


       

        # --- Input sequences (T, Z, Y, X) ---
        x_tracer = self.tracer[t_idx:t_idx + t_in, z_window] #add z_window
        x_u = self.u[t_idx:t_idx + t_in, z_window]
        x_v = self.v[t_idx:t_idx + t_in, z_window]
        x_w = self.w[t_idx:t_idx + t_in, z_window]
        x_dif = self.dif[t_idx:t_idx + t_in, z_window]
        x_mld_distance = self.mld_distance[t_idx:t_idx + t_in, z_window]


        x_mld_distance = np.nan_to_num(
            x_mld_distance,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        # concatenate along channel axis
        #x = np.concatenate([x_tracer, x_u, x_v, x_w], axis=1) # (TIME, CHANNEL/VARIABLE=4*3=12, HEIGHT BOX, WIDTH BOX)

        # x_tracer, x_u, x_v, x_w: (T, Z, H, W)

        x = np.stack([x_tracer, x_u, x_v, x_w, x_dif, x_mld_distance], axis=2)
        # shape (T, Z, C, H, W)

        """
        Axis 0 → Time
        Axis 1 → Depth
        Axis 2 → Variables
        Axis 3 → Height
        Axis 4 → Width
        """
        x = np.transpose(x, (0, 2, 1, 3, 4))
        # shape (T, C, Z, H, W)
        """
        New axis 0 = old axis 0 (Time)

        New axis 1 = old axis 2 (Variables)

        New axis 2 = old axis 1 (Depth)

        New axis 3 = old axis 3 (Height)

        New axis 4 = old axis 4 (Width)
        """

        #x = torch.tensor(x, dtype=torch.float32)

        x = torch.from_numpy(x).float() #supposed to be more efficient

        # after transposeshape: (T, Z, C, H, W)

        # --- Target ---
        y_true = self.tracer[t_idx+t_in:t_idx+t_in+t_out, z_center] #(use sliding depth for target)


        # --- DEBUG (only first few samples) ---
        #if idx == 1:


        #    print("x_tracer :", x_tracer.shape)
        #    print("x_u      :", x_u.shape)
        #    print("x_v      :", x_v.shape)
        #    print("x_w      :", x_w.shape)
        #    print("x_dif      :", x_dif.shape)
        #    print("x_mld_dist      :", x_mld_distance.shape)
        #    print("target   :", y_true.shape)
        
        # --- Convert to tensors ---
        #x = torch.tensor(x, dtype=torch.float32)
        #print(f"@data.py - Line 184 - y_true.shape {y_true.shape} ")
        #y = torch.tensor(y_true, dtype=torch.float32)
        
       
        # --- Build mask for valid ocean points ---
        #mask = np.broadcast_to(self.land_mask, y_true.shape).astype(np.float32)


        y = torch.tensor(y_true, dtype=torch.float32)

        mask = torch.tensor(self.land_mask, dtype=torch.float32)

        # ensure same spatial size (safe guard)
        H = min(mask.shape[0], y.shape[-2])
        W = min(mask.shape[1], y.shape[-1])

        mask = mask[:H, :W]
        y    = y[..., :H, :W]

        # --- CRITICAL: make both (1, H, W) ---
        y    = y.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return x, y, mask



    # helper to load & safely slice a variable by dim names
    def nan_load_and_slice(self,nc_path, varname, z_indices, y_slice, x_slice):



        with xr.open_dataset(nc_path, engine="netcdf4") as ds_local:
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
                #print(f"�~Z| �~O slice [{y0}:{y1c}, {x0}:{x1c}] out of bounds for {nc_path} var {varname} dims {dims} sizes {sizes}")
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
                isel_dict[depth_dim] = z_indices #It should return now shape of (T,Z,Y,X) (time steps,3,Height of Patch, Width Patch)

            isel_dict[y_dim] = slice(y0, y1c)
            isel_dict[x_dim] = slice(x0, x1c)
            #print(f"@data.py - line 247 -def safe_load_slice() - isel_dict: {isel_dict}")
            arr = var.isel(isel_dict).values  # numpy array

            # Replace large missing values with NaN
            # convert to float32/float64 for NaN support
            arr = arr.astype(np.float64)


            # prefer explicit attrs in the NetCDF: _FillValue or missing_value
            missing_value = var.attrs.get("_FillValue", var.attrs.get("missing_value", None))

            # If attribute present, convert sentinel to NaN
            if missing_value is not None:
                # If arr contains exact sentinel values, replace them with NaN
                try:
                    arr[arr == missing_value] = np.nan
                except Exception:
                    # fallback safe replacement (handles different dtypes)
                    arr = np.where(arr == missing_value, np.nan, arr)


            # Also convert impossible large values (�~I� 1e19) into NaN as safety
            arr[np.abs(arr) > 1e19] = np.nan

            arr[~np.isfinite(arr)] = np.nan         # convert inf/-inf to NaN

            return arr






    # helper to load & safely slice a variable by dim names
    def safe_load_and_slice(self,nc_path, varname, z_indices, y_slice, x_slice):
        


        with xr.open_dataset(nc_path, engine="netcdf4") as ds_local:
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
                isel_dict[depth_dim] = z_indices

            isel_dict[y_dim] = slice(y0, y1c)
            isel_dict[x_dim] = slice(x0, x1c)
            #print(f"@data.py - line 247 -def safe_load_slice() - isel_dict: {isel_dict}")
            arr = var.isel(isel_dict).values  # numpy array
            
            # Replace large missing values with NaN
            # convert to float32/float64 for NaN support
            arr = arr.astype(np.float64)


            # prefer explicit attrs in the NetCDF: _FillValue or missing_value
            missing_value = var.attrs.get("_FillValue", var.attrs.get("missing_value", None))

            # If attribute present, convert sentinel to NaN
            if missing_value is not None:
                # If arr contains exact sentinel values, replace them with NaN
                try:
                    arr[arr == missing_value] = 0
                except Exception:
                    # fallback safe replacement (handles different dtypes)
                    arr = np.where(arr == missing_value, np.nan, arr)


            # Also convert impossible large values (≥ 1e19) into NaN as safety
            arr[np.abs(arr) > 1e19] = 0

            arr[~np.isfinite(arr)] = 0         # convert inf/-inf to NaN
            
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

        with xr.open_dataset(nc_path, engine="netcdf4") as ds:
            ny, nx = ds[self.cfg.data.tracers.variables[0]].shape[-2:] 
            depths = ds["olevel"].values            
          

        plt.figure(figsize=(9, 5))
   
        plt.title("Defined Regions in Grid Index Space")
        plt.xlabel("X index")
        plt.ylabel("Y index")

        for reg in regions:
            name = reg.get("name", "region")
            # ✅ Support both YAML-style (x_range, y_range) and flattened x1/x2 format
           
            #zmin, zmax = reg["z_range"]
            x1, x2 = reg["x_range"]
            y1, y2 = reg["y_range"]
                      
            


 
            #z_index = int(np.argmin(np.abs(depths - np.mean([zmin, zmax]))))

            if "z_index" in reg:
                z_index = int(reg["z_index"])
            elif "z_range" in reg:
                zmin, zmax = reg["z_range"]
                z_index = int(np.argmin(np.abs(depths - np.mean([zmin, zmax]))))
            else:
                raise ValueError("@data.py PLOT BOX Region must define z_index or z_range")



            self.z_indices = [max(z_index-1, 0), z_index, min(z_index+1, len(depths)-1)]
            self.z_index = z_index
            self.z_level = depths[z_index]
            print(f"🌊 @data.py PLOT BOX self.z_level [m]: {self.z_level:.2f}")

    
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




