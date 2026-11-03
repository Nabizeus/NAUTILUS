"""
data_v2.py -- data pipeline for the v2 full-column experiments.

Changes against src/data.py (WindowedOceanDataset):

  * 3-D ocean mask. NEMO stores cells below the sea floor (and land) as
    the fill value 1e20. data.py turned those into 0 and only used a 2-D
    surface land mask, so every column counted as ocean down to 6000 m and
    the loss, the normalization and the plots saw fake zero tracer below
    the sea floor. Here the mask is (Z, H, W), taken from the tracer fill
    values of the first time step (the NEMO z-grid is static).
  * Normalization statistics are computed from ocean cells only, over the
    training period only.
  * The mask is an extra (7th) input channel, so the model knows where the
    sea floor and the coast are. Non-ocean cells are 0 in every channel.
  * The region is read from disk once and shared by the train/val/test
    datasets (data.py read and kept the full timeline three times).
  * Random-patch training: the train split can return a random
    patch_size x patch_size crop of the region, so a large box fits on the
    GPU and the fully convolutional model learns local behaviour that can
    be applied to any other region.
  * Time split from cfg.data.split (default 70/15/15).
  * cfg.data.mask_channel (default true): false drops the 7th channel. The
    mask is then used only in the data (non-ocean cells 0 in every input
    channel, left out of loss and metrics), as in the earlier 6-variable runs.
  * cfg.data.target (default "full"): "increment" makes the target the
    change of the tracer since the newest input step, divided by its
    per-depth RMS over the training period (stats["inc_rms"]). Predicting 0
    is then the "no change" forecast, with a normalized MSE of about 1.
    to_full() turns a predicted increment back into the full field.
  * v3: several training regions with one model (cfg.data.training_group).
    compute_stats() then pools the statistics over the ocean cells of all
    training regions, so every region gets the same scaling; read_boxes()
    reads the boxes in one pass over the global fields; model_name() gives
    the name used for the checkpoint and the statistics file.
  * v5: lead time k (cfg.data.lead_steps, default 1). The input snapshots are
    k steps apart (t, t+k, ..., t+(T_in-1)k) and the target is k steps after
    the last one; the increment target is scaled by the RMS of the k-step
    change. cfg.data.forcing_average (default false): u, v, w, dif and the
    MLD distance of each input are the mean over the k steps up to that input
    (s-k, s] instead of the snapshot at s; the tracer stays a snapshot.

Input  x : (T_in, C, Z, h, w)     channels: tracer, u, v, w, dif, mld_distance[, ocean_mask]
Target y : (1, Z, h, w)           normalized tracer at t + T_in, or its scaled increment
Mask   m : (1, Z, h, w)           1 = ocean cell, 0 = land or below the sea floor
"""

import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

FILL_ABS = 1e19
CHANNELS = ["tracer", "u", "v", "w", "dif", "mld_distance", "ocean_mask"]


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def _variables(cfg):
    """(name, path, NetCDF variable) for the five 3-D fields, in channel order."""
    u_path, v_path, w_path = cfg.data.forcing.velocity_paths
    return [
        ("tracer", cfg.data.tracers.paths[0], cfg.data.tracers.variables[0]),
        ("u", u_path, cfg.data.forcing.u_var),
        ("v", v_path, cfg.data.forcing.v_var),
        ("w", w_path, cfg.data.forcing.w_var),
        ("dif", w_path, cfg.data.forcing.dif_var),
    ]


def read_box(path, var, t0, t1, y0, y1, x0, x1):
    """Read var[t0:t1, ..., y0:y1, x0:x1] as float32 with fill values set to NaN."""
    with xr.open_dataset(path, engine="netcdf4", mask_and_scale=False,
                         decode_times=False) as ds:
        da = ds[var]
        sel = {da.dims[0]: slice(t0, t1), da.dims[-2]: slice(y0, y1),
               da.dims[-1]: slice(x0, x1)}
        a = np.asarray(da.isel(sel).values, dtype=np.float32)
    bad = ~np.isfinite(a)
    bad |= np.abs(a) >= FILL_ABS
    a[bad] = np.nan
    return a


def read_boxes(cfg, rows, t0, t1, chunk=40):
    """
    Read time steps [t0, t1) of every field for several boxes at once.
    rows: dicts with name, x0, x1, y0, y1. Reads full global chunks (how the
    NEMO files are stored), which is much faster than one read per box.
    Returns {name: {"tracer", "u", "v", "w", "dif", "mld": float32 array, NaN = fill}}.
    """
    import time
    out = {r["name"]: {} for r in rows}
    items = list(_variables(cfg))
    items.append(("mld", cfg.data.forcing.mld_path, cfg.data.forcing.mld_var))
    for key, path, var in items:
        tic = time.time()
        parts = {r["name"]: [] for r in rows}
        for a in range(t0, t1, chunk):
            b = min(t1, a + chunk)
            with xr.open_dataset(path, engine="netcdf4", mask_and_scale=False,
                                 decode_times=False) as ds:
                g = np.asarray(ds[var].isel({ds[var].dims[0]: slice(a, b)}).values, dtype=np.float32)
            for r in rows:
                parts[r["name"]].append(g[..., r["y0"]:r["y1"], r["x0"]:r["x1"]].copy())
            del g
        for r in rows:
            arr = np.concatenate(parts[r["name"]], axis=0)
            parts[r["name"]] = None
            bad = ~np.isfinite(arr) | (np.abs(arr) >= FILL_ABS)
            arr[bad] = np.nan
            out[r["name"]][key] = arr
        print(f"[data_v2] read {key:6s} ({var}) t {t0}:{t1} in {time.time() - tic:.0f} s", flush=True)
    return out


def read_depths(cfg):
    with xr.open_dataset(cfg.data.tracers.paths[0], engine="netcdf4",
                         decode_times=False) as ds:
        return np.asarray(ds["olevel"].values, dtype=np.float64)


def n_timesteps(cfg):
    with xr.open_dataset(cfg.data.tracers.paths[0], engine="netcdf4",
                         decode_times=False) as ds:
        return int(ds.sizes[cfg.data.coords.time])


def lead_steps(cfg):
    """v5: time steps between the input snapshots and from the last input to the target."""
    return max(1, int(cfg.data.get("lead_steps", 1) or 1))


def split_bounds(cfg, total_len):
    """Window start indices [0, train_end), [train_end, val_end), [val_end, usable)."""
    t_in = int(cfg.data.window.input_steps)
    t_out = int(cfg.data.window.pred_steps)
    usable = total_len - (t_in * lead_steps(cfg) + t_out)   # v5: inputs and target k steps apart
    fr = list(cfg.data.get("split", [0.70, 0.15, 0.15]))
    train_end = int(fr[0] * usable)
    val_end = int((fr[0] + fr[1]) * usable)
    return train_end, val_end, usable


# ----------------------------------------------------------------------
# Normalization statistics
# ----------------------------------------------------------------------
def compute_stats(raw, ocean, depths, t_end, lead=1):
    """
    Statistics from time steps [0, t_end) and ocean cells only.

    tracer: per-depth min/max (Z,). A level with no ocean cell in the
            region takes the values of the nearest level above that has
            one, so the scaling is defined if another region is deeper.
    inc_rms: per-depth RMS of the `lead`-step tracer change (v5; 1 = one
            step), in normalized units (Z,), same filling; used by the
            increment target.
    u, v, w, dif, mld_distance: one mean/std each.

    v3: raw and ocean can also be lists (one entry per training region); the
    statistics are then pooled over the ocean cells of all regions.
    """
    raws = raw if isinstance(raw, (list, tuple)) else [raw]
    oceans = ocean if isinstance(ocean, (list, tuple)) else [ocean]
    trs = [r["tracer"][:t_end] for r in raws]
    Z = trs[0].shape[1]
    tr_min = np.full(Z, np.nan)
    tr_max = np.full(Z, np.nan)
    for z in range(Z):
        for tr, oc in zip(trs, oceans):
            if oc[z].any():
                vals = tr[:, z][:, oc[z]]
                tr_min[z] = np.fmin(tr_min[z], np.nanmin(vals))
                tr_max[z] = np.fmax(tr_max[z], np.nanmax(vals))
    ocean_levels = np.isfinite(tr_min)
    inc_rms = np.full(Z, np.nan)
    for z in range(Z):
        if ocean_levels[z]:
            rng = max(tr_max[z] - tr_min[z], 1e-12)
            s2 = n = 0.0
            for tr, oc in zip(trs, oceans):
                if oc[z].any():
                    a = tr[:, z].astype(np.float64)
                    d = (a[lead:] - a[:-lead])[:, oc[z]] / rng
                    s2 += float(np.nansum(d ** 2))
                    n += float(np.isfinite(d).sum())
            inc_rms[z] = np.sqrt(s2 / n) if n > 0 else np.nan
    last = int(np.where(ocean_levels)[0].max())
    for z in range(Z):
        if not ocean_levels[z]:
            src = max(i for i in range(min(z, last) + 1) if ocean_levels[i])
            tr_min[z], tr_max[z] = tr_min[src], tr_max[src]
            inc_rms[z] = inc_rms[src]
    # floor, so a level that hardly changes is not blown up into noise
    inc_rms = np.maximum(inc_rms, 1e-3 * np.nanmedian(inc_rms))

    stats = {"tracer_min": tr_min, "tracer_max": tr_max, "inc_rms": inc_rms,
             "ocean_levels_in_training_region": ocean_levels}
    for name in ("u", "v", "w", "dif"):
        if len(raws) == 1:
            vals = raws[0][name][:t_end][:, oceans[0]]
            mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))
        else:
            s = ss = n = 0.0
            for r, oc in zip(raws, oceans):
                vals = r[name][:t_end][:, oc].astype(np.float64)
                s += float(np.nansum(vals))
                ss += float(np.nansum(vals ** 2))
                n += float(np.isfinite(vals).sum())
            mean = s / n
            std = float(np.sqrt(max(ss / n - mean ** 2, 0.0)))
        stats[f"{name}_mean"] = mean
        stats[f"{name}_std"] = std if np.isfinite(std) and std > 1e-12 else 1.0

    s = n = ss = 0.0
    for r, oc in zip(raws, oceans):
        mld = r["mld"][:t_end]
        for z in range(Z):
            if not oc[z].any():
                continue
            d = depths[z] - mld[:, oc[z]]
            d = d[np.isfinite(d)]
            s += d.sum(dtype=np.float64)
            ss += (d.astype(np.float64) ** 2).sum()
            n += d.size
    mean = s / n
    std = float(np.sqrt(max(ss / n - mean ** 2, 0.0)))
    stats["mld_distance_mean"] = float(mean)
    stats["mld_distance_std"] = std if std > 1e-12 else 1.0
    return stats


def save_stats(path, stats):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **stats)


def load_stats(path):
    return {k: v for k, v in np.load(path).items()}


def model_name(cfg):
    """
    Name of the trained model in file names (checkpoint, statistics, logs):
    cfg.data.training_group when several regions train one model (v3),
    otherwise the single training region.
    """
    group = cfg.data.get("training_group", None)
    if group:
        return str(group)
    return list(cfg.data.training_regions)[0]


def stats_path(cfg, region_name):
    tag = cfg.training.get("experiment_name", "") or "run"
    return os.path.join(cfg.data.get("norms_dir", "norms"),
                        f"{tag}_norm_stats_{region_name}.npz")


# ----------------------------------------------------------------------
# A region in memory, normalized
# ----------------------------------------------------------------------
class RegionData:
    """
    One box of the global grid over a range of time steps, read once and
    normalized. Arrays are float32 (T, Z, H, W); non-ocean cells are 0.

    stats=None computes the statistics from this region's training period
    (and saves them to stats_file). Pass the training region's statistics
    to evaluate any other region with the same scaling the model learned.
    """

    def __init__(self, cfg, x_range, y_range, name, t_range=None, stats=None,
                 stats_file=None, verbose=True, raw=None):
        self.cfg = cfg
        self.name = name
        self.x_range = tuple(int(v) for v in x_range)
        self.y_range = tuple(int(v) for v in y_range)
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        T_total = n_timesteps(cfg)
        t0, t1 = (0, T_total) if t_range is None else (int(t_range[0]), int(t_range[1]))
        self.t0, self.t1 = t0, t1
        self.depths = read_depths(cfg)

        log = print if verbose else (lambda *a, **k: None)
        if raw is None:
            # raw given = arrays already read by the caller (e.g. cut from
            # global fields), as float32 with fill values set to NaN
            log(f"[data_v2] {name}: reading x {x0}:{x1}, y {y0}:{y1}, t {t0}:{t1}")
            raw = {}
            for key, path, var in _variables(cfg):
                raw[key] = read_box(path, var, t0, t1, y0, y1, x0, x1)
                log(f"[data_v2]   {key:6s} {var:7s} {raw[key].shape}")
            raw["mld"] = read_box(cfg.data.forcing.mld_path, cfg.data.forcing.mld_var,
                                  t0, t1, y0, y1, x0, x1)
        T = min(a.shape[0] for a in raw.values())
        for k in raw:
            raw[k] = raw[k][:T]

        # 3-D ocean mask from the tracer fill values (static z-grid)
        self.ocean = np.isfinite(raw["tracer"][0])          # (Z, H, W)
        self.land_mask = self.ocean[0]                      # (H, W) surface, True = ocean
        self.n_ocean_levels = self.ocean.sum(axis=0)        # (H, W)
        log(f"[data_v2]   ocean cells: {int(self.ocean.sum())} of {self.ocean.size}; "
            f"surface ocean {int(self.land_mask.sum())} of {self.land_mask.size}")

        if stats is None:
            train_end, _, _ = split_bounds(cfg, T)
            stats = compute_stats(raw, self.ocean, self.depths, train_end, lead=lead_steps(cfg))
            if stats_file:
                save_stats(stats_file, stats)
                log(f"[data_v2]   saved normalization stats -> {stats_file}")
        self.stats = stats

        rng = stats["tracer_max"] - stats["tracer_min"]
        rng = np.where(rng <= 1e-12, 1.0, rng)
        self.tracer_min = stats["tracer_min"].astype(np.float64)
        self.tracer_range = rng.astype(np.float64)

        ocean4 = self.ocean[None]
        tr = raw.pop("tracer")
        tr -= self.tracer_min[None, :, None, None].astype(np.float32)
        tr /= self.tracer_range[None, :, None, None].astype(np.float32)
        self.tracer = np.where(ocean4 & np.isfinite(tr), tr, np.float32(0.0)).astype(np.float32, copy=False)
        del tr
        self.fields = {"tracer": self.tracer}
        for key in ("u", "v", "w", "dif"):
            a = raw.pop(key)
            a -= np.float32(stats[f"{key}_mean"])
            a /= np.float32(stats[f"{key}_std"])
            self.fields[key] = np.where(ocean4 & np.isfinite(a), a, np.float32(0.0)).astype(np.float32, copy=False)
            del a
        mld = raw.pop("mld")                                  # (T, H, W)
        self.mld = mld
        md = (self.depths.astype(np.float32)[None, :, None, None] - mld[:, None])
        md -= np.float32(stats["mld_distance_mean"])
        md /= np.float32(stats["mld_distance_std"])
        self.fields["mld_distance"] = np.where(ocean4 & np.isfinite(md), md, np.float32(0.0)).astype(np.float32, copy=False)
        del md
        self.mask_f = self.ocean.astype(np.float32)          # (Z, H, W)

        self.T = T
        self.Z, self.H, self.W = self.ocean.shape
        self.lead = lead_steps(cfg)                                  # v5
        self.forcing_average = bool(cfg.data.get("forcing_average", False))
        self.mask_channel = bool(cfg.data.get("mask_channel", True))
        self.channels = CHANNELS if self.mask_channel else CHANNELS[:-1]
        self.n_channels = len(self.channels)
        self.target_mode = str(cfg.data.get("target", "full"))
        if self.target_mode not in ("full", "increment"):
            raise ValueError(f"data.target must be 'full' or 'increment', not {self.target_mode!r}")
        self.inc_scale = None
        if self.target_mode == "increment":
            if "inc_rms" not in stats:
                raise ValueError("normalization stats have no 'inc_rms'; recompute them "
                                 "(delete the norms file of this experiment)")
            self.inc_scale = np.asarray(stats["inc_rms"], dtype=np.float32)   # (Z,)

    def denormalize(self, tracer_norm):
        """(…, Z, H, W) normalized -> physical tracer units."""
        return (np.asarray(tracer_norm, dtype=np.float64)
                * self.tracer_range[:, None, None] + self.tracer_min[:, None, None])

    def to_full(self, pred, last):
        """
        Model output -> normalized full field.
        pred: (..., Z, H, W) model output; last: newest input tracer, same shape.
        Works for numpy arrays and torch tensors.
        """
        if self.target_mode == "full":
            return pred
        s = self.inc_scale[:, None, None]
        if isinstance(pred, torch.Tensor):
            s = torch.as_tensor(s, dtype=pred.dtype, device=pred.device)
        return last + pred * s

    def window(self, t, ys=slice(None), xs=slice(None), t_in=4, full_target=False):
        """
        Model input/target/mask for the window starting at local time index t.
        full_target=True returns the full-field target also in increment mode
        (for evaluation).
        """
        k = self.lead
        times = [t + i * k for i in range(t_in)]             # v5: input snapshots k steps apart
        sl = (slice(t, t + (t_in - 1) * k + 1, k), slice(None), ys, xs)
        chans = [self.tracer[sl]]
        for key in CHANNELS[1:-1]:
            if self.forcing_average and k > 1:
                # mean over the k steps up to each input time, (s-k, s]
                f = self.fields[key]
                chans.append(np.stack([f[max(0, s - k + 1):s + 1, :, ys, xs].mean(axis=0)
                                       for s in times]).astype(np.float32, copy=False))
            else:
                chans.append(self.fields[key][sl])
        m = self.mask_f[:, ys, xs]
        if self.mask_channel:
            chans.append(np.broadcast_to(m[None], chans[0].shape))
        x = np.stack(chans, axis=1)                          # (T_in, C, Z, h, w)
        last = times[-1]
        y = self.tracer[last + k, :, ys, xs]                 # (Z, h, w)
        if self.target_mode == "increment" and not full_target:
            y = (y - self.tracer[last, :, ys, xs]) / self.inc_scale[:, None, None]
            y = y * m                                        # 0 outside the ocean
        return x, y[None].astype(np.float32, copy=False), m[None]


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class MaskedOceanDataset(Dataset):
    """
    Windows of one RegionData for one split.

    patch_size: if set, every item is a random patch_size x patch_size crop
    with at least min_ocean_frac surface ocean (train split). Otherwise the
    whole box (or the crop given by ys/xs) is returned.
    """

    def __init__(self, region, split, cfg, patch_size=None, min_ocean_frac=0.3,
                 ys=slice(None), xs=slice(None), window_starts=None, full_target=False):
        self.region = region
        self.full_target = bool(full_target)
        self.split = split
        self.t_in = int(cfg.data.window.input_steps)
        self.patch_size = int(patch_size) if patch_size else None
        self.min_ocean_frac = float(min_ocean_frac)
        self.ys, self.xs = ys, xs
        train_end, val_end, usable = split_bounds(cfg, region.T)
        if window_starts is not None:
            self.indices = list(window_starts)
        else:
            self.t_start, self.t_end = {"train": (0, train_end), "val": (train_end, val_end),
                                        "test": (val_end, usable)}[split]
            self.indices = list(range(self.t_start, self.t_end))
        if self.indices:
            self.t_start, self.t_end = self.indices[0], self.indices[-1] + 1

        # attributes train.py / plotting code expect
        self.depths = region.depths
        self.n_depths = region.Z
        self.n_channels = region.n_channels
        self.H = self.patch_size or region.H
        self.W = self.patch_size or region.W
        self.x_range, self.y_range = region.x_range, region.y_range
        self.land_mask = region.land_mask

        if self.patch_size:
            p = self.patch_size
            if p > region.H or p > region.W:
                raise ValueError(f"patch_size {p} larger than region {region.H}x{region.W}")
            # precompute admissible patch corners
            surf = region.land_mask.astype(np.float64)
            c = np.pad(surf.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            frac = (c[p:, p:] - c[:-p, p:] - c[p:, :-p] + c[:-p, :-p]) / (p * p)
            self.corners = np.argwhere(frac >= self.min_ocean_frac)
            if len(self.corners) == 0:
                raise ValueError("no patch position with enough ocean")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        ys, xs = self.ys, self.xs
        if self.patch_size:
            k = int(torch.randint(len(self.corners), (1,)).item())
            y0, x0 = self.corners[k]
            ys = slice(int(y0), int(y0) + self.patch_size)
            xs = slice(int(x0), int(x0) + self.patch_size)
        x, y, m = self.region.window(t, ys, xs, self.t_in, full_target=self.full_target)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)))


def region_cfg_by_name(cfg, name):
    for r in cfg.data.regions:
        if r["name"] == name:
            return r
    raise ValueError(f"Region '{name}' not found in config.data.regions")


class PredictDataset(MaskedOceanDataset):
    """
    Test period of one box, with the attributes src/predict.py expects from
    data.py's WindowedOceanDataset. Normalized with the training region's
    statistics; targets are always the full field (to_full() converts an
    increment prediction). Time indices are local to the test period.
    """

    def __init__(self, cfg, region_name, stats):
        rc = region_cfg_by_name(cfg, region_name)
        T_total = n_timesteps(cfg)
        _, val_end, usable = split_bounds(cfg, T_total)
        region = RegionData(cfg, rc["x_range"], rc["y_range"], region_name,
                            t_range=(val_end, T_total), stats=stats)
        super().__init__(region, "test", cfg, window_starts=range(usable - val_end),
                         full_target=True)
        self.valid_z = np.arange(region.Z)
        self.z_to_local = {int(z): int(z) for z in self.valid_z}
        self.stats = region.stats
        self.mld_distance = region.fields["mld_distance"]   # normalized, (T, Z, H, W)
        self.z_index = rc.get("z_index", None)
        self.time_offset = val_end                           # global index of local t = 0
        self.target_mode = region.target_mode
        self.ocean = region.ocean

    def to_full(self, pred, inputs):
        """pred (B, 1, Z, H, W) model output, inputs (B, T, C, Z, H, W) -> full field."""
        return self.region.to_full(pred, inputs[:, -1:, 0])
