"""
evaluate_global.py -- evaluate a model trained on one region (v2: the large
North Atlantic box) or on several regions with one model (v3:
data.training_group) on randomly chosen boxes over the whole globe and on the
named boxes of the config, over the TEST period only.

v3: the random boxes are drawn avoiding only evaluation_global.sampling_exclude_regions
(default: all training regions). Setting it to the v2 training box keeps the
same 50 boxes as v2; random boxes that overlap any training region are then
kind "seen" and left out of the random-box scores.

    python -m src.evaluate_global --config config/nautilus_v2_NA_mask3d.yaml
    python -m src.evaluate_global --config ... --map-only     # only draw the boxes

Every box is normalized with the TRAINING region's statistics (what the model
learned), cells below the sea floor and on land are excluded from every score
and blank in every plot (3-D ocean mask), and the model is compared with the
"no change" (persistence) forecast.

Outputs in cfg.evaluation_global.out_dir:
    regions.csv                 every box: kind, indices, lat/lon range, ocean fraction
    map_boxes.png               training box, named boxes and random boxes on the grid
    metrics_per_region.csv      RMSE of model and persistence per box (normalized and physical)
    metrics_per_depthband.csv   the same per depth band
    rmse_by_level.csv/.png      mean RMSE per model level, model vs persistence
    metrics_per_level_per_region.csv   RMSE, no-change RMSE, ratio, skill per box and level
    heatmap_skill_depth_region.png     skill 1 - MSE/MSE(no change), depth level x box
    heatmap_ratio_band_region.png      RMSE(model)/RMSE(no change), depth band x box
    heatmap_rmse_depth_region.png      model RMSE in tracer units, depth level x box
    heatmap_ssim_change_depth_region.png  SSIM of predicted vs true change, depth level x box
    heatmap_ssim_full_depth_region.png    SSIM of the full field, depth level x box
    map_skill.png               boxes coloured by RMSE(model) / RMSE(no change)
    watercolumn/*.png           profile and section figures (sea floor blank)
    summary.json
"""

import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import xarray as xr

from src.data_v2 import (RegionData, MaskedOceanDataset, read_box, read_boxes,
                         n_timesteps, split_bounds, load_stats, stats_path,
                         region_cfg_by_name, model_name)
from src.model_conv2d_asym_hybrid import OceanConv2DAsymXPU
from src.model_light import build_model
from src.utils import load_config, get_device

DEPTH_BANDS = [(0, 200), (200, 1000), (1000, 2000), (2000, 4000), (4000, 6500)]


# ----------------------------------------------------------------------
# Boxes
# ----------------------------------------------------------------------
def overlaps(a, b):
    (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = a, b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def sample_boxes(surface_ocean, n, size, seed, min_frac, y_max, exclude):
    ny, nx = surface_ocean.shape
    c = np.pad(surface_ocean.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    rng = np.random.default_rng(seed)
    chosen = []
    for _ in range(200000):
        if len(chosen) == n:
            break
        x0 = int(rng.integers(0, nx - size + 1))
        y0 = int(rng.integers(0, min(ny, y_max) - size + 1))
        box = (x0, x0 + size, y0, y0 + size)
        frac = (c[y0 + size, x0 + size] - c[y0, x0 + size] - c[y0 + size, x0] + c[y0, x0]) / size ** 2
        if frac < min_frac:
            continue
        if any(overlaps(box, e) for e in exclude) or any(overlaps(box, b) for b in chosen):
            continue
        chosen.append(box)
    return chosen


def read_grid_info(cfg):
    p = cfg.data.tracers.paths[0]
    with xr.open_dataset(p, engine="netcdf4", decode_times=False) as ds:
        lat = ds["nav_lat"].values
        lon = ds["nav_lon"].values
    sfc = read_box(p, cfg.data.tracers.variables[0], 0, 1, 0, lat.shape[0], 0, lat.shape[1])
    return lat, lon, np.isfinite(sfc[0, 0])


def build_box_list(cfg, surface_ocean, lat, lon):
    ev = cfg.evaluation_global
    train_names = list(cfg.data.training_regions)
    boxes = []
    for name in ev.named_regions:
        r = region_cfg_by_name(cfg, name)
        kind = "training" if name in train_names else "named"
        boxes.append((name, kind, *r["x_range"], *r["y_range"]))
    exclude = []
    if ev.get("exclude_training_box", True):
        for name in list(ev.get("sampling_exclude_regions", None) or train_names):
            r = region_cfg_by_name(cfg, name)
            exclude.append((*r["x_range"], *r["y_range"]))
    rand = sample_boxes(surface_ocean, int(ev.n_random_regions), int(ev.box_size),
                        int(ev.seed), float(ev.min_surface_ocean_frac), int(ev.y_max), exclude)
    train_boxes = []
    for name in train_names:
        r = region_cfg_by_name(cfg, name)
        train_boxes.append((*r["x_range"], *r["y_range"]))
    for i, (x0, x1, y0, y1) in enumerate(rand):
        # v3: a random box that overlaps a training region is not unseen
        seen = any(overlaps((x0, x1, y0, y1), tb) for tb in train_boxes)
        boxes.append((f"R{i:02d}", "seen" if seen else "random", x0, x1, y0, y1))
    rows = []
    for name, kind, x0, x1, y0, y1 in boxes:
        la, lo = lat[y0:y1, x0:x1], lon[y0:y1, x0:x1]
        rows.append(dict(name=name, kind=kind, x0=x0, x1=x1, y0=y0, y1=y1,
                         lat_min=round(float(la.min()), 1), lat_max=round(float(la.max()), 1),
                         lon_min=round(float(lo.min()), 1), lon_max=round(float(lo.max()), 1),
                         surface_ocean_frac=round(float(surface_ocean[y0:y1, x0:x1].mean()), 2)))
    return rows


def draw_map(rows, surface_ocean, lat, path, values=None, vlabel=None):
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(np.where(surface_ocean, 1.0, 0.0), origin="lower", cmap="Greys_r",
              vmin=-0.6, vmax=1.15, interpolation="nearest")
    cmap = plt.get_cmap("RdYlGn_r")
    norm = None
    if values is not None:
        finite = [v for v in values.values() if np.isfinite(v) and v > 0]
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(min(finite), 1e-1), vmax=max(finite)) if finite else None
        if finite and min(finite) < 1.0 < max(finite):
            # some boxes better, some worse than "no change": green < 1 < red
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=min(finite), vcenter=1.0, vmax=max(finite))
    for r in rows:
        w, h = r["x1"] - r["x0"], r["y1"] - r["y0"]
        if r["kind"] == "training":
            ax.add_patch(Rectangle((r["x0"], r["y0"]), w, h, fill=False, ec="black", lw=2.8))
            ax.text(r["x0"] + 1, r["y1"] + 1.5, f"training: {r['name'].replace('_', ' ')}",
                    fontsize=8, weight="bold")
            continue
        if r["kind"] == "seen":
            # random box overlapping a training region: drawn, but not scored as unseen
            ax.add_patch(Rectangle((r["x0"], r["y0"]), w, h, fill=False, ec="grey", lw=0.9,
                                   ls="--", hatch="//"))
            ax.text(r["x0"] + 0.8, r["y0"] + 0.8, r["name"], fontsize=6, color="grey")
            continue
        if values is not None and r["name"] in values and norm is not None and np.isfinite(values[r["name"]]):
            fc = cmap(norm(values[r["name"]]))
            ax.add_patch(Rectangle((r["x0"], r["y0"]), w, h, fc=fc, ec="k", lw=0.6, alpha=0.85))
        else:
            ec = "tab:blue" if r["kind"] == "named" else "tab:orange"
            ax.add_patch(Rectangle((r["x0"], r["y0"]), w, h, fill=False, ec=ec,
                                   lw=1.8 if r["kind"] == "named" else 1.1))
        ax.text(r["x0"] + 0.8, r["y0"] + 0.8, r["name"].replace("_", " ") if r["kind"] == "named" else r["name"],
                fontsize=6.5 if r["kind"] == "random" else 7.5)
    # approximate latitude labels (ORCA grid: the y index is close to latitude
    # away from the north fold); taken along a central column
    xc = surface_ocean.shape[1] // 2
    yt = np.arange(0, surface_ocean.shape[0], 25)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{j}\n({lat[j, xc]:.0f}°)" for j in yt], fontsize=7)
    ax.set_xlabel("x index (ORCA1 grid)")
    ax.set_ylabel("y index (approx. latitude)")
    if values is not None and norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, shrink=0.7)
        cb.set_label(vlabel or "")
        # plain decimals on the colorbar (a log scale would label 6x10^-1)
        from matplotlib.ticker import FormatStrFormatter, NullFormatter
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb.ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f") if isinstance(norm, LogNorm) else NullFormatter())
        ax.set_title("Skill per box: RMSE(model) / RMSE(no change), test period  (<1 = better than no change)")
    else:
        ax.set_title("Training boxes (black), named evaluation boxes (blue), random evaluation boxes (orange), "
                     "random boxes overlapping training (grey, hatched)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Depth x box heatmaps
# ----------------------------------------------------------------------
def _box_order(region_rows):
    """training, named, then random boxes from south to north."""
    def lat_c(rw):
        return 0.5 * (float(rw["lat_min"]) + float(rw["lat_max"]))
    kinds = {"training": 0, "named": 1, "random": 2, "seen": 3}
    rows = sorted(region_rows, key=lambda rw: (kinds.get(rw["kind"], 3),
                                               lat_c(rw) if rw["kind"] == "random" else 0))
    labels = []
    for rw in rows:
        lc = lat_c(rw)
        hemi = "N" if lc >= 0 else "S"
        name = rw["name"].replace("North_Atlantic_big", "NA big").replace("_", " ")
        if rw["kind"] == "training":
            name += " (train)"
        elif rw["kind"] == "seen":
            name += " (seen)"
        labels.append(f"{name} ({abs(lc):.0f}°{hemi})")
    return rows, labels


def plot_depth_region_heatmaps(out_dir, region_rows, per_level, depths, tag=""):
    """
    per_level[name] = dict(sse_m, sse_p, sse_m_phys, n), arrays (Z,).
    Writes the per-level CSV and three heatmaps (boxes on x, depth on y).
    """
    from matplotlib.colors import TwoSlopeNorm, LogNorm
    rows, labels = _box_order(region_rows)
    names = [rw["name"] for rw in rows]
    Z = len(depths)
    skill = np.full((Z, len(names)), np.nan)
    rmse_phys = np.full((Z, len(names)), np.nan)
    csv_rows = []
    for j, nm in enumerate(names):
        a = per_level[nm]
        for z in range(Z):
            if a["n"][z] == 0:
                continue
            rm = np.sqrt(a["sse_m"][z] / a["n"][z])
            rp = np.sqrt(a["sse_p"][z] / a["n"][z])
            sk = 1.0 - a["sse_m"][z] / a["sse_p"][z] if a["sse_p"][z] > 0 else np.nan
            skill[z, j] = sk
            rmse_phys[z, j] = np.sqrt(a["sse_m_phys"][z] / a["n"][z])
            csv_rows.append(dict(name=nm, kind=rows[j]["kind"], level=z,
                                 depth_m=round(float(depths[z]), 1), ocean_cells_x_windows=int(a["n"][z]),
                                 rmse_model=rm, rmse_nochange=rp,
                                 ratio=rm / rp if rp > 0 else np.nan, skill=sk,
                                 rmse_model_phys=float(rmse_phys[z, j]),
                                 ssim_full=float(a["ssim_full"][z]),
                                 ssim_full_nochange=float(a["ssim_full_p"][z]),
                                 ssim_change=float(a["ssim_change"][z])))
    with open(os.path.join(out_dir, "metrics_per_level_per_region.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # boundary lines between training / named / random groups
    kinds = [rw["kind"] for rw in rows]
    seps = [j - 0.5 for j in range(1, len(kinds)) if kinds[j] != kinds[j - 1]]
    yt = [z for z in range(Z) if z % 5 == 0]
    width = max(10, 0.22 * len(names) + 3)

    def finish(fig, ax, im, cblabel, title, fname, ylabels):
        for sx in seps:
            ax.axvline(sx, color="black", lw=1.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label(cblabel)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.75")
    # 1) skill per level
    fig, ax = plt.subplots(figsize=(width, 8))
    # colour range from the data (98th percentile of |skill|), between 0.02 and 1
    fs = np.abs(skill[np.isfinite(skill)])
    vlim = float(np.clip(np.percentile(fs, 98), 0.02, 1.0)) if fs.size else 1.0
    im = ax.imshow(np.clip(skill, -vlim, vlim), aspect="auto", cmap=cmap,
                   norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim), interpolation="nearest")
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{depths[z]:.0f} m" for z in yt], fontsize=7)
    ax.set_ylabel("depth (model level)")
    finish(fig, ax, im, f"skill = 1 - MSE(model)/MSE(no change)\n>0 better than no change; clipped to ±{vlim:.2g}",
           f"{tag} skill per box and level, test period (grey: no ocean)",
           "heatmap_skill_depth_region.png", None)

    # 2) ratio per depth band, annotated
    band_ratio = np.full((len(DEPTH_BANDS), len(names)), np.nan)
    for j, nm in enumerate(names):
        a = per_level[nm]
        for b, (lo, hi) in enumerate(DEPTH_BANDS):
            sel = (depths >= lo) & (depths < hi)
            n = a["n"][sel].sum()
            if n > 0 and a["sse_p"][sel].sum() > 0:
                band_ratio[b, j] = np.sqrt(a["sse_m"][sel].sum() / a["sse_p"][sel].sum())
    finite = band_ratio[np.isfinite(band_ratio)]
    lo_v, hi_v = (float(finite.min()), float(finite.max())) if finite.size else (0.5, 2.0)
    lo_v, hi_v = min(lo_v, 0.99), max(hi_v, 1.01)
    cmap2 = plt.get_cmap("RdYlGn_r").copy()
    cmap2.set_bad("0.75")
    fig, ax = plt.subplots(figsize=(width, 3.6))
    im = ax.imshow(band_ratio, aspect="auto", cmap=cmap2,
                   norm=TwoSlopeNorm(vmin=lo_v, vcenter=1.0, vmax=hi_v), interpolation="nearest")
    for b in range(band_ratio.shape[0]):
        for j in range(band_ratio.shape[1]):
            v = band_ratio[b, j]
            if np.isfinite(v):
                ax.text(j, b, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="center",
                        fontsize=4.5, rotation=90)
    ax.set_yticks(range(len(DEPTH_BANDS)))
    ax.set_yticklabels([f"{lo}-{hi} m" for lo, hi in DEPTH_BANDS], fontsize=7)
    finish(fig, ax, im, "RMSE(model) / RMSE(no change)\n<1 better",
           f"{tag} RMSE ratio per box and depth band, test period",
           "heatmap_ratio_band_region.png", None)

    # 3) SSIM per level: change field and full field
    for key, fname, title, lab in (
            ("ssim_change", "heatmap_ssim_change_depth_region.png",
             "SSIM of the CHANGE field (predicted vs true change since the last input step)",
             "SSIM of the change field\n(0 = no pattern skill, 1 = perfect)"),
            ("ssim_full", "heatmap_ssim_full_depth_region.png",
             "SSIM of the FULL field (close to 1 also for \"no change\")",
             "SSIM of the full field")):
        arr = np.stack([per_level[nm][key] for nm in names], axis=1)       # (Z, boxes)
        arr = np.where(np.isfinite(skill), arr, np.nan)
        fa = arr[np.isfinite(arr)]
        if key == "ssim_change":
            vmin, vmax = -1.0, 1.0
            norm_ = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            cm = cmap
        else:
            vmin = float(np.percentile(fa, 2)) if fa.size else 0.0
            norm_ = None
            cm = plt.get_cmap("viridis").copy()
            cm.set_bad("0.75")
        fig, ax = plt.subplots(figsize=(width, 8))
        im = ax.imshow(arr, aspect="auto", cmap=cm, interpolation="nearest", norm=norm_,
                       **({} if norm_ is not None else dict(vmin=vmin, vmax=1.0)))
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{depths[z]:.0f} m" for z in yt], fontsize=7)
        ax.set_ylabel("depth (model level)")
        finish(fig, ax, im, lab, f"{tag} {title}, test period", fname, None)

    # 4) physical RMSE per level
    fin = rmse_phys[np.isfinite(rmse_phys) & (rmse_phys > 0)]
    cmap3 = plt.get_cmap("viridis").copy()
    cmap3.set_bad("0.75")
    fig, ax = plt.subplots(figsize=(width, 8))
    im = ax.imshow(rmse_phys, aspect="auto", cmap=cmap3, interpolation="nearest",
                   norm=LogNorm(vmin=float(fin.min()), vmax=float(fin.max())) if fin.size else None)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{depths[z]:.0f} m" for z in yt], fontsize=7)
    ax.set_ylabel("depth (model level)")
    finish(fig, ax, im, "RMSE of the model, tracer units",
           f"{tag} model RMSE per box and level, test period (grey: no ocean)",
           "heatmap_rmse_depth_region.png", None)


# ----------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------
def masked_ssim(x, y, m, sigma=1.5):
    """
    SSIM per level of two (Z, H, W) fields over ocean cells only.
    Gaussian window (sigma in cells) with normalized convolution, so land and
    sea-floor cells neither enter the local means nor are scored. Data range L
    per level = max - min of y over the ocean cells. Returns (Z,), NaN for a
    level without ocean or with a constant y.
    """
    from scipy.ndimage import gaussian_filter
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mf = m.astype(np.float64)
    g = lambda a: gaussian_filter(a, sigma=(0, sigma, sigma), mode="constant", truncate=3.0)
    w = g(mf)
    w = np.where(w > 1e-6, w, np.nan)
    mx = g(x * mf) / w
    my = g(y * mf) / w
    sxx = g(x * x * mf) / w - mx * mx
    syy = g(y * y * mf) / w - my * my
    sxy = g(x * y * mf) / w - mx * my
    yo = np.where(m, y, np.nan)
    has = m.any(axis=(1, 2))                    # levels with ocean; nanmax over an all-NaN level warns
    L = np.full(y.shape[0], np.nan)
    L[has] = np.nanmax(yo[has], axis=(1, 2)) - np.nanmin(yo[has], axis=(1, 2))
    L = np.where(np.isfinite(L) & (L > 0), L, np.nan)[:, None, None]
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    smap = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    smap = np.where(m, smap, np.nan)
    with np.errstate(all="ignore"):
        cnt = m.sum(axis=(1, 2))
        out = np.nansum(smap, axis=(1, 2)) / np.where(cnt > 0, cnt, np.nan)
    return np.where(np.isfinite(L[:, 0, 0]), out, np.nan)


class Acc:
    """Running sums of squared errors (and SSIM) per level, over ocean cells."""

    def __init__(self, Z):
        self.sse_m = np.zeros(Z)
        self.sse_p = np.zeros(Z)
        self.sse_m_phys = np.zeros(Z)
        self.sse_p_phys = np.zeros(Z)
        self.n = np.zeros(Z)
        self.out_of_range = 0
        self.n_cells = 0
        # SSIM sums per level: full field (model, no change) and change field
        self.ssim_full = np.zeros(Z)
        self.ssim_full_p = np.zeros(Z)
        self.ssim_change = np.zeros(Z)
        self.n_ssim_full = np.zeros(Z)
        self.n_ssim_change = np.zeros(Z)

    def add_ssim(self, pred, true, pers, mask):
        for arr, key, n_key in ((masked_ssim(pred, true, mask), "ssim_full", "n_ssim_full"),
                                (masked_ssim(pers, true, mask), "ssim_full_p", None),
                                (masked_ssim(pred - pers, true - pers, mask), "ssim_change", "n_ssim_change")):
            ok = np.isfinite(arr)
            getattr(self, key)[ok] += arr[ok]
            if n_key:
                getattr(self, n_key)[ok] += 1

    def ssim(self, key, sel=None):
        """Per-level mean SSIM (Z,), or the mean over the selected levels."""
        n = self.n_ssim_change if key == "ssim_change" else self.n_ssim_full
        with np.errstate(all="ignore"):
            per_level = np.where(n > 0, getattr(self, key) / np.where(n > 0, n, 1), np.nan)
        if sel is None:
            return per_level
        v = per_level[sel]
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    def add(self, pred, true, pers, mask, rng, with_ssim=False):
        # (Z, H, W) normalized; mask bool (Z, H, W); rng (Z,) physical range
        for z in range(pred.shape[0]):
            m = mask[z]
            if not m.any():
                continue
            em = (pred[z] - true[z])[m]
            ep = (pers[z] - true[z])[m]
            self.sse_m[z] += float((em.astype(np.float64) ** 2).sum())
            self.sse_p[z] += float((ep.astype(np.float64) ** 2).sum())
            self.sse_m_phys[z] += float(((em * rng[z]).astype(np.float64) ** 2).sum())
            self.sse_p_phys[z] += float(((ep * rng[z]).astype(np.float64) ** 2).sum())
            self.n[z] += m.sum()
        tv = true[mask]
        self.out_of_range += int(((tv < -0.05) | (tv > 1.05)).sum())
        self.n_cells += tv.size
        if with_ssim:
            self.add_ssim(pred, true, pers, mask)

    def rmse(self, sel=None, phys=False):
        sel = np.ones_like(self.n, bool) if sel is None else sel
        n = self.n[sel].sum()
        if n == 0:
            return np.nan, np.nan
        sm = (self.sse_m_phys if phys else self.sse_m)[sel].sum()
        sp = (self.sse_p_phys if phys else self.sse_p)[sel].sum()
        return float(np.sqrt(sm / n)), float(np.sqrt(sp / n))


def load_model(cfg, ckpt, device, n_channels):
    model = build_model(cfg, input_channels=n_channels).to(device)   # v6: cfg.model.arch
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/nautilus_v2_NA_mask3d.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--map-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ev = cfg.evaluation_global
    out_dir = ev.get("out_dir", "eval_global")
    os.makedirs(os.path.join(out_dir, "watercolumn"), exist_ok=True)
    train_region = model_name(cfg)          # v3: the training group for a multi-region model
    tag = cfg.training.get("experiment_name", "")

    lat, lon, surface_ocean = read_grid_info(cfg)
    rows = build_box_list(cfg, surface_ocean, lat, lon)
    with open(os.path.join(out_dir, "regions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    draw_map(rows, surface_ocean, lat, os.path.join(out_dir, "map_boxes.png"))
    print(f"[eval] {len(rows)} boxes ({sum(r['kind'] == 'random' for r in rows)} random unseen, "
          f"{sum(r['kind'] == 'seen' for r in rows)} random overlapping training) "
          f"-> {out_dir}/regions.csv, map_boxes.png", flush=True)
    if args.map_only:
        return

    device = get_device(cfg)
    ckpt = args.checkpoint or os.path.join(cfg.training.checkpoint_dir,
                                           f"{tag}_{train_region}_best_model.pt")
    stats = load_stats(args.stats or stats_path(cfg, train_region))
    print(f"[eval] checkpoint {ckpt}\n[eval] normalization of every box: {train_region} statistics", flush=True)

    T_total = n_timesteps(cfg)
    train_end, val_end, usable = split_bounds(cfg, T_total)
    t_in = int(cfg.data.window.input_steps)
    t0, t1 = val_end, T_total
    n_windows = usable - val_end
    print(f"[eval] test period: windows {val_end}:{usable} ({n_windows}), time steps {t0}:{t1}", flush=True)

    raw_all = read_boxes(cfg, rows, t0, t1)

    model = None
    Z = None
    level_acc = {"random": None, "named": None, "training": None, "seen": None}
    region_rows, band_rows = [], []
    per_level = {}
    n_plots = int(ev.get("plot_regions", 6))
    plot_names = [r["name"] for r in rows if r["kind"] != "random"] + \
                 [r["name"] for r in rows if r["kind"] == "random"][:max(0, n_plots - 2)]
    depths = None

    for r in rows:
        tic = time.time()
        rd = RegionData(cfg, (r["x0"], r["x1"]), (r["y0"], r["y1"]), r["name"],
                        t_range=(t0, t1), stats=stats, verbose=False,
                        raw=raw_all.pop(r["name"]))
        depths = rd.depths
        Z = rd.Z
        if model is None:
            model = load_model(cfg, ckpt, device, rd.n_channels)
            for k in level_acc:
                level_acc[k] = Acc(Z)
        ds = MaskedOceanDataset(rd, "test", cfg, window_starts=range(n_windows), full_target=True)
        bs = max(1, int(4096 // (rd.H * rd.W)))
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
        acc = Acc(Z)
        mask = rd.ocean
        rng_z = rd.tracer_range
        sample_for_plot = None
        mid = n_windows // 2
        k = 0
        with torch.no_grad():
            for x, y, m in loader:
                out = model(x.to(device)).float().cpu().numpy()[:, 0]      # (B, Z, H, W)
                true = y.numpy()[:, 0]
                pers = x.numpy()[:, -1, 0]                                 # newest input tracer
                pred = rd.to_full(out, pers)                               # increment -> full field
                for b in range(pred.shape[0]):
                    acc.add(pred[b], true[b], pers[b], mask, rng_z, with_ssim=True)
                    level_acc[r["kind"]].add(pred[b], true[b], pers[b], mask, rng_z)
                    if k == mid:
                        sample_for_plot = (pred[b].copy(), true[b].copy(), pers[b].copy())
                    k += 1
        rm, rp = acc.rmse()
        rmp, rpp = acc.rmse(phys=True)
        per_level[r["name"]] = dict(sse_m=acc.sse_m.copy(), sse_p=acc.sse_p.copy(),
                                    sse_m_phys=acc.sse_m_phys.copy(), n=acc.n.copy(),
                                    ssim_full=acc.ssim("ssim_full"),
                                    ssim_full_p=acc.ssim("ssim_full_p"),
                                    ssim_change=acc.ssim("ssim_change"))
        row = dict(name=r["name"], kind=r["kind"], lat_min=r["lat_min"], lat_max=r["lat_max"],
                   lon_min=r["lon_min"], lon_max=r["lon_max"],
                   surface_ocean_frac=r["surface_ocean_frac"],
                   ocean_levels_max=int(rd.n_ocean_levels.max()),
                   rmse_model=rm, rmse_persistence=rp,
                   ratio=rm / rp if rp and rp > 0 else np.nan,
                   rmse_model_phys=rmp, rmse_persistence_phys=rpp,
                   true_outside_0_1_frac=acc.out_of_range / max(1, acc.n_cells),
                   ssim_full=acc.ssim("ssim_full", slice(None)),
                   ssim_full_nochange=acc.ssim("ssim_full_p", slice(None)),
                   ssim_change=acc.ssim("ssim_change", slice(None)))
        region_rows.append(row)
        for lo_, hi_ in DEPTH_BANDS:
            sel = (depths >= lo_) & (depths < hi_)
            bm, bp = acc.rmse(sel)
            band_ssim = dict(ssim_full=acc.ssim("ssim_full", sel),
                             ssim_change=acc.ssim("ssim_change", sel))
            band_rows.append(dict(name=r["name"], kind=r["kind"], band=f"{lo_}-{hi_} m",
                                  ocean_cells_x_windows=int(acc.n[sel].sum()),
                                  rmse_model=bm, rmse_persistence=bp,
                                  ratio=bm / bp if bp and bp > 0 else np.nan, **band_ssim))
        print(f"[eval] {r['name']:20s} {r['kind']:8s} RMSE model {rm:.5f}  no-change {rp:.5f}  "
              f"ratio {row['ratio']:.4f}  SSIM full {row['ssim_full']:.5f} "
              f"change {row['ssim_change']:.3f}  ({time.time() - tic:.0f} s)", flush=True)

        if r["name"] in plot_names and sample_for_plot is not None:
            try:
                from src.water_column_viz import plot_water_column_comparison
                p, t, _ = sample_for_plot
                tp = np.where(mask, rd.denormalize(t), np.nan)
                pp = np.where(mask, rd.denormalize(p), np.nan)
                plot_water_column_comparison(
                    true_field=tp, pred_field=pp, depths=depths,
                    title=f"{r['name']} ({r['kind']}) | test window {val_end + mid} | sea floor masked",
                    save_path=os.path.join(out_dir, "watercolumn", f"{r['name']}_watercolumn.png"),
                )
            except Exception as e:  # a plot must not stop the evaluation
                print(f"[eval] water-column plot failed for {r['name']}: {e}", flush=True)
        del rd, ds, loader

    def write(path, rows_):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader()
            w.writerows(rows_)

    write(os.path.join(out_dir, "metrics_per_region.csv"), region_rows)
    write(os.path.join(out_dir, "metrics_per_depthband.csv"), band_rows)

    lvl_rows = []
    for z in range(Z):
        row = dict(level=z, depth_m=round(float(depths[z]), 1))
        for kind, a in level_acc.items():
            if a is None or a.n[z] == 0:
                row[f"{kind}_rmse_model"] = row[f"{kind}_rmse_persistence"] = np.nan
            else:
                row[f"{kind}_rmse_model"] = float(np.sqrt(a.sse_m[z] / a.n[z]))
                row[f"{kind}_rmse_persistence"] = float(np.sqrt(a.sse_p[z] / a.n[z]))
        lvl_rows.append(row)
    write(os.path.join(out_dir, "rmse_by_level.csv"), lvl_rows)

    fig, ax = plt.subplots(figsize=(6, 7))
    for kind, col in (("training", "black"), ("random", "tab:orange"), ("named", "tab:blue")):
        if level_acc.get(kind) is None or level_acc[kind].n.sum() == 0:
            continue
        mm = [rw[f"{kind}_rmse_model"] for rw in lvl_rows]
        pp = [rw[f"{kind}_rmse_persistence"] for rw in lvl_rows]
        ax.plot(mm, depths, "-", color=col, label=f"{kind} boxes: model")
        ax.plot(pp, depths, ":", color=col, label=f"{kind} boxes: no change")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE, normalized tracer (training-region scaling)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("RMSE per level, ocean cells only, test period")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rmse_by_level.png"), dpi=150)
    plt.close(fig)

    try:
        plot_depth_region_heatmaps(out_dir, region_rows, per_level, np.asarray(depths), tag=tag)
        print(f"[eval] depth x box heatmaps -> {out_dir}/heatmap_*.png", flush=True)
    except Exception as e:  # a plot must not stop the evaluation
        print(f"[eval] heatmaps failed: {e}", flush=True)

    draw_map(rows, surface_ocean, lat, os.path.join(out_dir, "map_skill.png"),
             values={rw["name"]: rw["ratio"] for rw in region_rows},
             vlabel="RMSE(model) / RMSE(no change)")

    rnd = [rw for rw in region_rows if rw["kind"] == "random"]
    summary = dict(
        checkpoint=ckpt, training_region=train_region, test_windows=n_windows,
        n_random=len(rnd), n_seen=sum(rw["kind"] == "seen" for rw in region_rows),
        training_regions=list(cfg.data.training_regions),
        random_ratio_median=float(np.nanmedian([rw["ratio"] for rw in rnd])) if rnd else None,
        random_ratio_min=float(np.nanmin([rw["ratio"] for rw in rnd])) if rnd else None,
        random_ratio_max=float(np.nanmax([rw["ratio"] for rw in rnd])) if rnd else None,
        random_better_than_no_change=int(sum(rw["ratio"] < 1 for rw in rnd)),
        per_kind={k: dict(zip(("rmse_model", "rmse_persistence"), level_acc[k].rmse()))
                  for k in level_acc if level_acc[k] is not None and level_acc[k].n.sum() > 0},
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[eval] summary:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
