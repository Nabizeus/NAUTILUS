"""
water_column_viz.py

Side-by-side True-vs-Predicted visualization for full-column (all-depth)
ocean tracer fields, adapted from the "Water Column Tracer Visualization
Module" reference design.

Differences from the reference design (and why):
  - Panel 2 in the reference ("Plan View / Section Line" on a real
    lat/lon map) is dropped. The dataset here is a small regional
    patch indexed by grid points (x_range/y_range), not real
    lat/lon, so a world map with a section line doesn't apply.
    The Hovmöller section itself (depth vs. distance along a
    transect through the patch) is kept.
  - Panel 5 ("3D Isosurface View") needs mayavi/pyvista, which
    aren't in this project's requirements. It's approximated with
    layered matplotlib surfaces (same technique as panel 1, fewer/
    wider-spaced levels) and is off by default since it's largely
    redundant with panel 1 for a quick forecast-quality check.
  - Every panel is rendered for True and Predicted side by side
    (plus an Error panel where that's meaningful) using SHARED color
    limits, so the two columns are visually comparable.
  - Axes are labeled in grid points, not km, since the dataset
    doesn't carry real grid spacing.

Fields are expected as (Z, H, W) arrays already in physical units
(i.e. after denormalize_tracer(...) in predict.py), with land/invalid
points as NaN. `depths` is a 1D array of length Z, in meters.

Typical usage from predict.py, once per sample you want to visualize
(e.g. i == 0), after the full-column true_np/pred_np have been
denormalized and land-masked:

    from src.water_column_viz import plot_water_column_comparison

    true_masked = np.where(mask_np, true_np, np.nan)
    pred_masked = np.where(mask_np, pred_np, np.nan)

    plot_water_column_comparison(
        true_field=true_masked,
        pred_field=pred_masked,
        depths=dataset.depths[dataset.valid_z],
        title=f"{test_region} | forecast quality | t={t_idx}",
        save_path=os.path.join(
            output_dir, "watercolumn",
            f"{test_region}_t{t_idx:04d}_watercolumn_compare.png",
        ),
    )
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

TRACER_CMAP = "Blues"  # oceanic light-to-dark blue instead of "turbo"
ERROR_CMAP = "RdBu_r"


# =====================================================================
# Helpers
# =====================================================================

def _safe_vlims(*fields, pct=1.0):
    """Shared color limits across one or more fields, ignoring NaNs."""
    stacked = np.concatenate([np.asarray(f).ravel() for f in fields])
    stacked = stacked[np.isfinite(stacked)]
    if stacked.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(stacked, pct))
    vmax = float(np.percentile(stacked, 100 - pct))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return vmin, vmax


def _symmetric_error_vlim(error_field, pct=1.0):
    """Symmetric color limit around 0 for a diverging error colormap."""
    err = np.asarray(error_field).ravel()
    err = err[np.isfinite(err)]
    if err.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(err), 100 - pct))
    return v if v > 0 else 1.0


def _check_field(field, depths, name):
    field = np.asarray(field)
    depths = np.asarray(depths)
    if field.ndim != 3:
        raise ValueError(f"{name} must have shape (Z,H,W), got {field.shape}")
    if field.shape[0] != len(depths):
        raise ValueError(
            f"{name} has {field.shape[0]} depth levels but `depths` has "
            f"{len(depths)} — pass the same depth subset used to build the field "
            f"(e.g. dataset.depths[dataset.valid_z])."
        )
    return field, depths


# =====================================================================
# Panel builders (each draws into an Axes handed to it)
# =====================================================================

def mld_field_from_distance(mld_distance, depths, z_index=0):
    """
    Recover the raw mixed-layer depth (meters), as a full (H, W) map,
    from a (Z, H, W) mld_distance array for ONE sample/timestep.

    data.py computes (see src/depth_skill_profile.estimate_mean_mld_m
    for the same derivation): mld_distance[z,y,x] = depth[z] - mld[y,x],
    so for ANY fixed z: mld[y,x] = depth[z] - mld_distance[z,y,x].
    """
    mld_distance = np.asarray(mld_distance)
    depth_z = float(np.asarray(depths)[z_index])
    return depth_z - mld_distance[z_index]


def _mld_transect(mld_map, transect, idx):
    """Slice a (H,W) MLD map along the same row/col a Hovmöller section uses."""
    if mld_map is None:
        return None
    return mld_map[idx, :] if transect == "row" else mld_map[:, idx]


def plot_pseudo3d_volume(ax, field, depths, vmin, vmax, n_layers=8, cmap=TRACER_CMAP,
                          label=""):
    """Panel 1 style: stacked semi-transparent depth-layer maps."""
    Z, H, W = field.shape
    layer_idx = np.unique(np.linspace(0, Z - 1, min(n_layers, Z)).astype(int))

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)

    for zi in layer_idx:
        layer = field[zi]
        zdata = np.full_like(xx, fill_value=depths[zi], dtype=float)
        facecolors = cmap_obj(norm(np.nan_to_num(layer, nan=vmin)))
        facecolors[..., 3] = np.where(np.isnan(layer), 0.0, 0.85)
        ax.plot_surface(
            xx, yy, zdata,
            facecolors=facecolors,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False,
        )

    ax.set_zlim(float(depths.max()) + 100, float(depths.min()) - 100)  # depth downward
    ax.set_xlabel("X (grid pts)")
    ax.set_ylabel("Y (grid pts)")
    ax.set_zlabel("Depth (m)")
    ax.set_title(f"Pseudo-3D Volume{(' — ' + label) if label else ''}", fontsize=11)


def plot_hovmoller_section(ax, field, depths, transect="row", index=None,
                            vmin=None, vmax=None, cmap=TRACER_CMAP, label="",
                            mld_map=None, depth_max_m=None, show_ylabel=True,
                            mld_color="orangered"):
    """
    Panel 2 style: depth vs. distance along a transect through the patch.

    mld_map: optional (H, W) mixed-layer-depth field in meters (see
        mld_field_from_distance) — drawn as a dashed line following the
        SAME transect as the tracer field, since MLD varies spatially
        and a flat horizontal line would misrepresent that.
    depth_max_m: optional — crop the depth axis to [0, depth_max_m],
        e.g. for a zoomed-in near-surface view where the MLD (usually
        only tens to a few hundred meters) would otherwise be squeezed
        into a thin sliver against a multi-thousand-meter full column.
    show_ylabel: set False to hide the depth-axis tick labels/label —
        use this on the middle/right panels of a True|Predicted|Error
        row, since they share the same depth axis as the left panel
        and repeating it just crowds the plot next to the colorbars.
    mld_color: color of the MLD line — a warm color (orange/red) shows
        up clearly against the blue tracer colormap; the default
        black blended in too easily.
    """
    Z, H, W = field.shape
    if transect == "row":
        idx = H // 2 if index is None else index
        section = field[:, idx, :]
        distance = np.arange(W)
        xlabel = "Distance along row (grid pts)"
    elif transect == "col":
        idx = W // 2 if index is None else index
        section = field[:, :, idx]
        distance = np.arange(H)
        xlabel = "Distance along column (grid pts)"
    else:
        raise ValueError(f"Unknown transect '{transect}' (use 'row' or 'col')")

    im = ax.pcolormesh(distance, depths, section, shading="auto",
                        cmap=cmap, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    if depth_max_m is not None:
        ax.set_ylim(depth_max_m, float(np.min(depths)))
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Depth (m)")
    else:
        ax.tick_params(labelleft=False)
    zoom_tag = f" (0-{depth_max_m:.0f}m)" if depth_max_m is not None else ""
    ax.set_title(f"Hovmöller Section{(' — ' + label) if label else ''}{zoom_tag}", fontsize=11)

    mld_line = _mld_transect(mld_map, transect, idx)
    if mld_line is not None:
        ax.plot(distance, mld_line, color=mld_color, linestyle="--",
                linewidth=1.8, alpha=0.95, label="MLD")
        ax.legend(fontsize=7, loc="upper right")

    return im


def plot_vertical_profile_compare(ax, true_field, pred_field, depths, y0=None, x0=None,
                                   mld_depth_m=None, mld_color="orangered", show_ylabel=True):
    """
    Panel 3 style: True vs Predicted profiles overlaid at one location.

    mld_depth_m: optional scalar (meters) — the mixed-layer depth AT
        THIS SPECIFIC (y0, x0) point, drawn as a horizontal reference
        line (unlike the Hovmöller section, a single profile is one
        point, so one number is the correct — not simplified — MLD
        here, not an average).
    mld_color: see plot_hovmoller_section — a warm color shows up
        better than black against the profile lines.
    """
    Z, H, W = true_field.shape
    y0 = H // 2 if y0 is None else y0
    x0 = W // 2 if x0 is None else x0

    true_profile = true_field[:, y0, x0]
    pred_profile = pred_field[:, y0, x0]

    ax.plot(true_profile, depths, "-o", color="#4E79A7", markersize=3, label="True")
    ax.plot(pred_profile, depths, "--o", color="#F28E2B", markersize=3, label="Predicted")
    if mld_depth_m is not None:
        ax.axhline(mld_depth_m, color=mld_color, linestyle="--", linewidth=1.8,
                   alpha=0.9, label=f"MLD ({mld_depth_m:.0f} m)")
    ax.invert_yaxis()
    ax.set_xlabel("Tracer")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    if show_ylabel:
        ax.set_ylabel("Depth (m)")
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(f"Vertical Profile @ (y={y0}, x={x0})", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_error_profile_compare(ax, true_field, pred_field, depths, y0=None, x0=None,
                                mld_depth_m=None, mld_color="orangered", show_ylabel=False):
    """
    Companion to plot_vertical_profile_compare: (Predicted - True) —
    i.e. the forecast bias/error — as its own line vs. depth at the
    same point, so over/under-prediction is directly visible instead
    of having to eyeball the gap between two overlapping profile lines.
    Also reports the RMSE at that single point in the title.
    """
    Z, H, W = true_field.shape
    y0 = H // 2 if y0 is None else y0
    x0 = W // 2 if x0 is None else x0

    error_profile = pred_field[:, y0, x0] - true_field[:, y0, x0]
    finite = np.isfinite(error_profile)
    point_rmse = float(np.sqrt(np.mean(error_profile[finite] ** 2))) if finite.any() else float("nan")

    ax.plot(error_profile, depths, "-o", color="#C0392B", markersize=3, label="Error (Pred-True)")
    ax.axvline(0, color="gray", linestyle="-", linewidth=1, alpha=0.6)
    if mld_depth_m is not None:
        ax.axhline(mld_depth_m, color=mld_color, linestyle="--", linewidth=1.8,
                   alpha=0.9, label=f"MLD ({mld_depth_m:.0f} m)")
    ax.invert_yaxis()
    ax.set_xlabel("Error (Pred-True)")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    if show_ylabel:
        ax.set_ylabel("Depth (m)")
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(f"Error Profile @ (y={y0}, x={x0})\nRMSE={point_rmse:.3g}", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_depth_map(ax, field, depths, dmin, dmax, vmin, vmax, cmap=TRACER_CMAP,
                    label="", show_xlabel=True):
    """One depth-integrated (averaged) map over [dmin, dmax] meters."""
    mask = (depths >= dmin) & (depths <= dmax)
    if not mask.any():
        ax.text(0.5, 0.5, "no levels\nin range", ha="center", va="center")
        ax.axis("off")
        return None
    if mask.sum() > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg = np.nanmean(field[mask], axis=0)
    else:
        avg = field[mask][0]
    im = ax.imshow(avg, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(f"{label} {dmin:.0f}-{dmax:.0f} m" if label else f"{dmin:.0f}-{dmax:.0f} m",
                 fontsize=10)
    if show_xlabel:
        ax.set_xlabel("X (grid pts)", fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("Y (grid pts)", fontsize=8)
    return im


# =====================================================================
# Top-level: True vs Predicted comparison dashboard
# =====================================================================

def plot_water_column_comparison(
    true_field,
    pred_field,
    depths,
    title="Water Column Forecast Quality",
    save_path="water_column_compare.png",
    depth_bins=((0, 500), (500, 2000), (2000, 5000)),
    transect="row",
    transect_index=None,
    profile_yx=None,
    include_isosurface=False,
    cmap=TRACER_CMAP,
    error_cmap=ERROR_CMAP,
    dpi=150,
    mld_distance=None,
    mld_z_index=0,
    zoom_depth_max_m=1000.0,
    mld_color="orangered",
):
    """
    Build a multi-panel dashboard comparing True vs Predicted full-column
    tracer fields, so forecast quality can be assessed at a glance:

      Row 1: Pseudo-3D volume        [True] [Predicted]
      Row 2: Hovmöller section       [True] [Predicted] [Error]
      Row 3: Vertical profile        [True & Predicted overlaid]
      Row 4: Depth-integrated maps   for each depth bin: [True] [Predicted] [Error]

    Parameters
    ----------
    true_field, pred_field : (Z, H, W) arrays, physical units, NaN = land/invalid
    depths : (Z,) array, meters
    depth_bins : sequence of (min_m, max_m) tuples for the depth-integrated row
    transect : "row" or "col" — which axis the Hovmöller section cuts along
    transect_index : index along the other axis (defaults to the center)
    profile_yx : (y0, x0) pixel location for the vertical profile (defaults to center)
    include_isosurface : add an extra True/Predicted layered-volume row
        (matplotlib approximation, not a true isosurface — see module docstring)
    mld_distance : optional (Z, H, W) array, the SAME sample's raw
        mld_distance forcing field (e.g. mld_distance_np from predict.py) —
        if given, overlays the mixed-layer depth: as a curve following
        the transect on the Hovmöller section (MLD varies spatially, a
        flat line would misrepresent it), and as a single-point
        reference line at (y0, x0) on the vertical profile. MLD is a
        forcing input, not something the model predicts, so the same
        line is shown on both True and Predicted panels.
    mld_z_index : which z-level's mld_distance to use for the recovery
        formula (any level gives the same answer in principle — see
        mld_field_from_distance — 0 is just a safe default).
    zoom_depth_max_m : adds a second Hovmöller row cropped to
        [0, zoom_depth_max_m] — the MLD is usually only tens to a few
        hundred meters deep, easily lost as a thin sliver against a
        multi-thousand-meter full column. Set to None to disable.
    """
    true_field, depths = _check_field(true_field, depths, "true_field")
    pred_field, _ = _check_field(pred_field, depths, "pred_field")
    if pred_field.shape != true_field.shape:
        raise ValueError(
            f"true_field {true_field.shape} and pred_field {pred_field.shape} "
            "must have the same shape"
        )

    error_field = pred_field - true_field
    vmin, vmax = _safe_vlims(true_field, pred_field)
    evlim = _symmetric_error_vlim(error_field)

    mld_map = None
    if mld_distance is not None:
        mld_map = mld_field_from_distance(mld_distance, depths, z_index=mld_z_index)

    n_rows = 4 + (1 if include_isosurface else 0) + (1 if zoom_depth_max_m else 0)
    height_ratios = (
        [1.1]
        + ([1.0] if include_isosurface else [])
        + [1.0]
        + ([0.85] if zoom_depth_max_m else [])
        + [0.85, 2.1]
    )
    fig = plt.figure(figsize=(16, 4.6 * n_rows))
    outer_gs = fig.add_gridspec(n_rows, 12, height_ratios=height_ratios, hspace=0.6, wspace=0.7)

    row = 0

    # -----------------------------------------------------------
    # Row: Pseudo-3D volume (True | Predicted)
    # -----------------------------------------------------------
    ax_t = fig.add_subplot(outer_gs[row, 0:6], projection="3d")
    ax_p = fig.add_subplot(outer_gs[row, 6:12], projection="3d")
    plot_pseudo3d_volume(ax_t, true_field, depths, vmin, vmax, cmap=cmap, label="True")
    plot_pseudo3d_volume(ax_p, pred_field, depths, vmin, vmax, cmap=cmap, label="Predicted")
    row += 1

    if include_isosurface:
        ax_t2 = fig.add_subplot(outer_gs[row, 0:6], projection="3d")
        ax_p2 = fig.add_subplot(outer_gs[row, 6:12], projection="3d")
        plot_pseudo3d_volume(ax_t2, true_field, depths, vmin, vmax, n_layers=5,
                              cmap=cmap, label="True (layered approx.)")
        plot_pseudo3d_volume(ax_p2, pred_field, depths, vmin, vmax, n_layers=5,
                              cmap=cmap, label="Predicted (layered approx.)")
        row += 1

    # -----------------------------------------------------------
    # Row: Hovmöller section (True | Predicted | Error)
    # -----------------------------------------------------------
    ax_t = fig.add_subplot(outer_gs[row, 0:4])
    ax_p = fig.add_subplot(outer_gs[row, 4:8])
    ax_e = fig.add_subplot(outer_gs[row, 8:12])

    im_t = plot_hovmoller_section(ax_t, true_field, depths, transect, transect_index,
                                   vmin, vmax, cmap=cmap, label="True", mld_map=mld_map,
                                   mld_color=mld_color, show_ylabel=True)
    im_p = plot_hovmoller_section(ax_p, pred_field, depths, transect, transect_index,
                                   vmin, vmax, cmap=cmap, label="Predicted", mld_map=mld_map,
                                   mld_color=mld_color, show_ylabel=False)
    im_e = plot_hovmoller_section(ax_e, error_field, depths, transect, transect_index,
                                   -evlim, evlim, cmap=error_cmap, label="Error (Pred-True)",
                                   mld_map=mld_map, mld_color=mld_color, show_ylabel=False)

    fig.colorbar(im_t, ax=ax_t, label="Tracer", fraction=0.046, pad=0.04)
    fig.colorbar(im_p, ax=ax_p, label="Tracer", fraction=0.046, pad=0.04)
    fig.colorbar(im_e, ax=ax_e, label="Error", fraction=0.046, pad=0.04)
    row += 1

    if zoom_depth_max_m:
        # -----------------------------------------------------------
        # Row: Zoomed-in Hovmöller section, near-surface only — the MLD
        # line is otherwise squeezed into a thin sliver against a
        # multi-thousand-meter full column. Uses its OWN color scale,
        # computed just from this shallower slice, since the full-depth
        # scale is dominated by deeper values and would wash out
        # near-surface contrast.
        # -----------------------------------------------------------
        zoom_mask = depths <= zoom_depth_max_m
        if zoom_mask.any():
            zoom_vmin, zoom_vmax = _safe_vlims(true_field[zoom_mask], pred_field[zoom_mask])
            zoom_evlim = _symmetric_error_vlim(error_field[zoom_mask])

            ax_tz = fig.add_subplot(outer_gs[row, 0:4])
            ax_pz = fig.add_subplot(outer_gs[row, 4:8])
            ax_ez = fig.add_subplot(outer_gs[row, 8:12])

            im_tz = plot_hovmoller_section(ax_tz, true_field, depths, transect, transect_index,
                                            zoom_vmin, zoom_vmax, cmap=cmap, label="True",
                                            mld_map=mld_map, depth_max_m=zoom_depth_max_m,
                                            mld_color=mld_color, show_ylabel=True)
            im_pz = plot_hovmoller_section(ax_pz, pred_field, depths, transect, transect_index,
                                            zoom_vmin, zoom_vmax, cmap=cmap, label="Predicted",
                                            mld_map=mld_map, depth_max_m=zoom_depth_max_m,
                                            mld_color=mld_color, show_ylabel=False)
            im_ez = plot_hovmoller_section(ax_ez, error_field, depths, transect, transect_index,
                                            -zoom_evlim, zoom_evlim, cmap=error_cmap,
                                            label="Error (Pred-True)",
                                            mld_map=mld_map, depth_max_m=zoom_depth_max_m,
                                            mld_color=mld_color, show_ylabel=False)

            fig.colorbar(im_tz, ax=ax_tz, label="Tracer", fraction=0.046, pad=0.04)
            fig.colorbar(im_pz, ax=ax_pz, label="Tracer", fraction=0.046, pad=0.04)
            fig.colorbar(im_ez, ax=ax_ez, label="Error", fraction=0.046, pad=0.04)
        else:
            print(f"⚠ No depth levels ≤ {zoom_depth_max_m}m — skipping zoomed Hovmöller row.")
        row += 1

    # -----------------------------------------------------------
    # Row: Vertical profile (overlaid) + error profile + summary stats box
    # -----------------------------------------------------------
    ax_profile = fig.add_subplot(outer_gs[row, 0:5])
    y0, x0 = profile_yx if profile_yx is not None else (true_field.shape[1] // 2,
                                                          true_field.shape[2] // 2)
    mld_at_point = float(mld_map[y0, x0]) if mld_map is not None else None
    plot_vertical_profile_compare(ax_profile, true_field, pred_field, depths, y0, x0,
                                   mld_depth_m=mld_at_point, mld_color=mld_color,
                                   show_ylabel=True)

    ax_error_profile = fig.add_subplot(outer_gs[row, 5:9])
    plot_error_profile_compare(ax_error_profile, true_field, pred_field, depths, y0, x0,
                                mld_depth_m=mld_at_point, mld_color=mld_color,
                                show_ylabel=False)

    ax_stats = fig.add_subplot(outer_gs[row, 9:12])
    ax_stats.axis("off")
    finite = np.isfinite(error_field)
    rmse = float(np.sqrt(np.nanmean(error_field[finite] ** 2))) if finite.any() else float("nan")
    mae = float(np.nanmean(np.abs(error_field[finite]))) if finite.any() else float("nan")
    bias = float(np.nanmean(error_field[finite])) if finite.any() else float("nan")
    stats_txt = (
        f"{title}\n\n"
        f"Depth levels: {len(depths)}\n"
        f"Depth range: {depths.min():.0f}-{depths.max():.0f} m\n"
        f"Field shape: {true_field.shape}\n\n"
        f"RMSE:  {rmse:.3e}\n"
        f"MAE:   {mae:.3e}\n"
        f"Bias:  {bias:+.3e}"
    )
    ax_stats.text(0, 1, stats_txt, va="top", family="monospace", fontsize=9)
    row += 1

    # -----------------------------------------------------------
    # Row: Depth-integrated maps (per bin: True | Predicted | Error)
    # -----------------------------------------------------------
    n_bins = len(depth_bins)
    inner = outer_gs[row, :].subgridspec(3, n_bins, hspace=0.7, wspace=0.4)

    true_axes, pred_axes, err_axes = [], [], []
    last_im, last_err_im = None, None
    for k, (dmin, dmax) in enumerate(depth_bins):
        ax_true = fig.add_subplot(inner[0, k])
        ax_pred = fig.add_subplot(inner[1, k])
        ax_err = fig.add_subplot(inner[2, k])
        true_axes.append(ax_true)
        pred_axes.append(ax_pred)
        err_axes.append(ax_err)

        im = plot_depth_map(ax_true, true_field, depths, dmin, dmax, vmin, vmax,
                             cmap=cmap, label="True", show_xlabel=False)
        last_im = im if im is not None else last_im
        im = plot_depth_map(ax_pred, pred_field, depths, dmin, dmax, vmin, vmax,
                             cmap=cmap, label="Predicted", show_xlabel=False)
        last_im = im if im is not None else last_im
        im_e = plot_depth_map(ax_err, error_field, depths, dmin, dmax, -evlim, evlim,
                               cmap=error_cmap, label="Error", show_xlabel=True)
        last_err_im = im_e if im_e is not None else last_err_im

    if last_im is not None:
        fig.colorbar(last_im, ax=true_axes + pred_axes, label="Tracer",
                     fraction=0.02, pad=0.03, shrink=0.8)
    if last_err_im is not None:
        fig.colorbar(last_err_im, ax=err_axes, label="Error",
                     fraction=0.05, pad=0.03, shrink=0.8)

    fig.suptitle(title, fontsize=18, fontweight="bold")

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"🌊 Saved True-vs-Predicted water column comparison → {save_path}")
    return save_path
