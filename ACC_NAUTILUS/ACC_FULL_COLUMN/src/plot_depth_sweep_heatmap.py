"""
plot_depth_sweep_heatmap.py

Turns the depth-skill sweep CSV (plots/heatmaps/depth_skill_binned_all_runs.csv,
produced across every predict.py run in a depth-budget/banded-window
sweep by src/depth_skill_profile.py) into a heatmap: rows = training
config (tagged by its trained depth interval), columns = test depth
bin, cell = a chosen skill metric — the depth-axis analog of
src/plot_cross_region_heatmap.py.

One structural difference from the region heatmap this is modeled on:
a training config's "home" isn't a single (i,i) cell. Depth-training
ranges are CONTIGUOUS INTERVALS that can span several test bins (a
cumulative budget covers many) or exactly one (a banded window, if its
width matches the evaluation bin width — see
generate_depth_sweep_configs.py's `eval_bin_width` handling). So
instead of one diagonal box, every bin whose range overlaps
[train_depth_min_m, train_depth_max_m] for that row gets outlined, and
the generalization score becomes "mean skill on bins inside the
trained interval" ("Own") vs. "mean skill on bins outside it"
("Cross" / extrapolated) — the depth-axis analog of the region
heatmap's Own/Cross columns.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

# Metrics where higher = better vs. lower = better — same convention
# as plot_cross_region_heatmap.py
HIGHER_IS_BETTER = {"ssim", "r2"}


def _pretty_metric_label(metric):
    """Display label for plot titles/colorbars — R2 renders as proper R², everything else is just upper-cased."""
    if metric.lower() == "r2":
        return r"$R^2$"
    return metric.upper()


def _bin_order(df):
    """Order depth bins by their numeric lower edge, not alphabetically."""
    return (
        df[["depth_bin", "depth_bin_min_m"]]
        .drop_duplicates()
        .sort_values("depth_bin_min_m")["depth_bin"]
        .tolist()
    )


def _row_order(df):
    """Order training configs by their trained depth range (min, then max)."""
    return (
        df[["run_name", "train_depth_min_m", "train_depth_max_m"]]
        .drop_duplicates()
        .sort_values(["train_depth_min_m", "train_depth_max_m"])["run_name"]
        .tolist()
    )


def _region_tag(region=None, checkpoint_region=None):
    """
    Filename suffix identifying the region filter, following the
    <checkpoint>_to_<test> convention of the run names, e.g.
    '_North_Atlantic_to_North_Pacific'. A side that wasn't filtered is
    written as 'all'; with no filter at all the suffix is empty, so the
    unfiltered report keeps its old generic filenames.
    """
    if region is None and checkpoint_region is None:
        return ""
    return f"_{checkpoint_region or 'all'}_to_{region or 'all'}"


def load_depth_sweep_csv(csv_path, metric, region=None, checkpoint_region=None):
    """
    Load the binned depth-skill CSV and filter to one metric (metric
    names in the CSV are lowercase: 'rmse', 'mae', 'r2', 'ssim',
    'rmse_pct', 'mae_pct'). Optionally filter to one test region and/or
    one checkpoint (training) region — recommended whenever the sweep
    covers more than one, since averaging different regions' skill
    together in one row would mix unrelated physical regimes.
    """
    df = pd.read_csv(csv_path)
    df = df[df["metric"] == metric.lower()].copy()

    if region is not None:
        df = df[df["region"] == region]
    if checkpoint_region is not None:
        df = df[df["checkpoint_region"] == checkpoint_region]

    if df.empty:
        raise ValueError(
            f"No rows found for metric='{metric}'"
            f"{f', region={region}' if region else ''}"
            f"{f', checkpoint_region={checkpoint_region}' if checkpoint_region else ''}"
            f" in {csv_path}"
        )

    n_regions = df[["region", "checkpoint_region"]].drop_duplicates().shape[0]
    if n_regions > 1 and (region is None or checkpoint_region is None):
        print(
            f"⚠ {csv_path} has {n_regions} distinct (region, checkpoint_region) "
            f"combinations for metric={metric}, but no filter was given — rows "
            f"for different regions will be averaged together in the same cell. "
            f"Pass region=/checkpoint_region= to plot one combination at a time."
        )

    return df


def plot_depth_sweep_heatmap(df, metric, outdir="plots/heatmaps",
                              higher_is_better=None, label_prefix="",
                              vmin=None, vmax=None, cmap=None, file_tag=""):
    """
    df: filtered long-format DataFrame from load_depth_sweep_csv (a
        single metric, ideally a single region/checkpoint_region pair).

    file_tag: suffix appended to the output filename (see _region_tag)
        so heatmaps for different regions don't overwrite each other.

    vmin/vmax: explicit color-scale bounds. If omitted, auto-scaled to
        the data's min/max — which for a metric like R2 (unbounded
        below) means a handful of catastrophic extrapolation cells
        (e.g. -148) stretch the whole scale and wash out the
        difference between "good" (0.9) and "mediocre" (0.5) cells.
        Pass e.g. vmin=0, vmax=1 to fix that; values outside the range
        are simply clipped to the nearest end color (their raw value
        is still shown as text).
    cmap: any matplotlib colormap name. Defaults to "copper" (higher
        is better) / "RdYlGn_r" (lower is better) if not given.
    """
    os.makedirs(outdir, exist_ok=True)

    if higher_is_better is None:
        higher_is_better = metric.lower() in HIGHER_IS_BETTER

    bins = _bin_order(df)
    rows = _row_order(df)

    matrix = (
        df.pivot_table(index="run_name", columns="depth_bin", values="value", aggfunc="mean")
        .reindex(index=rows, columns=bins)
    )

    bounds = (
        df[["run_name", "train_depth_min_m", "train_depth_max_m"]]
        .drop_duplicates()
        .set_index("run_name")
        .loc[rows]
    )
    bin_bounds = (
        df[["depth_bin", "depth_bin_min_m", "depth_bin_max_m"]]
        .drop_duplicates()
        .set_index("depth_bin")
        .loc[bins]
    )

    finite_vals = matrix.values[np.isfinite(matrix.values)]
    if finite_vals.size == 0:
        raise ValueError(f"No finite values to plot for metric={metric}")
    if vmin is None:
        vmin = float(np.min(finite_vals))
    if vmax is None:
        vmax = float(np.max(finite_vals))
    if cmap is None:
        cmap = "copper" if higher_is_better else "RdYlGn_r"

    fig, ax = plt.subplots(figsize=(1.1 * len(bins) + 3, 0.6 * len(rows) + 3))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, extend="both" if (
        np.nanmin(matrix.values) < vmin or np.nanmax(matrix.values) > vmax
    ) else "neither")
    cbar.set_label(_pretty_metric_label(metric))

    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)

    ax.set_xlabel("Test depth bin")
    ax.set_ylabel("Training config (trained depth range)")
    ax.set_title(f"{label_prefix}Depth-Sweep Generalization — {_pretty_metric_label(metric)}",
                 fontsize=14, fontweight="bold")

    # ------------------------------------------------------------
    # Outline every cell INSIDE this row's trained interval — the
    # depth-sweep analog of the single diagonal box in the region
    # heatmap, since here "home" can span several columns.
    # ------------------------------------------------------------
    for i, run_name in enumerate(rows):
        tmin = bounds.loc[run_name, "train_depth_min_m"]
        tmax = bounds.loc[run_name, "train_depth_max_m"]
        for j, b in enumerate(bins):
            bmin = bin_bounds.loc[b, "depth_bin_min_m"]
            bmax = bin_bounds.loc[b, "depth_bin_max_m"]
            if bmax > tmin and bmin < tmax:  # bin overlaps the trained interval
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="lime", linewidth=2.5,
                ))

    # ------------------------------------------------------------
    # Hatch cells where the true field was essentially constant at that
    # depth (mean_true_variance ~ 0, e.g. a tracer that never reached
    # that depth) — R2 is undefined there by construction, and
    # RMSE/MAE/SSIM look trivially "perfect" without the model having
    # learned anything. See src.depth_skill_profile.summarize_depth_skill_per_z
    # for where this flag comes from.
    # ------------------------------------------------------------
    has_low_variance_col = "low_variance" in df.columns
    n_hatched = 0
    if has_low_variance_col:
        low_var_matrix = (
            df.pivot_table(index="run_name", columns="depth_bin",
                            values="low_variance", aggfunc="max")
            .reindex(index=rows, columns=bins)
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                cell_val = low_var_matrix.values[i, j]
                if pd.notna(cell_val) and bool(cell_val):
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        facecolor="none", edgecolor="dimgray",
                        hatch="////", linewidth=0, alpha=0.6, zorder=3,
                    ))
                    n_hatched += 1

    # ------------------------------------------------------------
    # Cell text — raw value always shown. Text color is picked from
    # the ACTUAL rendered cell color's luminance (not just the
    # normalized value), since brightness direction varies by
    # colormap — e.g. viridis is dark at low values and bright at
    # high values, the opposite of copper. Computing luminance
    # directly makes this correct for any colormap automatically.
    # ------------------------------------------------------------
    cmap_obj = cm.get_cmap(cmap)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if not np.isfinite(val):
                continue
            norm = (val - vmin) / (vmax - vmin + 1e-12)
            norm_clamped = min(max(norm, 0.0), 1.0)
            r, g, b, _ = cmap_obj(norm_clamped)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "black" if luminance > 0.5 else "white"
            ax.text(j, i, f"{val:.3g}", ha="center", va="center",
                    fontsize=8, color=color)

    if n_hatched:
        fig.text(
            0.5, -0.02,
            f"Hatched cells ({n_hatched}): true value ≈ constant at that depth "
            f"(R² undefined, RMSE/MAE/SSIM trivially \"perfect\") — excluded from "
            f"the generalization score.",
            ha="center", fontsize=8, style="italic", color="dimgray",
        )

    plt.tight_layout()
    out_path = os.path.join(outdir, f"depth_sweep_heatmap_{metric.lower()}{file_tag}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"🌡️ Saved depth-sweep heatmap → {out_path}")
    return out_path


def depth_generalization_score(df, metric, higher_is_better=None, outdir="plots/heatmaps",
                                exclude_low_variance=True, file_tag=""):
    """
    For each training config: mean skill on bins INSIDE its trained
    interval ("Own") vs. mean skill on bins OUTSIDE it ("Cross" /
    extrapolated), plus a "Generalization Gap" you can sort by —
    smaller is better (own performance carries over to unseen depths).
    Mirrors generalization_score() in plot_cross_region_heatmap.py,
    adapted for contiguous depth ranges instead of discrete regions.

    exclude_low_variance: drop bins flagged `low_variance` (true value
        essentially constant at that depth — R2 undefined,
        RMSE/MAE/SSIM trivially "perfect") before computing Own/Cross.
        Otherwise a training config that happens to cover a lot of
        trivial constant-zero depth would look like it generalizes
        well without actually having learned anything there. Set
        False to reproduce the old (unfiltered) behavior.
    """
    if higher_is_better is None:
        higher_is_better = metric.lower() in HIGHER_IS_BETTER

    own_col = f"Own {metric.upper()}"
    cross_col = f"Cross {metric.upper()}"

    n_excluded_total = 0
    if exclude_low_variance and "low_variance" in df.columns:
        low_var_mask = df["low_variance"].fillna(False).astype(bool)
        n_excluded_total = int(low_var_mask.sum())
        df = df[~low_var_mask]

    rows = []
    for run_name, g in df.groupby("run_name"):
        tmin = g["train_depth_min_m"].iloc[0]
        tmax = g["train_depth_max_m"].iloc[0]

        inside_mask = (g["depth_bin_max_m"] > tmin) & (g["depth_bin_min_m"] < tmax)
        inside = g[inside_mask]
        outside = g[~inside_mask]

        rows.append({
            "run_name": run_name,
            "train_depth_min_m": tmin,
            "train_depth_max_m": tmax,
            own_col: inside["value"].mean() if not inside.empty else np.nan,
            cross_col: outside["value"].mean() if not outside.empty else np.nan,
        })

    scores = pd.DataFrame(rows)

    if higher_is_better:
        scores["Generalization Gap"] = scores[own_col] - scores[cross_col]
    else:
        scores["Generalization Gap"] = scores[cross_col] - scores[own_col]

    scores = scores.sort_values("Generalization Gap", ascending=True)

    if n_excluded_total:
        print(
            f"ℹ Excluded {n_excluded_total} low-variance depth-bin row(s) "
            f"(metric={metric}) from the generalization score — see "
            f"'low_variance' in the depth-skill CSV for which ones."
        )

    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"depth_sweep_generalization_{metric.lower()}{file_tag}.csv")
    scores.to_csv(out_csv, index=False)
    print(f"📋 Depth-sweep generalization scores → {out_csv}")

    return scores


# Sensible fixed color scales for bounded metrics — R2 in particular is
# unbounded BELOW (a bad model can score arbitrarily negative), so
# auto-scaling stretches the whole colorbar to fit one catastrophic
# extrapolation cell and washes out the difference between "good" and
# "mediocre". SSIM is naturally bounded to [0,1] anyway. RMSE/MAE are
# left auto-scaled since their sensible range depends on your tracer's
# units/magnitude.
DEFAULT_METRIC_STYLE = {
    "r2": dict(vmin=0.0, vmax=1.0, cmap="viridis"),
    "ssim": dict(vmin=0.0, vmax=1.0, cmap="viridis"),
}


def build_depth_sweep_report(csv_path="plots/heatmaps/depth_skill_binned_all_runs.csv",
                              metrics=("rmse", "mae", "ssim", "r2"),
                              region=None, checkpoint_region=None,
                              outdir="plots/heatmaps",
                              metric_style=None):
    """
    Convenience wrapper: one heatmap + one generalization CSV per metric.

    metric_style: optional {metric_name: {"vmin":..., "vmax":..., "cmap":...}}
        to override DEFAULT_METRIC_STYLE, e.g.
        metric_style={"r2": {"cmap": "GnBu"}} to keep the 0-1 clipping
        but swap to a blue "ocean" palette instead of viridis.
    """
    style = {**DEFAULT_METRIC_STYLE, **(metric_style or {})}

    file_tag = _region_tag(region, checkpoint_region)

    # One folder per <checkpoint>_to_<region> pair under depth_sweep/, so
    # the 16 pairs x 4 metrics don't pile up in a single directory. The
    # unfiltered report goes to depth_sweep/ itself.
    outdir = os.path.join(outdir, "depth_sweep", file_tag.lstrip("_"))
    label_prefix = (
        f"{checkpoint_region or 'all'} → {region or 'all'}\n" if file_tag else ""
    )

    for metric in metrics:
        try:
            df = load_depth_sweep_csv(csv_path, metric, region=region,
                                       checkpoint_region=checkpoint_region)
        except ValueError as e:
            print(f"⚠ Skipping {metric}: {e}")
            continue

        style_kwargs = style.get(metric.lower(), {})
        plot_depth_sweep_heatmap(df, metric, outdir=outdir, label_prefix=label_prefix,
                                 file_tag=file_tag, **style_kwargs)
        depth_generalization_score(df, metric, outdir=outdir, file_tag=file_tag)

    print("\n✓ Depth-sweep report finished.")


def build_all_pairs_reports(csv_path="plots/heatmaps/depth_skill_binned_all_runs.csv",
                             outdir="plots/heatmaps", cross_only=False, **kwargs):
    """
    Build one depth-sweep report per (checkpoint_region, region) pair
    found in the CSV — own-region pairs and cross-region transfers alike.
    cross_only=True skips the pairs where the model is tested on its own
    training region.
    """
    pairs = (
        pd.read_csv(csv_path, usecols=["checkpoint_region", "region"])
        [["checkpoint_region", "region"]]
        .drop_duplicates()
        .sort_values(["checkpoint_region", "region"])
        .itertuples(index=False, name=None)
    )
    pairs = [(c, r) for c, r in pairs if not (cross_only and c == r)]

    print(f"🗺️ Building depth-sweep reports for {len(pairs)} region pair(s)")
    for checkpoint_region, region in pairs:
        print(f"\n── {checkpoint_region} → {region}")
        build_depth_sweep_report(csv_path, region=region,
                                 checkpoint_region=checkpoint_region,
                                 outdir=outdir, **kwargs)

    print(f"\n✓ All {len(pairs)} pair reports finished → {os.path.join(outdir, 'depth_sweep')}")


if __name__ == "__main__":
    build_depth_sweep_report()
