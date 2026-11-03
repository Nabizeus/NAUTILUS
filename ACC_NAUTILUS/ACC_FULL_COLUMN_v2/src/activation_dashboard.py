"""
activation_dashboard.py

Consolidates the per-layer activation maps produced by
`src.activation_maps.get_activation_maps` into ONE summary figure
(rows = layers, columns = channels), instead of the several separate
PNGs that `visualize_activation_maps` writes per sample. Useful as a
quick "is the network doing something sane" glance during prediction,
while `visualize_activation_maps`'s per-layer PNGs remain the ones
consumed by `make_activation_gifs.py` for animating a layer across
samples.

Rows, top to bottom:
  - Input variables: the 6 raw physical channels (Tracer, U, V, W,
    Dif, MLD) taken directly from the model's input tensor — no hook
    needed, these are the actual values the network receives, each
    normalized independently (their physical scales differ wildly, so
    a shared min/max would make most of them look flat).
  - input_reduce, conv1, conv2, convlstm_hidden, decoder_conv1,
    decoder_out: the FULL network, input to output (see
    src.activation_maps.register_hooks for exactly which submodule
    each name maps to).
  - Depth-activation profile (optional, last row): how strongly each
    layer fires, per depth level. This is possible in one forward pass
    because OceanConv2DAsymXPU folds the Z (depth) axis into the batch
    dimension before running its Conv2LSTM (`x.reshape(B*Z, T, C, H,
    W)`) — so a hooked activation's leading dimension is (B*Z), and
    simply reshaping it back to (B, Z, ...) recovers every depth
    level's activations from the SAME forward pass, no extra compute
    needed. Reducing over channels/space per depth gives a "how strong
    is this layer's response at each depth" curve — a rough proxy for
    whether whatever the network learned at one depth is actually
    firing similarly at other depths ("transferring"), vs. being
    depth-specific.

Caveat: for the per-channel image grid (rows x columns), we still only
show index 0 of the (B*Z) dimension, i.e. depth level 0 of the first
batch element, since showing every depth as its own image grid would
be enormous. The depth-profile panel is what actually uses all Z
depths; the image grid is just a spatial sanity-check at one depth.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.activation_maps import get_activation_maps, get_raw_input_channels, CHANNEL_NAMES

LAYER_ORDER = [
    "input_reduce",
    "conv1",
    "conv2",
    "convlstm_hidden",
    "decoder_conv1",
    "decoder_out",
]


def _prep_fmap(fmap, n_channels=4):
    """
    fmap: raw hooked tensor, batch-first.
    Returns (normalized (n_channels, H, W) numpy array ready to imshow,
    total channel count BEFORE truncation) — the total is what lets the
    dashboard label previews as "ch0/64" rather than a bare "ch0" that
    would wrongly imply there are only as many channels as shown.
    """
    fmap = fmap[0].detach().cpu()  # first (batch*Z) sample

    if fmap.dim() == 4:  # (C, Z, H, W) — not expected for this model, but handled
        z_mid = fmap.shape[1] // 2
        fmap = fmap[:, z_mid]
    elif fmap.dim() != 3:  # expected (C, H, W)
        raise ValueError(f"Unexpected activation shape: {fmap.shape}")

    fmap = fmap.numpy()
    fmin, fmax = np.nanmin(fmap), np.nanmax(fmap)
    fmap = (fmap - fmin) / (fmax - fmin + 1e-8)

    total_channels = fmap.shape[0]
    n_ch = min(n_channels, total_channels)
    return fmap[:n_ch], total_channels


def _prep_input_channels(x, timestep=-1, z_index=0):
    """
    Extract the 6 raw physical-variable channels directly from the
    model's input `x` (native 6D (B,T,C,Z,H,W), first batch element),
    normalized INDEPENDENTLY per channel — Tracer/U/V/W/Dif/MLD have
    very different physical scales, so a shared min/max would make
    most of them look uniformly flat.

    Returns a (6, H, W) numpy array, in CHANNEL_NAMES order.
    """
    raw = get_raw_input_channels(x, timestep=timestep, z_index=z_index)  # (C,H,W) tensor
    raw = raw.cpu().numpy()

    out = np.empty_like(raw)
    for c in range(raw.shape[0]):
        cmin, cmax = np.nanmin(raw[c]), np.nanmax(raw[c])
        out[c] = (raw[c] - cmin) / (cmax - cmin + 1e-8)
    return out


def _depth_profile_from_acts(acts, B, Z, layer_order=LAYER_ORDER, metric="mean_abs"):
    """
    Reduce each layer's (B*Z, C, H, W) hooked activation into a (Z,)
    "activation strength per depth" curve, without re-running the model.
    """
    profiles = {}
    for lname in layer_order:
        if lname not in acts:
            continue

        arr = acts[lname]
        arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)

        if arr.shape[0] != B * Z:
            raise ValueError(
                f"Activation '{lname}' has leading dim {arr.shape[0]}, expected "
                f"B*Z={B*Z}. The depth-folding assumption (OceanConv2DAsymXPU "
                "reshapes to (B*Z, ...) before its Conv2LSTM) may not hold for "
                "this model/layer — check register_hooks() in activation_maps.py."
            )

        arr = arr.reshape(B, Z, *arr.shape[1:])  # (B, Z, C, H, W)
        reduce_axes = tuple(a for a in range(arr.ndim) if a != 1)  # keep depth axis

        if metric == "mean_abs":
            per_depth = np.mean(np.abs(arr), axis=reduce_axes)
        elif metric == "std":
            per_depth = np.std(arr, axis=reduce_axes)
        elif metric == "rms":
            per_depth = np.sqrt(np.mean(arr ** 2, axis=reduce_axes))
        else:
            raise ValueError(f"Unknown metric '{metric}' (use 'mean_abs', 'std', or 'rms')")

        profiles[lname] = per_depth  # (Z,)

    return profiles


def compute_depth_activation_profile(model, x, layer_order=LAYER_ORDER, metric="mean_abs"):
    """
    Standalone entry point: run `model` on the native 6D (B,T,C,Z,H,W)
    input `x` and return {layer_name: (Z,) activation-strength array}.
    """
    if x.dim() != 6:
        raise ValueError(
            "compute_depth_activation_profile needs the native 6D (B,T,C,Z,H,W) "
            f"input so per-depth activations can be recovered from the batch-folded "
            f"activations; got {x.dim()}D."
        )
    B, T, C, Z, H, W = x.shape
    acts = get_activation_maps(model, x)
    return _depth_profile_from_acts(acts, B, Z, layer_order=layer_order, metric=metric)


def plot_depth_activation_profile(ax, profiles, depths=None, layer_order=LAYER_ORDER,
                                   metric_label="Mean |activation|"):
    """Overlay one activation-strength-vs-depth line per layer."""
    for lname in layer_order:
        if lname not in profiles:
            continue
        y = depths if depths is not None else np.arange(len(profiles[lname]))
        ax.plot(profiles[lname], y, "-o", markersize=3, label=lname)

    ax.invert_yaxis()
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Depth (m)" if depths is not None else "Depth index")
    ax.set_title(
        "Activation Strength vs. Depth\n"
        "(how strongly each layer fires at each depth level)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


def plot_activation_and_skill_profile(
    activation_profiles,
    depths,
    skill_df_per_z=None,
    skill_metrics=("ssim", "r2"),
    mld_depth_m=None,
    layer_order=LAYER_ORDER,
    activation_metric_label="Mean |activation|",
    save_path=None,
    dpi=150,
):
    """
    Side-by-side depth profiles, sharing the depth (y) axis, so you can
    visually line up "where does the network fire strangely" against
    "where does its actual skill fall apart":

      panel 1: activation strength vs. depth (from activation_profiles,
               e.g. the dict returned by
               plot_activation_dashboard(..., return_profiles=True) or
               compute_depth_activation_profile(...))
      panel 2..N: one panel per metric in `skill_metrics`, plotting that
               metric's value vs. depth from `skill_df_per_z` — a
               per-z long-format DataFrame for ONE run/region, i.e.
               `depth_skill_per_z_all_runs.csv` already filtered down
               (see src.depth_skill_profile.summarize_depth_skill_per_z)

    `mld_depth_m`: optional single number (meters) — draws a horizontal
    dashed reference line on every panel marking the (regionally
    averaged) mixed-layer depth, so you can see whether a firing
    change or a skill drop lines up with the mixed layer boundary
    rather than some arbitrary depth. See
    src.depth_skill_profile.estimate_mean_mld_m for how to compute
    this from the dataset.

    Note: this deliberately does NOT re-run the model — the activation
    profile has to come from a forward pass you already did (e.g. at
    sample i==0), and the skill profile has to come from the full test
    set (only available after predict.py's test loop finishes), so
    combining them here happens after both already exist.
    """
    n_skill_panels = len(skill_metrics) if skill_df_per_z is not None else 0
    n_panels = 1 + n_skill_panels

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    plot_depth_activation_profile(
        axes[0], activation_profiles, depths=depths, layer_order=layer_order,
        metric_label=activation_metric_label,
    )

    if skill_df_per_z is not None:
        for k, metric in enumerate(skill_metrics):
            ax = axes[k + 1]
            sub = skill_df_per_z[skill_df_per_z["metric"] == metric.lower()].sort_values("depth_m")
            if sub.empty:
                ax.text(0.5, 0.5, f"no '{metric}' rows found", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                ax.plot(sub["value"], sub["depth_m"], "-o", markersize=3, color="tab:red")
            ax.set_xlabel(metric.upper())
            ax.set_title(f"{metric.upper()} vs. Depth", fontsize=11)
            ax.grid(alpha=0.3)

    if mld_depth_m is not None:
        for i, ax in enumerate(axes):
            ax.axhline(
                mld_depth_m, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Avg. MLD ({mld_depth_m:.0f} m)" if i == 0 else None,
            )
        axes[0].legend(fontsize=8, loc="lower right")

    fig.suptitle("Activation Strength & Skill vs. Depth", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"📉 Saved activation+skill depth profile → {save_path}")
        return save_path

    return fig


def plot_all_channels_grid(
    model,
    x,
    layer_name,
    sample=0,
    region_name="unknown",
    out_dir="activations_dashboard",
    cmap="copper",
    dpi=150,
):
    """
    Show EVERY channel of ONE chosen layer (e.g. all 64 of conv1), laid
    out in a roughly square grid — the full version of what the main
    dashboard's 4-channel preview (ch0/64 ... ch3/64) is only a sample
    of. Useful as a one-off "let me actually see the whole mixed
    feature space" figure rather than something you'd generate every
    predict.py run (64 small panels is a lot of image for a routine
    diagnostic — the 4-channel preview in plot_activation_dashboard is
    the one meant for that).

    layer_name: one of the names in LAYER_ORDER (e.g. "conv1",
        "conv2", "convlstm_hidden", "decoder_conv1"). "decoder_out"
        works too but only has 1 channel by construction (see
        src.weight_viz's docstring on why) — the grid degenerates to a
        single panel, which is correct, not a bug.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    acts = get_activation_maps(model, x)
    if layer_name not in acts:
        raise KeyError(f"'{layer_name}' not found. Available: {list(acts.keys())}")

    fmap, total_channels = _prep_fmap(acts[layer_name], n_channels=10_000)  # no truncation

    n_cols = int(np.ceil(np.sqrt(total_channels)))
    n_rows = int(np.ceil(total_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows), squeeze=False)

    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        if i < total_channels:
            ax.imshow(fmap[i], cmap=cmap)
            ax.set_title(f"ch{i}", fontsize=7)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{layer_name} — all {total_channels} channels — {region_name} | sample {sample}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"{region_name}_{layer_name}_all{total_channels}ch_s{sample:05d}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"🔬 Saved full {total_channels}-channel grid for '{layer_name}' → {out_path}")
    return out_path


def plot_activation_dashboard(
    model,
    x,
    sample=0,
    region_name="unknown",
    out_dir="activations_dashboard",
    layer_order=LAYER_ORDER,
    channel_names=CHANNEL_NAMES,
    n_channels=4,
    cmap="copper",
    dpi=150,
    depths=None,
    include_depth_profile=True,
    depth_profile_metric="mean_abs",
    input_timestep=-1,
    input_z_index=0,
    return_profiles=False,
):
    """
    Run `model` on `x` (its NATIVE input shape — see
    src.activation_maps.get_activation_maps docstring, do not pre-slice
    T/Z) and save one combined dashboard figure:
      - row 0: the 6 raw input variables (Tracer, U, V, W, Dif, MLD),
        each independently normalized, at `input_timestep`/`input_z_index`
      - rows 1..N: the full network's layers, columns = the first
        `n_channels` feature-map channels of that layer (spatial
        sanity-check at depth 0) — these are LEARNED FEATURES, not
        physical variables, so they're still labeled ch0/ch1/... rather
        than by variable name (a feature-map channel is some mixture of
        all 6 inputs, not any single one of them)
      - an extra bottom row: activation strength vs. depth, one line
        per layer, using every depth level from the same forward pass

    `depths`: optional (Z,) array in meters for labeling the y-axis of
    the depth-profile row (e.g. dataset.depths[dataset.valid_z]). If
    omitted, depth index is used instead.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    acts = get_activation_maps(model, x)

    missing = [l for l in layer_order if l not in acts]
    if missing:
        raise KeyError(
            f"Missing expected activation layer(s) {missing}. "
            f"Hooks captured: {list(acts.keys())}. Check register_hooks() "
            f"in src/activation_maps.py still targets the right submodules."
        )

    n_layers = len(layer_order)
    show_profile = include_depth_profile and x.dim() == 6
    if include_depth_profile and not show_profile:
        print(
            "⚠ Skipping depth-activation profile: input is not the native 6D "
            "(B,T,C,Z,H,W) tensor, so per-depth activations can't be recovered."
        )

    # The input row needs all 6 named variables even if n_channels (the
    # feature-map channel count) is set smaller for brevity.
    n_cols = max(n_channels, len(channel_names))

    n_rows = 1 + n_layers + (1 if show_profile else 0)  # +1 for the input row
    fig = plt.figure(
        figsize=(2.6 * n_cols, 2.6 * (n_layers + 1) + (3.4 if show_profile else 0))
    )
    height_ratios = [1] * (n_layers + 1) + ([1.4] if show_profile else [])
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios)

    # ------------------------------------------------------------
    # Row 0: raw input variables, labeled by name
    # ------------------------------------------------------------
    input_map = _prep_input_channels(x, timestep=input_timestep, z_index=input_z_index)
    for col in range(n_cols):
        ax = fig.add_subplot(gs[0, col])
        if col < input_map.shape[0]:
            ax.imshow(input_map[col], cmap=cmap)
            ax.set_title(channel_names[col], fontsize=9)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("input", fontsize=11, fontweight="bold")

    # ------------------------------------------------------------
    # Rows 1..n_layers: full network, learned feature-map channels
    # ------------------------------------------------------------
    for row_offset, lname in enumerate(layer_order):
        row = row_offset + 1
        fmap, total_channels = _prep_fmap(acts[lname], n_channels=n_channels)
        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            if col < fmap.shape[0]:
                ax.imshow(fmap[col], cmap=cmap)
                ax.set_title(f"ch{col}/{total_channels}", fontsize=9)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{lname}\n({total_channels} ch total)", fontsize=10, fontweight="bold")

    profiles = None
    if show_profile:
        B, T, C, Z, H, W = x.shape
        try:
            profiles = _depth_profile_from_acts(
                acts, B, Z, layer_order=layer_order, metric=depth_profile_metric
            )
            ax_profile = fig.add_subplot(gs[n_layers + 1, :])
            plot_depth_activation_profile(ax_profile, profiles, depths=depths,
                                           layer_order=layer_order)
        except ValueError as e:
            print(f"⚠ Skipping depth-activation profile: {e}")

    fig.suptitle(
        f"Activation Map Dashboard — {region_name} | sample {sample}",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"{region_name}_activation_dashboard_s{sample:05d}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"🧠 Saved activation dashboard → {out_path}")

    if return_profiles:
        return out_path, profiles
    return out_path
