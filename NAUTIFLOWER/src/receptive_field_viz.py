"""
receptive_field_viz.py

Schematic: for a handful of sampled pixels on the predicted tracer
field, show WHERE that pixel's value could have come from, and
visually connect it to the decoder.

Important, read before trusting the picture: `decoder_out` itself is a
1x1 conv — it does NOT spatially mix anything. Each output pixel only
depends on the SAME pixel location's decoder_conv1 features at that
step. All of the network's actual spatial mixing happens EARLIER, in
the 3x3 conv layers (conv1, conv2, decoder_conv1) and in the
ConvLSTM cell's 3x3 gate convolution, which gets applied once per
input timestep (recurrently) — so its contribution to the spatial
footprint scales with how many input timesteps the model was given,
not just "one layer's worth."

So this module draws two DIFFERENT things, layered together, and they
mean different things:
  - a translucent CIRCLE around each sampled pixel: the network's
    THEORETICAL receptive field, computed analytically from the
    kernel sizes and the number of input timesteps. This is an
    architecture-derived upper bound, not a measured/exact per-pixel
    attribution (that would need e.g. an input-gradient saliency map,
    which this is not).
  - a thin LINE from each sampled pixel to a "Decoder Out" box: purely
    SCHEMATIC, illustrating "this pixel's value passed through the
    decoder's final step" — not a claim that decoder_out reaches out
    spatially to gather information from elsewhere.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def compute_effective_receptive_field(kernel_size=3, input_steps=4, spatial_conv_layers=3):
    """
    Theoretical spatial receptive-field radius (in pixels) of one
    predicted tracer pixel, derived from the architecture:
      - conv1, conv2, decoder_conv1 (spatial_conv_layers=3 by default):
        each a kxk, padding=k//2 (shape-preserving) conv — each
        contributes (k//2) pixels of radius.
      - the ConvLSTM cell's kxk gate conv runs ONCE PER INPUT TIMESTEP
        (recurrently) — its total contribution is
        input_steps * (k//2), not just one layer's worth.
      - decoder_out (kernel_size=1) contributes 0 — it does not
        spatially mix at all.

    This is an architecture-derived estimate, not an empirically
    traced/measured receptive field.
    """
    half_k = kernel_size // 2
    return int(half_k * (spatial_conv_layers + input_steps))


def plot_receptive_field_diagram(
    tracer_field,
    n_pixels=10,
    kernel_size=3,
    input_steps=4,
    spatial_conv_layers=3,
    seed=0,
    cmap="Blues",
    title="Where does each predicted pixel's value come from?",
    save_path=None,
    dpi=150,
):
    """
    tracer_field: (H, W) 2D array — the predicted tracer at one
        depth/timestep (e.g. pred_field[z] from predict.py, or a
        depth-integrated map from water_column_viz.py).
    n_pixels: how many pixels to sample and annotate (10-12 is a good
        default — enough to see the pattern without cluttering the
        image).
    kernel_size, input_steps, spatial_conv_layers: passed straight to
        compute_effective_receptive_field — set input_steps to match
        your actual cfg.data.window.input_steps for an accurate circle.
    """
    rng = np.random.default_rng(seed)
    H, W = tracer_field.shape
    radius = compute_effective_receptive_field(kernel_size, input_steps, spatial_conv_layers)

    margin = max(1, min(radius + 1, min(H, W) // 2 - 1))
    if H - margin <= margin or W - margin <= margin:
        print(f"⚠ Field too small ({H}x{W}) for receptive-field radius {radius} — "
              f"circles will be clipped to the field edges.")
        margin = 0

    n_pixels = min(n_pixels, H * W)
    ys = rng.integers(margin, max(margin + 1, H - margin), size=n_pixels)
    xs = rng.integers(margin, max(margin + 1, W - margin), size=n_pixels)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(tracer_field, cmap=cmap, origin="upper")
    fig.colorbar(im, ax=ax, label="Predicted tracer", fraction=0.046, pad=0.04)

    decoder_x = W - 1 + max(3.0, W * 0.15)
    decoder_y = H / 2
    ax.set_xlim(-0.5, decoder_x + max(3.0, W * 0.12))

    for y, x in zip(ys, xs):
        ax.add_patch(plt.Circle((x, y), radius, color="orangered",
                                 fill=True, alpha=0.15, zorder=2, linewidth=0))
        ax.plot(x, y, "o", color="orangered", markersize=4, zorder=3)
        ax.plot([x, decoder_x], [y, decoder_y], color="orangered",
                linewidth=0.7, alpha=0.6, zorder=1)

    ax.plot(decoder_x, decoder_y, "s", color="black", markersize=22, zorder=4)
    ax.text(decoder_x, decoder_y, "Decoder\nOut", ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold", zorder=5)

    ax.set_title(
        f"{title}\n"
        f"Circle radius = {radius}px (theoretical), from {spatial_conv_layers} conv "
        f"layers + {input_steps} ConvLSTM steps, kernel={kernel_size}x{kernel_size}",
        fontsize=11,
    )
    ax.set_xlabel("X (grid pts)")
    ax.set_ylabel("Y (grid pts)")

    fig.text(
        0.5, -0.04,
        "Lines to \"Decoder Out\" are schematic — it's a 1x1 conv, it only touches the SAME pixel, no spatial "
        "mixing there.\nAll spatial spread (circles) comes from the upstream 3x3 layers and the ConvLSTM's "
        f"recurrence over {input_steps} input timesteps — this is a theoretical bound, not a measured attribution.",
        ha="center", fontsize=8, style="italic", color="dimgray",
    )

    plt.tight_layout()

    if save_path is not None:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"🔗 Saved receptive-field diagram → {save_path}")
        return save_path

    return fig
