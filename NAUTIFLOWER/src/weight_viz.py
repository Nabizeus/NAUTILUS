"""
weight_viz.py

Visualizes the model's LEARNED WEIGHTS — not activations. Weights are
the fixed parameters baked in during training: the same numbers no
matter what input you feed the model. (Activations, by contrast, are
the output signal for one specific input right now — see
src.activation_dashboard for those.)

Every conv layer's weight tensor has shape (out_channels, in_channels,
kH, kW). For a 1x1 conv (like input_reduce) this is already, in
effect, a genuine (out_channels, in_channels) MATRIX. For a kxk conv
(k>1), we summarize each (out_channel, in_channel) filter's overall
connection strength as its L2 norm over the kxk spatial extent —
this deliberately throws away the filter's spatial SHAPE (edge
detector vs. blob detector, etc.) in exchange for something directly
comparable, as one heatmap, across every layer in the network:
"how strongly is each input channel connected to each output channel."

For input_reduce specifically, the in_channels axis has KNOWN physical
meaning — 6 variables (Tracer, U, V, W, Dif, MLD) x a small window of
neighboring depths — so that one layer's matrix gets real axis labels
instead of anonymous indices. Every other layer's channels are learned
mixtures with no fixed identity, so they stay unlabeled.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.activation_maps import CHANNEL_NAMES


def _weight_to_matrix(weight):
    """
    weight: a tensor with a `.detach().cpu().numpy()`-style API,
        shape (out_channels, in_channels, kH, kW) or already 2D.
    Returns a (out_channels, in_channels) numpy array. For a kxk
    filter (k>1), each entry is the L2 norm over the kxk spatial
    extent — "how strong is this input->output connection overall",
    not the filter's spatial shape.
    """
    w = weight.detach().cpu().numpy()
    if w.ndim == 4:
        return np.sqrt(np.sum(w ** 2, axis=(2, 3)))
    if w.ndim == 2:
        return w
    raise ValueError(f"Unexpected weight shape {w.shape}; expected 2D or 4D.")


def _input_reduce_channel_labels(model, channel_names=CHANNEL_NAMES):
    """
    input_reduce's in_channels = len(channel_names) variables x a
    window of (max_below + max_above + 1) neighboring depths, ordered
    [var0 @ offset -max_below..+max_above, var1 @ same offsets, ...] —
    verified against OceanConv2DAsymXPU.forward's reshape order
    (src/model_conv2d_asym_hybrid.py): the window axis is the
    fastest-varying one when (C, window) get flattened together.

    Returns None if the model doesn't expose max_below/max_above
    (labels are skipped rather than guessed).
    """
    max_below = getattr(model, "max_below", None)
    max_above = getattr(model, "max_above", None)
    if max_below is None or max_above is None:
        return None

    offsets = list(range(-max_below, max_above + 1))
    return [f"{name}(z{off:+d})" for name in channel_names for off in offsets]


def plot_weight_matrix(ax, weight, title="", xlabel="Input channel", ylabel="Output channel",
                        xticklabels=None, cmap="coolwarm"):
    """
    Draw one layer's weight matrix on the given Axes. Returns the
    plotted handle (imshow AxesImage, or a BarContainer for the
    single-output-channel case).

    When out_channels == 1 (e.g. decoder_out, which maps down to the
    single final tracer value), a 2D heatmap is just one row stretched
    to fill the panel — a bar chart shows the exact same numbers more
    directly, so that case is handled separately below.
    """
    matrix = _weight_to_matrix(weight)

    if matrix.shape[0] == 1:
        values = matrix[0]
        vmax = float(np.max(np.abs(values))) or 1.0
        colors = plt.get_cmap(cmap)((values / vmax) * 0.5 + 0.5)  # map [-vmax,vmax] -> [0,1]
        bars = ax.bar(np.arange(len(values)), values, color=colors, edgecolor="none")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"{title}\n(1x{matrix.shape[1]})", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Weight value", fontsize=8)
        if xticklabels is not None and len(xticklabels) <= 40:
            ax.set_xticks(np.arange(len(xticklabels)))
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=5)
        else:
            ax.set_xticks([])
        return bars

    vmax = float(np.max(np.abs(matrix))) or 1.0
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_title(f"{title}\n({matrix.shape[0]}x{matrix.shape[1]})", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

    if xticklabels is not None and len(xticklabels) <= 40:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=5)
    else:
        ax.set_xticks([])
    ax.set_yticks([])

    return im


def plot_network_weights_dashboard(model, out_dir="weights_dashboard", region_name="unknown",
                                    dpi=150):
    """
    One figure: the weight matrix of every named conv layer in the
    network, in the ACTUAL forward-pass order (encoder -> ConvLSTM
    cell -> decoder, ending at decoder_out, the true pipeline
    endpoint) — the WEIGHTS analog of
    activation_dashboard.plot_activation_dashboard.

    `model` must be the full OceanConv2DAsymXPU instance (same object
    passed to get_activation_maps elsewhere in this pipeline).
    """
    os.makedirs(out_dir, exist_ok=True)

    layers = [
        ("input_reduce", model.input_reduce.weight, _input_reduce_channel_labels(model)),
        ("conv1", model.model2d.encoder[0].weight, None),
        ("conv2", model.model2d.encoder[4].weight, None),
        # The ConvLSTM cell's 4 gates (input/forget/output/candidate) share
        # ONE Conv2d over the concatenated [input_features, hidden_state] —
        # shown as one combined matrix rather than split into 4, since
        # splitting requires slicing the output channels into 4 equal
        # blocks in the exact order ConvLSTMCell.forward expects, which is
        # easy to get subtly wrong without re-deriving it from that code.
        # Placed here (not at the end) because the cell runs BETWEEN the
        # encoder and the decoder in the real forward pass, not after
        # everything else.
        ("convlstm_cell_gates", model.model2d.cell.conv.weight, None),
        ("decoder_conv1", model.model2d.decoder[0].weight, None),
        ("decoder_out", model.model2d.decoder[4].weight, None),
    ]

    fig, axes = plt.subplots(1, len(layers), figsize=(4.5 * len(layers), 5.5))
    if len(layers) == 1:
        axes = [axes]

    for ax, (name, weight, labels) in zip(axes, layers):
        handle = plot_weight_matrix(ax, weight, title=name, xticklabels=labels)
        if hasattr(handle, "get_array"):  # imshow case — real colorbar
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
        else:  # bar-chart case (single output channel) — build one manually
            matrix = _weight_to_matrix(weight)
            vmax = float(np.max(np.abs(matrix))) or 1.0
            sm = plt.cm.ScalarMappable(
                norm=plt.Normalize(vmin=-vmax, vmax=vmax), cmap="coolwarm"
            )
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Learned Weight Matrices — {region_name}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(out_dir, f"{region_name}_weights_dashboard.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"⚖️  Saved weight matrix dashboard → {out_path}")
    return out_path
