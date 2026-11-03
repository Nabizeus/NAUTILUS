import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Physical variable order, matching the exact stacking order in
# WindowedOceanDataset.__getitem__ (src/data.py):
#   np.stack([x_tracer, x_u, x_v, x_w, x_dif, x_mld_distance], axis=2)
# Channel 0=Tracer, 1=U, 2=V, 3=W, 4=Dif, 5=MLD — do not reorder this
# without checking data.py again, mislabeling here silently mislabels
# every input-channel plot.
CHANNEL_NAMES = ["Tracer", "U", "V", "W", "Dif", "MLD"]

activations = {}

def hook(name):
    def fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]  # ConvLSTM returns (h,c)
        activations[name] = out.detach()
    return fn


def register_hooks(model):
    """
    Hooks the FULL network, input to output:
      input_reduce  -> depth-window channel reduction (raw stacked
                        variables*window -> hidden_dim), the very
                        first learned layer
      conv1         -> encoder's first Conv2d
      conv2         -> encoder's SECOND Conv2d (encoder[4] — encoder[2]
                        is actually the ReLU between the two convs, a
                        bug in an earlier version of this function that
                        silently mislabeled the ReLU's output as "conv2")
      convlstm_hidden -> ConvLSTM cell's hidden state
      decoder_conv1 -> decoder's first Conv2d (hidden_dim -> hidden_dim/2)
      decoder_out   -> decoder's final output (hidden_dim/2 -> 1 channel)
    """
    activations.clear()

    model.input_reduce.register_forward_hook(hook("input_reduce"))

    model.model2d.encoder[0].register_forward_hook(hook("conv1"))
    model.model2d.encoder[4].register_forward_hook(hook("conv2"))

    model.model2d.cell.register_forward_hook(hook("convlstm_hidden"))

    model.model2d.decoder[0].register_forward_hook(hook("decoder_conv1"))
    model.model2d.decoder.register_forward_hook(hook("decoder_out"))


def get_raw_input_channels(x, timestep=-1, z_index=0, batch_index=0):
    """
    Extract the raw physical-variable channels (Tracer, U, V, W, Dif,
    MLD) directly from the model's native input `x` — no hook needed,
    this is just indexing the input tensor itself at one timestep and
    depth level.

    x: native (B,T,C,Z,H,W) tensor (do not pre-slice before calling).
    Returns a (C,H,W) tensor, C in CHANNEL_NAMES order.
    """
    if x.dim() != 6:
        raise ValueError(
            f"get_raw_input_channels needs the native 6D (B,T,C,Z,H,W) "
            f"input; got {x.dim()}D."
        )
    return x[batch_index, timestep, :, z_index].detach()


def get_activation_maps(model, x):
    """
    Run `model` on `x` with hooks registered, returning the captured
    intermediate activations.

    IMPORTANT: `x` must already be in the shape `model.forward` expects
    natively (for OceanConv2DAsymXPU that's the raw (B,T,C,Z,H,W)
    tensor — it does its own internal depth/window reshaping). Do not
    pre-slice out the T or Z dimensions before calling this function,
    or the forward pass will fail on a shape mismatch.
    """
    register_hooks(model)
    with torch.no_grad():
        # Previously this function reshaped/sliced `x` down to a shape
        # that no longer matched the model's forward signature, and
        # never actually called the model — so `activations` was always
        # empty (or stale from a previous call). Now we just run the
        # real forward pass so the registered hooks fire.
        model(x)
    return activations.copy()



def visualize_activation_maps(
    model,
    x,
    sample,
    region_name="unknown",
    out_dir="activations",
):
    """
    Visualize feature maps from OceanPredictor:

    input_reduce → conv1 → conv2 → convlstm_hidden → decoder_conv1 → decoder_out

    Supports:
      x shape = (B,T,C,H,W) or (B,T,C,Z,H,W)
    """

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # ---------------------------------------------------------
    # Compute a depth tag for filenames/titles only. This must NOT be
    # used to slice the tensor fed into the model — OceanConv2DAsymXPU
    # needs the full (B,T,C,Z,H,W) input and builds its own local
    # depth windows internally.
    # ---------------------------------------------------------
    if x.dim() == 6:
        # (B,T,C,Z,H,W)
        z_center = x.shape[3] // 2
        depth_tag = f"z{z_center}"
    elif x.dim() == 5:
        depth_tag = "z0"
    else:
        raise ValueError(f"Unsupported input shape {x.shape}")

    x = x.to(next(model.parameters()).device)

    # ---------------------------------------------------------
    # Get activations (run the model on its native input shape)
    # ---------------------------------------------------------

    acts = get_activation_maps(model, x)

    # ---------------------------------------------------------
    # Collect activations — full network, input to output
    # ---------------------------------------------------------

    layer_names = [
        "input_reduce",
        "conv1",
        "conv2",
        "convlstm_hidden",
        "decoder_conv1",
        "decoder_out",
    ]
    feature_maps = [acts[lname] for lname in layer_names]

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    for fmap, lname in zip(feature_maps, layer_names):

        fmap = fmap[0].detach().cpu()  # (C,Z,H,W)


        if fmap.dim() == 4:   # (C,Z,H,W)
            z_mid = fmap.shape[1] // 2
            fmap = fmap[:, z_mid]

        elif fmap.dim() == 3:  # already (C,H,W)
            pass

        else:
            raise ValueError(f"Unexpected fmap shape: {fmap.shape}")
       

        # normalize for visualization
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
 
        n_ch = min(4, fmap.shape[0])

        fig, axes = plt.subplots(1, n_ch, figsize=(3 * n_ch, 3), squeeze=False)

        for i in range(n_ch):
            axes[0, i].imshow(fmap[i], cmap="copper") #coolwarm viridis copper afmhot hot bone
            axes[0, i].axis("off")
            axes[0, i].set_title(f"{lname} ch{i}")

        plt.suptitle(f"{region_name} | sample {sample} | {lname} | {depth_tag}")

        fname = f"{region_name}_{lname}_{depth_tag}_s{sample:05d}.png"
        out_path = os.path.join(out_dir, fname)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"🧠 Activation maps saved for {region_name} sample {sample}")

