import os
import torch
import numpy as np
import matplotlib.pyplot as plt

activations = {}

def hook(name):
    def fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]  # ConvLSTM returns (h,c)
        activations[name] = out.detach()
    return fn


def register_hooks(model):
    activations.clear()

    # Encoder layers
    model.model2d.encoder[0].register_forward_hook(hook("conv1"))
    model.model2d.encoder[2].register_forward_hook(hook("conv2"))
    # ConvLSTM hidden state (hook the CELL output)
    model.model2d.cell.register_forward_hook(hook("convlstm_hidden"))
    # Decoder output
    model.model2d.decoder.register_forward_hook(hook("decoder_out"))


def get_activation_maps(model, x):
    register_hooks(model)
    with torch.no_grad():
        # flatten input like training
        if x.dim() == 6:
            B, T, C, Z, H, W = x.shape
            x = x.view(B, T, C*Z, H, W)
        # then take last timestep
        x = x[:, -1]
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

    conv1 → conv2  → convlstm_hidden → decoder_out

    Supports:
      x shape = (B,T,C,H,W) or (B,T,C,Z,H,W)
    """

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # ---------------------------------------------------------
    # Select last timestep
    # ---------------------------------------------------------
    if x.dim() == 6:
        # (B,T,C,Z,H,W)
        x_last = x[:, -1]
        z_center = x_last.shape[2] // 2
        x_last = x_last[:, :, z_center]
        depth_tag = f"z{z_center}"
    elif x.dim() == 5:
        x_last = x[:, -1]
        depth_tag = "z0"
    else:
        raise ValueError(f"Unsupported input shape {x.shape}")

    x_last = x_last.to(next(model.parameters()).device)

    # ---------------------------------------------------------
    # Get activations
    # ---------------------------------------------------------

    acts = get_activation_maps(model, x_last)

    # ---------------------------------------------------------
    # Collect activations
    # ---------------------------------------------------------
    
    feature_maps = [
        acts["conv1"],
        acts["conv2"],
        acts["convlstm_hidden"],
        acts["decoder_out"]
]



   

    layer_names = [
        "conv1",
        "conv2",
        "convlstm_hidden",
        "decoder_out",
    ]

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

