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
    model.model.encoder[0].register_forward_hook(hook("conv1"))
    model.model.encoder[2].register_forward_hook(hook("conv2"))
    # ConvLSTM hidden state (hook the CELL output)
    model.model.cell.register_forward_hook(hook("convlstm_hidden"))
    # Decoder output
    model.model.decoder.register_forward_hook(hook("decoder_out"))


def get_activation_maps(model, x):
    register_hooks(model)
    with torch.no_grad():
        _ = model(x)
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

    conv1 → relu1 → conv2 → relu2 → convlstm_hidden → decoder_out

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
    # Forward through encoder manually
    # ---------------------------------------------------------
    encoder = model.model.encoder
    decoder = model.model.decoder
    cell    = model.model.cell

    with torch.no_grad():
        conv1 = encoder[0](x_last)
        relu1 = encoder[1](conv1)
        conv2 = encoder[2](relu1)
        relu2 = encoder[3](conv2)

        # ConvLSTM
        B, C, H, W = relu2.shape
        h0 = torch.zeros(B, cell.hidden_dim, H, W, device=relu2.device)
        c0 = torch.zeros_like(h0)

        h, c = cell(relu2, h0, c0)

        decoded = decoder(h)

    # ---------------------------------------------------------
    # Collect activations
    # ---------------------------------------------------------
    feature_maps = [
        conv1,
        relu1,
        conv2,
        relu2,
        h,
        decoded,
    ]

    layer_names = [
        "conv1",
        "relu1",
        "conv2",
        "relu2",
        "convlstm_hidden",
        "decoder_out",
    ]

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    for fmap, lname in zip(feature_maps, layer_names):

        fmap = fmap[0].detach().cpu()  # (C,H,W)

        n_ch = min(4, fmap.shape[0])

        fig, axes = plt.subplots(1, n_ch, figsize=(3 * n_ch, 3), squeeze=False)

        for i in range(n_ch):
            axes[0, i].imshow(fmap[i], cmap="viridis")
            axes[0, i].axis("off")
            axes[0, i].set_title(f"{lname} ch{i}")

        plt.suptitle(f"{region_name} | sample {sample} | {lname} | {depth_tag}")

        fname = f"{region_name}_{lname}_{depth_tag}_s{sample:05d}.png"
        out_path = os.path.join(out_dir, fname)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"🧠 Activation maps saved for {region_name} sample {sample}")

