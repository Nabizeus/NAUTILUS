# src/activation_maps.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_grid(arr, path, title, max_ch=8):
    C = arr.shape[0]
    n = min(C, max_ch)
    idxs = np.linspace(0, C - 1, n).astype(int)

    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 2.5))
    if n == 1:
        axes = [axes]

    for ax, ch in zip(axes, idxs):
        ax.imshow(arr[ch], cmap="viridis")
        ax.set_title(f"ch{ch}")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def visualize_activation_maps(
    model,
    x,
    step,
    region,
    out_dir="activations",
):
    """
    Works with the NEW OceanPredictor architecture.
    x shape: (B, T, C, H, W)
    """
    os.makedirs(out_dir, exist_ok=True)
    output_dir_region = os.path.join(out_dir, f"{region}")
    os.makedirs(output_dir_region, exist_ok=True)

    model.eval()
    x = x.to(next(model.parameters()).device)

    with torch.no_grad():
        B, T, C, H, W = x.shape
        h, c = model.convlstm.init_hidden(B, H, W)
        h, c = h.to(x.device), c.to(x.device)

        # take last batch element
        xt = x[:, -1]

        # ---- Encoder 1 ----
        a1 = model.enc1[0](xt)      # conv1
        g1 = model.enc1[1](a1)      # gelu1
        a2 = model.enc1[2](g1)      # conv2
        g2 = model.enc1[3](a2)      # gelu2

        # ---- Encoder 2 ----
        e2 = model.enc2(g2)

        # ---- ConvLSTM ----
        h, (h, c) = model.convlstm(e2, (h, c))

        # ---- Decoder ----
        d = model.dec1(h)
        out = model.output_conv(d)

    # Save activations
    save_grid(a1[0].cpu().numpy(),
              f"{output_dir_region}/{region}_conv1_{step}.png",
              f"{region} conv1")

    save_grid(g2[0].cpu().numpy(),
              f"{output_dir_region}/{region}_relu2_{step}.png",
              f"{region} relu2")

    save_grid(h[0].cpu().numpy(),
              f"{output_dir_region}/{region}_convlstm_hidden_{step}.png",
              f"{region} convlstm hidden")

    plt.figure(figsize=(4, 4))
    plt.imshow(out[0, 0].cpu(), cmap="viridis")
    plt.colorbar()
    plt.title(f"{region} decoder output")
    plt.tight_layout()
    plt.savefig(f"{output_dir_region}/{region}_decoder_out_{step}.png", dpi=150)
    plt.close()

    #print(f"🧠 Saved activations for {region} @ step {step}")

