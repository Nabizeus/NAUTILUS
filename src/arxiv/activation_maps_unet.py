# src/activation_maps.py
import torch
import matplotlib.pyplot as plt
import os

def register_activation_hooks(model, layer_names):
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            # ConvLSTM Returns (h, c)
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu() # keep h only
            else:
                activations[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            module.register_forward_hook(hook_fn(name))

    return activations


def visualize_activation_maps_unet(
    model,
    inputs,
    region,
    step,
    layers=None,
    outdir="activations"
):
    """
    inputs: (B, T, C, H, W)
    """
    os.makedirs(outdir, exist_ok=True)
    model.eval()

    if layers is None:
        layers = [
            "inc",                  # First UNet block
            "down1.conv",
            "down2.conv",
            "down3.conv",
            "bottleneck_conv",
            "convlstm",            # special handling h only
            "up1.conv",
            "up2.conv",
            "up3.conv",
            "outc"                 # final prediction
        ]

    activations = register_activation_hooks(model, layers)

    with torch.no_grad():
        _ = model(inputs)

    for lname, act in activations.items():
        # act shape: (B, C, H, W) or ConvLSTM hidden
        fmap = act[0]           # first batch
        fmap = fmap.mean(0)     # average channels → (H, W)

        plt.figure(figsize=(4, 4))
        plt.imshow(fmap, cmap="magma")
        plt.colorbar()
        plt.title(f"{region} | {lname} | step {step}")
        plt.tight_layout()

        out = f"{outdir}/{region}_{lname.replace('.','_')}_{step:04d}.png"
        plt.savefig(out, dpi=150)
        plt.close()

    print(f"🧠 Saved activation maps for {region} @ step {step}")

