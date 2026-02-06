import os
import numpy as np
import matplotlib.pyplot as plt

def save_pred_activation_compare(pred, act, target, step, region, layer, out_dir="compare"):
    os.makedirs(out_dir, exist_ok=True)

    pred = pred[0,0].cpu().numpy()
    target = target[0,0].cpu().numpy()
    act = act[0,0].cpu().numpy()

    vmin = np.nanmin(target)
    vmax = np.nanmax(target)

    fig, axs = plt.subplots(1, 4, figsize=(14,4))

    axs[0].imshow(target, vmin=vmin, vmax=vmax)
    axs[0].set_title("Target")

    axs[1].imshow(pred, vmin=vmin, vmax=vmax)
    axs[1].set_title("Prediction")

    axs[2].imshow(pred-target)
    axs[2].set_title("Error")

    axs[3].imshow(act)
    axs[3].set_title(f"Activation: {layer}")

    for a in axs:
        a.axis("off")

    fname = f"{region}_{layer}_{step:05d}.png"
    path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

    return path

