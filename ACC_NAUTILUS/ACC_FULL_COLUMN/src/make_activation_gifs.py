import glob
import os
from PIL import Image

def build_gif_for_layer(region, layer, out_dir="activations", gif_dir="activation_gifs"):
    os.makedirs(gif_dir, exist_ok=True)

    pattern = os.path.join(out_dir, f"{region}_{layer}_z0_s*.png")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"⚠ No files for {region} {layer}")
        return

    images = [Image.open(f) for f in files]

    out_path = os.path.join(gif_dir, f"{region}_{layer}.gif")
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=150,
        loop=0
    )

    print(f"🎞 Saved GIF → {out_path}")

