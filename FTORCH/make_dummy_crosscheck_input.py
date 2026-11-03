"""
make_dummy_crosscheck_input.py

Generates a deterministic dummy input tensor of the exact shape the
exported TorchScript model expects, (1, T*C*Z, H, W), runs it through
the model in Python, and writes both the input and the reference output
as raw float32 binary files that test_nautilus_crosscheck.F90 can read
byte-for-byte with a plain Fortran `read(unit) array` statement.

IMPORTANT ON BINARY LAYOUT:
Fortran arrays are column-major (first index varies fastest); NumPy's
.tofile() always writes in C order regardless of the array's internal
layout. To make a flat file that a Fortran `read` into an array of the
SAME shape reproduces exactly, we must ravel in Fortran order ourselves
before writing:  array.ravel(order="F").tofile(path)
Reading a Fortran-written array back into NumPy for comparison is the
mirror image: np.fromfile(...).reshape(shape, order="F")

Usage:
    python make_dummy_crosscheck_input.py \
        --export-dir ftorch_export \
        --region North_Atlantic \
        --seed 0 \
        --out-dir crosscheck
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, help="Dir with the exported .pt + metadata.json")
    ap.add_argument("--region", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="crosscheck")
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    meta_path = export_dir / f"nautilus_{args.region}_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    shapes = meta["tensor_shapes"]
    T = shapes["input_steps_T"]
    C = shapes["channels_C"]
    Z = shapes["depth_levels_Z"]
    H = shapes["height_H"]
    W = shapes["width_W"]

    model_path = export_dir / meta["model_path"]
    model = torch.jit.load(str(model_path))
    model.eval()

    print(f"[crosscheck] T={T} C={C} Z={Z} H={H} W={W}")
    print(f"[crosscheck] loaded {model_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Deterministic dummy input in [0, 1) -- values in this range are
    # representative of already-normalized channels, so this exercises
    # the model+layout without worrying about normalization at all.
    x = torch.rand(1, T * C * Z, H, W, dtype=torch.float32)

    with torch.no_grad():
        y = model(x)   # (1, Z, H, W)

    assert y.shape == (1, Z, H, W), f"unexpected output shape {y.shape}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_np = x.numpy().astype(np.float32)
    y_np = y.numpy().astype(np.float32)

    in_path = out_dir / "dummy_input.bin"
    out_path = out_dir / "dummy_output_reference.bin"
    shape_path = out_dir / "dummy_shapes.json"

    x_np.ravel(order="F").tofile(in_path)
    y_np.ravel(order="F").tofile(out_path)

    with open(shape_path, "w") as f:
        json.dump({
            "T": T, "C": C, "Z": Z, "H": H, "W": W,
            "input_shape": list(x_np.shape),
            "output_shape": list(y_np.shape),
            "input_file": in_path.name,
            "reference_output_file": out_path.name,
        }, f, indent=2)

    print(f"[crosscheck] wrote {in_path}  ({x_np.nbytes} bytes)")
    print(f"[crosscheck] wrote {out_path} ({y_np.nbytes} bytes)")
    print(f"[crosscheck] wrote {shape_path}")
    print(f"[crosscheck] input  stats: min={x_np.min():.6f} max={x_np.max():.6f} mean={x_np.mean():.6f}")
    print(f"[crosscheck] output stats: min={y_np.min():.6f} max={y_np.max():.6f} mean={y_np.mean():.6f}")


if __name__ == "__main__":
    main()
