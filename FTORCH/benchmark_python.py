"""
benchmark_python.py

Times the forward pass of the exported TorchScript model in Python, on the
same dummy input and with the same thread count as test_nautilus_benchmark
(the Fortran/FTorch timing), so the two can be compared directly.

Usage:
    python benchmark_python.py --export-dir ftorch_export --region North_Atlantic \
        --input crosscheck/dummy_input.bin --threads 1 --reps 5 --warmup 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="ftorch_export")
    ap.add_argument("--region", required=True)
    ap.add_argument("--input", default="crosscheck/dummy_input.bin")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    export_dir = Path(args.export_dir)
    meta = json.load(open(export_dir / f"nautilus_{args.region}_metadata.json"))
    s = meta["tensor_shapes"]
    shape = (1, s["input_steps_T"] * s["channels_C"] * s["depth_levels_Z"], s["height_H"], s["width_W"])
    x = torch.from_numpy(
        np.fromfile(args.input, dtype=np.float32).reshape(shape, order="F").copy())
    model = torch.jit.load(str(export_dir / meta["model_path"])).eval()

    ms = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        for i in range(args.reps):
            t0 = time.perf_counter()
            model(x)
            ms.append(1000.0 * (time.perf_counter() - t0))
            print(f"   run {i + 1}: {ms[-1]:10.2f} ms", flush=True)

    print(f"BENCH who=Python-torch-{torch.__version__.split('+')[0]} threads={torch.get_num_threads()} "
          f"reps={args.reps} mean_ms={np.mean(ms):.2f} min_ms={np.min(ms):.2f} max_ms={np.max(ms):.2f}")


if __name__ == "__main__":
    main()
