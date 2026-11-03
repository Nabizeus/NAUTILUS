"""
benchmark_inference.py -- inference speed of a trained full-column model.

Times one forward step (4 input snapshots -> next step, all 75 levels) over a
domain of the given size, on GPU (fp32 and fp16 autocast) and CPU (fp32), for
1 tracer and for N tracers at once. With a single-tracer model, N tracers are
N independent forecasts run as one batch (or in chunks if the batch does not
fit in memory), so this measures speed only, not accuracy.

    python -m src.benchmark_inference --config config/nautilus_v4_multi4_long.yaml \
        --checkpoint checkpoints/v4_multi4_long_multi4_best_model.pt \
        --tracers 1 24 --height 292 --width 362 --out logs/benchmark_inference.json

Inputs are random numbers with the shape of the real data; the speed of the
convolutional model does not depend on the values.

--tiles 0 128 64: 0 = the whole domain in one piece; a number = square tiles of
that size plus a halo of --halo cells on every side (the model's spatial reach),
as a coupled run would split the domain. One step then is all tiles x tracers,
run in batches.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch

from src.model_conv2d_asym_hybrid import OceanConv2DAsymXPU
from src.model_light import build_model
from src.utils import load_config


def load_model(cfg, ckpt, device, n_channels):
    model = build_model(cfg, input_channels=n_channels).to(device)   # v6: cfg.model.arch
    if ckpt and str(ckpt).lower() != "none":
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
    # else: random weights (the speed does not depend on the weights)
    model.eval()
    return model


def count_flops(model, shape, device):
    """FLOPs of one forward pass for one tracer on the given input shape (None if unavailable)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(1, *shape, device=device)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        return float(fc.get_total_flops())
    except Exception as e:
        print(f"[bench] FLOP count not available: {e}", flush=True)
        return None


def time_forward(model, n_items, shape, device, amp, repeats, chunk):
    """Seconds per step for n_items forward passes (tiles x tracers), run in batches of `chunk`."""
    T, C, Z, H, W = shape
    x = torch.randn(chunk, T, C, Z, H, W, device=device)
    n_calls = int(np.ceil(n_items / chunk))

    def step():
        for i in range(n_calls):
            b = min(chunk, n_items - i * chunk)
            # autocast only when used: older CPU builds reject a disabled fp16 autocast
            ctx = (torch.autocast(device_type=device.type, dtype=torch.float16) if amp
                   else contextlib.nullcontext())
            with torch.no_grad(), ctx:
                model(x[:b])
        if device.type == "cuda":
            torch.cuda.synchronize()

    step()                                   # warm-up (kernel selection, allocation)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.min(times))


def run(model, n_tracers, shape, device, amp, repeats, max_batch=None, n_tiles=1):
    """Largest batch that fits (GPU; on CPU at most max_batch), then time it."""
    n_items = n_tracers * n_tiles
    chunk = n_items if not max_batch else min(n_items, max_batch)
    while chunk >= 1:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            med, best = time_forward(model, n_items, shape, device, amp, repeats, chunk)
            mem = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else float("nan")
            return dict(tracers=n_tracers, tiles=n_tiles, tile_shape=list(shape[-2:]),
                        batch=chunk, calls=int(np.ceil(n_items / chunk)),
                        seconds_per_step=med, best_seconds_per_step=best,
                        seconds_per_tracer=med / n_tracers, peak_gpu_memory_gb=mem)
        except RuntimeError as e:            # out of memory: halve the batch
            if "out of memory" not in str(e).lower():
                raise
            chunk //= 2
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return dict(tracers=n_tracers, error="out of memory even with batch 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="none", help="'none' = random weights")
    ap.add_argument("--share", type=int, default=1,
                    help="this process runs 1/share of the tiles (several processes on one node, like MPI ranks)")
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    ap.add_argument("--tracers", type=int, nargs="+", default=[1, 24])
    ap.add_argument("--height", type=int, default=292)
    ap.add_argument("--width", type=int, default=362)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tiles", type=int, nargs="+", default=[0], help="0 = whole domain, else tile size")
    ap.add_argument("--halo", type=int, default=8)
    ap.add_argument("--cpu-repeats", type=int, default=1)
    ap.add_argument("--cpu-batch", type=int, default=4,
                    help="largest CPU batch (out of memory on CPU kills the job instead of raising)")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--cpu-tracers", type=int, nargs="+", default=[1],
                    help="tracer counts timed on CPU (the time is linear in the number of tracers)")
    ap.add_argument("--out", default="logs/benchmark_inference.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_ch = 7 if bool(cfg.data.get("mask_channel", True)) else 6
    T, Z = int(cfg.data.window.input_steps), int(cfg.data.vertical_levels)
    print(f"[bench] domain {args.height} x {args.width}, {Z} levels, {T} inputs, {n_ch} channels", flush=True)
    results = dict(config=args.config, checkpoint=args.checkpoint, domain=[args.height, args.width],
                   halo=args.halo, arch=str(cfg.model.get("arch", "conv2dlstm")),
                   hidden_dim=int(cfg.model.get("hidden_dim", 128)), share=args.share,
                   rank=int(os.environ.get("SLURM_PROCID", 0)), runs=[])
    m0 = load_model(cfg, args.checkpoint, torch.device("cpu"), n_ch)
    results["parameters"] = int(sum(p.numel() for p in m0.parameters()))
    f64 = count_flops(m0, (T, n_ch, Z, 64, 64), torch.device("cpu"))
    if f64:
        # convolutions scale with the number of grid cells: scale a 64 x 64 count to the domain
        results["gflops_per_call_global"] = f64 * args.height * args.width / (64 * 64) / 1e9
    print(f"[bench] arch {results['arch']}, hidden {results['hidden_dim']}: {results['parameters']:,} parameters, "
          f"{results.get('gflops_per_call_global', float('nan')):.0f} GFLOP per call (global domain, 1 tracer)", flush=True)
    del m0

    devices = []
    if torch.cuda.is_available() and "cuda" in args.devices:
        devices += [(torch.device("cuda"), False), (torch.device("cuda"), True)]
        results["gpu"] = torch.cuda.get_device_name(0)
    if not args.skip_cpu and "cpu" in args.devices:
        devices.append((torch.device("cpu"), False))
    results["cpu_threads"] = torch.get_num_threads()

    for device, amp in devices:
        model = load_model(cfg, args.checkpoint, device, n_ch)
        reps = args.repeats if device.type == "cuda" else args.cpu_repeats
        for tile in args.tiles:
            if tile <= 0:
                n_tiles, shape = 1, (T, n_ch, Z, args.height, args.width)
            else:
                n_tiles = int(np.ceil(args.height / tile) * np.ceil(args.width / tile))
                n_tiles = int(np.ceil(n_tiles / max(1, args.share)))     # this process's share
                shape = (T, n_ch, Z, tile + 2 * args.halo, tile + 2 * args.halo)
            for n in (args.tracers if device.type == "cuda" else args.cpu_tracers):
                r = run(model, n, shape, device, amp, reps,
                        max_batch=args.cpu_batch if device.type == "cpu" else None, n_tiles=n_tiles)
                r.update(device=device.type, precision="fp16 autocast" if amp else "fp32",
                         tile=tile if tile > 0 else "whole domain")
                results["runs"].append(r)
                print(f"[bench] {device.type:4s} {r['precision']:13s} tile {str(r['tile']):12s} "
                      f"({n_tiles} x {shape[-2]}x{shape[-1]}) tracers {n:3d}: "
                      + (f"{r['seconds_per_step']:.3f} s per step (batch {r['batch']}, {r['calls']} call(s)), "
                         f"{r['seconds_per_tracer']:.3f} s per tracer, peak GPU memory {r['peak_gpu_memory_gb']:.1f} GB"
                         if "error" not in r else r["error"]), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out = args.out.replace("{rank}", str(results["rank"]))   # one file per process
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
