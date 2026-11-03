"""
benchmark_cpu_coupling.py -- CPU inference speed as a coupled NEMO run would see it (v7).

Every process is one MPI rank: the global ORCA1 domain (292 x 362) is split into
px x py subdomains as NEMO would split it (the factorization of the number of ranks
with the most square subdomains), and each rank times one model call on its own
subdomain plus a halo of --halo cells (east-west periodic, closed at the south and
north edges). The node time of a call is that of the slowest rank.

The real ORCA1 land mask (--mask, from make_mask) sets the ocean columns, so the
graph models (arch gnn) only compute ocean columns and land-heavy subdomains are
cheaper; the convolutional models compute every cell.

Each rank times 4 settings: fp32 / bf16 (model and input in bfloat16, as an FTorch
model exported in bf16 would run), eager / TorchScript (torch.jit.trace + freeze,
how FTorch runs the model). Random weights, 1 tracer (CPU time is linear in tracers).

    python -m src.benchmark_cpu_coupling --make-mask --config config/nautilus_v6_k5_col256.yaml
    OMP_NUM_THREADS=1 srun -n 112 python -m src.benchmark_cpu_coupling \
        --config config/nautilus_v7_k5_gnn128.yaml --out "logs/benchcc_{tag}_{rank}.json"
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from src.model_light import build_model

MASK_FILE = "logs/orca1_surface_ocean_mask.npy"


def make_mask(cfg, out=MASK_FILE):
    """Surface ocean mask (H, W) from the fill values of the tracer at the first time step."""
    import xarray as xr
    path, var = cfg.data.tracers.paths[0], cfg.data.tracers.variables[0]
    with xr.open_dataset(path, engine="netcdf4", mask_and_scale=False, decode_times=False) as ds:
        da = ds[var]
        a = np.asarray(da.isel({da.dims[0]: 0, da.dims[1]: 0}).values, dtype=np.float64)
    mask = np.isfinite(a) & (np.abs(a) < 1e19)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, mask)
    print(f"[benchcc] surface ocean mask {mask.shape}: {int(mask.sum())} ocean columns "
          f"of {mask.size} ({100 * mask.mean():.1f} %) -> {out}", flush=True)


def decompose(P, H, W):
    """(py, px) with py * px = P and the most square subdomains (as NEMO's default)."""
    best = None
    for py in range(1, P + 1):
        if P % py:
            continue
        px = P // py
        score = abs(np.log((H / py) / (W / px)))
        if best is None or score < best[0]:
            best = (score, py, px)
    return best[1], best[2]


def subdomain(mask, P, rank, halo):
    """Rows and columns (with halo) of this rank's subdomain, and its inner size."""
    H, W = mask.shape
    py, px = decompose(P, H, W)
    iy, ix = divmod(rank, px)
    y0, y1 = iy * H // py, (iy + 1) * H // py
    x0, x1 = ix * W // px, (ix + 1) * W // px
    rows = np.arange(max(0, y0 - halo), min(H, y1 + halo))
    cols = np.arange(x0 - halo, x1 + halo) % W if px > 1 else np.arange(W)
    return rows, cols, (py, px), (y1 - y0, x1 - x0)


def time_call(fn, x, repeats):
    with torch.no_grad():
        fn(x)                                          # warm-up
        fn(x)
        t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(x)
            t.append(time.perf_counter() - t0)
    return float(np.median(t)), float(np.min(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--make-mask", action="store_true", help="write the ORCA1 surface mask and stop")
    ap.add_argument("--mask", default=MASK_FILE)
    ap.add_argument("--ranks", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    ap.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--halo", type=int, default=None, help="default: spatial reach of the model")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--settings", nargs="+", default=["fp32-eager", "fp32-jit", "bf16-eager", "bf16-jit"])
    ap.add_argument("--check", action="store_true", help="also compare bf16 and jit outputs with fp32 eager")
    ap.add_argument("--out", default="logs/benchcc_{tag}_{ranks}_{rank}.json")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.make_mask:
        make_mask(cfg, args.mask)
        return
    torch.manual_seed(0)
    arch = str(cfg.model.get("arch", "conv2dlstm"))
    is_graph = arch.startswith("gnn")
    if is_graph:
        reach = int(cfg.model.get("n_layers", 6))
    elif arch == "column":
        reach = 2 * int(cfg.model.get("n_blocks", 3))
    else:
        reach = 8                                      # no fixed reach (U-Nets, warps): the halo of the v5/v6 benchmarks
    halo = reach if args.halo is None else args.halo
    mask_g = np.load(args.mask)
    rows, cols, (py, px), inner = subdomain(mask_g, args.ranks, args.rank, halo)
    mask = mask_g[np.ix_(rows, cols)]
    if is_graph:
        cfg.model.periodic_x = args.ranks == 1 or px == 1     # wrap only if the subdomain spans all longitudes

    T, Z = int(cfg.data.window.input_steps), int(cfg.data.vertical_levels)
    C = 7 if bool(cfg.data.get("mask_channel", True)) else 6
    h, w = mask.shape
    x = torch.randn(1, T, C, Z, h, w) * torch.as_tensor(mask, dtype=torch.float32)   # land 0, as in the data
    if C == 7:
        x[:, :, -1] = torch.as_tensor(mask, dtype=torch.float32)   # the ocean-mask channel (surface mask at every level)

    model = build_model(cfg, input_channels=C).eval()
    if is_graph:
        model.set_mask(torch.as_tensor(mask))
    res = dict(config=args.config, arch=arch, hidden_dim=int(cfg.model.get("hidden_dim", 0)),
               n_layers=int(cfg.model.get("n_layers", 0)), n_blocks=int(cfg.model.get("n_blocks", 0)),
               ranks=args.ranks, rank=args.rank, decomposition=[py, px], halo=halo,
               inner=list(inner), shape=[h, w], ocean_columns=int(mask.sum()), cells=h * w,
               threads=torch.get_num_threads(), parameters=int(sum(p.numel() for p in model.parameters())),
               runs=[])
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        res["gflop_per_call"] = fc.get_total_flops() / 1e9
    except Exception as e:                             # FLOP counter not available
        print(f"[benchcc] no FLOP count: {e}", flush=True)

    ref = None
    for setting in args.settings:
        prec, mode = setting.split("-")
        dt = torch.bfloat16 if prec == "bf16" else torch.float32
        m = build_model(cfg, input_channels=C).eval()
        m.load_state_dict(model.state_dict())
        m = m.to(dt)
        if is_graph:
            m.set_mask(torch.as_tensor(mask))
        xi = x.to(dt)
        fn = m
        tic = time.perf_counter()
        try:
            if mode == "jit":
                with torch.no_grad():
                    fn = torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(m, xi, check_trace=False)))
            med, best = time_call(fn, xi, args.repeats)
            r = dict(setting=setting, seconds_per_call=med, best_seconds_per_call=best,
                     prepare_seconds=time.perf_counter() - tic)
            if args.check:
                with torch.no_grad():
                    y = fn(xi).float()
                if ref is None and setting == "fp32-eager":
                    ref = y
                elif ref is not None:
                    r["rel_error_vs_fp32_eager"] = float((y - ref).norm() / ref.norm().clamp_min(1e-30))
        except Exception as e:
            r = dict(setting=setting, error=f"{type(e).__name__}: {str(e)[:300]}")
        res["runs"].append(r)
        if args.rank == 0 or "error" in r:
            print(f"[benchcc] {arch} rank {args.rank}/{args.ranks} ({py}x{px}) {h}x{w} ocean {res['ocean_columns']} "
                  f"{setting}: " + (f"{r['seconds_per_call']:.4f} s per call" if "error" not in r else r["error"]),
                  flush=True)

    tag = os.path.splitext(os.path.basename(args.config))[0].replace("nautilus_", "")
    out = args.out.format(tag=tag, ranks=args.ranks, rank=args.rank)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
