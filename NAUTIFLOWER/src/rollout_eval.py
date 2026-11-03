"""
rollout_eval.py -- auto-regressive (roll-out) evaluation over the test period.

From many start times in the test period the model is run forward on its own
predictions: the tracer input of every step is the model's earlier output,
while u, v, w, dif and the MLD distance come from the NEMO data (as when the
model is coupled to NEMO). A model with lead time k (cfg.data.lead_steps)
needs horizon / k calls. At every lead time the forecast is compared with the
truth and with "no change" from the start time (persistence).

    python -m src.rollout_eval --config config/nautilus_v4_multi4_long.yaml \
        [--horizon 96] [--start-every 24] [--kinds training named random]

Outputs in <evaluation_global.out_dir>/rollout/:
    rollout_per_lead.csv     per box, kind, lead (time steps) and depth band:
                             RMSE model, RMSE persistence, ratio, tracer-sum error
    rollout_ratio_vs_lead.png  median ratio against lead time per box kind
    rollout_summary.json
"""

import argparse
import contextlib
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from src.data_v2 import (RegionData, read_boxes, n_timesteps, split_bounds, load_stats,
                         stats_path, model_name, lead_steps)
from src.evaluate_global import read_grid_info, build_box_list, load_model
from src.utils import load_config, get_device

BANDS = [("all", 0, 1e9), ("0-200 m", 0, 200), ("200-2000 m", 200, 2000), ("2000+ m", 2000, 1e9)]


def time_step_seconds(cfg):
    with xr.open_dataset(cfg.data.tracers.paths[0], engine="netcdf4", decode_times=False) as ds:
        t = np.asarray(ds[cfg.data.coords.time].values[:3], dtype=np.float64)
    return float(np.median(np.diff(t)))


def rollout_box(model, rd, starts, t_in, k, n_calls, device, use_amp):
    """
    Roll the model out from every start (local window-start index) for n_calls
    calls. Returns {lead_steps: dict(band: (sse_m, sse_p, n), sum_m, sum_t)}.
    """
    Z = rd.Z
    ocean = rd.ocean                                           # (Z, H, W)
    m = rd.mask_f                                              # (Z, H, W) float
    bands = {name: (rd.depths >= lo) & (rd.depths < hi) for name, lo, hi in BANDS}
    # tracer history per start: the t_in true input snapshots
    hist = [[rd.tracer[s + i * k].copy() for i in range(t_in)] for s in starts]
    pers = [h[-1] for h in hist]                               # "no change" from the start
    out = {}
    for j in range(1, n_calls + 1):
        xs = []
        for b, s in enumerate(starts):
            last = s + (t_in - 1) * k + (j - 1) * k             # newest input time
            w0 = last - (t_in - 1) * k                          # window start
            x, _, _ = rd.window(w0, t_in=t_in, full_target=True)
            x = np.array(x, copy=True)
            x[:, 0] = np.stack(hist[b][-t_in:]) * m[None]       # model's own tracer history
            xs.append(x)
        xb = torch.from_numpy(np.stack(xs)).to(device)
        ctx = (torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp
               else contextlib.nullcontext())
        with torch.no_grad(), ctx:
            pred = model(xb).float().cpu().numpy()[:, 0]       # (B, Z, H, W) scaled change
        acc = {name: [0.0, 0.0, 0] for name in bands}
        sum_m = sum_t = 0.0
        for b, s in enumerate(starts):
            last = s + (t_in - 1) * k + (j - 1) * k
            full = rd.to_full(pred[b], hist[b][-1]) * m          # normalized full field
            hist[b].append(full)
            truth = rd.tracer[last + k]
            for name, sel in bands.items():
                o = ocean & sel[:, None, None]
                acc[name][0] += float(((full - truth)[o] ** 2).sum())
                acc[name][1] += float(((pers[b] - truth)[o] ** 2).sum())
                acc[name][2] += int(o.sum())
            sum_m += float(rd.denormalize(full)[ocean].sum())
            sum_t += float(rd.denormalize(truth)[ocean].sum())
        out[j * k] = dict(bands=acc, sum_m=sum_m, sum_t=sum_t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--horizon", type=int, default=96, help="roll-out length in time steps")
    ap.add_argument("--start-every", type=int, default=24, help="time steps between start times")
    ap.add_argument("--kinds", nargs="+", default=["training", "named", "random", "seen"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    k = lead_steps(cfg)
    t_in = int(cfg.data.window.input_steps)
    n_calls = args.horizon // k
    out_dir = os.path.join(cfg.evaluation_global.get("out_dir", "eval_global"), "rollout")
    os.makedirs(out_dir, exist_ok=True)
    tag = cfg.training.get("experiment_name", "")
    mname = model_name(cfg)
    ckpt = args.checkpoint or os.path.join(cfg.training.checkpoint_dir, f"{tag}_{mname}_best_model.pt")
    stats = load_stats(stats_path(cfg, mname))
    device = get_device(cfg)
    use_amp = device.type == "cuda" and bool(cfg.hardware.get("use_amp", False))
    dt = time_step_seconds(cfg)
    print(f"[rollout] {tag}: lead {k} step(s), {n_calls} calls up to {n_calls * k} steps "
          f"({n_calls * k * dt / 3600:.1f} h); time step {dt / 60:.1f} min; checkpoint {ckpt}", flush=True)

    lat, lon, sfc = read_grid_info(cfg)
    rows = [r for r in build_box_list(cfg, sfc, lat, lon) if r["kind"] in args.kinds]
    T_total = n_timesteps(cfg)
    _, val_end, _ = split_bounds(cfg, T_total)
    pad = k - 1                               # history for the first averaged forcing
    t0 = max(0, val_end - pad)
    T_loc = T_total - t0
    last_start = T_loc - 1 - (t_in - 1) * k - n_calls * k
    starts = list(range(val_end - t0, last_start + 1, args.start_every))
    print(f"[rollout] {len(rows)} boxes, {len(starts)} start times (local {starts[0]}..{starts[-1]})", flush=True)
    raw_all = read_boxes(cfg, rows, t0, T_total)

    model = None
    csv_rows = []
    for r in rows:
        tic = time.time()
        rd = RegionData(cfg, (r["x0"], r["x1"]), (r["y0"], r["y1"]), r["name"],
                        t_range=(t0, T_total), stats=stats, verbose=False, raw=raw_all.pop(r["name"]))
        if model is None:
            model = load_model(cfg, ckpt, device, rd.n_channels)
        res = rollout_box(model, rd, starts, t_in, k, n_calls, device, use_amp)
        for lead, d in res.items():
            for band, (sm, sp, n) in d["bands"].items():
                if n == 0:
                    continue
                rm, rp = np.sqrt(sm / n), np.sqrt(sp / n)
                csv_rows.append(dict(name=r["name"], kind=r["kind"], lead_steps=lead,
                                     lead_hours=round(lead * dt / 3600, 2), band=band,
                                     rmse_model=rm, rmse_persistence=rp,
                                     ratio=rm / rp if rp > 0 else np.nan,
                                     tracer_sum_error_pct=(100 * (d["sum_m"] - d["sum_t"]) / abs(d["sum_t"])
                                                           if band == "all" and d["sum_t"] else np.nan)))
        last = res[max(res)]["bands"]["all"]
        print(f"[rollout] {r['name']:20s} {r['kind']:8s} ratio at {max(res)} steps "
              f"{np.sqrt(last[0] / last[2]) / np.sqrt(last[1] / last[2]):.3f} ({time.time() - tic:.0f} s)", flush=True)
        del rd

    with open(os.path.join(out_dir, "rollout_per_lead.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # median ratio against lead per kind and band
    import pandas as pd
    df = pd.DataFrame(csv_rows)
    fig, axs = plt.subplots(1, len(BANDS), figsize=(4.2 * len(BANDS), 4), sharey=True)
    colors = {"training": "black", "named": "#eb6834", "random": "#2a78d6", "seen": "#52514e"}
    summary = dict(experiment=tag, lead_steps=k, calls=n_calls, horizon_steps=n_calls * k,
                   time_step_minutes=dt / 60, starts=len(starts), median_ratio={})
    for ax, (band, _, _) in zip(axs, BANDS):
        for kind, c in colors.items():
            s = df[(df.kind == kind) & (df.band == band)]
            if s.empty:
                continue
            med = s.groupby("lead_steps").ratio.median()
            ax.plot(med.index * dt / 3600, med.values, "-o", ms=3, c=c, label=kind)
            summary["median_ratio"].setdefault(band, {})[kind] = {int(i): float(v) for i, v in med.items()}
        ax.axhline(1, c="#52514e", ls="--", lw=1)
        ax.set(title=band, xlabel="lead time (h)")
        ax.grid(alpha=.3)
    axs[0].set_ylabel('median RMSE ratio model / "no change"')
    axs[0].legend(fontsize=8)
    fig.suptitle(f"{tag}: roll-out (lead {k} step(s) per call)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rollout_ratio_vs_lead.png"), dpi=130)
    with open(os.path.join(out_dir, "rollout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[rollout] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
