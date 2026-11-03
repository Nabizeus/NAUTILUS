"""
compare_realdata.py

Compares what test_nautilus_realdata.F90 produced from RAW NEMO fields with
the Python reference from make_realdata_crosscheck.py.

  CHECK 1  normalized input tensor: Fortran push_timestep() vs the real
           training data loader, per channel.
  CHECK 2  prediction: Fortran forward() vs the exported model run in Python.
  INFO     skill against the NEMO truth for that time step, with a
           "persistence" baseline (predict no change) for scale.

Exits 0 and prints PASS if checks 1 and 2 are within tolerance, else 1.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def rd(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape, order="F").astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="realdata")
    ap.add_argument("--atol-input", type=float, default=1e-5)
    ap.add_argument("--atol-pred", type=float, default=1e-4,
                    help="Tolerance on the prediction in normalized tracer units")
    args = ap.parse_args()

    d = Path(args.dir)
    info = json.load(open(d / "realdata_info.json"))
    T, C, Z, H, W = (info[k] for k in "TCZHW")
    roles = info["channel_roles"]
    print(f"Region {info['region']}, test sample {info['sample_in_test_split']} "
          f"(time index {info['time_index']}); target time {info['target_time']}")

    ok = True

    # ---- CHECK 1: normalized inputs ----
    ref_in = rd(d / "ref_input.bin", (1, T * C * Z, H, W)).reshape(T, C, Z, H, W)
    f_in = rd(d / "fortran_input.bin", (1, T * C * Z, H, W)).reshape(T, C, Z, H, W)
    print("\nCHECK 1: normalized input tensor (Fortran push_timestep vs training data loader)")
    for c, role in enumerate(roles):
        diff = np.abs(f_in[:, c] - ref_in[:, c])
        good = diff.max() <= args.atol_input
        ok &= bool(good)
        print(f"  channel {c + 1} {role:6s}: max abs diff = {diff.max():.3e}   "
              f"(values span {ref_in[:, c].min():.3f} .. {ref_in[:, c].max():.3f})   "
              f"{'ok' if good else 'MISMATCH'}")
        if not good:
            t, z, h, w = np.unravel_index(diff.argmax(), diff.shape)
            print(f"      worst at t={t} z={z} y={h} x={w}: "
                  f"fortran={f_in[t, c, z, h, w]:.6f} python={ref_in[t, c, z, h, w]:.6f}")

    # ---- CHECK 2: prediction ----
    tmin = np.array(info["tracer_min"])[:, None, None]
    rng = (np.array(info["tracer_max"]) - np.array(info["tracer_min"]))[:, None, None]
    live = rng[:, 0, 0] > 1e-12                      # levels with a real tracer range
    safe = np.where(rng > 1e-12, rng, 1.0)
    f_phys = rd(d / "fortran_pred_phys.bin", (Z, H, W))
    r_phys = rd(d / "ref_pred_phys.bin", (Z, H, W))
    f_norm = (f_phys - tmin) / safe
    r_norm = (r_phys - tmin) / safe
    dn = np.abs(f_norm - r_norm)[live]
    good = dn.max() <= args.atol_pred
    ok &= bool(good)
    print("\nCHECK 2: prediction (Fortran forward vs exported model in Python)")
    print(f"  max abs diff, normalized units = {dn.max():.3e}   {'ok' if good else 'MISMATCH'}")
    print(f"  max abs diff, physical units   = {np.abs(f_phys - r_phys).max():.3e}   "
          f"(tracer values up to {np.abs(r_phys).max():.3e})")

    # ---- INFO: skill against NEMO ----
    truth = rd(d / "truth_norm.bin", (Z, H, W))
    ocean = np.fromfile(d / "land_mask.bin", dtype=np.int32).reshape((H, W), order="F") != 0
    depths = np.array(info["depths_m"])
    lev = live & (depths >= info["train_min_depth_m"]) & (depths <= info["train_max_depth_m"])
    persistence = f_in[T - 1, roles.index("tracer")]  # newest input tracer, normalized
    sel = lev[:, None, None] & ocean[None, :, :] & np.isfinite(truth)
    rmse_model = float(np.sqrt(np.mean((f_norm - truth)[sel] ** 2)))
    rmse_pers = float(np.sqrt(np.mean((persistence - truth)[sel] ** 2)))
    print("\nINFO: skill against the NEMO truth (normalized tracer units, ocean cells,")
    print(f"      the {int(lev.sum())} levels between {info['train_min_depth_m']:.0f} and "
          f"{info['train_max_depth_m']:.0f} m used in training)")
    print(f"  RMSE of the Fortran prediction : {rmse_model:.5f}")
    print(f"  RMSE of 'no change' baseline   : {rmse_pers:.5f}")

    print()
    if ok:
        print("PASS: Fortran reproduces the Python pipeline on real NEMO data.")
        sys.exit(0)
    print("FAIL: Fortran and Python disagree on real NEMO data (see MISMATCH lines above).")
    sys.exit(1)


if __name__ == "__main__":
    main()
