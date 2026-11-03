"""
compare_outputs.py

Compares the raw binary output produced by test_nautilus_crosscheck.F90
(fortran_output.bin) against the Python reference
(crosscheck/dummy_output_reference.bin) written by
make_dummy_crosscheck_input.py.

Exits with status 0 and prints PASS if they match within tolerance,
otherwise prints FAIL and diagnostic stats, and exits with status 1.

Usage:
    python compare_outputs.py --crosscheck-dir crosscheck \
        --fortran-output crosscheck/fortran_output.bin \
        --atol 1e-4 --rtol 1e-3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crosscheck-dir", default="crosscheck")
    ap.add_argument("--fortran-output", required=True,
                     help="Raw float32 binary written by the Fortran test program")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    cc_dir = Path(args.crosscheck_dir)
    with open(cc_dir / "dummy_shapes.json") as f:
        shapes = json.load(f)

    out_shape = tuple(shapes["output_shape"])
    ref_path = cc_dir / shapes["reference_output_file"]

    ref = np.fromfile(ref_path, dtype=np.float32).reshape(out_shape, order="F")
    fortran_out = np.fromfile(args.fortran_output, dtype=np.float32)

    if fortran_out.size != ref.size:
        print(f"FAIL: element count mismatch. Python ref has {ref.size} elements, "
              f"Fortran output has {fortran_out.size}. Check that the Fortran "
              f"program declared raw_out with shape {out_shape} and wrote all of it.")
        sys.exit(1)

    fortran_out = fortran_out.reshape(out_shape, order="F")

    diff = np.abs(fortran_out - ref)
    max_abs_diff = float(diff.max())
    max_rel_diff = float((diff / (np.abs(ref) + 1e-8)).max())
    close = np.allclose(fortran_out, ref, atol=args.atol, rtol=args.rtol)

    print(f"Reference  stats: min={ref.min():.6f} max={ref.max():.6f} mean={ref.mean():.6f}")
    print(f"Fortran    stats: min={fortran_out.min():.6f} max={fortran_out.max():.6f} mean={fortran_out.mean():.6f}")
    print(f"Max abs diff: {max_abs_diff:.6e}")
    print(f"Max rel diff: {max_rel_diff:.6e}")

    if close:
        print("PASS: Fortran output matches the Python-traced model within tolerance.")
        sys.exit(0)
    else:
        print("FAIL: Fortran output does NOT match the Python-traced model.")
        print("Likely cause: the flat_idx packing in forward()/forward_raw() "
              "(nautilus_ftorch_interface.F90) does not match the "
              "x_flat.view(B, T, C, Z, H, W) reshape used in "
              "export_nautilus_model.py's FTorchExportWrapper, or the "
              "in_layout/out_layout tensor_layout arrays need adjusting for "
              "your installed FTorch version. Inspect a few individual "
              "elements (not just aggregate stats) to localize which axis "
              "is transposed or reversed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
