"""
generate_depth_sweep_configs.py

Generates a set of config YAML variants from a base config, each
changing only:
  - data.depth_training  (which depth range/indices contribute to the
    training LOSS — the dataset still loads/predicts the full column)
  - training.experiment_name  (so each variant gets its own checkpoint
    file and run_name — see _checkpoint_prefix() in train.py — instead
    of silently overwriting another sweep config's checkpoint)
  - training.checkpoint_dir  (nested under a "depth_sweep" subfolder,
    as a second layer of separation)
  - evaluation.depth_bin_width_m  (read by predict.py to build its
    evaluation depth bins — set to match --step for banded sweeps, so
    each band's training range lines up EXACTLY with one evaluation
    bin; without this a band like "1000-1500m" could get merged into
    a coarser eval bin and make "trained vs extrapolated" ambiguous)

Two sweep modes:

  cumulative
      Growing budgets from 0m: 0-step, 0-2*step, ..., up to max-depth.
      e.g. --step 1000 --max-depth 6000 gives:
        0-1000, 0-2000, 0-3000, 0-4000, 0-5000, 0-6000

  banded
      Non-overlapping windows of width --step spanning the whole
      column. e.g. --step 500 --max-depth 6000 gives:
        0-500, 500-1000, 1000-1500, ..., 5500-6000

Usage:
    python generate_depth_sweep_configs.py \\
        --base conv2dlstm_s323D_fullcolumn.yaml \\
        --out-dir configs/depth_sweep/cumulative \\
        --mode cumulative --step 1000 --max-depth 6000

    python generate_depth_sweep_configs.py \\
        --base conv2dlstm_s323D_fullcolumn.yaml \\
        --out-dir configs/depth_sweep/banded \\
        --mode banded --step 500 --max-depth 6000

Each call writes one YAML per depth range plus a manifest CSV
(depth_sweep_manifest.csv) in --out-dir, listing every generated
config's path, tag, mode, and depth bounds. run_depth_sweep.py reads
that manifest to launch train.py + predict.py for each config in turn,
and src/plot_depth_sweep_heatmap.py reads it (indirectly, via each
run's rows in the depth-skill CSV) to build the comparison heatmap.
"""

import argparse
import copy
import csv
import os

import yaml


def _make_tag(mode, lo, hi):
    if mode == "cumulative":
        return f"depthbudget_0-{int(hi)}m"
    return f"depthband_{int(lo)}-{int(hi)}m"


def _depth_ranges(mode, step, max_depth):
    if mode == "cumulative":
        hi = step
        ranges = []
        while hi <= max_depth + 1e-6:
            ranges.append((0.0, float(hi)))
            hi += step
        if not ranges:
            ranges = [(0.0, float(max_depth))]
        return ranges

    if mode == "banded":
        ranges = []
        lo = 0.0
        while lo < max_depth - 1e-6:
            hi = min(lo + step, max_depth)
            ranges.append((float(lo), float(hi)))
            lo += step
        return ranges

    raise ValueError(f"Unknown mode '{mode}' (use 'cumulative' or 'banded')")


def generate_configs(base_path, out_dir, mode, step, max_depth, eval_bin_width=None,
                      epochs_override=None):
    """
    Returns the manifest CSV path. `eval_bin_width` defaults to `step`
    for banded sweeps (so each band = exactly one evaluation bin) and
    to 500 for cumulative sweeps unless overridden.

    `epochs_override`: if set, overrides training.epochs in every
    generated config. Use this for a cheap smoke test of the sweep
    PLUMBING (config loading, checkpoint naming, depth-skill CSV
    accumulation, heatmap) with e.g. epochs_override=1 on a short
    queue, before committing to the real multi-epoch sweep on a
    longer-walltime/GPU queue.
    """
    if eval_bin_width is None:
        eval_bin_width = step if mode == "banded" else 500

    with open(base_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(out_dir, exist_ok=True)
    ranges = _depth_ranges(mode, step, max_depth)

    manifest_rows = []

    for lo, hi in ranges:
        cfg = copy.deepcopy(base_cfg)
        tag = _make_tag(mode, lo, hi)
        if epochs_override is not None:
            tag = f"{tag}_smoketest"

        cfg.setdefault("data", {}).setdefault("depth_training", {})
        cfg["data"]["depth_training"]["mode"] = "range"
        cfg["data"]["depth_training"]["min_depth_m"] = float(lo)
        cfg["data"]["depth_training"]["max_depth_m"] = float(hi)
        cfg["data"]["depth_training"]["indices"] = []

        cfg.setdefault("training", {})
        cfg["training"]["experiment_name"] = tag
        base_ckpt_dir = cfg["training"].get("checkpoint_dir", "./checkpoints")
        cfg["training"]["checkpoint_dir"] = os.path.join(base_ckpt_dir, "depth_sweep")
        if epochs_override is not None:
            cfg["training"]["epochs"] = int(epochs_override)

        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["depth_bin_width_m"] = float(eval_bin_width)

        out_path = os.path.join(out_dir, f"{tag}.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        manifest_rows.append({
            "tag": tag,
            "mode": mode,
            "config_path": os.path.abspath(out_path),
            "depth_min_m": lo,
            "depth_max_m": hi,
            "eval_bin_width_m": eval_bin_width,
        })
        print(f"✅ Wrote {out_path}  (train depths {lo:.0f}-{hi:.0f} m, "
              f"eval bin width {eval_bin_width:.0f} m)")

    manifest_path = os.path.join(out_dir, "depth_sweep_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tag", "mode", "config_path", "depth_min_m",
                           "depth_max_m", "eval_bin_width_m"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"📋 Manifest written → {manifest_path} ({len(manifest_rows)} configs)")

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth-training config sweep YAMLs (cumulative budgets or banded windows)."
    )
    parser.add_argument("--base", required=True, help="Path to the base YAML config.")
    parser.add_argument("--out-dir", required=True, help="Directory to write generated configs + manifest.")
    parser.add_argument("--mode", choices=["cumulative", "banded"], required=True)
    parser.add_argument(
        "--step", type=float, required=True,
        help="Budget increment in meters (cumulative) or band width in meters (banded).",
    )
    parser.add_argument("--max-depth", type=float, required=True,
                         help="Deepest depth (meters) to sweep up to.")
    parser.add_argument(
        "--eval-bin-width", type=float, default=None,
        help="Evaluation bin width in meters written into each config's "
             "evaluation.depth_bin_width_m. Defaults to --step for banded "
             "sweeps, 500 for cumulative sweeps.",
    )
    parser.add_argument(
        "--epochs-override", type=int, default=None,
        help="Override training.epochs in every generated config, e.g. "
             "--epochs-override 1 for a cheap smoke test of the sweep "
             "plumbing (checkpoint naming, --config loading, depth-skill "
             "CSV, heatmap) on a short queue before running the real "
             "sweep on a longer/GPU queue.",
    )
    args = parser.parse_args()

    generate_configs(
        args.base, args.out_dir, args.mode, args.step, args.max_depth,
        eval_bin_width=args.eval_bin_width, epochs_override=args.epochs_override,
    )


if __name__ == "__main__":
    main()
