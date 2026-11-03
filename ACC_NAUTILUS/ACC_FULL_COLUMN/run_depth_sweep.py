"""
run_depth_sweep.py

Launches train.py + predict.py once per config in a manifest CSV
produced by generate_depth_sweep_configs.py, then (once every config
has finished) builds the depth-sweep comparison heatmap from the
accumulated depth-skill CSV.

If a config's training or prediction step fails, the sweep logs the
failure and continues with the remaining configs — one bad config
(out-of-memory on a huge depth range, a NaN loss, etc.) shouldn't
throw away the rest of an overnight sweep.

Usage:
    python run_depth_sweep.py --manifest configs/depth_sweep/cumulative/depth_sweep_manifest.csv
    python run_depth_sweep.py --manifest configs/depth_sweep/banded/depth_sweep_manifest.csv

    # Skip training (e.g. you already trained, just want to re-run eval):
    python run_depth_sweep.py --manifest ... --skip-train

    # Only build the heatmap from whatever's already in the CSV so far:
    python run_depth_sweep.py --manifest ... --heatmap-only
"""

import argparse
import csv
import subprocess
import sys
import time


def run_step(cmd, tag, step_name, log_path=None):
    print(f"\n{'=' * 90}\n▶ [{tag}] {step_name}\n{'=' * 90}")
    print(" ".join(cmd))

    t0 = time.time()
    if log_path:
        with open(log_path, "w") as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd)
    elapsed = (time.time() - t0) / 60.0

    if result.returncode != 0:
        print(f"❌ [{tag}] {step_name} FAILED (exit {result.returncode}, {elapsed:.1f} min)"
              + (f" — see {log_path}" if log_path else ""))
        return False

    print(f"✅ [{tag}] {step_name} finished ({elapsed:.1f} min)")
    return True


def run_sweep(manifest_path, python_exe="python", skip_train=False, skip_predict=False,
              log_dir="logs/depth_sweep"):
    import os
    os.makedirs(log_dir, exist_ok=True)

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    print(f"📋 Loaded manifest with {len(rows)} configs from {manifest_path}")

    results = []
    for row in rows:
        tag = row["tag"]
        config_path = row["config_path"]

        train_ok = True
        if not skip_train:
            train_ok = run_step(
                [python_exe, "-m", "src.train", "--config", config_path],
                tag, "src.train",
                log_path=os.path.join(log_dir, f"{tag}_train.log"),
            )

        predict_ok = True
        if not skip_predict:
            if train_ok:
                predict_ok = run_step(
                    [python_exe, "-m", "src.predict", "--config", config_path],
                    tag, "src.predict",
                    log_path=os.path.join(log_dir, f"{tag}_predict.log"),
                )
            else:
                print(f"⏭ [{tag}] Skipping src.predict — training failed.")
                predict_ok = False

        results.append({
            "tag": tag, "config_path": config_path,
            "train_ok": train_ok, "predict_ok": predict_ok,
        })

    print(f"\n{'=' * 90}\nSweep summary\n{'=' * 90}")
    for r in results:
        status = "✅" if (r["train_ok"] and r["predict_ok"]) else "❌"
        print(f"{status} {r['tag']}  (train={r['train_ok']}, predict={r['predict_ok']})")

    n_failed = sum(1 for r in results if not (r["train_ok"] and r["predict_ok"]))
    if n_failed:
        print(f"\n⚠ {n_failed}/{len(results)} configs had a failure — "
              f"check the logs in {log_dir} before trusting the heatmap.")

    return results


def build_heatmap(csv_path="plots/heatmaps/depth_skill_binned_all_runs.csv",
                   region=None, checkpoint_region=None, all_pairs=False, cross_only=False):
    from src.plot_depth_sweep_heatmap import build_depth_sweep_report, build_all_pairs_reports
    if all_pairs or cross_only:
        build_all_pairs_reports(csv_path, cross_only=cross_only)
    else:
        build_depth_sweep_report(csv_path, region=region, checkpoint_region=checkpoint_region)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                         help="Manifest CSV from generate_depth_sweep_configs.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--heatmap-only", action="store_true",
                         help="Skip running the sweep; just (re)build the heatmap "
                              "from whatever is already in the depth-skill CSV.")
    parser.add_argument("--region", default=None,
                         help="Filter the heatmap to one test region (recommended "
                              "if your sweep covers more than one region).")
    parser.add_argument("--checkpoint-region", default=None,
                         help="Filter the heatmap to one checkpoint/training region.")
    parser.add_argument("--all-pairs", action="store_true",
                         help="Build one heatmap report per (checkpoint region, test "
                              "region) pair found in the depth-skill CSV, instead of a "
                              "single --region/--checkpoint-region combination.")
    parser.add_argument("--cross-only", action="store_true",
                         help="Like --all-pairs, but skip the pairs where a model is "
                              "tested on its own training region.")
    args = parser.parse_args()

    if not args.heatmap_only:
        run_sweep(
            args.manifest, python_exe=args.python,
            skip_train=args.skip_train, skip_predict=args.skip_predict,
        )

    build_heatmap(region=args.region, checkpoint_region=args.checkpoint_region,
                  all_pairs=args.all_pairs, cross_only=args.cross_only)


if __name__ == "__main__":
    main()
