"""
depth_skill_profile.py

Turns predict.py's existing per-timestep, per-depth-level `histories`
accumulator (histories[z]["rmse"/"mae"/"r2"/"ssim"/...] = one value per
test timestep) into:

  1) a per-z-level long-format skill table — one row per depth level
     per metric, averaged over the WHOLE test set (not one sample), and
  2) a depth-BINNED skill table — grouping z levels into named ranges
     (e.g. "0-500m"), which is the granularity the future
     depth-training-range x depth-bin heatmap will actually plot.

Both are appended to a running CSV across separate predict.py
invocations (identified by run_name/region/checkpoint_region), so a
depth-budget or banded-window training sweep can be evaluated model by
model — each run just appends its rows — and only assembled into the
comparison heatmap once every model in the sweep has been evaluated.

Nothing here re-computes metrics: it only aggregates values predict.py
has already computed per z per timestep, using the same land-masking
predict.py already applied when building `histories`.
"""

import os
import time
import numpy as np
import pandas as pd

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:  # not available on Windows; sweep is HPC/Linux-only anyway
    _HAS_FCNTL = False

METRICS = ["rmse", "mae", "r2", "ssim", "rmse_pct", "mae_pct"]


def summarize_depth_skill_per_z(
    histories,
    dataset_depths,
    valid_z,
    run_name,
    region_name,
    checkpoint_region=None,
    train_depth_mode=None,
    train_depth_min=None,
    train_depth_max=None,
    train_depth_indices=None,
    metrics=METRICS,
    low_variance_threshold=1e-20,
):
    """
    histories: dict {z: {"rmse": [...], "mae": [...], ...}}, exactly as
        already built in predict.py's test loop (one value appended per
        test timestep for that z level).
    dataset_depths: dataset.depths (full vertical grid, meters)
    valid_z: dataset.valid_z (the z indices actually predicted)

    low_variance_threshold: a depth level is flagged `low_variance=True`
        when the mean true-value variance across test timesteps
        (predict.py's `den`, the R2 denominator / SS_tot) falls at or
        below this threshold. When true values are essentially
        constant (commonly ~0, e.g. a tracer that hasn't reached that
        depth), R2 is undefined by construction (see predict.py's
        `if den > 0: ... else: _r2 = np.nan`), and RMSE/MAE/SSIM
        become trivially "perfect" without the model having learned
        anything — Pred and True both collapse to the same constant.
        Flagged rows should generally be excluded from generalization
        scoring, not celebrated as good skill.

    Returns a long-format DataFrame: one row per (z, metric), averaged
    over every test timestep with NaNs (e.g. degenerate/all-land
    samples) dropped rather than propagated.
    """
    rows = []
    for z in valid_z:
        depth_m = float(dataset_depths[z])
        h = histories.get(z, {})

        variance_values = np.asarray(h.get("true_variance", []), dtype=float)
        variance_values = variance_values[np.isfinite(variance_values)]
        mean_true_variance = (
            float(np.mean(variance_values)) if variance_values.size > 0 else float("nan")
        )
        low_variance = bool(
            np.isfinite(mean_true_variance) and mean_true_variance <= low_variance_threshold
        )

        for metric in metrics:
            values = np.asarray(h.get(metric, []), dtype=float)
            values = values[np.isfinite(values)]
            n_valid = int(values.size)

            mean_val = float(np.mean(values)) if n_valid > 0 else float("nan")
            std_val = float(np.std(values)) if n_valid > 0 else float("nan")

            rows.append({
                "run_name": run_name,
                "region": region_name,
                "checkpoint_region": checkpoint_region,
                "train_depth_mode": train_depth_mode,
                "train_depth_min_m": train_depth_min,
                "train_depth_max_m": train_depth_max,
                "train_depth_indices": (
                    ",".join(map(str, train_depth_indices))
                    if train_depth_indices else None
                ),
                "z": int(z),
                "depth_m": depth_m,
                "metric": metric,
                "value": mean_val,
                "std": std_val,
                "n_timesteps": n_valid,
                "mean_true_variance": mean_true_variance,
                "low_variance": low_variance,
            })

    return pd.DataFrame(rows)


def bin_depth_skill(df_per_z, depth_bins, bin_labels=None):
    """
    Collapse a per-z long-format table (from summarize_depth_skill_per_z)
    into named depth bins, e.g. [(0,500), (500,2000), (2000,5000)].

    Values are averaged across the z levels falling in each bin,
    weighted by each level's n_timesteps (levels with fewer valid
    timesteps count less). Returns a long-format DataFrame with a
    "depth_bin" column instead of "z"/"depth_m".

    A bin's "low_variance" flag is True only if EVERY z level
    contributing to it was individually flagged — i.e. the whole bin's
    signal is trivial, not just part of it.
    """
    if df_per_z.empty:
        return df_per_z.copy()

    if bin_labels is None:
        bin_labels = [f"{lo:.0f}-{hi:.0f}m" for lo, hi in depth_bins]

    group_cols = [
        c for c in df_per_z.columns
        if c not in ("z", "depth_m", "value", "std", "n_timesteps",
                     "mean_true_variance", "low_variance")
    ]

    rows = []
    for (lo, hi), label in zip(depth_bins, bin_labels):
        mask = (df_per_z["depth_m"] >= lo) & (df_per_z["depth_m"] < hi)
        sub = df_per_z[mask]
        if sub.empty:
            continue

        for keys, group in sub.groupby(group_cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            weights = group["n_timesteps"].clip(lower=1)

            finite = np.isfinite(group["value"].values)
            if not finite.any():
                weighted_mean = float("nan")
            else:
                weighted_mean = float(np.average(
                    group["value"].values[finite], weights=weights.values[finite]
                ))

            var_finite = np.isfinite(group["mean_true_variance"].values)
            if var_finite.any():
                bin_true_variance = float(np.average(
                    group["mean_true_variance"].values[var_finite],
                    weights=weights.values[var_finite],
                ))
            else:
                bin_true_variance = float("nan")

            row = dict(zip(group_cols, keys))
            row["depth_bin"] = label
            row["depth_bin_min_m"] = lo
            row["depth_bin_max_m"] = hi
            row["value"] = weighted_mean
            row["n_z_levels"] = int(len(group))
            row["n_timesteps_total"] = int(group["n_timesteps"].sum())
            row["mean_true_variance"] = bin_true_variance
            row["low_variance"] = bool(group["low_variance"].all())
            rows.append(row)

    return pd.DataFrame(rows)


def estimate_mean_mld_m(dataset, z_index=0, land_mask=None):
    """
    Recover the average mixed-layer depth (meters) from
    `dataset.mld_distance`, without needing data.py to store the raw
    MLD field separately.

    data.py computes (verified against src/data.py's actual formula):
        mld_distance[t, z, y, x] = depth[z] - mld[t, y, x]
    so, for ANY fixed z:
        mld[t, y, x] = depth[z] - mld_distance[t, z, y, x]

    We use z_index=0 (shallowest level) by default and average over
    time/space (ignoring non-finite / masked-out points) to get one
    representative number for a region.
    """
    if getattr(dataset, "mld_distance", None) is None:
        raise AttributeError(
            "dataset.mld_distance is not set — this dataset wasn't built with "
            "forcing.use_mld_distance: true, so there's no MLD field to recover."
        )

    depth_z = float(dataset.depths[dataset.valid_z[z_index]])
    mld_field = depth_z - dataset.mld_distance[:, z_index]  # (T, H, W)

    if land_mask is not None:
        mld_field = np.where(land_mask[None, :, :] > 0, mld_field, np.nan)

    mld_field = np.asarray(mld_field)
    finite = np.isfinite(mld_field)
    if not finite.any():
        raise ValueError("No finite MLD values found — check land masking / z_index.")

    return float(np.mean(mld_field[finite]))


def append_to_master_csv(df, csv_path, dedupe_keys=None, lock_timeout=120):
    """
    Append `df` to a running CSV (creating it if needed). If
    `dedupe_keys` is given, existing rows matching the same keys as the
    incoming rows are dropped first — so re-running predict.py for the
    same run_name/region/... overwrites stale results instead of
    duplicating them across repeated runs.

    File-locked (POSIX flock on a sidecar .lock file) because a SLURM
    job array runs multiple configs' predict.py concurrently against
    this SAME shared CSV — without a lock, two tasks doing this
    read-modify-write at once can silently drop each other's rows.
    """
    if df.empty:
        print(f"⚠ Nothing to append to {csv_path} (empty DataFrame)")
        return csv_path

    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    lock_path = csv_path + ".lock"
    lock_file = open(lock_path, "a")

    if _HAS_FCNTL:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > lock_timeout:
                    raise TimeoutError(
                        f"Could not acquire lock on {lock_path} after "
                        f"{lock_timeout}s — another process may be stuck."
                    )
                time.sleep(0.5)

    try:
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            if dedupe_keys and not existing.empty:
                key_tuples = set(map(tuple, df[dedupe_keys].astype(str).values))
                existing_keys = existing[dedupe_keys].astype(str).apply(tuple, axis=1)
                existing = existing[~existing_keys.isin(key_tuples)]
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df

        combined.to_csv(csv_path, index=False)
        print(f"📐 Depth-skill table updated → {csv_path} (+{len(df)} rows, {len(combined)} total)")
    finally:
        if _HAS_FCNTL:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    return csv_path
