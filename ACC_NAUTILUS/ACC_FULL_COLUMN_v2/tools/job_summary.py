#!/usr/bin/env python3
"""Summary of a finished NAUTILUS / NAUTIFLOWER training job.

    python3 tools/job_summary.py --job 48068037

Reads the job's Slurm state, its job output and the evaluation files and writes
the summary to ~/NAUTILUS/mail_outbox/summary_<job>.txt (first line: subject).
The laptop script work_mail/laptop_mail.py runs it over ssh, makes a PDF and
e-mails it. Standard library only (system python3).
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys

HOME = os.path.expanduser("~")
REPOS = {
    "NAUTILUS": os.path.join(HOME, "NAUTILUS/ACC_NAUTILUS/ACC_FULL_COLUMN_v2"),
    "NAUTIFLOWER": os.path.join(HOME, "NAUTILUS/NAUTIFLOWER"),
}
OUTBOX = os.path.join(HOME, "NAUTILUS/mail_outbox")
BUDGET = 250  # PISCES share, CPU core-hours per simulated year

# Models compared in every summary. CPU cost: core-hours per simulated year,
# 24 tracers, k = 5, best layout (v7 CPU coupling benchmark, 2026-10-28; the
# reference and A-C from the v6 tiled benchmark, which overstates by ~1.7x).
MODELS = [
    ("Reference", "NAUTILUS", "config/nautilus_v5_k5_snap.yaml", "~29000", "-"),
    ("A", "NAUTILUS", "config/nautilus_v6_k5_h64.yaml", "~10800", "-"),
    ("B", "NAUTILUS", "config/nautilus_v6_k5_ds128.yaml", "~24900", "-"),
    ("C", "NAUTILUS", "config/nautilus_v6_k5_tc64.yaml", "~2100", "-"),
    ("E", "NAUTILUS", "config/nautilus_v6_k5_col256.yaml", "151", "65"),
    ("G128", "NAUTILUS", "config/nautilus_v7_k5_gnn128.yaml", "266", "85"),
    ("G64", "NAUTILUS", "config/nautilus_v7_k5_gnn64.yaml", "77", "31"),
    ("Flower 2-D", "NAUTIFLOWER", "config/nautiflower_2d_k5.yaml", "122", "83"),
    ("Flower 3-D", "NAUTIFLOWER", "config/nautiflower_3d_k5.yaml", "1695", "NaN"),
]


def yaml_value(path, key):
    """First `key: value` in a YAML file, as the job script reads it."""
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r'^\s*%s:\s*"?([^"#\s]+)' % key, line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def run(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def slurm_job(job):
    out = run(["sacct", "-X", "-n", "-P", "-j", str(job),
               "--format=JobName,State,Start,End,Elapsed,ExitCode,WorkDir"])
    line = out.strip().splitlines()[0] if out.strip() else ""
    keys = ["name", "state", "start", "end", "elapsed", "exit", "workdir"]
    info = dict(zip(keys, line.split("|"))) if line else {}
    info["state"] = info.get("state", "UNKNOWN").split()[0]
    return info


def tail(path, n):
    try:
        with open(path, errors="replace") as f:
            return [l.rstrip("\n") for l in f.readlines()[-n:]]
    except OSError:
        return []


def median(values):
    values = [v for v in values if v == v]
    return statistics.median(values) if values else None


def read_csv(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def eval_metrics(eval_dir):
    """Accuracy of one model from its global evaluation; None if not evaluated."""
    try:
        with open(os.path.join(eval_dir, "summary.json")) as f:
            s = json.load(f)
    except (OSError, ValueError):
        return None
    m = {"unseen": s.get("random_ratio_median"),
         "better": s.get("random_better_than_no_change"),
         "n_unseen": s.get("n_random")}
    regions = read_csv(os.path.join(eval_dir, "metrics_per_region.csv"))
    for kind in ("seen", "named", "training"):
        m[kind] = median([float(r["ratio"]) for r in regions if r["kind"] == kind])
    bands = {}
    for r in read_csv(os.path.join(eval_dir, "metrics_per_depthband.csv")):
        if r["kind"] == "random":
            bands.setdefault(r["band"], []).append(float(r["ratio"]))
    m["bands"] = [(b, median(v), sum(x < 1 for x in v)) for b, v in bands.items()]
    try:
        with open(os.path.join(eval_dir, "rollout/rollout_summary.json")) as f:
            ro = json.load(f)["median_ratio"]["all"]["random"]
        m["roll30"], m["roll90"] = ro.get("30"), ro.get("90")
    except (OSError, ValueError, KeyError):
        m["roll30"] = m["roll90"] = None
    leads = read_csv(os.path.join(eval_dir, "rollout/rollout_per_lead.csv"))
    last = max((int(r["lead_steps"]) for r in leads), default=None)
    drift = [float(r["tracer_sum_error_pct"]) for r in leads
             if r["band"] == "all" and int(r["lead_steps"]) == last
             and r["tracer_sum_error_pct"] not in ("", "nan")]
    m["drift"] = (median(drift), max(map(abs, drift))) if drift else None
    return m


def fmt(x, nd=3):
    return "-" if x is None else ("%.*f" % (nd, x))


def eval_path(repo_dir, config_path, exp):
    """Evaluation folder of a config: its out_dir, else the same name in eval/ or the repo root."""
    out = yaml_value(config_path, "out_dir") or ("eval_global_%s" % exp)
    candidates = [out, os.path.join("eval", os.path.basename(out)), os.path.basename(out)]
    for c in candidates:
        if os.path.isdir(os.path.join(repo_dir, c)):
            return os.path.join(repo_dir, c)
    return os.path.join(repo_dir, out)


def model_dirs(repo, config):
    path = os.path.join(REPOS[repo], config)
    exp = yaml_value(path, "experiment_name")
    return exp, eval_path(REPOS[repo], path, exp)


def summarize(job):
    info = slurm_job(job)
    repo_dir = os.path.normpath(info.get("workdir") or "")
    if not os.path.isfile(os.path.join(repo_dir, "server/v2_%s.out" % job)):
        repo_dir = next((d for d in REPOS.values()
                         if os.path.isfile(os.path.join(d, "server/v2_%s.out" % job))), repo_dir)
    out_file = os.path.join(repo_dir, "server/v2_%s.out" % job)
    err_file = os.path.join(repo_dir, "server/v2_%s.err" % job)
    out_lines = tail(out_file, 100000)
    config = next((l.split("=", 1)[1] for l in out_lines if l.startswith("CONFIG=")), None)
    commit = next((l.split(":", 1)[1].strip() for l in out_lines if l.startswith("git commit:")), "?")
    copied = [l for l in out_lines if "copied results to" in l]
    exp, eval_dir, metrics, train = None, None, None, {}
    if config:
        exp = yaml_value(os.path.join(repo_dir, config), "experiment_name")
        eval_dir = eval_path(repo_dir, os.path.join(repo_dir, config), exp)
        metrics = eval_metrics(eval_dir)
        for p in glob.glob(os.path.join(repo_dir, "logs", exp or "-", "training_summary_*.json")):
            with open(p) as f:
                train = json.load(f)
    ref = eval_metrics(model_dirs("NAUTILUS", MODELS[0][2])[1])
    state = info["state"]

    subject = "[NAUTILUS] %s job %s %s" % (exp or info.get("name", "?"), job, state)
    if metrics and metrics["unseen"] is not None:
        subject += ": unseen %.3f (ref %s)" % (metrics["unseen"], fmt(ref and ref["unseen"]))
    L = [subject, ""]
    L.append("Job %s (%s), %s, exit %s" % (job, info.get("name", "?"), state, info.get("exit", "?")))
    L.append("Start %s, end %s, elapsed %s" % (info.get("start", "?"), info.get("end", "?"),
                                                 info.get("elapsed", "?")))
    L.append("Config %s, commit %s, repo %s" % (config or "?", commit, repo_dir))
    if train:
        L.append("Training: %s epochs, best %s, stop: %s, %.0f min, AMP skipped %s"
                 % (train.get("epochs_run"), train.get("best_epoch"), train.get("stop_reason"),
                    train.get("minutes", 0), train.get("skipped_steps")))
    L.append(copied[-1].strip("# ") if copied else
             "WARNING: no 'copied results to' line in the job output - check the results copy "
             "and the scratch/projects quota.")
    L.append("")

    if metrics:
        L.append("ACCURACY (median RMSE ratio model / 'no change', normalized; below 1 is better)")
        L.append("%-24s %8s %8s" % ("", exp[:8] if exp else "model", "ref"))
        rows = [("Unseen random boxes", "unseen"), ("Seen random boxes", "seen"),
                ("Named boxes", "named"), ("Training boxes", "training"),
                ("Roll-out 22.5 h, unseen", "roll30"), ("Roll-out 67.5 h, unseen", "roll90")]
        for label, k in rows:
            L.append("%-24s %8s %8s" % (label, fmt(metrics.get(k)), fmt(ref and ref.get(k))))
        L.append("Unseen boxes better than 'no change': %s of %s (ref %s)"
                 % (metrics["better"], metrics["n_unseen"], ref and ref["better"]))
        if metrics["drift"]:
            L.append("Tracer total after the roll-out, unseen and other boxes: median %+.3f %%, "
                     "largest %.3f %%" % metrics["drift"])
        L.append("")
        L.append("By depth, unseen boxes (ratio, boxes better than 'no change'):")
        ref_bands = {b: (r, n) for b, r, n in (ref["bands"] if ref else [])}
        for b, r, n in metrics["bands"]:
            rr = ref_bands.get(b, (None, None))
            L.append("  %-12s %s (%s)   ref %s (%s)" % (b, fmt(r), n, fmt(rr[0]), rr[1]))
        L.append("")
    elif state == "COMPLETED":
        L.append("No evaluation summary found in %s." % eval_dir)
        L.append("")

    if state != "COMPLETED":
        errors = [l.strip() for l in out_lines + tail(err_file, 100000)
                  if re.match(r"^\s*\w*(Error|Exception)\b.*:", l)]
        if errors:
            L.append("First error: %s" % errors[0])
        L.append("Last lines of the error output (%s):" % err_file)
        L += ["  " + l for l in tail(err_file, 20)] or ["  (empty)"]
        L.append("Last lines of the job output:")
        L += ["  " + l for l in out_lines[-10:]] or ["  (empty)"]
        L.append("")

    L.append("ALL MODELS (normalized ratio, unseen boxes; CPU core-h per simulated year, "
             "budget %d)" % BUDGET)
    L.append("%-11s %-14s %7s %7s %8s %7s %6s" % ("Model", "Experiment", "Unseen", "Better",
                                                    "Roll67h", "CPUfp32", "bf16"))
    for label, repo, cfg, c32, c16 in MODELS:
        e, d = model_dirs(repo, cfg)
        m = eval_metrics(d)
        mark = " <-" if e == exp else ""
        L.append("%-11s %-14s %7s %7s %8s %7s %6s%s"
                 % (label, e, fmt(m and m["unseen"]), m["better"] if m else "-",
                    fmt(m and m["roll90"]), c32, c16, mark))
    L.append("('-': not trained or not evaluated yet.)")
    L.append("")

    L.append("OTHER WATCHED JOBS")
    for w in watched_jobs():
        if w != str(job):
            i = slurm_job(w)
            L.append("  %s %-14s %s %s" % (w, i.get("name", "?"), i["state"], i.get("elapsed", "")))
    L.append("")
    L.append("Evaluation: %s" % eval_dir)
    L.append("Written by tools/job_summary.py on %s." % os.uname()[1])
    return "\n".join(L) + "\n"


def watched_jobs():
    try:
        with open(os.path.join(OUTBOX, "watched_jobs.txt")) as f:
            return [l.split()[0] for l in f if l.strip() and not l.startswith("#")]
    except OSError:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    a = p.parse_args()
    os.makedirs(OUTBOX, exist_ok=True)
    path = os.path.join(OUTBOX, "summary_%s.txt" % a.job)
    with open(path, "w") as f:
        f.write(summarize(a.job))
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
