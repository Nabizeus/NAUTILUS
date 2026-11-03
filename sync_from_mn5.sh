#!/usr/bin/env bash
# Copy the git-worthy part of the MareNostrum 5 NAUTILUS work into this repo:
#   ACC_NAUTILUS/ACC_FULL_COLUMN_v2  <- mn5:NAUTILUS/ACC_NAUTILUS/ACC_FULL_COLUMN_v2  (current work)
#   ACC_NAUTILUS/ACC_FULL_COLUMN     <- mn5:NAUTILUS/ACC_NAUTILUS/ACC_FULL_COLUMN     (code only)
#   ACC_NAUTILUS/ACC_PARTIAL_COLUMN  <- mn5:NAUTILUS/ACC_NAUTILUS/ACC_PARTIAL_COLUMN  (code only)
#   FTORCH                           <- mn5:NAUTILUS/FTORCH
#   NAUTIFLOWER                      <- mn5:NAUTILUS/NAUTIFLOWER
# Code, configs, job scripts, reports (Markdown + figures) and small per-run results
# (CSV/JSON scores, training histories, loss curves). Never: checkpoints, predictions,
# exported models, binary data, build folders, anything over 10 MB.
# The filters are the same as the GitLab mirrors in ~/GitLab/ai4pisces/TRANSPORT/NAUTILUS.
#
# Usage:  ./sync_from_mn5.sh            (then: git add -A ACC_NAUTILUS FTORCH NAUTIFLOWER && git commit)
#         ./sync_from_mn5.sh --dry-run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
R="rsync -a --prune-empty-dirs --max-size=10m --exclude=.git/ --exclude=__pycache__/ $*"
EVAL_V2=(--include='/eval/' --include='/eval/eval_global*/' --include='/eval/eval_global*/*.csv' --include='/eval/eval_global*/*.json'
         --include='/eval/eval_global*/rollout/' --include='/eval/eval_global*/rollout/*.csv' --include='/eval/eval_global*/rollout/*.json' --include='/eval/eval_global*/rollout/*.png')

mkdir -p "$ROOT/ACC_NAUTILUS/ACC_FULL_COLUMN_v2" "$ROOT/ACC_NAUTILUS/ACC_FULL_COLUMN" "$ROOT/ACC_NAUTILUS/ACC_PARTIAL_COLUMN" "$ROOT/FTORCH" "$ROOT/NAUTIFLOWER"

echo "== ACC_FULL_COLUMN_v2"
$R --exclude='*smoketest*' \
  --include='/src/' --include='/src/*.py' --include='/config/' --include='/config/*.yaml' --include='/sbatch_*.job' \
  --include='/tools/' --include='/tools/*.py' \
  --include='/docs/' --include='/docs/*.md' --include='/docs/nautilus_*_figures/' --include='/docs/nautilus_*_figures/*.png' \
  "${EVAL_V2[@]}" \
  --include='/artifacts/' --include='/artifacts/v*/' --include='/artifacts/v*/*.png' \
  --include='/logs/' --include='/logs/benchmark_inference_*.json' --include='/logs/bench_*.json' --include='/logs/benchcc_*.json' \
  --include='/logs/v*/' --include='/logs/v*/*.csv' --include='/logs/v*/*.json' \
  --exclude='*' mn5:NAUTILUS/ACC_NAUTILUS/ACC_FULL_COLUMN_v2/ "$ROOT/ACC_NAUTILUS/ACC_FULL_COLUMN_v2/"

for d in ACC_FULL_COLUMN ACC_PARTIAL_COLUMN; do
  echo "== $d (code only)"
  $R --include='/src/' --include='/src/*.py' --include='/config/' --include='/config/*.yaml' \
    --include='/configs/' --include='/configs/*.yaml' --include='/*.py' --include='/*.job' --include='/*.md' \
    --exclude='*' mn5:NAUTILUS/ACC_NAUTILUS/$d/ "$ROOT/ACC_NAUTILUS/$d/"
done

echo "== FTORCH"
$R --no-links --exclude='.orig_*/' --exclude='build/' --exclude='*.bin' --exclude='*.pt' \
  --include='/*.py' --include='/*.F90' --include='/*.sh' --include='/*.job' --include='/CMakeLists.txt' --include='/*.md' \
  --include='/logs/***' --include='/artifacts/' --include='/artifacts/*.png' \
  --include='/ftorch_export/' --include='/ftorch_export/*.json' --include='/realdata/' --include='/realdata/*.json' \
  --include='/crosscheck/' --include='/crosscheck/*.json' --include='/runs/' --include='/runs/*/' \
  --include='/runs/*/artifacts/' --include='/runs/*/artifacts/*.png' --include='/runs/*/*/' --include='/runs/*/*/*.json' \
  --exclude='*' mn5:NAUTILUS/FTORCH/ "$ROOT/FTORCH/"

echo "== NAUTIFLOWER"
$R --exclude='*smoketest*.pt' \
  --include='/src/' --include='/src/*.py' --include='/config/' --include='/config/nautiflower_*.yaml' \
  --include='/sbatch_*.job' --include='/env_nautiflower.sh' --include='/NAUTIFLOWER.md' \
  --include='/docs/' --include='/docs/NAUTIFLOWER_*.md' --include='/docs/nautiflower_*_figures/' --include='/docs/nautiflower_*_figures/*.png' \
  --include='/eval/' --include='/eval/eval_global_nf*/' --include='/eval/eval_global_nf*/*.csv' --include='/eval/eval_global_nf*/*.json' \
  --include='/eval/eval_global_nf*/rollout/' --include='/eval/eval_global_nf*/rollout/*.csv' --include='/eval/eval_global_nf*/rollout/*.json' --include='/eval/eval_global_nf*/rollout/*.png' \
  --include='/artifacts/' --include='/artifacts/nf*/' --include='/artifacts/nf*/*.png' \
  --include='/logs/' --include='/logs/nf*/' --include='/logs/nf*/*.csv' --include='/logs/nf*/*.json' \
  --exclude='*' mn5:NAUTILUS/NAUTIFLOWER/ "$ROOT/NAUTIFLOWER/"

du -sh "$ROOT/ACC_NAUTILUS" "$ROOT/FTORCH" "$ROOT/NAUTIFLOWER"
