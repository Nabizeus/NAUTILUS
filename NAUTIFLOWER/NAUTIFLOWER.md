# NAUTIFLOWER

Flower models (Muser et al. 2026, *Flowers: A Warp Drive for Neural PDE Solvers*, arXiv:2603.04430) on the NAUTILUS ocean tracer problem. Their skill and speed can be compared directly with the NAUTILUS models.

- This directory is a git clone of `~/NAUTILUS/ACC_NAUTILUS/ACC_FULL_COLUMN_v2` (commit `7b07004`), on branch `nautiflower`. Data, boxes, split, normalization, loss, training, `evaluate_global.py`, `predict.py` and `benchmark_inference.py` are the NAUTILUS ones. Only the model differs.
- To take over NAUTILUS fixes: `git pull origin master` (or `git cherry-pick <commit>`).

## Models (`model.arch`)

| arch | model | config |
|---|---|---|
| `flower2d` | 2-D Flower U-Net. The whole column (75 levels × 6 variables × 4 input times) is the channels, as in v6 E. The warp moves values in (y, x). 75 output channels. | `config/nautiflower_2d_k5.yaml` |
| `flower3d` | 3-D Flower U-Net over (z, y, x), 24 input channels. The warp moves values horizontally and vertically. 1 output channel. | `config/nautiflower_3d_k5.yaml` |

Both configs are `nautilus_v6_k5_col256.yaml` (v6 E: k = 5, snapshot forcing, 4 training boxes), with only the model block and the names changed. There are `*_smoketest.yaml` versions with 2 epochs.

- `src/flower_standalone.py`: the Flower code from `~/Codex/flowers`, unchanged.
- `src/model_flower.py`: the NAUTILUS adapter. Compared with the original it:
    - works on any grid size (padded with zeros to a multiple of 2^(n_levels-1), then cropped back);
    - measures the warp displacement in grid cells, so it means the same distance on 48×48 patches, 78×78 boxes and global tiles;
    - adds no coordinate channels;
    - does not use `torch.compile` for `grid_sample`, which fails with icpc on MN5.
    - with `model.mask_warp: true` (needs `data.mask_channel: true`, set in all NAUTIFLOWER configs), uses the ocean mask in every warp. The mask channel is taken off the input, so the network sees the same 6 variables as the other models. Warps sample ocean values only: values are set to 0 outside the ocean, interpolated, then divided by the interpolated ocean share. Warped features and the output are 0 on land and below the sea floor. `flower3d` uses the 3-D mask; `flower2d` uses the surface mask in the warps and the 3-D mask on the output.

## Running

```bash
CONFIG=config/nautiflower_2d_k5_smoketest.yaml sbatch --qos=acc_debug --time=02:00:00 sbatch_v2_train_eval.job
CONFIG=config/nautiflower_2d_k5.yaml sbatch sbatch_v2_train_eval.job
```

Results are also copied to `/gpfs/scratch/bsc32/bsc032196/nautilus/NAUTIFLOWER/<experiment>/`. For interactive checks, run `source env_nautiflower.sh` (the modules of the job). Flower needs `einops`, which is installed in `~/python_extra`.
