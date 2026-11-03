# Running the trained NAUTILUS model from Fortran: a beginner's guide to the FTorch pipeline

Written 8 October 2026 from the files in `~/NAUTILUS/FTORCH` on MareNostrum 5 (MN5) and from successful runs of the pipeline that day (Slurm jobs 47076062, 47076764 and, with the real-data step, 47077991).
Updated 10 October 2026 for the layout introduced on 9 October: the job now takes the model to test at submit time (`SOURCE`, `REGION`, `TAG`), writes into `runs/`, `logs/` and `server/`, and has a timing step 7. Example numbers in the new parts come from jobs 47100670–47100673.
It is a companion to `NAUTILUS_pipeline_guide.md`, which explains how the model is trained. This guide starts where that one ends: you already have a trained model and want Fortran code to use it.

**How to read this guide.** Sections 1–3 explain the idea and the words. Section 4 lists the files. Sections 5–7 are the three things you actually do: install once, run the pipeline, read the result. Section 8 explains each pipeline step. Section 9 shows how to call the model from your own Fortran program. Sections 10–12 cover re-running, troubleshooting and known limits. A glossary is at the end.

---

## 1. What this pipeline does, in one paragraph

NAUTILUS is a neural network trained in Python (with the PyTorch library) to predict how an ocean tracer moves. The ocean model NEMO is written in Fortran. Fortran cannot read a Python model directly. This pipeline converts the trained model into a file Fortran can load, builds a small Fortran library that loads it and runs it, and then proves that Fortran and Python give the same numbers for the same input.

```
   Python world (training)                      Fortran world (ocean model)
   -----------------------                      ---------------------------
   trained weights  (checkpoint .pt)
   normalization numbers (.npz)
            |
            |  STEP 2: export
            v
   TorchScript model (.pt)  ------------------>  nautilus_ftorch_interface.F90
   metadata (.json)         ------------------>    load_model()
                                                   push_timestep()  x 4
                                                   forward()  -> predicted tracer
```

Nothing is retrained here. The model's knowledge is unchanged; only its file format and the language calling it change.

---

## 2. Words you need first

| Word | Plain meaning |
|---|---|
| **Pre-trained model** | A neural network whose internal numbers ("weights") have already been learned. Using it is called **inference**; it is much cheaper than training. |
| **Checkpoint** | The file training saves, holding the weights. Here: `checkpoints/ocean_transport_fullcolumn_North_Atlantic_best_model.pt`. Only Python can use it, because it needs the Python code that defines the network. |
| **TorchScript** | A self-contained version of the model: weights *and* the computation, in one file. Programs that are not Python can run it. |
| **Export** | Converting the checkpoint into a TorchScript file. |
| **libtorch** | The C++ engine underneath PyTorch. It does the actual arithmetic. |
| **FTorch** | A library that lets Fortran talk to libtorch. Fortran → FTorch → libtorch → model. |
| **json-fortran** | A library that lets Fortran read JSON text files. Used to read the metadata. |
| **Normalization** | Rescaling inputs to small numbers before they enter the network, exactly as was done in training. The network gives nonsense if this is skipped or done differently. |
| **Metadata** | A small text file saying what shape the input has and which numbers to use for normalization. |
| **Tensor** | An array of numbers with several dimensions. |
| **Module** (on MN5) | A software package you switch on with `module load <name>`. |
| **Slurm job** | A script you hand to the cluster's queue with `sbatch`; it runs on a compute node, not on the login node. |

---

## 3. The whole pipeline at a glance

```
 ONE-TIME SETUP (login node glogin4, a few minutes)
   bash install_ftorch_deps.sh
        builds json-fortran 9.0.2 and FTorch v1.0.0  ->  ~/nautilus_deps/
        writes ~/nautilus_deps/env_ftorch.sh  (paths the job needs)

 EVERY RUN (compute node, 10 to 12 minutes; most of it is step 6 loading data)
   sbatch sbatch_ftorch_pipeline.job                             (GPP model, North Atlantic)
   SOURCE=ACC REGION=South_Pacific sbatch sbatch_ftorch_pipeline.job     (another model)
        |
        |  the job makes  runs/<TAG>/  and  logs/<TAG>/   (TAG = SOURCE_REGION, e.g. ACC_South_Pacific)
        |  and links the training folder's src, config, checkpoints, norms into runs/<TAG>/
        |
        |  STEP 2  export_nautilus_model.py
        |          checkpoint + norms  ->  runs/<TAG>/ftorch_export/nautilus_<REGION>.pt
        |                                  runs/<TAG>/ftorch_export/nautilus_<REGION>_metadata.json
        |
        |  STEP 3  make_dummy_crosscheck_input.py
        |          random test input -> run in Python -> save input and Python's answer
        |
        |  STEP 4  cmake + make
        |          compile the Fortran interface and four test programs
        |
        |  STEP 5a test_nautilus_crosscheck  + compare_outputs.py
        |          run the same test input in Fortran, compare with Python  -> PASS / FAIL
        |
        |  STEP 5b test_nautilus_dummy_driver
        |          use the interface the way an ocean model would           -> PASS / FAIL
        |
        |  STEP 6  make_realdata_crosscheck.py + test_nautilus_realdata + compare_realdata.py
        |          feed REAL NEMO fields through Fortran, compare with Python  -> PASS / FAIL
        |
        |  STEP 7  test_nautilus_benchmark + benchmark_python.py
        |          time one forward pass in Fortran and in Python           -> a table, no PASS / FAIL
        v
   logs/<TAG>/crosscheck_compare.log, realdata_compare.log   <- the lines that tell you whether it worked
   logs/<TAG>/benchmark_summary.log                          <- the timing table
```

Step 1 is training, which is not part of this pipeline; it must already have been done.

**Where things go.** Three folders in `~/NAUTILUS/FTORCH` collect everything a job writes:

| Folder | Contents |
|---|---|
| `server/` | Slurm's own `.out` and `.err` file of every job, whatever model it tested. |
| `logs/<TAG>/` | One log file per step, for that model. |
| `runs/<TAG>/` | That model's exported files, test data and compiled programs. |

Running the same `TAG` again overwrites its `logs/<TAG>/` and `runs/<TAG>/`. The files in `server/` are never overwritten, because their names carry the job number.

---

## 4. The files and folders

Everything lives in `~/NAUTILUS/FTORCH` on MN5 (full path `/gpfs/home/bsc/bsc032196/NAUTILUS/FTORCH`).

**Files you run**

| File | What it is |
|---|---|
| `install_ftorch_deps.sh` | One-time installer for FTorch and json-fortran. |
| `sbatch_ftorch_pipeline.job` | The Slurm job that runs steps 2–7. |
| `monitor_ftorch.sh` | Follows a running job live and prints a summary at the end. |

**Files the job uses**

| File | Language | Role |
|---|---|---|
| `export_nautilus_model.py` | Python | Step 2: checkpoint → TorchScript + metadata. |
| `make_dummy_crosscheck_input.py` | Python | Step 3: makes the test input and Python's reference answer. |
| `CMakeLists.txt` | CMake | Step 4: build instructions for the Fortran code. |
| `nautilus_ftorch_interface.F90` | Fortran | **The interface itself.** This is the part you keep and use in real work. |
| `test_nautilus_crosscheck.F90` | Fortran | Step 5a test program. |
| `compare_outputs.py` | Python | Step 5a: compares Fortran's answer with Python's. |
| `test_nautilus_dummy_driver.F90` | Fortran | Step 5b test program, and a usage example. |
| `make_realdata_crosscheck.py` | Python | Step 6: extracts real NEMO fields for one test-period sample and builds the Python reference. |
| `test_nautilus_realdata.F90` | Fortran | Step 6: feeds those raw fields through the interface. |
| `compare_realdata.py` | Python | Step 6: compares Fortran with Python and with the NEMO truth. |
| `test_nautilus_benchmark.F90` | Fortran | Step 7: times the forward pass through FTorch. |
| `benchmark_python.py` | Python | Step 7: times the same forward pass in Python. |

**The two training folders**

The job reads the trained model from one of two training folders, chosen with `SOURCE` when you submit:

| `SOURCE` | Training folder | Config file | What it holds |
|---|---|---|---|
| `GPP` (default) | `~/NAUTILUS/GPP_NAUTILUS/FULL_COLUMN` | `config/conv2dlstm_s323D_fullcolumn.yaml` | Models trained on the CPU partition. |
| `ACC` | `~/NAUTILUS/ACC_NAUTILUS/ACC_FULL_COLUMN` | `config/conv2dlstm_s323D_fullcolumn_GPU.yaml` | Models trained on the GPU partition. |

A model trained on GPUs needs nothing special: it is exported and run on CPUs here like any other.

**Folders the pipeline creates**

| Folder | Contents |
|---|---|
| `server/` | `ftorch_pipeline_<jobid>.out` and `.err` of every job. |
| `logs/<TAG>/` | The per-step log files of one model (listed in section 7). |
| `logs/earlier_runs/` | Per-step logs from before 9 October, when they were written into `FTORCH/` itself. They hold the last GPP run (job 47099593). |
| `runs/<TAG>/` | Everything else belonging to one model; see the next table. |
| `.orig_2026-10-08/`, `.orig_2026-10-08b/`, `.orig_2026-10-08c/`, `.orig_2026-10-09/` | Backups of files as they were before each round of changes. |

Inside each `runs/<TAG>/`:

| Entry | Contents |
|---|---|
| `ftorch_export/` | The two files Fortran needs: the TorchScript model (about 6 MB) and the metadata JSON. |
| `crosscheck/` | Test input, Python's answer, Fortran's answer. |
| `realdata/` | Step 6: raw NEMO fields, Python reference, Fortran results, NEMO truth. |
| `artifacts/` | A land-mask picture written by the training data loader in step 6. |
| `build/` | Compiled programs. Safe to delete; it is rebuilt. |
| `src`, `config`, `checkpoints`, `norms` | Symbolic links (shortcuts) into the training folder, made by the job. |

The four links are what the Python steps read:

| Link | Needed for |
|---|---|
| `src/` | The Python code that defines the network (needed to rebuild it before export). |
| `config/` | The config file: region box, number of time steps, variable names. |
| `checkpoints/` | The trained weights. |
| `norms/` | `norm_stats_<REGION>.npz`: the normalization numbers from training. |

Do not delete or move the training folders; the export step would stop working. Because these are links, a retrained checkpoint in the training folder is picked up automatically on the next run.

Run folders that exist on 10 October 2026: `ACC_North_Atlantic`, `ACC_South_Pacific`, `ACC_North_Pacific`, `ACC_Tropical_Pacific`. `runs/GPP_North_Atlantic` appears the first time the job is submitted without `SOURCE`.

**Leftovers in `FTORCH/` from before 9 October.** `ftorch_export/`, `crosscheck/`, `realdata/`, `artifacts/`, `build/` and the links `src`, `config`, `checkpoints`, `norms` directly in `FTORCH/` belong to the old layout. The job no longer reads or writes them. `ftorch_export/` there still holds the GPP North Atlantic model exported by job 47099593.

Outside this folder, the installer creates `~/nautilus_deps/` (`ftorch/`, `json-fortran/`, `src/`, `env_ftorch.sh`).

---

## 5. One-time setup: install FTorch and json-fortran

FTorch and json-fortran are not available as modules on MN5, so they are built from source once.

**Use login node glogin4.** The installer downloads source code from GitHub. glogin4 can reach GitHub; glogin1 cannot (the download hangs).

```bash
ssh <your_user>@glogin4.bsc.es
cd ~/NAUTILUS/FTORCH
bash install_ftorch_deps.sh
```

It takes a few minutes and ends with:

```
Installed:
  json-fortran 9.0.2 -> /home/bsc/<user>/nautilus_deps/json-fortran
  FTorch       v1.0.0 -> /home/bsc/<user>/nautilus_deps/ftorch
Environment file: /home/bsc/<user>/nautilus_deps/env_ftorch.sh
```

What it does, in order:

1. Loads the module `PyTorch/2.1.2-foss-2023b-Python-3.11.5`. This brings the compilers (GCC 13.2.0, including the Fortran compiler `gfortran`) and a copy of libtorch.
2. Loads `CMake/3.27.6-GCCcore-13.2.0`. The version is pinned on purpose: the default CMake module switches the compiler to GCC 14.3.0, which no longer matches libtorch.
3. Downloads and builds json-fortran.
4. Downloads and builds FTorch against the libtorch from step 1 (CPU only, no GPU).
5. Writes `env_ftorch.sh`, a short file listing where everything was installed.

One message in the log is harmless: `Cannot load module "hdf5/1.14.1-2"`. The installer does not need it.

**This is already done** for user `bsc032196`. Repeat it only if you are a different user, if `~/nautilus_deps` is deleted, or if the PyTorch module version is changed.

To install somewhere else:

```bash
PREFIX=/gpfs/projects/bsc32/$USER/deps bash install_ftorch_deps.sh
# and later submit the job with:
NAUTILUS_DEPS=/gpfs/projects/bsc32/$USER/deps sbatch sbatch_ftorch_pipeline.job
```

---

## 6. Run the pipeline

```bash
cd ~/NAUTILUS/FTORCH
sbatch sbatch_ftorch_pipeline.job
```

`sbatch` prints a job number, for example `Submitted batch job 47100670`.

With nothing else on the line, the job tests the GPP model for the North Atlantic. To test another model, put settings in front of `sbatch`:

```bash
SOURCE=ACC sbatch sbatch_ftorch_pipeline.job                        # ACC model, North Atlantic
SOURCE=ACC REGION=South_Pacific sbatch sbatch_ftorch_pipeline.job   # ACC model, South Pacific
```

**Choosing what to test**

| Setting | Default | Meaning |
|---|---|---|
| `SOURCE` | `GPP` | Which training folder the model comes from: `GPP` or `ACC` (section 4). |
| `REGION` | `North_Atlantic` | Region name exactly as in the config. Trained so far on ACC: `North_Atlantic`, `South_Pacific`, `North_Pacific`, `Tropical_Pacific`. |
| `CHECKPOINT` | `checkpoints/ocean_transport_fullcolumn_<REGION>_best_model.pt` | Another checkpoint file, given relative to the training folder. |
| `TAG` | `<SOURCE>_<REGION>` | Name of the run. It decides the folder names `runs/<TAG>/` and `logs/<TAG>/`. |
| `TRAIN_DIR`, `CONFIG` | set by `SOURCE` | A training folder and config other than the two built in. |

When you set `CHECKPOINT`, set `TAG` too, otherwise the run overwrites the results of the default checkpoint for that region. For example, for one of the depth-sweep checkpoints:

```bash
SOURCE=ACC REGION=North_Atlantic \
CONFIG=$HOME/NAUTILUS/ACC_NAUTILUS/ACC_FULL_COLUMN/configs/depth_sweep/acc_cumulative/depthbudget_0-1000m.yaml \
CHECKPOINT=checkpoints/depth_sweep/depthbudget_0-1000m_North_Atlantic_best_model.pt \
TAG=ACC_North_Atlantic_0-1000m \
sbatch sbatch_ftorch_pipeline.job
```

`CONFIG` is given here with its full path because the sweep configs are in `configs/`, not in the `config/` folder the job links. Giving the checkpoint its own config makes step 6 score the depth range that checkpoint was trained on. All 24 depth-sweep checkpoints were run this way on 13 October 2026 (jobs 47248483–47248506, listed in `server/depth_sweep_jobs_2026-10-13.txt`).

The first lines of the job's output repeat what was chosen, including the date of the checkpoint file, so you can confirm it picked the model you meant:

```
SOURCE     = ACC
REGION     = North_Atlantic
TRAIN_DIR  = /gpfs/home/bsc/bsc032196/NAUTILUS/ACC_NAUTILUS/ACC_FULL_COLUMN
CONFIG     = config/conv2dlstm_s323D_fullcolumn_GPU.yaml
CHECKPOINT = checkpoints/ocean_transport_fullcolumn_North_Atlantic_best_model.pt  (2026-09-23 04:34)
RUN_DIR    = /gpfs/home/bsc/bsc032196/NAUTILUS/FTORCH/runs/ACC_North_Atlantic
LOG_DIR    = /gpfs/home/bsc/bsc032196/NAUTILUS/FTORCH/logs/ACC_North_Atlantic
```

**Several models.** Submit one job per model. The `gp_debug` queue runs one job per user at a time, so they run one after another; the waiting ones show the reason `QOSMaxJobsPerUserLimit` in `squeue`. A training job of yours in `gp_debug` holds them back in the same way.

**Watching it live**

```bash
bash monitor_ftorch.sh            # follows the most recent job
bash monitor_ftorch.sh 47100670   # or a specific job
```

This shows the queue state while the job waits, prints the job's output as it is written, and ends with a summary (job state, run time, every PASS/FAIL line). Ctrl-C stops the watching only; the job keeps running.

Each step starts with a banner carrying the clock time, so you can see where the job is:

```
######## 09:24:35  STEP 2/7: export
######## 09:24:55  STEP 3/7: build dummy cross-check input + Python reference
######## 09:24:57  STEP 4/7: build Fortran interface + test programs
######## 09:25:34  STEP 5/7: run cross-check + dummy driver smoke test
######## 09:25:36  STEP 6/7: real NEMO data through the Fortran interface
######## 09:33:23  STEP 7/7: timing, Fortran+FTorch vs Python
######## 09:35:08  FTorch pipeline finished for ACC_North_Atlantic ...
```

Doing the same by hand:

```bash
squeue -u $USER                                 # waiting or running? Empty list = finished.
tail -f server/ftorch_pipeline_<jobid>.out      # live output (Ctrl-C to stop watching)
sacct -j <jobid> -X -o JobID,State,ExitCode,Elapsed    # final state
```

**More detail and shortcuts**

These go in front of `sbatch` like the settings above, and can be combined with them:

| Setting | Effect |
|---|---|
| `VERBOSE=1` | Also echoes every shell command the job runs. |
| `REALDATA_SAMPLE=3` | Uses another test-period sample in step 6 (0–25). |
| `SKIP_REALDATA=1` | Skips step 6. The job then takes 2 to 3 minutes. |
| `BENCH_REPS=20` | Number of timed runs in step 7 (default 10; the 1-thread runs use half). |

Step 6 goes quiet for about 7 minutes after the line `building the training data loader`. That is normal: it is reading a full year of NEMO data for the region.

The job asks for 16 cores for at most 1 hour in the `gp_debug` queue. The four ACC runs took between 9 minutes 44 seconds and 12 minutes 16 seconds; an earlier run without step 7 peaked at about 8 GB of memory. It sends e-mail to the address on the `--mail-user` line of the job file; change that line if you are not the original author.

Always submit from `~/NAUTILUS/FTORCH`. The job finds its scripts, and creates `runs/`, `logs/` and `server/`, in the folder it is submitted from.

---

## 7. Did it work? How to read the result

All the logs of one model are in `logs/<TAG>/`. The examples below use `ACC_North_Atlantic`; put your own tag in its place. The fastest check:

```bash
cat logs/ACC_North_Atlantic/crosscheck_compare.log
```

A successful run looks like this (numbers from job 47076764):

```
Reference  stats: min=-0.543204 max=0.537093 mean=-0.108283
Fortran    stats: min=-0.543204 max=0.537093 mean=-0.108283
Max abs diff: 2.384186e-07
Max rel diff: 9.213981e-04
PASS: Fortran output matches the Python-traced model within tolerance.
```

How to read it:

- **Reference** is Python's answer; **Fortran** is Fortran's answer to the same input. They should look identical.
- **Max abs diff** is the largest difference in any single number. About `2e-07` is the rounding noise of single-precision arithmetic; it means the two agree.
- **PASS / FAIL** applies the tolerance (absolute `1e-4`, relative `1e-3`).

Then check the second test:

```bash
tail -3 logs/ACC_North_Atlantic/dummy_driver.log
```

```
 PASS: forward() ran and returned a finite prediction.
 pred_tracer stats: min=  -8.02E-02  max=   7.88E-02  mean=   1.12E-03
```

These numbers change from run to run because the test feeds random inputs. Only the `PASS` matters.

Then the real-data test:

```bash
cat logs/ACC_North_Atlantic/realdata_compare.log
```

The example below is from job 47077991 (the 2-epoch GPP model of 8 October):

```
CHECK 1: normalized input tensor (Fortran push_timestep vs training data loader)
  channel 1 tracer: max abs diff = 0.000e+00   (values span 0.000 .. 1.007)   ok
  channel 2 u     : max abs diff = 1.907e-06   (values span -28.656 .. 10.585)   ok
  ...
CHECK 2: prediction (Fortran forward vs exported model in Python)
  max abs diff, normalized units = 8.794e-07   ok
INFO: skill against the NEMO truth ...
  RMSE of the Fortran prediction : 0.04432
  RMSE of 'no change' baseline   : 0.00011
PASS: Fortran reproduces the Python pipeline on real NEMO data.
```

- **CHECK 1** compares the normalized numbers Fortran builds from raw NEMO fields with those the training code builds. A `MISMATCH` here means normalization or land handling differs.
- **CHECK 2** compares the final predictions.
- **PASS / FAIL** covers checks 1 and 2 only: it says Fortran does what Python does.
- **INFO** is about the *model*, not the interface: how far the prediction is from what NEMO actually computed, next to the error of simply predicting "no change". The line above the two RMSE values says which depth levels were scored; they are the training depth range in the config. See section 12 for what the numbers mean.

Finally, the timing table. It has no PASS or FAIL; it is information:

```bash
cat logs/ACC_North_Atlantic/benchmark_summary.log
```

```
who                           threads   runs      mean_ms       min_ms       max_ms
Fortran+FTorch                      1      5      3850.67      3797.14      3898.80
Python-torch-2.1.2                  1      5      3839.89      3772.86      3888.35
Python-torch-2.3.0                  1      5      3774.94      3743.49      3818.26
Fortran+FTorch                     16     10       280.12       275.34       282.45
Python-torch-2.1.2                 16     10       272.17       265.08       278.62
Python-torch-2.3.0                 16     10       268.54       265.39       271.88
```

Each row is one forward pass of the model, in milliseconds. `Fortran+FTorch` is the Fortran interface. The two Python rows are the same model run from Python: once with the same libtorch version Fortran uses (2.1.2), once with the Python the training uses (2.3.0). Section 8 explains what is and is not timed.

To see the verdicts of all models at once:

```bash
grep -h -E "^ ?(PASS|FAIL)" logs/*/crosscheck_compare.log logs/*/dummy_driver.log logs/*/realdata_compare.log
```

**All log files**

The two Slurm files are in `server/`; all the others are in `logs/<TAG>/`.

| File | Written by | Look here when |
|---|---|---|
| `server/ftorch_pipeline_<jobid>.out` | whole job | You want everything in order. It also shows which model the job tested. |
| `server/ftorch_pipeline_<jobid>.err` | whole job | Something crashed. On a good run it holds only a few `load ...` lines from the module system. |
| `export_nautilus.log` | step 2 | Export failed. |
| `crosscheck_python.log` | step 3 | The Python reference could not be made. |
| `cmake_configure.log`, `cmake_build.log` | step 4 | The Fortran did not compile. |
| `crosscheck_fortran.log` | step 5a | The Fortran test crashed. |
| `crosscheck_compare.log` | step 5a | **PASS / FAIL of the main test.** |
| `dummy_driver.log` | step 5b | PASS / FAIL of the usage test. |
| `realdata_python.log` | step 6 | The real-data extraction failed, or you want to see which time steps were used. |
| `realdata_fortran.log` | step 6 | The Fortran real-data run crashed. |
| `realdata_compare.log` | step 6 | **PASS / FAIL of the real-data test**, and the skill numbers. |
| `benchmark.log` | step 7 | You want the time of every single run, or the timing step crashed. |
| `benchmark_summary.log` | step 7 | **The timing table.** |

`export_nautilus.log` contains three `TracerWarning` messages on every run. They are expected; see section 12.

---

## 8. What each step does and why

### Step 2: export (`export_nautilus_model.py`)

1. Reads the config to find the region box and sizes.
2. Rebuilds the empty network from `src/` and fills it with the weights from the checkpoint.
3. Wraps it so that the input has 4 dimensions instead of 6. Fewer dimensions crossing the Python/Fortran border means fewer ways to get the order wrong.
4. **Traces** it: runs one example input through and records every operation. The recording is the TorchScript file.
5. Writes the metadata JSON: shapes, the channel list with each channel's normalization numbers, and the depth in metres of each of the 75 levels (read from the NEMO tracer file).

The sizes for the North Atlantic model:

| Symbol | Meaning | Value |
|---|---|---|
| T | time steps the model looks back | 4 |
| C | input channels (variables) | 6 |
| Z | depth levels | 75 |
| H | grid points in y | 25 |
| W | grid points in x | 20 |

T, C and Z are the same for every region. H and W depend on the region's box:

| Region | Grid indices x (0-based, end not included) | Grid indices y | H × W |
|---|---|---|---|
| North Atlantic | 253–273 | 223–248 | 25 × 20 |
| South Pacific | 110–151 | 31–55 | 24 × 41 |
| North Pacific | 102–129 | 202–222 | 20 × 27 |
| Tropical Pacific | 152–166 | 124–160 | 36 × 14 |

The Fortran interface reads H and W from the metadata, so the same Fortran code serves every region.

The six channels, in order:

| # | Role | NEMO variable | Normalization |
|---|---|---|---|
| 1 | tracer | `mytrc` | per depth level: `(x - min[z]) / (max[z] - min[z])` |
| 2 | u (flow along x) | `umo` | `(x - mean) / std` |
| 3 | v (flow along y) | `vmo` | `(x - mean) / std` |
| 4 | w (vertical flow) | `wmo` | `(x - mean) / std` |
| 5 | dif (vertical mixing) | `difvso` | `(x - mean) / std` |
| 6 | mld (distance from the mixed layer) | from `mlotst` | `(x - mean) / std` |

Channel 6 is not a NEMO field as such. It is computed as `depth[z] - mixed_layer_depth[y,x]`: negative above the mixed layer, positive below. The Fortran interface computes it for you.

### Step 3: make the test input (`make_dummy_crosscheck_input.py`)

Creates one random input of the right shape, runs it through the exported model in Python, and saves both the input and the answer as raw binary files. This answer is the "truth" Fortran must reproduce.

### Step 4: build (`cmake` + `make`)

Compiles `nautilus_ftorch_interface.F90` into a library and links the four test programs against it, FTorch, json-fortran and libtorch. The job first does `source ~/nautilus_deps/env_ftorch.sh` so that CMake knows where those are. The Fortran source files stay in `FTORCH/`; the compiled programs go to `runs/<TAG>/build/`, so each model has its own copy.

### Step 5a: the cross-check

`test_nautilus_crosscheck` loads the model in Fortran, reads the same test input, runs it, and writes its answer. `compare_outputs.py` compares the two answers.

Why this matters: Python and Fortran store multi-dimensional arrays in opposite orders. If one dimension were swapped, the model would still run and still return plausible-looking numbers, only wrong ones. Comparing against Python is the only way to know the layout is right. This test deliberately skips normalization so that it checks exactly one thing.

### Step 5b: the dummy driver

`test_nautilus_dummy_driver` uses the interface the way a real ocean model would: load, feed four time steps, ask for a prediction. The fields are random numbers, so the prediction means nothing physically. It checks that the full sequence runs and returns finite numbers.

### Step 6: the real-data test

1. `make_realdata_crosscheck.py` builds the same data loader the training used, takes one sample from the **test period** (the last 1 % of the year, which the model never trained on), and saves the loader's normalized input. It also reads the *raw* NEMO values for the same four time steps straight from the NetCDF files, and runs the exported model in Python to get the reference prediction. By default it takes the middle test sample: time index 2541, inputs 21 March 1971 10:30–12:45, target 13:30.
2. `test_nautilus_realdata` gives the raw values to the Fortran interface exactly as an ocean model would (`push_timestep` four times, then `forward`).
3. `compare_realdata.py` compares the two, and scores the prediction against NEMO.

This is the test that covers what step 5 leaves out: normalization, the mixed-layer channel and land cells, on real numbers.

### Step 7: timing

Runs one forward pass of the exported model several times and reports the average time. The input is the random test input from step 3. It is done in three setups:

| Row in the table | What runs |
|---|---|
| `Fortran+FTorch` | `test_nautilus_benchmark`: the model called from Fortran, using libtorch 2.1.2. |
| `Python-torch-2.1.2` | `benchmark_python.py` with the PyTorch 2.1.2 module, the same libtorch Fortran uses. |
| `Python-torch-2.3.0` | `benchmark_python.py` with the Python the training uses. |

Each setup is timed twice: with 1 thread, which is what a single NEMO process would have, and with all 16 cores of the job. Two warm-up runs are thrown away before the timed ones.

Only the network itself is timed. Reading files, normalization and loading the model are left out, so the numbers say how long `forward` takes, not how long a coupled NEMO time step would take.

---

## 9. Using the model from your own Fortran code

The interface is a Fortran **module** called `nautilus_ftorch_interface`. You need five calls.

| Call | When | What it does |
|---|---|---|
| `call load_model(metadata_path)` | once, at start | Reads the metadata JSON and loads the TorchScript model named in it. |
| `call push_timestep(tracer, u, v, w, dif, mld, land_mask)` | once per time step | Normalizes the fields and stores them as the newest of the last 4 steps. |
| `history_ready()` | any time | `.true.` once 4 steps have been pushed. |
| `call forward(pred_tracer, ok)` | when you want a prediction | Runs the network on the stored 4 steps. Returns the predicted tracer in physical units. `ok` is `.false.` if fewer than 4 steps have been pushed. |
| `call cleanup_model()` | once, at end | Frees the model. |

**The model has no memory between calls.** It needs the last 4 time steps every time. The interface keeps those for you: each `push_timestep` drops the oldest step and adds the newest.

**Array shapes and types**

| Argument | Shape | Type | Notes |
|---|---|---|---|
| `tracer`, `u`, `v`, `w`, `dif` | `(Z, H, W)`; `(75, 25, 20)` for the North Atlantic | `real(wp)` (single precision) | Raw NEMO values, same variables and units as in training. No normalization by you. |
| `mld` | `(H, W)`; `(25, 20)` for the North Atlantic | `real(wp)` | Mixed layer depth in metres (`mlotst`). |
| `land_mask` | `(H, W)` | `logical` | `.true.` = ocean, `.false.` = land. |
| `pred_tracer` | `(Z, H, W)` | `real(wp)` | Output, physical tracer units. |

Three things to get right:

- **Index order is depth, then y, then x.** NEMO's own arrays are usually `(x, y, depth)`, so they must be reordered before being passed in.
- **Only the trained box.** The North Atlantic model was trained on grid indices x = 253–273, y = 223–248 (0-based, end not included, as in the config). Pass exactly that 25 × 20 cut-out. The boxes of the other regions are in section 8.
- **Values on land.** Land cells are overwritten by the interface to match training, so their input values do not matter. Predictions on land cells are not meaningful.

**Minimal example**

```fortran
program my_driver
    use nautilus_ftorch_interface
    implicit none

    real(wp), allocatable :: tracer(:,:,:), u(:,:,:), v(:,:,:), w(:,:,:), dif(:,:,:)
    real(wp), allocatable :: mld(:,:), pred(:,:,:)
    logical,  allocatable :: land_mask(:,:)
    logical :: ok
    integer :: step

    ! 1. Load once. After this, n_depths, grid_H, grid_W and T_steps are set.
    call load_model("runs/ACC_North_Atlantic/ftorch_export/nautilus_North_Atlantic_metadata.json")

    allocate(tracer(n_depths, grid_H, grid_W), u(n_depths, grid_H, grid_W), &
             v(n_depths, grid_H, grid_W), w(n_depths, grid_H, grid_W), &
             dif(n_depths, grid_H, grid_W), pred(n_depths, grid_H, grid_W))
    allocate(mld(grid_H, grid_W), land_mask(grid_H, grid_W))

    do step = 1, 100
        ! 2. Fill tracer, u, v, w, dif, mld, land_mask from your ocean model here.

        ! 3. Hand the newest time step to the interface.
        call push_timestep(tracer, u, v, w, dif, mld, land_mask)

        ! 4. Predict once 4 steps are stored.
        if (history_ready()) then
            call forward(pred, ok)
            ! pred now holds the tracer predicted for the next time step.
        end if
    end do

    ! 5. Clean up.
    call cleanup_model()
end program my_driver
```

The path given to `load_model` is the metadata file of the model you want, in that model's run folder. Written as above it is relative to `~/NAUTILUS/FTORCH`, so the program must be started from there; use the full path to start it from anywhere.

To build it, put `my_driver.F90` in `~/NAUTILUS/FTORCH`, add two lines to the end of `CMakeLists.txt` and re-run the job (or `cmake` and `make` by hand in an environment set up like the job's):

```cmake
add_executable(my_driver my_driver.F90)
target_link_libraries(my_driver PRIVATE nautilus_ftorch_interface)
```

The compiled program appears as `runs/<TAG>/build/my_driver`.

`test_nautilus_dummy_driver.F90` is a complete, working version of this pattern and a good file to copy from.

---

## 10. When to re-run what

"The job" below means `sbatch sbatch_ftorch_pipeline.job` with the same `SOURCE`, `REGION` and other settings as before.

| You changed… | Do this |
|---|---|
| Nothing, you just want to test again | The job. |
| Retrained the model (new checkpoint / norms in the training folder) | The job. The export step picks up the new files through the links. |
| The Fortran interface or test programs | The job, once for each model you want rebuilt. Each `runs/<TAG>/build/` is compiled separately. |
| The region | Train that region first, then submit with `REGION=<name>`. Each region gets its own exported model. |
| The number of time steps or depth levels | Retrain first, then the job. The exported model is fixed to one shape. |
| The PyTorch module version in the scripts | Re-run `bash install_ftorch_deps.sh` on glogin4, then the job. |
| `CMakeLists.txt`, and the build then behaves oddly | `rm -rf runs/<TAG>/build`, then the job. |
| You want to start one model from scratch | `rm -rf runs/<TAG> logs/<TAG>`, then the job. |

---

## 11. Troubleshooting

| What you see | Likely cause | Fix |
|---|---|---|
| `ERROR: .../env_ftorch.sh not found` | The one-time install was not done for this user or location. | Section 5. |
| Installer hangs or fails at `git clone` | You are on a login node without internet (for example glogin1). | Log in to glogin4 and run it there. |
| `ERROR: <folder>/<file> not found` right at the start of the job | The checkpoint, the norms file or the config does not exist in the training folder: training was never run for that region, or `SOURCE`, `REGION` or `CHECKPOINT` is misspelled. | Check the `SOURCE`, `REGION`, `TRAIN_DIR` and `CHECKPOINT` lines at the top of `server/ftorch_pipeline_<jobid>.out`, and `ls` the training folder's `checkpoints/` and `norms/`. |
| `ERROR: SOURCE=... is not GPP or ACC` | `SOURCE` has another value and `TRAIN_DIR` and `CONFIG` were not given. | Use `GPP` or `ACC`, or set both `TRAIN_DIR` and `CONFIG`. |
| `Region '...' not found in config.data.regions` | `REGION` does not match a region that is active in the YAML config. | Use a region name that is not commented out in the config. |
| The job stays pending with reason `QOSMaxJobsPerUserLimit` | Another job of yours is running in `gp_debug`, which allows one at a time. | Wait; it starts by itself when the other job ends. |
| No `.out` file appears and the job vanishes at once | The job was submitted from a folder without a `server/` folder, so Slurm could not write its output. | Submit from `~/NAUTILUS/FTORCH`. In a fresh copy of the pipeline, `mkdir server` first. |
| The results are from the wrong model | Two runs used the same `TAG`, and the later one overwrote the earlier. | Give each checkpoint its own `TAG`. |
| `ERROR: required key '...' missing from metadata` | The metadata JSON is from an older export or was edited by hand. | Re-run the job so step 2 rewrites it. |
| `WARNING: valid 'depths_m' not in metadata` then `ERROR: depths_m not set` | The export could not read the depth levels from the NEMO tracer file (file moved or unreadable). | Check the tracer path under `data.tracers.paths` in the config, then re-run. |
| `FAIL: Fortran output does NOT match` | The array layout in Fortran and the reshape in the export no longer agree (someone edited one of them), or libtorch was changed. | Compare the current files against the `.orig_*` backups; re-run the installer if the PyTorch module changed. |
| `error while loading shared libraries: libftorch.so` or `libtorch.so` when running a program by hand | The environment is not set up in that shell. | Load the same modules as the job, then `source ~/nautilus_deps/env_ftorch.sh`. |
| Compile errors mentioning a different `gfortran` version, or `.mod` files "created by a different version" | A different compiler got loaded (for example through an unpinned `module load CMake`). | Keep `CMake/3.27.6-GCCcore-13.2.0`; `gfortran --version` should say 13.2.0. Then `rm -rf runs/<TAG>/build`. |
| `NAUTILUS forward(): history buffer not yet full` | `forward` was called before 4 `push_timestep` calls. | Not an error. Check `history_ready()` first. |

---

## 12. What has and has not been verified

**Verified on MN5 on 8 October 2026**

- The installer builds both libraries with GCC 13.2.0.
- The whole job runs from `~/NAUTILUS/FTORCH` and finishes with all three tests passing.
- Fortran and Python agree on the model output to about `2e-07` for a random input.
- On real NEMO data (one test-period sample), the normalized inputs Fortran builds match the training data loader to `2e-06` or better in every channel, and the predictions match to `9e-07`.

**Verified on MN5 on 9 October 2026**

- **GPP, retrained with 4 epochs** (job 47099593, old layout; its logs are in `logs/earlier_runs/`): all tests passed. Cross-check difference `1.8e-07`, real-data prediction difference `3.1e-07`.
- **The four ACC models, trained on GPUs** (jobs 47100670–47100673, new layout): all tests passed, with exit code 0.

| Run | H × W | Cross-check, max difference | Real-data prediction, max difference | Verdict |
|---|---|---|---|---|
| `ACC_North_Atlantic` | 25 × 20 | 1.8e-07 | 4.2e-07 | PASS |
| `ACC_South_Pacific` | 24 × 41 | 1.8e-07 | 5.8e-07 | PASS |
| `ACC_North_Pacific` | 20 × 27 | 3.6e-07 | 2.1e-06 | PASS |
| `ACC_Tropical_Pacific` | 36 × 14 | 2.4e-07 | 2.2e-06 | PASS |

This shows that GPU-trained checkpoints load and run on CPUs through Fortran unchanged, and that one Fortran interface handles four different box sizes.

**Verified on MN5 on 13 October 2026**

- **The 24 ACC depth-sweep checkpoints** (four regions, trained on 0–1000 m up to 0–6000 m; jobs 47248483–47248506): all passed the cross-check, the smoke test and the real-data test. The largest Fortran-versus-Python differences were `4.2e-07` in the cross-check and `3.1e-06` in the real-data prediction. Their logs are in `logs/ACC_<Region>_0-<depth>m/`.
- **Skill:** on the one sample scored, every one of them is 47 to 450 times worse than "no change", like the models in the table below.

**Timing (step 7, 9 October 2026)**

One forward pass through the Fortran interface, mean of the timed runs:

| Run | 1 thread | 16 threads |
|---|---|---|
| `ACC_North_Atlantic` | 3.85 s | 0.28 s |
| `ACC_South_Pacific` | 7.55 s | 0.58 s |
| `ACC_North_Pacific` | 4.19 s | 0.30 s |
| `ACC_Tropical_Pacific` | 3.71 s | 0.27 s |

- **Fortran is as fast as Python, not faster.** In the North Atlantic table of section 7 the three setups are within 5 % of each other. Both spend their time in the same libtorch arithmetic; Fortran only changes who calls it.
- **Time grows with the box.** The South Pacific box has about twice the cells of the North Atlantic one and takes about twice as long.

**A finding about the models, not the interface**

On the one test sample scored in each run, every model predicts far worse than simply predicting "no change". RMSE is in normalized tracer units over ocean cells:

| Model | RMSE of the prediction | RMSE of "no change" | Levels scored |
|---|---|---|---|
| GPP North Atlantic, 2 epochs (8 October) | 0.044 | 0.00011 | 0–2000 m |
| GPP North Atlantic, 4 epochs (9 October) | 0.027 | 0.00010 | 0–6000 m (65 levels) |
| ACC North Atlantic | 0.049 | 0.00011 | 0–2000 m (54 levels) |
| ACC South Pacific | 0.055 | 0.00020 | 0–2000 m (54 levels) |
| ACC North Pacific | 0.039 | 0.00037 | 0–2000 m (54 levels) |
| ACC Tropical Pacific | 0.035 | 0.00078 | 0–2000 m (54 levels) |

Snapshots are 45 minutes apart, so the tracer barely moves between them and "no change" is a very strong baseline. This is one sample per model and says nothing about the Fortran side, which reproduces Python exactly; it does mean none of these checkpoints should be used for real predictions without looking at the training results first.

Two cautions when reading the table:

- **The two GPP rows are not directly comparable.** They were scored over different depth ranges, because the config was changed between the runs.
- **The ACC depth range is the config's current one.** The ACC config was edited after those checkpoints were written, so it is not confirmed that they were trained on 0–2000 m.

**Not yet verified**

- **More than one real sample.** Step 6 tests a single time window per run. Use `REALDATA_SAMPLE=<n>` to try others.
- **The GPP model under the new layout.** Only the four ACC runs have used `runs/`, `logs/` and `server/`. The default submission (`sbatch sbatch_ftorch_pipeline.job` with no settings) uses the same job script but has not been run since the change.
- **Reliable timings for the depth-sweep runs.** Several of the 13 October jobs ran on a busy node and show forward passes up to 16 times slower than normal, in Fortran and Python alike. Use the 9 October timings above.
- **Timing inside NEMO.** Step 7 times the network alone, on an otherwise idle node.
- **Coupling to NEMO.** Nothing here runs inside NEMO. Reordering NEMO's arrays to `(depth, y, x)` and cutting out the region box is the job of whoever writes that coupling.

**Things to know**

- **One region, one shape.** Each exported file works only for its own region's box (75 × 25 × 20 for the North Atlantic) with 4 input steps. The `TracerWarning` messages in `export_nautilus.log` say exactly this: the shape was fixed at export time.
- **Two PyTorch versions are involved.** The Python that exports the model is version 2.3.0 (it comes from the `python` module loaded last in the job). The libtorch that Fortran uses is 2.1.2 (from the PyTorch module). The cross-check passing shows this combination works for this model. If either changes, the cross-check is what will tell you.
- **CPU only.** FTorch was built without GPU support.
- **The ten deepest levels (66–75) have no tracer in the GPP North Atlantic training data** (minimum and maximum are both zero there). The interface returns zero for those levels. This was checked for that model only.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| ABI | The low-level conventions two compiled pieces of code must share to work together. The reason the compiler version must match libtorch's. |
| Channel | One input variable (tracer, u, v, w, dif, mld). |
| Checkpoint | File of trained weights saved by Python. |
| CMake | A tool that works out how to compile and link a program. |
| Column-major / row-major | The two orders in which a multi-dimensional array can be laid out in memory. Fortran uses the first, Python the second. |
| Cross-check | Running the same input through Python and Fortran and comparing. |
| Denormalize | Convert the network's output back to physical units. `forward` does this. |
| Forward pass | One run of the network from input to output. |
| FTorch | Library connecting Fortran to libtorch. |
| History buffer | The last 4 time steps the interface keeps in memory. |
| Inference | Using a trained model to make a prediction. |
| JSON | A plain-text format for structured data. The metadata file is JSON. |
| Land mask | A map marking each grid cell as ocean or land. |
| libtorch | The C++ core of PyTorch. |
| Login node | The machine you land on with `ssh`. For light work only. |
| Mixed layer depth (MLD) | Depth of the well-mixed surface layer of the ocean; NEMO variable `mlotst`. |
| Module (Fortran) | A unit of Fortran code that other code brings in with `use`. |
| Module (MN5) | A software package enabled with `module load`. |
| Symbolic link | A shortcut to a file or folder elsewhere. |
| Tag | The name of one run of the pipeline, by default `<SOURCE>_<REGION>`. It names the run's folders in `runs/` and `logs/`. |
| Tracer | A dissolved substance carried by the water; NEMO variable `mytrc`. |
| TorchScript | A self-contained, Python-independent model file. |
| Tracing | Recording the operations of one run of the model to produce TorchScript. |
