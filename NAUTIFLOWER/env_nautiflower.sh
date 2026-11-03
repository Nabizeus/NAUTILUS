# Python environment of sbatch_v2_train_eval.job (same modules), for interactive checks: source env_nautiflower.sh
module purge
module load PyYAML/6.0.1-GCCcore-13.2.0 tqdm/4.66.2-GCCcore-13.2.0 PyTorch/2.3.0-foss-2023b-CUDA-12.4.0
module load imageio/2.34.1-gfbf-2023b scikit-learn/1.4.0-gfbf-2023b Seaborn/0.13.2-gfbf-2023b-Python-3.11.5
module load xarray/2024.5.0-gfbf-2023b-Python-3.11.5 SciPy-bundle/2023.11-gfbf-2023b netcdf4-python/1.7.1.post2-foss-2023b h5netcdf/1.6.1-foss-2023b-Python-3.11.5
export PYTHONPATH="$HOME/python_extra:$PYTHONPATH"
