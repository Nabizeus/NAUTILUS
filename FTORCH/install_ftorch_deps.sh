#!/bin/bash
# install_ftorch_deps.sh
#
# One-time install of json-fortran and FTorch from source on MareNostrum 5
# (neither is available as a module there). Run this on a LOGIN node or
# inside a short interactive/sbatch job -- NOT as part of every pipeline run.
#
#   bash install_ftorch_deps.sh                # installs under $HOME/nautilus_deps
#   PREFIX=/gpfs/projects/bsc32/$USER/deps bash install_ftorch_deps.sh
#
# When it finishes it writes ${PREFIX}/env_ftorch.sh. Source that file (the
# sbatch pipeline job already does) to get FTORCH_DIR / JSON_FORTRAN_DIR /
# CMAKE_PREFIX_PATH / LD_LIBRARY_PATH set up.
#
# IMPORTANT -- toolchain must match the Python job:
# FTorch is compiled against the libtorch that ships INSIDE the PyTorch
# module, and your Fortran code must use the same GCC as that module. The
# training job uses PyTorch/2.1.2-foss-2023b-Python-3.11.5, i.e. GCC 13.2.0,
# so we load exactly that here. If you change the PyTorch module, re-run this
# script with the matching toolchain, otherwise you get ABI/link errors.
#
# FTorch version: pinned to v1.0.0 because its API
#   call torch_model_load(model, path, device)
#   call torch_tensor_from_array(tensor, array, layout, device)
# (subroutines) is what nautilus_ftorch_interface.F90 and your original CAMP
# interface use. Newer FTorch releases changed this style; don't bump the
# version without porting the interface.

set -euo pipefail

PREFIX="${PREFIX:-$HOME/nautilus_deps}"
FTORCH_TAG="${FTORCH_TAG:-v1.0.0}"
JSON_FORTRAN_TAG="${JSON_FORTRAN_TAG:-9.0.2}"
FTORCH_REPO="${FTORCH_REPO:-https://github.com/Cambridge-ICCS/FTorch.git}"
JSON_FORTRAN_REPO="${JSON_FORTRAN_REPO:-https://github.com/jacobwilliams/json-fortran.git}"
JOBS="${JOBS:-8}"

SRC_DIR="${PREFIX}/src"
mkdir -p "${SRC_DIR}"

# ---------------------------------------------------------------------
# Toolchain: same as the training/export job (foss-2023b -> GCC 13.2.0)
# ---------------------------------------------------------------------
module purge || true
module load PyTorch/2.1.2-foss-2023b-Python-3.11.5
# Pinned: the default CMake module is built on GCCcore-14.3.0 and would
# replace the GCC 13.2.0 toolchain that libtorch was built with.
module load CMake/3.27.6-GCCcore-13.2.0
module load hdf5 || true

echo "gfortran: $(command -v gfortran)  ($(gfortran --version | head -1))"
echo "cmake   : $(command -v cmake)  ($(cmake --version | head -1))"

# libtorch (the C++ library FTorch wraps) -> CMAKE_PREFIX_PATH for FTorch.
# Default: the libtorch that ships inside the PyTorch module's Python package
# (<site-packages>/torch/{lib,include,share/cmake/Torch}). This is the best
# choice because it is the exact build that trained/exported your model and
# it matches the module's GCC and C++ ABI.
# Override: LIBTORCH_DIR=/path/to/unzipped/libtorch bash install_ftorch_deps.sh
# Only use a standalone libtorch if the module's copy lacks the CMake files;
# then download the CPU build of the SAME version (2.1.2) with the cxx11 ABI
# from pytorch.org on a login node (compute nodes have no internet).
if [ -n "${LIBTORCH_DIR:-}" ]; then
    TORCH_DIR="${LIBTORCH_DIR}"
else
    TORCH_DIR="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')"
fi
echo "libtorch: ${TORCH_DIR}"
test -d "${TORCH_DIR}/share/cmake/Torch" || {
    echo "ERROR: ${TORCH_DIR}/share/cmake/Torch not found; cannot build FTorch." >&2
    echo "Set LIBTORCH_DIR to a standalone libtorch (CPU, v2.1.2, cxx11 ABI) and re-run." >&2
    exit 1
}

# ---------------------------------------------------------------------
# 1/2: json-fortran
# ---------------------------------------------------------------------
if [ ! -d "${SRC_DIR}/json-fortran" ]; then
    git clone --depth 1 --branch "${JSON_FORTRAN_TAG}" "${JSON_FORTRAN_REPO}" "${SRC_DIR}/json-fortran"
fi
rm -rf "${SRC_DIR}/json-fortran/build"
cmake -S "${SRC_DIR}/json-fortran" -B "${SRC_DIR}/json-fortran/build" \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}/json-fortran" \
    -DUSE_GNU_INSTALL_CONVENTION=ON \
    -DSKIP_DOC_GEN=ON \
    -DJSONFORTRAN_ENABLE_TESTS=OFF
cmake --build "${SRC_DIR}/json-fortran/build" -j "${JOBS}"
cmake --install "${SRC_DIR}/json-fortran/build"

# ---------------------------------------------------------------------
# 2/2: FTorch (CPU only)
# ---------------------------------------------------------------------
if [ ! -d "${SRC_DIR}/FTorch" ]; then
    git clone --depth 1 --branch "${FTORCH_TAG}" "${FTORCH_REPO}" "${SRC_DIR}/FTorch"
fi
rm -rf "${SRC_DIR}/FTorch/build"
cmake -S "${SRC_DIR}/FTorch" -B "${SRC_DIR}/FTorch/build" \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${TORCH_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}/ftorch" \
    -DGPU_DEVICE=NONE
cmake --build "${SRC_DIR}/FTorch/build" -j "${JOBS}"
cmake --install "${SRC_DIR}/FTorch/build"

# ---------------------------------------------------------------------
# Environment file for the pipeline job
# ---------------------------------------------------------------------
# lib vs lib64 differs per package/system; add whichever exist.
ENV_FILE="${PREFIX}/env_ftorch.sh"
cat > "${ENV_FILE}" <<EOF
# Generated by install_ftorch_deps.sh -- source this after loading the
# PyTorch/foss-2023b modules.
export NAUTILUS_DEPS="${PREFIX}"
export FTORCH_DIR="${PREFIX}/ftorch"
export JSON_FORTRAN_DIR="${PREFIX}/json-fortran"
export TORCH_DIR="${TORCH_DIR}"
export CMAKE_PREFIX_PATH="\${FTORCH_DIR}:\${JSON_FORTRAN_DIR}:\${TORCH_DIR}:\${CMAKE_PREFIX_PATH:-}"
for d in "\${FTORCH_DIR}/lib" "\${FTORCH_DIR}/lib64" "\${JSON_FORTRAN_DIR}/lib" "\${JSON_FORTRAN_DIR}/lib64" "\${TORCH_DIR}/lib"; do
    [ -d "\$d" ] && export LD_LIBRARY_PATH="\$d:\${LD_LIBRARY_PATH:-}"
done
EOF

echo
echo "Installed:"
echo "  json-fortran ${JSON_FORTRAN_TAG} -> ${PREFIX}/json-fortran"
echo "  FTorch       ${FTORCH_TAG}        -> ${PREFIX}/ftorch"
echo "Environment file: ${ENV_FILE}"
echo "Next: sbatch sbatch_ftorch_pipeline.job"
