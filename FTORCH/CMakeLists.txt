# CMakeLists.txt for the NAUTILUS FTorch interface + test programs.
#
# Dependencies (FTorch v1.0.0, json-fortran, and the libtorch inside the
# PyTorch module) are NOT available as modules on MareNostrum 5. Install them
# once with install_ftorch_deps.sh, then `source ${PREFIX}/env_ftorch.sh`
# (the sbatch pipeline does this for you) which sets FTORCH_DIR,
# JSON_FORTRAN_DIR, TORCH_DIR and CMAKE_PREFIX_PATH. This file picks them up
# from the environment, or from -D options if you pass them explicitly:
#
#   mkdir -p build && cd build
#   cmake .. \
#       -DCMAKE_Fortran_COMPILER=gfortran \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DFTORCH_DIR=$HOME/nautilus_deps/ftorch \
#       -DJSON_FORTRAN_DIR=$HOME/nautilus_deps/json-fortran \
#       -DTORCH_DIR=<site-packages>/torch
#   make -j4
#
cmake_minimum_required(VERSION 3.18)

# C and CXX are enabled because FTorch's exported config pulls in Torch,
# whose CMake package expects those languages to be available.
project(nautilus_ftorch_test LANGUAGES Fortran C CXX)

set(CMAKE_Fortran_STANDARD 2008)

# ---------------------------------------------------------------------
# Locate dependencies. -D options win, then environment variables.
# ---------------------------------------------------------------------
foreach(dep FTORCH_DIR JSON_FORTRAN_DIR TORCH_DIR)
    if(NOT DEFINED ${dep} AND DEFINED ENV{${dep}})
        set(${dep} "$ENV{${dep}}")
    endif()
    if(DEFINED ${dep} AND NOT "${${dep}}" STREQUAL "")
        list(PREPEND CMAKE_PREFIX_PATH "${${dep}}")
    endif()
endforeach()

find_package(FTorch REQUIRED)       # provides target FTorch::ftorch

# json-fortran installs a CMake package named "jsonfortran-<compiler id>"
# (e.g. jsonfortran-gnu) under <prefix>/lib/cmake/jsonfortran-gnu-<version>/,
# found via CMAKE_PREFIX_PATH. The exact package/target spelling has varied
# between releases, so try the plausible ones and take whichever target exists.
string(TOLOWER "${CMAKE_Fortran_COMPILER_ID}" _fc_id_lower)
foreach(_pkg jsonfortran-${_fc_id_lower} jsonfortran-${CMAKE_Fortran_COMPILER_ID} jsonfortran)
    if(NOT NAUTILUS_JSON_TARGET)
        find_package(${_pkg} QUIET)
        foreach(_tgt ${_pkg}::jsonfortran jsonfortran::jsonfortran jsonfortran)
            if(NOT NAUTILUS_JSON_TARGET AND TARGET ${_tgt})
                set(NAUTILUS_JSON_TARGET ${_tgt})
            endif()
        endforeach()
    endif()
endforeach()

if(NOT NAUTILUS_JSON_TARGET)
    # Fallback for installs that don't ship CMake config files: look for the
    # library + .mod files directly under JSON_FORTRAN_DIR.
    find_library(JSONFORTRAN_LIB
        NAMES jsonfortran
        HINTS "${JSON_FORTRAN_DIR}"
        PATH_SUFFIXES lib lib64)
    find_path(JSONFORTRAN_INC
        NAMES json_module.mod
        HINTS "${JSON_FORTRAN_DIR}"
        PATH_SUFFIXES include lib lib64 include/jsonfortran-gnu-9.0.2)
    if(NOT JSONFORTRAN_LIB OR NOT JSONFORTRAN_INC)
        message(FATAL_ERROR
            "json-fortran not found. Run install_ftorch_deps.sh and source "
            "its env_ftorch.sh, or pass -DJSON_FORTRAN_DIR=<install prefix>.")
    endif()
    add_library(nautilus_jsonfortran UNKNOWN IMPORTED)
    set_target_properties(nautilus_jsonfortran PROPERTIES
        IMPORTED_LOCATION "${JSONFORTRAN_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${JSONFORTRAN_INC}")
    set(NAUTILUS_JSON_TARGET nautilus_jsonfortran)
endif()
message(STATUS "FTorch found; json-fortran target: ${NAUTILUS_JSON_TARGET}")

# Make executables find the shared libs at runtime without LD_LIBRARY_PATH.
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

add_library(nautilus_ftorch_interface nautilus_ftorch_interface.F90)
target_compile_definitions(nautilus_ftorch_interface PUBLIC NAUTILUS_USE_JSON)
target_link_libraries(nautilus_ftorch_interface PUBLIC FTorch::ftorch ${NAUTILUS_JSON_TARGET})

add_executable(test_nautilus_crosscheck test_nautilus_crosscheck.F90)
target_link_libraries(test_nautilus_crosscheck PRIVATE nautilus_ftorch_interface)

add_executable(test_nautilus_dummy_driver test_nautilus_dummy_driver.F90)
target_link_libraries(test_nautilus_dummy_driver PRIVATE nautilus_ftorch_interface)

add_executable(test_nautilus_realdata test_nautilus_realdata.F90)
target_link_libraries(test_nautilus_realdata PRIVATE nautilus_ftorch_interface)

add_executable(test_nautilus_benchmark test_nautilus_benchmark.F90)
target_link_libraries(test_nautilus_benchmark PRIVATE nautilus_ftorch_interface)
