# Author: Shiva Subramanian, Antariksh Krishnan
# GTID: 903780288
# Class: ECE 4122
# Last Date Modified: 4/23/26

# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Define the project name and enable C++ and CUDA languages
project(BranchPredictor LANGUAGES CXX CUDA)

# Require C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture optimization flags
set(CMAKE_CUDA_ARCHITECTURES "all")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

# Tell CMake where to find your header (.h and .cuh) files
include_directories(include src)

# List all your source files with their correct folder paths!
set(SOURCES
    src/main.cu
    src/trace_reader.cpp
    src/baseline.cpp
    src/predictors.cu
    src/transfer.cu
    src/validator.cpp
)

# Create the executable named 'predictor'
add_executable(predictor ${SOURCES})

# Explicitly tell CMake that these files need the CUDA compiler
set_target_properties(predictor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)