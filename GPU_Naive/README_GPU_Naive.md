GPU Naive Implementation

Description:
This folder contains the naive GPU implementation of molecular dynamics simulation. The algorithm calculates forces between all pairs of particles, similar to the CPU naive version, but leverages CUDA for parallel computation.

Contents:

gpu_naive.cu – main CUDA source code.

graphs/ – benchmark graphs comparing CPU Naive and GPU Naive.

cpu_vs_gpu.csv – timing results for different particle counts.

Usage:

Compile with CUDA:

nvcc gpu_naive.cu -o gpu_naive


Run the simulation:

./gpu_naive


Notes:

This implementation does not use cell lists or other optimizations.

Useful for performance comparison against CPU naive and optimized implementations.
