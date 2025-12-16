CUDA Particle Integrator

This repository contains CUDA implementations of a particle dynamics integrator, including both a naive and an optimized version leveraging GPU shared memory for improved performance. The code is designed for simulations with pairwise forces, such as Lennard-Jones interactions.

Features

Naive CUDA Version

Simple implementation.

Each thread accesses global memory for all computations.

Integrates particle positions and velocities using the Velocity Verlet method.

Optimized CUDA Version

Utilizes shared memory to reduce global memory accesses.

Minimizes redundant computation of pairwise forces.

Uses rsqrtf for fast approximate reciprocal square root instead of standard sqrt.

Supports a cutoff radius for interactions to reduce the number of force calculations.

Stores previous acceleration values for Velocity Verlet integration.

Algorithms
Velocity Verlet Integration

The integrator updates particle positions and velocities according to:

x(t+dt) = x(t) + v(t)*dt + 0.5 * a(t) * dt^2
v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt)) * dt


Where a(t) is the acceleration at the current time step, and a(t+dt) is the acceleration at the next step, calculated from particle interactions.

Force Computation (Optimized Version)

Positions of particles in each block are loaded into shared memory.

Pairwise distances are computed, respecting a cutoff radius.

Force is calculated using a simplified Lennard-Jones formula:

F = 24 * epsilon * (2 * (sigma / r)^12 - (sigma / r)^6) * (1/r)


Normalized direction vectors are applied to compute force components along x, y, z.

Acceleration is updated as a = F / m.

Optimizations

Shared Memory: Reduces repeated global memory accesses.

Reciprocal Square Root (rsqrtf): Faster approximation than standard sqrt.

Block-level parallelization: Forces between particles are computed per block with threads synchronized using __syncthreads().

Code Structure

integrate kernel: Naive velocity Verlet integrator.

optimized_gpu kernel: Shared memory optimized version with force cutoff.

Constants:

block = 256: Number of threads per CUDA block.

Usage

Compile with nvcc:

nvcc -o particle_sim particle_sim.cu


Set up particle arrays (x, y, z, v_x, v_y, v_z, a_x, a_y, a_z, m) on the host and copy them to the device.

Launch kernels:

integrate<<<grid, block>>>(...);
optimized_gpu<<<grid, block>>>(...);


Copy results back to the host.

Notes

This implementation is designed for educational purposes and performance experimentation.

Further improvements can include neighbor lists, atomic updates, and multi-GPU scaling.
