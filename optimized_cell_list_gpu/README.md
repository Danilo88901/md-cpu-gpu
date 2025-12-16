# Optimized Cell List GPU

This repository contains a **high-performance GPU implementation** of a cell-list based molecular dynamics (MD) simulation. The code is designed to efficiently compute pairwise interactions between particles using a **Lennard-Jones potential** with periodic boundary conditions.  

## Features

- **GPU-accelerated computation** using CUDA for massive speedup over CPU-only implementations.  
- **Optimized cell-list algorithm** to reduce the computational complexity from O(N²) to O(N) per time step.  
- Handles **periodic boundary conditions** in 3D simulations.  
- Flexible simulation box sizes and cutoff distances.  
- Supports large numbers of particles (up to hundreds of thousands) efficiently.  

## Algorithm Overview

1. **Cell List Construction**  
   - The simulation box is divided into 3D cells, each storing indices of particles contained within.  
   - Only particles in neighboring cells are considered for interaction calculations, dramatically reducing the number of distance checks.  

2. **Force Computation (GPU)**  
   - Each GPU thread computes forces for one particle.  
   - Interactions are calculated only with particles in the same or neighboring cells.  
   - Small distances are clamped to avoid division by zero and numerical instabilities.  

3. **Integration**  
   - Positions and velocities are updated using a velocity-Verlet-like integration scheme.  

## Performance

The GPU implementation achieves significant acceleration compared to a CPU-only version, especially for large particle systems. The cell-list algorithm ensures that computational complexity scales **linearly with the number of particles**, making it suitable for high-performance MD simulations.  

## Usage

- Configure the number of particles, time steps, and simulation box parameters in the `main.cpp`.  
- Compile with CUDA support.  
- Run the executable and observe the high-speed MD simulation.

---

> **Note:** This implementation is optimized for benchmarking and demonstration purposes. For production-level simulations, additional features such as temperature control, neighbor list updates, and energy computation may be added.
