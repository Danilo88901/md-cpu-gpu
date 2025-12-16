# CPU Optimized Molecular Dynamics (Cell List)

This implementation provides a CPU-optimized Molecular Dynamics force computation
using a **Cell List (linked-cell) algorithm**.

Unlike brute-force O(N²) approaches, this method reduces the computational complexity
to approximately **O(N)** for typical particle densities by limiting force evaluations
to nearby particles only.

## Key Features

- Cell List (spatial decomposition) neighbor search
- Periodic Boundary Conditions (PBC)
- Lennard-Jones potential
- Velocity Verlet integration
- Independent force computation per particle (no Newton's 3rd law optimization)

## Design Choices

Each particle computes its acceleration independently.
This results in a **double counting of particle pairs**, which is less optimal
in terms of raw arithmetic operations but has important advantages:

- No race conditions
- No atomic operations required
- Easily portable to GPU implementations
- Better cache locality and vectorization on CPU

This approach is commonly preferred in high-performance GPU and hybrid MD codes,
where atomic updates significantly degrade performance.

## Performance Notes

For typical simulations (10,000–20,000 particles), this CPU Cell List implementation
significantly outperforms GPU brute-force methods, even when the GPU version uses
shared memory and register tiling.

Performance depends on particle density, cutoff radius, and hardware, but the algorithm
scales well for moderately large systems.

## Typical Use Cases

- Medium to large molecular dynamics simulations
- CPU-based MD benchmarks
- Baseline implementation for GPU Cell List development
- Comparison against brute-force force calculations

## Complexity

- Time complexity: ~O(N)
- Memory overhead: O(N)
