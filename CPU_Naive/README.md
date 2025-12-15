This implementation uses a naive brute-force algorithm with **quadratic time complexity O(N²)** to compute particle interactions.

The main idea is straightforward: every particle interacts with every other particle. Due to its simplicity, this approach is well suited for:
- quick prototyping,
- validation of physical models,
- simulations with a small number of particles.

However, as the number of particles increases, the computational cost grows rapidly. For systems with more than a few hundred or ~1000 particles, this approach becomes inefficient and impractical for real simulations.

Therefore, the naive CPU implementation mainly serves as a **baseline reference** for performance comparison with optimized GPU implementations and algorithms based on spatial decomposition (e.g. Cell Lists).
