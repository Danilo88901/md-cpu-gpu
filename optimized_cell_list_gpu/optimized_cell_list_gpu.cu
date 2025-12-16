// optimized_cell_list_gpu.cu
//
// GPU-accelerated cell-list based force computation.
// This file is intended for performance benchmarking of
// molecular-dynamics-like workloads on CUDA-capable GPUs.
//
// The implementation focuses on realistic interaction patterns
// (neighbor search, cutoff, periodic boundary conditions),
// while prioritizing performance and scalability over strict
// physical accuracy.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <chrono>

// Simple CUDA error check macro for debugging kernel launches
#define CUDA_CHECK() { cudaError_t e = cudaGetLastError(); if(e != cudaSuccess) { \
  std::cerr << "CUDA ERR: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1);} }

// Global constants used in force computation and integration
float MIN_DIST2 = 1e-5;        // Small distance cutoff to avoid numerical instabilities
const float dt = 1e-4;         // Integration timestep
const float sigma = 1;         // Lennard-Jones sigma parameter
const float epsilon = 1;       // Lennard-Jones epsilon parameter
const float r_cutoff = 2.5f * sigma; // Interaction cutoff radius

// -----------------------------------------------------------------------------
// CPU-side construction of a cell list (linked-cell structure)
//
// This function:
// 1. Applies periodic boundary conditions to particle positions
// 2. Divides the simulation domain into uniform 3D cells
// 3. Assigns each particle to a corresponding cell
//
// The resulting cell lists are later transferred to the GPU and
// used for efficient neighbor search during force computation.
// -----------------------------------------------------------------------------
void cpu_list(int n, float* x, float* y, float* z, float r_cutoff,
              int* cell_count, int* cell_start, int* cell_end, int* cell_particles) {

	const float min_x = -200, max_x = 200;
	const float min_y = -50,  max_y = 50;
	const float min_z = -10,  max_z = 45;

	const float box_x = max_x - min_x;
	const float box_y = max_y - min_y;
	const float box_z = max_z - min_z;

	// Apply periodic boundary conditions (wrap positions into the box)
	for (int i = 0; i < n; i++) {
		x[i] = fmodf((x[i] - min_x), box_x);
		if (x[i] < 0) x[i] += box_x;
		x[i] += min_x;

		y[i] = fmodf((y[i] - min_y), box_y);
		if (y[i] < 0) y[i] += box_y;
		y[i] += min_y;

		z[i] = fmodf((z[i] - min_z), box_z);
		if (z[i] < 0) z[i] += box_z;
		z[i] += min_z;
	}

	// Determine number of cells in each dimension
	int cells_x = std::ceil(box_x / (1.5f * r_cutoff));
	int cells_y = std::ceil(box_y / (1.5f * r_cutoff));
	int cells_z = std::ceil(box_z / (1.5f * r_cutoff));

	// Reset cell counters
	for (int i = 0; i < cells_x * cells_y * cells_z; i++) {
		cell_count[i] = 0;
	}

	// First pass: count how many particles fall into each cell
	for (int i = 0; i < n; i++) {
		int ix = std::floor((x[i] - min_x) / (1.5f * r_cutoff));
		int iy = std::floor((y[i] - min_y) / (1.5f * r_cutoff));
		int iz = std::floor((z[i] - min_z) / (1.5f * r_cutoff));

		if (ix >= cells_x) ix = cells_x - 1;
		if (iy >= cells_y) iy = cells_y - 1;
		if (iz >= cells_z) iz = cells_z - 1;
		if (ix < 0) ix = 0;
		if (iy < 0) iy = 0;
		if (iz < 0) iz = 0;

		int cell_index = iz * cells_x * cells_y + iy * cells_x + ix;
		cell_count[cell_index]++;
	}

	// Prefix sum to determine start/end offsets for each cell
	int sum = 0;
	for (int i = 0; i < cells_x * cells_y * cells_z; i++) {
		cell_start[i] = sum;
		sum += cell_count[i];
		cell_end[i] = sum;
		cell_count[i] = 0;
	}

	// Second pass: store particle indices in cell_particles array
	for (int i = 0; i < n; i++) {
		int ix = std::floor((x[i] - min_x) / (1.5f * r_cutoff));
		int iy = std::floor((y[i] - min_y) / (1.5f * r_cutoff));
		int iz = std::floor((z[i] - min_z) / (1.5f * r_cutoff));

		if (ix >= cells_x) ix = cells_x - 1;
		if (iy >= cells_y) iy = cells_y - 1;
		if (iz >= cells_z) iz = cells_z - 1;
		if (ix < 0) ix = 0;
		if (iy < 0) iy = 0;
		if (iz < 0) iz = 0;

		int cell_index = iz * cells_x * cells_y + iy * cells_x + ix;
		int pos = cell_start[cell_index] + cell_count[cell_index];
		cell_particles[pos] = i;
		cell_count[cell_index]++;
	}
}

// -----------------------------------------------------------------------------
// GPU kernel: force computation using a cell list
//
// Each thread computes forces acting on a single particle.
// Only particles in the current and neighboring cells (3x3x3)
// are considered, reducing complexity from O(N^2) to ~O(N).
//
// Lennard-Jones (12-6) potential is used with a cutoff radius
// and periodic boundary conditions (minimum image convention).
// -----------------------------------------------------------------------------
__global__ void super_optimized_gpu(int n, float* m, float* x, float* y, float* z,
	float* a_x, float* a_y, float* a_z,
	float sigma, float epsilon, float r_cutoff, float dt,
	float* ax_out, float* ay_out, float* az_out,
	int* cell_count, int* cell_start, int* cell_end, int* cell_particles,
	float min_x, float min_y, float min_z,
	int cells_x, int cells_y, int cells_z,
	float box_x, float box_y, float box_z) {

	const float MIN_DIST2 = 1e-5f;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n) {
		float ax_new = 0.0f;
		float ay_new = 0.0f;
		float az_new = 0.0f;

		// Store previous acceleration for integration
		ax_out[idx] = a_x[idx];
		ay_out[idx] = a_y[idx];
		az_out[idx] = a_z[idx];

		// Determine cell index of the current particle
		int ix = floor((x[idx] - min_x) / (1.5f * r_cutoff));
		int iy = floor((y[idx] - min_y) / (1.5f * r_cutoff));
		int iz = floor((z[idx] - min_z) / (1.5f * r_cutoff));

		if (ix >= cells_x) ix = cells_x - 1;
		if (iy >= cells_y) iy = cells_y - 1;
		if (iz >= cells_z) iz = cells_z - 1;
		if (ix < 0) ix = 0;
		if (iy < 0) iy = 0;
		if (iz < 0) iz = 0;

		// Iterate over neighboring cells
		for (int dzz = -1; dzz <= 1; dzz++) {
			for (int dyy = -1; dyy <= 1; dyy++) {
				for (int dxx = -1; dxx <= 1; dxx++) {

					int nix = ix + dxx;
					int niy = iy + dyy;
					int niz = iz + dzz;

					// Periodic wrapping of cell indices
					if (nix < 0) nix += cells_x;
					if (niy < 0) niy += cells_y;
					if (niz < 0) niz += cells_z;
					if (nix >= cells_x) nix -= cells_x;
					if (niy >= cells_y) niy -= cells_y;
					if (niz >= cells_z) niz -= cells_z;

					int index1 = niz * cells_x * cells_y + niy * cells_x + nix;

					// Iterate over particles in the neighboring cell
					for (int k = cell_start[index1]; k < cell_end[index1]; k++) {
						int j = cell_particles[k];
						if (idx == j) continue;

						float dx = x[j] - x[idx];
						float dy = y[j] - y[idx];
						float dz = z[j] - z[idx];

						// Minimum image convention
						if (dx > box_x / 2) dx -= box_x;
						if (dx < -box_x / 2) dx += box_x;
						if (dy > box_y / 2) dy -= box_y;
						if (dy < -box_y / 2) dy += box_y;
						if (dz > box_z / 2) dz -= box_z;
						if (dz < -box_z / 2) dz += box_z;

						float r2 = dx * dx + dy * dy + dz * dz;
						if (r2 > r_cutoff * r_cutoff || r2 < MIN_DIST2) continue;

						float inv_r = rsqrtf(r2);
						float sr2 = (sigma * inv_r) * (sigma * inv_r);
						float sr6 = sr2 * sr2 * sr2;
						float sr12 = sr6 * sr6;

						float F = 24.0f * epsilon * (2.0f * sr12 - sr6) * inv_r;

						ax_new += F * dx * inv_r / m[idx];
						ay_new += F * dy * inv_r / m[idx];
						az_new += F * dz * inv_r / m[idx];
					}
				}
			}
		}

		// Write updated acceleration
		a_x[idx] = ax_new;
		a_y[idx] = ay_new;
		a_z[idx] = az_new;
	}
}

// -----------------------------------------------------------------------------
// GPU kernel: velocity-Verlet integration
//
// Updates particle positions and velocities using old and new
// accelerations computed in the force kernel.
// -----------------------------------------------------------------------------
__global__ void integrate(int n, float* x, float* y, float* z,
	float* v_x, float* v_y, float* v_z,
	float* a_x, float* a_y, float* a_z,
	float dt, float* ax_old, float* ay_old, float* az_old) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n) {
		x[idx] += v_x[idx] * dt + 0.5f * ax_old[idx] * dt * dt;
		y[idx] += v_y[idx] * dt + 0.5f * ay_old[idx] * dt * dt;
		z[idx] += v_z[idx] * dt + 0.5f * az_old[idx] * dt * dt;

		v_x[idx] += 0.5f * (ax_old[idx] + a_x[idx]) * dt;
		v_y[idx] += 0.5f * (ay_old[idx] + a_y[idx]) * dt;
		v_z[idx] += 0.5f * (az_old[idx] + a_z[idx]) * dt;
	}
}
