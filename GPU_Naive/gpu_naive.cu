#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>  // Для sqrt

// --- Simulation constants ---
const float dt = 1e-4;           // Time step
const float sigma = 1.0f;        // Lennard-Jones parameter σ
const float epsilon = 1.0f;      // Lennard-Jones parameter ε
const float r_cutoff = 2.5f * sigma; // Cutoff distance for interactions

// --- Naive GPU kernel to compute particle accelerations ---
// Each thread calculates acceleration for one particle
__global__ void naive_kernel(int n,
                             float* m, float* x, float* y, float* z,
                             float* a_x, float* a_y, float* a_z,
                             float sigma, float epsilon, float r_cutoff, float dt,
                             float* axx_out, float* ayy_out, float* azz_out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // Global thread index

    if (idx < n) {
        // Save previous acceleration to temporary arrays
        float ax_old = a_x[idx];
        float ay_old = a_y[idx];
        float az_old = a_z[idx];
        axx_out[idx] = ax_old;
        ayy_out[idx] = ay_old;
        azz_out[idx] = az_old;

        // Reset acceleration for this particle
        a_x[idx] = 0.0f;
        a_y[idx] = 0.0f;
        a_z[idx] = 0.0f;

        // --- Loop over all other particles to compute pairwise force ---
        for (int i = 0; i < n; i++) {
            if (i == idx) continue; // Skip self-interaction

            // Compute distance vector
            float dx = x[i] - x[idx];
            float dy = y[i] - y[idx];
            float dz = z[i] - z[idx];
            float r = sqrt(dx*dx + dy*dy + dz*dz);

            // Skip if beyond cutoff or extremely small distance
            if (r > r_cutoff || r < 1e-6f) continue;

            // Compute Lennard-Jones force
            float sr2 = (sigma / r) * (sigma / r);
            float sr6 = sr2 * sr2 * sr2;
            float sr12 = sr6 * sr6;
            float F = 24.0f * epsilon * (2.0f * sr12 - sr6) / r;

            // Normalize distance vector to get direction
            float nx = dx / r;
            float ny = dy / r;
            float nz = dz / r;

            // Force components
            float F_x = F * nx;
            float F_y = F * ny;
            float F_z = F * nz;

            // Update acceleration
            a_x[idx] += F_x / m[idx];
            a_y[idx] += F_y / m[idx];
            a_z[idx] += F_z / m[idx];
        }
    }
}

// --- GPU kernel for Velocity-Verlet integration ---
// Updates positions and velocities based on previous and current accelerations
__global__ void integrate(int n,
                          float* x, float* y, float* z,
                          float* v_x, float* v_y, float* v_z,
                          float* a_x, float* a_y, float* a_z,
                          float dt,
                          float* ax_old, float* ay_old, float* az_old) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // Global thread index

    if (idx < n) {
        // Update positions
        x[idx] += v_x[idx] * dt + 0.5f * ax_old[idx] * dt * dt;
        y[idx] += v_y[idx] * dt + 0.5f * ay_old[idx] * dt * dt;
        z[idx] += v_z[idx] * dt + 0.5f * az_old[idx] * dt * dt;

        // Update velocities
        v_x[idx] += 0.5f * (ax_old[idx] + a_x[idx]) * dt;
        v_y[idx] += 0.5f * (ay_old[idx] + a_y[idx]) * dt;
        v_z[idx] += 0.5f * (az_old[idx] + a_z[idx]) * dt;
    }
}
