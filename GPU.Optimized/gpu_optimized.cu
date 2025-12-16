#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

const int block = 256;          // Number of threads per CUDA block
const float dt = 1e-4;          // Time step
const float sigma = 1;           // Lennard-Jones parameter
const float epsilon = 1;         // Lennard-Jones parameter
const float r_cutoff = 2.5f * sigma; // Cutoff radius for force calculation

// =======================
// Naive Velocity Verlet integrator
// =======================
__global__ void integrate(
    int n, 
    float* x, float* y, float* z, 
    float* v_x, float* v_y, float* v_z, 
    float* a_x, float* a_y, float* a_z, 
    float dt,
    float* ax_old, float* ay_old, float* az_old) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
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

// =======================
// Optimized GPU kernel using shared memory
// =======================
__global__ void optimized_gpu(
    int n, 
    float* m, 
    float* x, float* y, float* z, 
    float* a_x, float* a_y, float* a_z,
    float sigma, float epsilon, float r_cutoff, float dt,
    float* ax_out, float* ay_out, float* az_out) 
{
    // Shared memory to store particle positions for current block
    __shared__ float x_pos[block];
    __shared__ float y_pos[block];
    __shared__ float z_pos[block];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    bool active = (idx < n);

    // Load particle data or set defaults for inactive threads
    float mass_id = active ? m[idx] : 0.0f;
    float x_id = active ? x[idx] : 0.0f;
    float y_id = active ? y[idx] : 0.0f;
    float z_id = active ? z[idx] : 0.0f;
    float ax_old = active ? a_x[idx] : 0.0f;
    float ay_old = active ? a_y[idx] : 0.0f;
    float az_old = active ? a_z[idx] : 0.0f;

    if (active) {
        // Store old acceleration for Velocity Verlet integration
        ax_out[idx] = ax_old;
        ay_out[idx] = ay_old;
        az_out[idx] = az_old;

        // Reset current acceleration
        a_x[idx] = 0;
        a_y[idx] = 0;
        a_z[idx] = 0;
    }

    float ax_new = 0.0f;
    float ay_new = 0.0f;
    float az_new = 0.0f;

    // Loop over blocks of particles for shared memory computation
    for (int j = 0; j < (n + block - 1) / block; j++) {
        int tx = threadIdx.x;
        int global = j * blockDim.x + tx;

        // Load positions of particles in the current block into shared memory
        if (global < n) {
            x_pos[tx] = x[global];
            y_pos[tx] = y[global];
            z_pos[tx] = z[global];
        }

        __syncthreads();

        if (active) {
            // Compute forces with particles in this block
            for (int i = 0; i < blockDim.x; i++) {
                int idx2 = j * blockDim.x + i;
                if (idx2 >= n || idx2 == idx) continue;

                float dx = x_pos[i] - x_id;
                float dy = y_pos[i] - y_id;
                float dz = z_pos[i] - z_id;
                float dis = dx * dx + dy * dy + dz * dz;

                // Skip if beyond cutoff
                if (dis > r_cutoff * r_cutoff) continue;

                float inv_r = rsqrtf(dis + 1e-12f); // Fast inverse square root
                float sr2 = (sigma * inv_r) * (sigma * inv_r);
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                float F = 24.0f * epsilon * (2.0f * sr12 - sr6) * inv_r;

                // Force components
                float nx = inv_r * dx;
                float ny = inv_r * dy;
                float nz = inv_r * dz;
                ax_new += F * nx / mass_id;
                ay_new += F * ny / mass_id;
                az_new += F * nz / mass_id;
            }

            // Update acceleration
            a_x[idx] = ax_new;
            a_y[idx] = ay_new;
            a_z[idx] = az_new;
        }

        __syncthreads();
    }
}
