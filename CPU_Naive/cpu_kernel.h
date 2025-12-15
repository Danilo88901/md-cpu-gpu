#pragma once
#include <cmath>

/*
 * CPU Naive Molecular Dynamics Kernel
 * -----------------------------------
 * This file contains the simplest, non-optimized CPU implementation 
 * for Molecular Dynamics simulations using Lennard-Jones potential.
 * 
 * Functions:
 *   - cpu_kernel       : computes forces between particles
 *   - cpu_integration  : updates positions and velocities using Verlet integration
 *
 * Constants:
 *   dt       : time step
 *   sigma    : Lennard-Jones sigma parameter
 *   epsilon  : Lennard-Jones epsilon parameter
 *   r_cutoff : interaction cutoff distance
 */

// --- Simulation constants ---
const float dt = 1e-4f;              // time step
const float sigma = 1.0f;             // Lennard-Jones sigma
const float epsilon = 1.0f;           // Lennard-Jones epsilon
const float r_cutoff = 2.5f * sigma;  // cutoff distance

// --- CPU Kernel: computes pairwise forces using Lennard-Jones potential ---
void cpu_kernel(int n, float* m, float* x, float* y, float* z,
                float* a_x, float* a_y, float* a_z,
                float sigma, float epsilon, float r_cutoff, float dt,
                float* ax_out, float* ay_out, float* az_out) {

    for (int i = 0; i < n; i++) {
        float ax_old = a_x[i];
        float ay_old = a_y[i];
        float az_old = a_z[i];

        // save previous accelerations for Verlet integration
        ax_out[i] = ax_old;
        ay_out[i] = ay_old;
        az_out[i] = az_old;

        // reset accelerations before summing new forces
        a_x[i] = 0; 
        a_y[i] = 0; 
        a_z[i] = 0;

        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            float dx = x[j] - x[i];
            float dy = y[j] - y[i];
            float dz = z[j] - z[i];
            float r = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (r_cutoff < r || r < 1e-6f) continue;  // ignore very distant or overlapping particles

            float sr2 = (sigma / r)*(sigma / r);
            float sr6 = sr2*sr2*sr2;
            float sr12 = sr6*sr6;
            float F = 24 * epsilon * (2 * sr12 - sr6) / r; // Lennard-Jones force magnitude

            // normalize vector for direction
            float nx = dx / r;
            float ny = dy / r;
            float nz = dz / r;

            // accumulate accelerations
            a_x[i] += nx * F / m[i];
            a_y[i] += ny * F / m[i];
            a_z[i] += nz * F / m[i];
        }
    }
}

// --- CPU Integration: updates positions and velocities (Verlet) ---
void cpu_integration(int n, float* x, float* y, float* z,
                     float* v_x, float* v_y, float* v_z,
                     float* a_x, float* a_y, float* a_z,
                     float dt, float* ax_old, float* ay_old, float* az_old) {

    for (int i = 0; i < n; i++) {
        // update positions
        x[i] += v_x[i] * dt + 0.5f * ax_old[i] * dt * dt;
        y[i] += v_y[i] * dt + 0.5f * ay_old[i] * dt * dt;
        z[i] += v_z[i] * dt + 0.5f * az_old[i] * dt * dt;

        // update velocities
        v_x[i] += 0.5f * (ax_old[i] + a_x[i]) * dt;
        v_y[i] += 0.5f * (ay_old[i] + a_y[i]) * dt;
        v_z[i] += 0.5f * (az_old[i] + a_z[i]) * dt;
    }
}
