#include <cmath>      // std::sqrt, std::floor, std::ceil, std::fmod
#include <cstdlib>    // std::exit
#include <iostream>   // std::cerr
#include <cstddef>    // std::size_t
#include <limits>     // std::isfinite

// ============================================================================
// Simulation constants
// ============================================================================

// Time step
constexpr float DT = 1e-4f;

// Lennard-Jones parameters
constexpr float SIGMA   = 1.0f;
constexpr float EPSILON = 1.0f;
constexpr float R_CUTOFF = 2.5f * SIGMA;

// Minimal squared distance to avoid numerical instabilities
constexpr float MIN_DIST2 = 1e-5f;

// ============================================================================
// CPU Cell List force computation
// ============================================================================
//
// n               - number of particles
// m               - particle masses
// x, y, z         - particle positions
// a_x, a_y, a_z   - output accelerations
// sigma, epsilon  - Lennard-Jones parameters
// r_cutoff        - cutoff radius
// ax_out, ay_out,
// az_out          - previous accelerations (for Verlet integration)
// cell_*          - cell list data structures
//

inline void cpu_list_algorithm(
    int n,
    float* m,
    float* x, float* y, float* z,
    float* a_x, float* a_y, float* a_z,
    float sigma, float epsilon, float r_cutoff,
    float* ax_out, float* ay_out, float* az_out,
    int* cell_count, int* cell_start, int* cell_end,
    int* cell_particles)
{
    // Simulation box limits
    constexpr float min_x = -200.0f, max_x = 200.0f;
    constexpr float min_y = -50.0f,  max_y = 50.0f;
    constexpr float min_z = -10.0f,  max_z = 45.0f;

    const float box_x = max_x - min_x;
    const float box_y = max_y - min_y;
    const float box_z = max_z - min_z;

    const float cell_size = 1.5f * r_cutoff;

    // ------------------------------------------------------------------------
    // 1. Apply periodic boundary conditions to particle positions
    // ------------------------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        x[i] = std::fmod(x[i] - min_x, box_x);
        if (x[i] < 0) x[i] += box_x;
        x[i] += min_x;

        y[i] = std::fmod(y[i] - min_y, box_y);
        if (y[i] < 0) y[i] += box_y;
        y[i] += min_y;

        z[i] = std::fmod(z[i] - min_z, box_z);
        if (z[i] < 0) z[i] += box_z;
        z[i] += min_z;
    }

    // ------------------------------------------------------------------------
    // 2. Compute number of cells in each direction
    // ------------------------------------------------------------------------
    const int cells_x = static_cast<int>(std::ceil(box_x / cell_size));
    const int cells_y = static_cast<int>(std::ceil(box_y / cell_size));
    const int cells_z = static_cast<int>(std::ceil(box_z / cell_size));

    const int total_cells = cells_x * cells_y * cells_z;

    for (int i = 0; i < total_cells; ++i) {
        cell_count[i] = 0;
    }

    // ------------------------------------------------------------------------
    // 3. Count particles per cell and store old accelerations
    // ------------------------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        ax_out[i] = a_x[i];
        ay_out[i] = a_y[i];
        az_out[i] = a_z[i];

        int ix = static_cast<int>(std::floor((x[i] - min_x) / cell_size));
        int iy = static_cast<int>(std::floor((y[i] - min_y) / cell_size));
        int iz = static_cast<int>(std::floor((z[i] - min_z) / cell_size));

        ix = std::min(ix, cells_x - 1);
        iy = std::min(iy, cells_y - 1);
        iz = std::min(iz, cells_z - 1);

        const int cell_index = iz * cells_x * cells_y + iy * cells_x + ix;
        ++cell_count[cell_index];
    }

    // ------------------------------------------------------------------------
    // 4. Compute prefix sums for cell particle storage
    // ------------------------------------------------------------------------
    int sum = 0;
    for (int i = 0; i < total_cells; ++i) {
        cell_start[i] = sum;
        sum += cell_count[i];
        cell_end[i] = sum;
        cell_count[i] = 0;
    }

    // ------------------------------------------------------------------------
    // 5. Fill cell particle list
    // ------------------------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        int ix = static_cast<int>(std::floor((x[i] - min_x) / cell_size));
        int iy = static_cast<int>(std::floor((y[i] - min_y) / cell_size));
        int iz = static_cast<int>(std::floor((z[i] - min_z) / cell_size));

        ix = std::min(ix, cells_x - 1);
        iy = std::min(iy, cells_y - 1);
        iz = std::min(iz, cells_z - 1);

        const int cell_index = iz * cells_x * cells_y + iy * cells_x + ix;
        const int pos = cell_start[cell_index] + cell_count[cell_index];

        if (pos >= n) {
            std::cerr << "cell_particles overflow\n";
            std::exit(EXIT_FAILURE);
        }

        cell_particles[pos] = i;
        ++cell_count[cell_index];
    }

    // ------------------------------------------------------------------------
    // 6. Force computation using neighboring cells
    // ------------------------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        int ix = static_cast<int>(std::floor((x[i] - min_x) / cell_size));
        int iy = static_cast<int>(std::floor((y[i] - min_y) / cell_size));
        int iz = static_cast<int>(std::floor((z[i] - min_z) / cell_size));

        ix = std::min(ix, cells_x - 1);
        iy = std::min(iy, cells_y - 1);
        iz = std::min(iz, cells_z - 1);

        float ax_new = 0.0f;
        float ay_new = 0.0f;
        float az_new = 0.0f;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {

                    int nix = (ix + dx + cells_x) % cells_x;
                    int niy = (iy + dy + cells_y) % cells_y;
                    int niz = (iz + dz + cells_z) % cells_z;

                    const int cell_index =
                        niz * cells_x * cells_y + niy * cells_x + nix;

                    for (int k = cell_start[cell_index];
                         k < cell_end[cell_index]; ++k) {

                        const int j = cell_particles[k];
                        if (i == j) continue;

                        float dx = x[j] - x[i];
                        float dy = y[j] - y[i];
                        float dz = z[j] - z[i];

                        // Minimum image convention
                        if (dx >  box_x * 0.5f) dx -= box_x;
                        if (dx < -box_x * 0.5f) dx += box_x;
                        if (dy >  box_y * 0.5f) dy -= box_y;
                        if (dy < -box_y * 0.5f) dy += box_y;
                        if (dz >  box_z * 0.5f) dz -= box_z;
                        if (dz < -box_z * 0.5f) dz += box_z;

                        const float r2 = dx*dx + dy*dy + dz*dz;
                        if (r2 > r_cutoff*r_cutoff || r2 < MIN_DIST2) continue;

                        const float inv_r = 1.0f / std::sqrt(r2);
                        const float sr2   = (sigma * inv_r) * (sigma * inv_r);
                        const float sr6   = sr2 * sr2 * sr2;
                        const float sr12  = sr6 * sr6;

                        const float F =
                            24.0f * epsilon * (2.0f * sr12 - sr6) * inv_r;

                        const float inv_m = 1.0f / m[i];
                        ax_new += F * dx * inv_r * inv_m;
                        ay_new += F * dy * inv_r * inv_m;
                        az_new += F * dz * inv_r * inv_m;
                    }
                }
            }
        }

        if (!std::isfinite(ax_new) ||
            !std::isfinite(ay_new) ||
            !std::isfinite(az_new)) {

            std::cerr << "NaN detected in acceleration for particle "
                      << i << std::endl;
            std::exit(EXIT_FAILURE);
        }

        a_x[i] = ax_new;
        a_y[i] = ay_new;
        a_z[i] = az_new;
    }
}

// ============================================================================
// Velocity Verlet integration
// ============================================================================

inline void cpu_integration(
    int n,
    float* x, float* y, float* z,
    float* v_x, float* v_y, float* v_z,
    float* a_x, float* a_y, float* a_z,
    float dt,
    float* ax_old, float* ay_old, float* az_old)
{
    for (int i = 0; i < n; ++i) {
        x[i] += v_x[i] * dt + 0.5f * ax_old[i] * dt * dt;
        y[i] += v_y[i] * dt + 0.5f * ay_old[i] * dt * dt;
        z[i] += v_z[i] * dt + 0.5f * az_old[i] * dt * dt;

        v_x[i] += 0.5f * (ax_old[i] + a_x[i]) * dt;
        v_y[i] += 0.5f * (ay_old[i] + a_y[i]) * dt;
        v_z[i] += 0.5f * (az_old[i] + a_z[i]) * dt;
    }
}
