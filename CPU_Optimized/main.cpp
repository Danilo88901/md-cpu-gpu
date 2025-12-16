#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>

#include "cpu_optimized.h"
#include "utils.h"

// ============================================================================
// Benchmark driver for CPU optimized Cell List MD implementation
// ============================================================================
//
// This program benchmarks the CPU Cell List force computation for different
// particle counts and stores timing results in CSV format.
//
// ============================================================================

int main() {

    std::ofstream fout("cpu_optimized_results.csv");
    fout << "n,cpu_optimized_time\n";

    const int steps = 500;

    // Particle counts to benchmark
    std::vector<int> particle_counts = {
        5, 10, 20, 50, 100, 200, 300, 500, 600,
        1000, 1500, 2000, 4000, 6000, 8000, 10000, 12000
    };

    std::vector<double> cpu_optimized_time;

    std::cout << "Benchmarking CPU Optimized Cell List\n";

    for (int n : particle_counts) {

        // Simulation box limits
        constexpr float min_x = -200.0f, max_x = 200.0f;
        constexpr float min_y = -50.0f,  max_y = 50.0f;
        constexpr float min_z = -10.0f,  max_z = 45.0f;

        const float box_x = max_x - min_x;
        const float box_y = max_y - min_y;
        const float box_z = max_z - min_z;

        const int cells_x = static_cast<int>(std::ceil(box_x / (R_CUTOFF * 1.5f)));
        const int cells_y = static_cast<int>(std::ceil(box_y / (R_CUTOFF * 1.5f)));
        const int cells_z = static_cast<int>(std::ceil(box_z / (R_CUTOFF * 1.5f)));

        const int num_cells = cells_x * cells_y * cells_z;

        // Allocate particle data
        float* m  = create_array(n);
        float* x  = create_array(n);
        float* y  = create_array(n);
        float* z  = create_array(n);
        float* vx = create_array(n);
        float* vy = create_array(n);
        float* vz = create_array(n);
        float* ax = create_array(n);
        float* ay = create_array(n);
        float* az = create_array(n);

        float* ax_old = create_array(n);
        float* ay_old = create_array(n);
        float* az_old = create_array(n);

        int* cell_count     = (int*)calloc(num_cells, sizeof(int));
        int* cell_start     = (int*)calloc(num_cells, sizeof(int));
        int* cell_end       = (int*)calloc(num_cells, sizeof(int));
        int* cell_particles = (int*)calloc(n, sizeof(int));

        // Initialize particles
        srand(42);
        for (int i = 0; i < n; ++i) {
            m[i]  = 1.0f + rand() % 5;
            x[i]  = min_x + static_cast<float>(rand()) / RAND_MAX * box_x;
            y[i]  = min_y + static_cast<float>(rand()) / RAND_MAX * box_y;
            z[i]  = min_z + static_cast<float>(rand()) / RAND_MAX * box_z;

            vx[i] = -0.5f + static_cast<float>(rand()) / RAND_MAX;
            vy[i] = -0.5f + static_cast<float>(rand()) / RAND_MAX;
            vz[i] = -0.5f + static_cast<float>(rand()) / RAND_MAX;

            ax[i] = ay[i] = az[i] = 0.0f;
            ax_old[i] = ay_old[i] = az_old[i] = 0.0f;
        }

        // Run benchmark
        auto start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < steps; ++step) {
            cpu_list_algorithm(
                n, m, x, y, z,
                ax, ay, az,
                SIGMA, EPSILON, R_CUTOFF,
                ax_old, ay_old, az_old,
                cell_count, cell_start, cell_end, cell_particles
            );

            cpu_integration(
                n, x, y, z,
                vx, vy, vz,
                ax, ay, az,
                DT,
                ax_old, ay_old, az_old
            );
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            / 1000.0;

        std::cout << "N = " << n
                  << " | Time = " << elapsed << " s\n";

        cpu_optimized_time.push_back(elapsed);

        // Free memory
        free_cpu_arrays(m, x, y, z, vx, vy, vz, ax, ay, az);
        free(ax_old); free(ay_old); free(az_old);
        free(cell_count); free(cell_start); free(cell_end); free(cell_particles);
    }

    // Save results to CSV
    for (size_t i = 0; i < particle_counts.size(); ++i) {
        fout << particle_counts[i] << "," << cpu_optimized_time[i] << "\n";
    }

    return 0;
}
