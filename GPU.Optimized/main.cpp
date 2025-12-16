#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include<gpu_optimized.cu>
#include<utils.h>

int main() {
    std::ofstream fout("gpu_optimized_results.csv");
    fout << "n,gpu_optimized_time\n";

    const int steps = 500;
    const int block_size = 256;

    // Array of particle counts to test
    std::vector<int> particle_counts {5,10,20,50,100,200,300,500,600,1000,1500,2000,4000,6000,8000,10000,12000};
    std::vector<double> gpu_optimized_times;

    std::cout << "Timing Optimized GPU kernel..." << std::endl;

    for (int n : particle_counts) {
        int size = n * sizeof(float);

        // Allocate CPU arrays
        float *m = create_array(n);
        float *x = create_array(n);
        float *y = create_array(n);
        float *z = create_array(n);
        float *vx = create_array(n);
        float *vy = create_array(n);
        float *vz = create_array(n);
        float *ax = create_array(n);
        float *ay = create_array(n);
        float *az = create_array(n);
        float *ax_out = create_array(n);
        float *ay_out = create_array(n);
        float *az_out = create_array(n);

        // Initialize arrays with random values
        srand(42);
        for (int j = 0; j < n; j++) {
            m[j] = 1 + rand() % 5;
            x[j] = -100 + rand() % 200;
            y[j] = -50 + rand() % 80;
            z[j] = -10 + rand() % 45;
            vx[j] = -0.5f + static_cast<float>(rand()) / RAND_MAX;
            vy[j] = -0.5f + static_cast<float>(rand()) / RAND_MAX;
            vz[j] = -0.5f + static_cast<float>(rand()) / RAND_MAX;
            ax[j] = ay[j] = az[j] = 0.0f;
            ax_out[j] = ay_out[j] = az_out[j] = 0.0f;
        }

        // Allocate GPU arrays
        float *d_m, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_ax, *d_ay, *d_az, *d_ax_out, *d_ay_out, *d_az_out;
        cudaMalloc(&d_m, size); cudaMalloc(&d_x, size); cudaMalloc(&d_y, size); cudaMalloc(&d_z, size);
        cudaMalloc(&d_vx, size); cudaMalloc(&d_vy, size); cudaMalloc(&d_vz, size);
        cudaMalloc(&d_ax, size); cudaMalloc(&d_ay, size); cudaMalloc(&d_az, size);
        cudaMalloc(&d_ax_out, size); cudaMalloc(&d_ay_out, size); cudaMalloc(&d_az_out, size);

        // Copy data from CPU to GPU
        cudaMemcpy(d_m, m, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, z, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vx, vx, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vy, vy, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vz, vz, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ax, ax, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ay, ay, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_az, az, size, cudaMemcpyHostToDevice);

        // Determine grid size
        int grid = (n + block_size - 1) / block_size;

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < steps; k++) {
            optimized_gpu<<<grid, block_size>>>(n, d_m, d_x, d_y, d_z, d_ax, d_ay, d_az,
                                                sigma, epsilon, r_cutoff, dt,
                                                d_ax_out, d_ay_out, d_az_out);
            integrate<<<grid, block_size>>>(n, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                            d_ax, d_ay, d_az,
                                            dt, d_ax_out, d_ay_out, d_az_out);
        }

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double time_for = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        std::cout << "It took: " << time_for << " seconds for " << n << " particles on Optimized GPU" << std::endl;
        gpu_optimized_times.push_back(time_for);

        // Free memory
        free_gpu_arrays(d_m, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);
        cudaFree(d_ax_out); cudaFree(d_ay_out); cudaFree(d_az_out);
        free_cpu_arrays(m, x, y, z, vx, vy, vz, ax, ay, az);
        free(ax_out); free(ay_out); free(az_out);
    }

    // Write results to CSV
    for (size_t i = 0; i < particle_counts.size(); i++) {
        fout << particle_counts[i] << "," << gpu_optimized_times[i] << "\n";
    }

    fout.close();
    return 0;
}
