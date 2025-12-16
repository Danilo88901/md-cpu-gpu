#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h" // Ваши функции create_array, free_gpu_arrays и free_cpu_arrays
#include "gpu_naive.cu" // Ядра naive_kernel и integrate

// --- Simulation constants ---
const float dt = 1e-4f;
const float sigma = 1.0f;
const float epsilon = 1.0f;
const float r_cutoff = 2.5f * sigma;

int main() {
    // --- Файл для записи результатов ---
    std::ofstream fout("gpu_naive_results.csv");
    fout << "n,gpu_naive_time\n";

    const int steps = 500;           // Количество шагов интеграции
    const int block_size = 256;      // Размер блока CUDA

    // --- Размеры систем для тестирования ---
    std::vector<int> arr{ 5,10,20,50,100,200,300,500,600,1000,1500,2000,4000,6000,8000 };
    std::vector<double> gpu_naive_time;

    std::cout << "Starting Naive GPU kernel timing..." << std::endl;

    // --- Основной цикл по разным размерам систем ---
    for (int n : arr) {
        int size = n * sizeof(float);

        // --- Выделение памяти на CPU ---
        float* m = create_array(n);
        float* x = create_array(n);
        float* y = create_array(n);
        float* z = create_array(n);
        float* vx = create_array(n);
        float* vy = create_array(n);
        float* vz = create_array(n);
        float* ax = create_array(n);
        float* ay = create_array(n);
        float* az = create_array(n);
        float* ax_out = create_array(n);
        float* ay_out = create_array(n);
        float* az_out = create_array(n);

        // --- Инициализация случайными значениями ---
        srand(42);
        for (int j = 0; j < n; j++) {
            m[j] = 1 + rand() % 5;
            x[j] = -100 + rand() % 200;
            y[j] = -50 + rand() % 80;
            z[j] = -10 + rand() % 45;
            vx[j] = (static_cast<float>(rand()) / RAND_MAX) - 0.5f;
            vy[j] = (static_cast<float>(rand()) / RAND_MAX) - 0.5f;
            vz[j] = (static_cast<float>(rand()) / RAND_MAX) - 0.5f;
            ax[j] = ay[j] = az[j] = 0.0f;
            ax_out[j] = ay_out[j] = az_out[j] = 0.0f;
        }

        // --- Выделение памяти на GPU ---
        float *d_m, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_ax, *d_ay, *d_az;
        float *d_ax_out, *d_ay_out, *d_az_out;
        cudaMalloc(&d_m, size);
        cudaMalloc(&d_x, size);
        cudaMalloc(&d_y, size);
        cudaMalloc(&d_z, size);
        cudaMalloc(&d_vx, size);
        cudaMalloc(&d_vy, size);
        cudaMalloc(&d_vz, size);
        cudaMalloc(&d_ax, size);
        cudaMalloc(&d_ay, size);
        cudaMalloc(&d_az, size);
        cudaMalloc(&d_ax_out, size);
        cudaMalloc(&d_ay_out, size);
        cudaMalloc(&d_az_out, size);

        // --- Копирование данных с CPU на GPU ---
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

        int grid = (n + block_size - 1) / block_size;

        // --- Измерение времени работы ядра ---
        auto start = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < steps; k++) {
            naive_kernel<<<grid, block_size>>>(
                n, d_m, d_x, d_y, d_z, d_ax, d_ay, d_az,
                sigma, epsilon, r_cutoff, dt,
                d_ax_out, d_ay_out, d_az_out
            );
            integrate<<<grid, block_size>>>(
                n, d_x, d_y, d_z, d_vx, d_vy, d_vz, 
                d_ax, d_ay, d_az, dt,
                d_ax_out, d_ay_out, d_az_out
            );
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        double time_for = dur.count();

        std::cout << "It took: " << time_for << " seconds for " << n << " particles on Naive GPU" << std::endl;
        gpu_naive_time.push_back(time_for);

        // --- Освобождение памяти ---
        free_gpu_arrays(d_m, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);
        cudaFree(d_ax_out);
        cudaFree(d_ay_out);
        cudaFree(d_az_out);
        free_cpu_arrays(m, x, y, z, vx, vy, vz, ax, ay, az);
        free(ax_out); free(ay_out); free(az_out);
    }

    // --- Запись результатов в CSV ---
    for (size_t i = 0; i < arr.size(); i++) {
        fout << arr[i] << "," << gpu_naive_time[i] << std::endl;
    }

    fout.close();
    std::cout << "Results saved to gpu_naive_results.csv" << std::endl;

    return 0;
}
