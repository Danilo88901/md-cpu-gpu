
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include "cpu_kernel.h"
#include "utils.h"
int main() {
    // --- Константы симуляции ---
    const int steps = 500;
    std::vector<int> arr{5,10,20,50,100,200,300,500,600,1000,1500,2000};

    std::cout << "=== CPU Naive Benchmark ===" << std::endl;

    // Вектор для хранения времени выполнения
    std::vector<double> cpu_naive_time;

    for (size_t idx = 0; idx < arr.size(); ++idx) {
        int n = arr[idx];

        // --- Создание массивов ---
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
        for (int i = 0; i < n; ++i) {
            m[i] = 1 + rand() % 5;
            x[i] = -100 + rand() % 200;
            y[i] = -50 + rand() % 80;
            z[i] = -10 + rand() % 45;
            vx[i] = -0.5f + static_cast<float>(rand()) / RAND_MAX;
            vy[i] = -0.5f + static_cast<float>(rand()) / RAND_MAX;
            vz[i] = -0.5f + static_cast<float>(rand()) / RAND_MAX;
            ax[i] = ay[i] = az[i] = 0.0f;
            ax_out[i] = ay_out[i] = az_out[i] = 0.0f;
        }

        // --- Замер времени ---
        auto start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < steps; ++step) {
            cpu_kernel(n, m, x, y, z, ax, ay, az, sigma, epsilon, r_cutoff, dt, ax_out, ay_out, az_out);
            cpu_integration(n, x, y, z, vx, vy, vz, ax, ay, az, dt, ax_out, ay_out, az_out);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

        std::cout << "N = " << n << " particles: " << elapsed_sec << " seconds" << std::endl;
        cpu_naive_time.push_back(elapsed_sec);

        // --- Очистка памяти ---
        free_cpu_arrays(m, x, y, z, vx, vy, vz, ax, ay, az);
        free(ax_out); free(ay_out); free(az_out);
    }

    // --- Сохранение результатов в CSV ---
    std::ofstream fout("cpu_naive.csv");
    fout << "n,time\n";
    for (size_t i = 0; i < arr.size(); ++i)
        fout << arr[i] << "," << cpu_naive_time[i] << "\n";
    fout.close();

    std::cout << "Benchmark finished. Results saved to cpu_naive.csv" << std::endl;

    return 0;
}
