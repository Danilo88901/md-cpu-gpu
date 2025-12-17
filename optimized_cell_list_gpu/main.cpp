#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include<cmath>
#include<vector>
#include<chrono>
#include <fstream>
#include<utils.h>
#include<optimized_cell_list_gpu.cu>

#define CUDA_CHECK() { cudaError_t e = cudaGetLastError(); if(e != cudaSuccess) { \
  std::cerr << "CUDA ERR: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1);} }
float MIN_DIST2 = 1e-5;
const float dt = 1e-4;
const float sigma = 1;
const float epsilon = 1;
const float r_cutoff = 2.5f * sigma;
int main() {
	std::ofstream fout("gpu_optimized_results.csv");
	fout << "n,gpu_optimized_time\n";
	const int block_size = 256;
	int steps = 500;
	std::vector<int>arr{ 5,20,100,500,1000,2000,4000,6000,8000,10000,15000,20000,25000,30000,40000,50000,70000,100000 };
	std::vector<double>gpu_optimized;
	for (int i{ 0 };i < arr.size();i++) {
		const float min_x = -200, max_x = 200;  // длина по x = 400
		const float min_y = -50, max_y = 50;   // длина по y = 100
		const float min_z = -10, max_z = 45;
		float box_x = max_x - min_x; // 400
		float box_y = max_y - min_y; // 100
		float box_z = max_z - min_z; // 55
		int cells_x = std::ceil(box_x / (r_cutoff * 1.5f));
		int cells_y = std::ceil(box_y / (r_cutoff * 1.5f));
		int cells_z = std::ceil(box_z / (r_cutoff * 1.5f));
		int num_cells = cells_x * cells_y * cells_z;
		int n = arr[i];
		int size = n * sizeof(float);
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
		int* cell_count = (int*)calloc(num_cells, sizeof(int));
		int* cell_start = (int*)calloc(num_cells, sizeof(int));
		int* cell_end = (int*)calloc(num_cells, sizeof(int));
		int* cell_particles = (int*)calloc(n, sizeof(int));
		for (int j{ 0 };j < n;j++) {
			m[j] = 1 + rand() % 5;
			x[j] = -100 + rand() % 200;
			y[j] = -50 + rand() % 80;
			z[j] = -10 + rand() % 45;
			vx[j] = -0.5 + static_cast<float>(rand()) / RAND_MAX;
			vy[j] = -0.5 + static_cast<float>(rand()) / RAND_MAX;
			vz[j] = -0.5 + static_cast<float>(rand()) / RAND_MAX;
			ax[j] = 0;
			ay[j] = 0;
			az[j] = 0;
			ax_out[j] = 0;
			ay_out[j] = 0;
			az_out[j] = 0;
		}
		float* d_m, * d_x, * d_y, * d_z, * d_vx, * d_vy, * d_vz, * d_ax, * d_ay, * d_az, * d_ax_out, * d_ay_out, * d_az_out;
		int* d_cell_start, * d_cell_end, * d_cell_count, * d_cell_particles;
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
		cudaMalloc(&d_cell_count, num_cells * sizeof(int));
		cudaMalloc(&d_cell_start, num_cells * sizeof(int));
		cudaMalloc(&d_cell_end, num_cells * sizeof(int));
		cudaMalloc(&d_cell_particles, n * sizeof(int));

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
		cudaMemcpy(d_cell_count, cell_count, num_cells * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_cell_start, cell_start, num_cells * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_cell_end, cell_end, num_cells * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_cell_particles, cell_particles, n * sizeof(int), cudaMemcpyHostToDevice);
		int grid = (n + block_size - 1) / block_size;
		auto start = std::chrono::high_resolution_clock::now();
		for (int k{ 0 };k < steps;k++) {
			cpu_list(n, x, y, z, r_cutoff, cell_count, cell_start, cell_end, cell_particles);

			// синхронизируем координаты: host -> device
			cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_z, z, size, cudaMemcpyHostToDevice);

			// затем копируем cell list
			cudaMemcpy(d_cell_count, cell_count, num_cells * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_cell_start, cell_start, num_cells * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_cell_end, cell_end, num_cells * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_cell_particles, cell_particles, n * sizeof(int), cudaMemcpyHostToDevice);

			// launch kernels
			super_optimized_gpu << <grid, block_size >> > (n, d_m, d_x, d_y, d_z, d_ax, d_ay, d_az, sigma, epsilon, r_cutoff, dt, d_ax_out, d_ay_out, d_az_out,
				d_cell_count, d_cell_start, d_cell_end, d_cell_particles, min_x, min_y, min_z, cells_x, cells_y, cells_z,
				box_x, box_y, box_z);
			CUDA_CHECK();
			integrate << <grid, block_size >> > (n, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, dt, d_ax_out, d_ay_out, d_az_out);
			CUDA_CHECK();

			// sync and copy coordinates back for next iteration (если оставляем построение списка на CPU)
			cudaDeviceSynchronize();
			cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(z, d_z, size, cudaMemcpyDeviceToHost);
		}
		cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();
		double time = 1000;//to change miliseconds into seconds
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double time_for = dur.count() / time;
		std::cout << "It took:" << time_for << " seconds" << " for " << n << " elements on Super-Optimized GPU" << std::endl;
		gpu_optimized.push_back(time_for);
		free_gpu_arrays(d_m, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);
		free_cpu_arrays(m, x, y, z, vx, vy, vz, ax, ay, az);

		cudaFree(d_ax_out);
		cudaFree(d_ay_out);
		cudaFree(d_az_out);
		cudaFree(d_cell_count);
		cudaFree(d_cell_end);
		cudaFree(d_cell_particles);
		cudaFree(d_cell_start);
		free(ax_out);
		free(ay_out);
		free(az_out);
		free(cell_count);
		free(cell_end);
		free(cell_start);
		free(cell_particles);
	}
	for (int i{ 0 };i < arr.size();i++) {
		fout << arr[i] << "," << gpu_optimized[i] << std::endl;
	}
}
