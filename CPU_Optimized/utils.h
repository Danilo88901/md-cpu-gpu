// --- Utility functions for memory management ---
// Allocate float array on host
float* create_array(int n) {
    return (float*)malloc(n * sizeof(float));
}

// Free GPU device arrays
void free_gpu_arrays(float* d_m, float* d_x, float* d_y, float* d_z,
                     float* d_vx, float* d_vy, float* d_vz,
                     float* d_ax, float* d_ay, float* d_az) {
    cudaFree(d_m); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
}

// Free CPU host arrays
void free_cpu_arrays(float* m, float* x, float* y, float* z,
                     float* vx, float* vy, float* vz,
                     float* ax, float* ay, float* az) {
    free(m); free(x); free(y); free(z);
    free(vx); free(vy); free(vz);
    free(ax); free(ay); free(az);
}
