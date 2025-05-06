/**
 * CUDA Vector Addition Example
 * 
 * This program demonstrates vector addition using CUDA.
 * It compares the performance of CPU and GPU implementations.
 * 
 * Compile with:
 * nvcc -o cuda_vector_add cuda_vector_add.cu
 * 
 * Run with:
 * ./cuda_vector_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Get global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

// CPU implementation of vector addition
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Function to verify results
bool verifyResults(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            printf("Verification failed at element %d! CPU: %f, GPU: %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Vector sizes to test
    int sizes[] = {1000000, 10000000, 50000000, 100000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("\n=================================================================\n");
    printf("CUDA VECTOR ADDITION BENCHMARK\n");
    printf("=================================================================\n\n");
    
    // Print device information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device Information:\n");
    printf("  Device name: %s\n", deviceProp.name);
    printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("  Clock rate: %.2f GHz\n\n", deviceProp.clockRate / 1000000.0);
    
    printf("Vector Addition Benchmark:\n");
    printf("  This test adds two vectors of increasing sizes\n");
    printf("  and compares the time taken on CPU vs GPU.\n\n");
    
    // Print header for results table
    printf("%-15s %-15s %-15s %-15s\n", "Vector Size", "CPU Time (ms)", "GPU Time (ms)", "Speedup");
    printf("----------------------------------------------------------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        size_t bytes = n * sizeof(float);
        
        // Allocate host memory
        float *h_a = (float*)malloc(bytes);
        float *h_b = (float*)malloc(bytes);
        float *h_c_cpu = (float*)malloc(bytes);
        float *h_c_gpu = (float*)malloc(bytes);
        
        // Initialize vectors
        for (int j = 0; j < n; j++) {
            h_a[j] = rand() / (float)RAND_MAX;
            h_b[j] = rand() / (float)RAND_MAX;
        }
        
        // CPU implementation
        clock_t cpu_start = clock();
        vectorAddCPU(h_a, h_b, h_c_cpu, n);
        clock_t cpu_end = clock();
        double cpu_time_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
        
        // Allocate device memory
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        
        // Copy data from host to device
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
        
        // Launch kernel
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Record start event
        cudaEventRecord(start);
        
        // Launch kernel
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        
        // Record stop event
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        float gpu_time_ms = 0;
        cudaEventElapsedTime(&gpu_time_ms, start, stop);
        
        // Copy result back to host
        cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
        
        // Verify results
        bool correct = verifyResults(h_c_cpu, h_c_gpu, n);
        
        // Calculate speedup
        float speedup = cpu_time_ms / gpu_time_ms;
        
        // Print results
        printf("%-15d %-15.2f %-15.2f %-15.2f %s\n", 
               n, 
               cpu_time_ms, 
               gpu_time_ms, 
               speedup,
               correct ? "" : "VERIFICATION FAILED!");
        
        // Free memory
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("\n=================================================================\n");
    printf("Benchmark completed!\n");
    printf("=================================================================\n\n");
    
    return 0;
}
