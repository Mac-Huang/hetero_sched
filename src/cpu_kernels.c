#include <stdio.h>
#include <math.h>
#include <string.h>
#include "scheduler.h"

// ============================================================================
// CPU Kernel Implementations
// ============================================================================

void cpu_vector_add(float *a, float *b, float *result, size_t size) {
    for (size_t i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void cpu_matrix_mult(float *a, float *b, float *result, size_t rows, size_t cols) {
    // Initialize result matrix to zero
    memset(result, 0, rows * cols * sizeof(float));
    
    // Standard matrix multiplication: C = A * B
    // Assuming square matrices for simplicity
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            for (size_t k = 0; k < cols; k++) {
                result[i * cols + j] += a[i * cols + k] * b[k * cols + j];
            }
        }
    }
}

void cpu_vector_scale(float *input, float *output, float scale, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] * scale;
    }
}

void cpu_relu(float *input, float *output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

// ============================================================================
// Optimized CPU Kernels (Optional)
// ============================================================================

// Blocked matrix multiplication for better cache performance
void cpu_matrix_mult_blocked(float *a, float *b, float *result, size_t rows, size_t cols) {
    const size_t BLOCK_SIZE = 64; // Tune this based on cache size
    
    // Initialize result matrix to zero
    memset(result, 0, rows * cols * sizeof(float));
    
    for (size_t i = 0; i < rows; i += BLOCK_SIZE) {
        for (size_t j = 0; j < cols; j += BLOCK_SIZE) {
            for (size_t k = 0; k < cols; k += BLOCK_SIZE) {
                
                // Process block
                size_t i_end = (i + BLOCK_SIZE < rows) ? i + BLOCK_SIZE : rows;
                size_t j_end = (j + BLOCK_SIZE < cols) ? j + BLOCK_SIZE : cols;
                size_t k_end = (k + BLOCK_SIZE < cols) ? k + BLOCK_SIZE : cols;
                
                for (size_t ii = i; ii < i_end; ii++) {
                    for (size_t jj = j; jj < j_end; jj++) {
                        float sum = result[ii * cols + jj];
                        for (size_t kk = k; kk < k_end; kk++) {
                            sum += a[ii * cols + kk] * b[kk * cols + jj];
                        }
                        result[ii * cols + jj] = sum;
                    }
                }
            }
        }
    }
}

// SIMD-optimized vector operations (if compiler supports auto-vectorization)
void cpu_vector_add_optimized(float *a, float *b, float *result, size_t size) {
    // Enable compiler auto-vectorization with pragma
    #pragma GCC ivdep
    for (size_t i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

// ============================================================================
// CPU Performance Testing
// ============================================================================

void cpu_warmup(void) {
    // Warm up CPU caches and frequency scaling
    const size_t warmup_size = 10000;
    float *a = malloc(warmup_size * sizeof(float));
    float *b = malloc(warmup_size * sizeof(float));
    float *result = malloc(warmup_size * sizeof(float));
    
    for (size_t i = 0; i < warmup_size; i++) {
        a[i] = (float)i;
        b[i] = (float)(i + 1);
    }
    
    // Perform several iterations to warm up
    for (int iter = 0; iter < 100; iter++) {
        cpu_vector_add(a, b, result, warmup_size);
    }
    
    free(a);
    free(b);
    free(result);
}

// ============================================================================
// CPU Kernel Benchmarking
// ============================================================================

typedef struct {
    const char *name;
    double time;
    size_t operations;
} BenchmarkResult;

BenchmarkResult benchmark_cpu_vector_add(size_t size, int iterations) {
    float *a = malloc(size * sizeof(float));
    float *b = malloc(size * sizeof(float));
    float *result = malloc(size * sizeof(float));
    
    // Initialize test data
    for (size_t i = 0; i < size; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)i * 0.3f;
    }
    
    double start_time = get_time();
    
    for (int i = 0; i < iterations; i++) {
        cpu_vector_add(a, b, result, size);
    }
    
    double end_time = get_time();
    
    BenchmarkResult bench_result;
    bench_result.name = "CPU Vector Add";
    bench_result.time = end_time - start_time;
    bench_result.operations = size * iterations;
    
    free(a);
    free(b);
    free(result);
    
    return bench_result;
}

BenchmarkResult benchmark_cpu_matrix_mult(size_t size, int iterations) {
    size_t total_elements = size * size;
    float *a = malloc(total_elements * sizeof(float));
    float *b = malloc(total_elements * sizeof(float));
    float *result = malloc(total_elements * sizeof(float));
    
    // Initialize test data
    for (size_t i = 0; i < total_elements; i++) {
        a[i] = (float)i * 0.01f;
        b[i] = (float)i * 0.02f;
    }
    
    double start_time = get_time();
    
    for (int i = 0; i < iterations; i++) {
        cpu_matrix_mult(a, b, result, size, size);
    }
    
    double end_time = get_time();
    
    BenchmarkResult bench_result;
    bench_result.name = "CPU Matrix Mult";
    bench_result.time = end_time - start_time;
    bench_result.operations = size * size * size * iterations; // O(n^3) operations
    
    free(a);
    free(b);
    free(result);
    
    return bench_result;
}

void print_benchmark_result(BenchmarkResult result) {
    double ops_per_sec = result.operations / result.time;
    double gflops = ops_per_sec / 1e9;
    
    printf("%s: %.4f seconds, %.2f GFLOPS\n", 
           result.name, result.time, gflops);
}

// Run comprehensive CPU benchmarks
void run_cpu_benchmarks(void) {
    printf("\n=== CPU Kernel Benchmarks ===\n");
    
    cpu_warmup();
    
    // Vector addition benchmarks
    size_t vec_sizes[] = {1000, 10000, 100000, 1000000};
    int vec_iterations[] = {10000, 1000, 100, 10};
    
    printf("\nVector Addition:\n");
    for (int i = 0; i < 4; i++) {
        BenchmarkResult result = benchmark_cpu_vector_add(vec_sizes[i], vec_iterations[i]);
        printf("Size %zu: ", vec_sizes[i]);
        print_benchmark_result(result);
    }
    
    // Matrix multiplication benchmarks
    size_t mat_sizes[] = {32, 64, 128, 256};
    int mat_iterations[] = {100, 10, 2, 1};
    
    printf("\nMatrix Multiplication:\n");
    for (int i = 0; i < 4; i++) {
        BenchmarkResult result = benchmark_cpu_matrix_mult(mat_sizes[i], mat_iterations[i]);
        printf("Size %zux%zu: ", mat_sizes[i], mat_sizes[i]);
        print_benchmark_result(result);
    }
}
