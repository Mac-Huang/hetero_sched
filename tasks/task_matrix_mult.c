#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "scheduler.h"

// ============================================================================
// Matrix Multiplication Task Implementation
// ============================================================================

typedef struct {
    float *matrix_a;
    float *matrix_b;
    float *result;
    size_t rows;
    size_t cols;
    size_t inner_dim;
} MatrixMultTask;

// Generate test matrices with known patterns for verification
void generate_test_matrices(float *a, float *b, size_t rows, size_t cols) {
    // Initialize matrix A with row index pattern
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            a[i * cols + j] = (float)(i + 1);
        }
    }
    
    // Initialize matrix B with column index pattern  
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            b[i * cols + j] = (float)(j + 1);
        }
    }
}

// Verify matrix multiplication result
bool verify_matrix_result(float *result, size_t rows, size_t cols) {
    // For our test pattern, result[i][j] should be (i+1) * sum(1 to cols)
    float expected_sum = (float)(cols * (cols + 1)) / 2.0f;
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float expected = (float)(i + 1) * expected_sum;
            float actual = result[i * cols + j];
            float error = fabsf(actual - expected) / expected;
            
            if (error > 1e-5) {
                printf("Verification failed at [%zu][%zu]: expected %.2f, got %.2f\n",
                       i, j, expected, actual);
                return false;
            }
        }
    }
    return true;
}

// Benchmark different matrix sizes
void benchmark_matrix_sizes(void) {
    printf("\n=== Matrix Multiplication Benchmark ===\n");
    
    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Size\tCPU Time (ms)\tGPU Time (ms)\tSpeedup\tRecommendation\n");
    printf("----\t-------------\t-------------\t-------\t--------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];
        size_t total_elements = size * size;
        
        // Allocate matrices
        float *a = malloc(total_elements * sizeof(float));
        float *b = malloc(total_elements * sizeof(float));
        float *cpu_result = malloc(total_elements * sizeof(float));
        float *gpu_result = malloc(total_elements * sizeof(float));
        
        generate_test_matrices(a, b, size, size);
        
        // Benchmark CPU
        double cpu_start = get_time();
        cpu_matrix_mult(a, b, cpu_result, size, size);
        double cpu_time = get_time() - cpu_start;
        
        // Verify CPU result
        if (!verify_matrix_result(cpu_result, size, size)) {
            printf("CPU result verification failed for size %zu\n", size);
        }
        
        double gpu_time = -1.0;
        bool gpu_verified = false;
        
#ifdef USE_CUDA
        if (cuda_initialized) {
            // Benchmark GPU
            double gpu_start = get_time();
            gpu_matrix_mult(a, b, gpu_result, size, size);
            gpu_time = get_time() - gpu_start;
            
            // Verify GPU result
            gpu_verified = verify_matrix_result(gpu_result, size, size);
            if (!gpu_verified) {
                printf("GPU result verification failed for size %zu\n", size);
            }
        }
#endif
        
        // Calculate speedup and recommendation
        double speedup = (gpu_time > 0) ? cpu_time / gpu_time : 0.0;
        const char *recommendation = "CPU";
        
        if (gpu_time > 0 && speedup > 1.2) {
            recommendation = "GPU";
        } else if (gpu_time > 0 && speedup > 0.8) {
            recommendation = "Either";
        }
        
        printf("%zu\t%.2f\t\t", size, cpu_time * 1000);
        if (gpu_time > 0) {
            printf("%.2f\t\t%.2fx\t%s\n", gpu_time * 1000, speedup, recommendation);
        } else {
            printf("N/A\t\tN/A\t%s\n", recommendation);
        }
        
        free(a);
        free(b);
        free(cpu_result);
        free(gpu_result);
    }
}

// Test matrix multiplication with different algorithms
void test_matrix_algorithms(void) {
    printf("\n=== Matrix Algorithm Comparison ===\n");
    
    const size_t test_size = 256;
    size_t total_elements = test_size * test_size;
    
    float *a = malloc(total_elements * sizeof(float));
    float *b = malloc(total_elements * sizeof(float));
    float *result1 = malloc(total_elements * sizeof(float));
    float *result2 = malloc(total_elements * sizeof(float));
    
    // Initialize with random values
    srand(42); // Fixed seed for reproducibility
    for (size_t i = 0; i < total_elements; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }
    
    // Test standard CPU implementation
    double start = get_time();
    cpu_matrix_mult(a, b, result1, test_size, test_size);
    double cpu_standard_time = get_time() - start;
    
    printf("Standard CPU algorithm: %.2f ms\n", cpu_standard_time * 1000);
    
    // Test blocked CPU implementation (if available)
    start = get_time();
    cpu_matrix_mult_blocked(a, b, result2, test_size, test_size);
    double cpu_blocked_time = get_time() - start;
    
    printf("Blocked CPU algorithm: %.2f ms (%.2fx speedup)\n", 
           cpu_blocked_time * 1000, cpu_standard_time / cpu_blocked_time);
    
    // Verify results match
    bool results_match = true;
    for (size_t i = 0; i < total_elements; i++) {
        if (fabsf(result1[i] - result2[i]) > 1e-5) {
            results_match = false;
            break;
        }
    }
    
    printf("Algorithm results match: %s\n", results_match ? "Yes" : "No");
    
#ifdef USE_CUDA
    if (cuda_initialized) {
        // Test GPU implementation
        start = get_time();
        gpu_matrix_mult(a, b, result2, test_size, test_size);
        double gpu_time = get_time() - start;
        
        printf("GPU algorithm: %.2f ms (%.2fx speedup vs standard CPU)\n",
               gpu_time * 1000, cpu_standard_time / gpu_time);
        
        // Verify GPU result
        results_match = true;
        for (size_t i = 0; i < total_elements; i++) {
            if (fabsf(result1[i] - result2[i]) > 1e-4) { // Slightly looser tolerance for GPU
                results_match = false;
                break;
            }
        }
        
        printf("GPU vs CPU results match: %s\n", results_match ? "Yes" : "No");
    }
#endif
    
    free(a);
    free(b);
    free(result1);
    free(result2);
}

// Performance analysis for different workload patterns
void analyze_matrix_workloads(void) {
    printf("\n=== Matrix Workload Analysis ===\n");
    
    // Test square matrices
    printf("Square matrices:\n");
    size_t square_sizes[] = {64, 128, 256, 512};
    for (int i = 0; i < 4; i++) {
        size_t size = square_sizes[i];
        float *a = malloc(size * size * sizeof(float));
        float *b = malloc(size * size * sizeof(float));
        float *result = malloc(size * size * sizeof(float));
        
        generate_test_matrices(a, b, size, size);
        
        double start = get_time();
        cpu_matrix_mult(a, b, result, size, size);
        double cpu_time = get_time() - start;
        
        printf("  %zux%zu: %.2f GFLOPS\n", size, size, 
               (2.0 * size * size * size) / (cpu_time * 1e9));
        
        free(a);
        free(b);
        free(result);
    }
    
    // Test rectangular matrices
    printf("\nRectangular matrices:\n");
    struct { size_t rows, cols; } rect_sizes[] = {
        {128, 256}, {256, 128}, {512, 128}, {128, 512}
    };
    
    for (int i = 0; i < 4; i++) {
        size_t rows = rect_sizes[i].rows;
        size_t cols = rect_sizes[i].cols;
        
        float *a = malloc(rows * cols * sizeof(float));
        float *b = malloc(cols * rows * sizeof(float)); // Transposed for multiplication
        float *result = malloc(rows * rows * sizeof(float));
        
        generate_test_matrices(a, b, rows, cols);
        
        double start = get_time();
        cpu_matrix_mult(a, b, result, rows, rows);
        double cpu_time = get_time() - start;
        
        printf("  %zux%zu: %.2f GFLOPS\n", rows, cols,
               (2.0 * rows * cols * rows) / (cpu_time * 1e9));
        
        free(a);
        free(b);
        free(result);
    }
}

// Main matrix multiplication task test
int main_matrix_test(void) {
    printf("=== Matrix Multiplication Task Tests ===\n");
    
    benchmark_matrix_sizes();
    test_matrix_algorithms();
    analyze_matrix_workloads();
    
    return 0;
}

// Integration with scheduler system
Task* create_random_matrix_task(int task_id, size_t min_size, size_t max_size) {
    size_t size = min_size + rand() % (max_size - min_size + 1);
    
    float *a = malloc(size * size * sizeof(float));
    float *b = malloc(size * size * sizeof(float));
    float *result = malloc(size * size * sizeof(float));
    
    // Initialize with random values
    for (size_t i = 0; i < size * size; i++) {
        a[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f; // Range [-5, 5]
        b[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    }
    
    return create_matrix_mult_task(task_id, size, size, a, b, result);
}

// Batch processing for multiple matrix tasks
void process_matrix_batch(Task **tasks, int num_tasks) {
    printf("Processing batch of %d matrix multiplication tasks\n", num_tasks);
    
    double total_cpu_time = 0.0;
    double total_gpu_time = 0.0;
    int cpu_count = 0;
    int gpu_count = 0;
    
    for (int i = 0; i < num_tasks; i++) {
        Task *task = tasks[i];
        bool use_gpu = use_gpu_for(task);
        
        double start = get_time();
        
        if (use_gpu) {
#ifdef USE_CUDA
            gpu_matrix_mult((float*)task->input1, (float*)task->input2,
                           (float*)task->output, task->rows, task->cols);
            gpu_count++;
            total_gpu_time += get_time() - start;
#endif
        } else {
            cpu_matrix_mult((float*)task->input1, (float*)task->input2,
                           (float*)task->output, task->rows, task->cols);
            cpu_count++;
            total_cpu_time += get_time() - start;
        }
    }
    
    printf("Batch results:\n");
    printf("  CPU tasks: %d (%.2f ms average)\n", 
           cpu_count, cpu_count > 0 ? (total_cpu_time * 1000 / cpu_count) : 0);
    printf("  GPU tasks: %d (%.2f ms average)\n",
           gpu_count, gpu_count > 0 ? (total_gpu_time * 1000 / gpu_count) : 0);
    printf("  Total time: %.2f ms\n", (total_cpu_time + total_gpu_time) * 1000);
}