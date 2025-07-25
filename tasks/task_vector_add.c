#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "scheduler.h"

// ============================================================================
// Vector Addition Task Implementation
// ============================================================================

typedef struct {
    float *vector_a;
    float *vector_b;
    float *result;
    size_t size;
    float scale_factor;
} VectorAddTask;

// Generate test vectors with known patterns for verification
void generate_test_vectors(float *a, float *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)i * 0.3f;
    }
}

// Verify vector addition result
bool verify_vector_result(float *result, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float expected = (float)i * 0.8f; // 0.5 + 0.3 = 0.8
        float actual = result[i];
        float error = fabsf(actual - expected);
        
        if (error > 1e-5) {
            printf("Verification failed at index %zu: expected %.6f, got %.6f\n",
                   i, expected, actual);
            return false;
        }
    }
    return true;
}

// Benchmark different vector sizes
void benchmark_vector_sizes(void) {
    printf("\n=== Vector Addition Benchmark ===\n");
    
    size_t sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Size\t\tCPU Time (ms)\tGPU Time (ms)\tBandwidth (GB/s)\tSpeedup\tRecommendation\n");
    printf("----\t\t-------------\t-------------\t----------------\t-------\t--------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];
        
        // Allocate vectors
        float *a = malloc(size * sizeof(float));
        float *b = malloc(size * sizeof(float));
        float *cpu_result = malloc(size * sizeof(float));
        float *gpu_result = malloc(size * sizeof(float));
        
        generate_test_vectors(a, b, size);
        
        // Benchmark CPU
        double cpu_start = get_time();
        cpu_vector_add(a, b, cpu_result, size);
        double cpu_time = get_time() - cpu_start;
        
        // Verify CPU result
        if (!verify_vector_result(cpu_result, size)) {
            printf("CPU result verification failed for size %zu\n", size);
        }
        
        // Calculate memory bandwidth (3 arrays: 2 reads + 1 write)
        double bytes_transferred = size * 3 * sizeof(float);
        double cpu_bandwidth = bytes_transferred / (cpu_time * 1e9);
        
        double gpu_time = -1.0;
        double gpu_bandwidth = 0.0;
        bool gpu_verified = false;
        
#ifdef USE_CUDA
        if (cuda_initialized) {
            // Benchmark GPU
            double gpu_start = get_time();
            gpu_vector_add(a, b, gpu_result, size);
            gpu_time = get_time() - gpu_start;
            
            // Verify GPU result
            gpu_verified = verify_vector_result(gpu_result, size);
            if (!gpu_verified) {
                printf("GPU result verification failed for size %zu\n", size);
            }
            
            gpu_bandwidth = bytes_transferred / (gpu_time * 1e9);
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
        
        printf("%zu\t\t%.2f\t\t", size, cpu_time * 1000);
        if (gpu_time > 0) {
            printf("%.2f\t\t%.2f\t\t\t%.2fx\t%s\n", 
                   gpu_time * 1000, gpu_bandwidth, speedup, recommendation);
        } else {
            printf("N/A\t\t%.2f\t\t\tN/A\t%s\n", cpu_bandwidth, recommendation);
        }
        
        free(a);
        free(b);
        free(cpu_result);
        free(gpu_result);
    }
}

// Test vector operations with different data patterns
void test_vector_patterns(void) {
    printf("\n=== Vector Pattern Analysis ===\n");
    
    const size_t test_size = 1000000;
    float *a = malloc(test_size * sizeof(float));
    float *b = malloc(test_size * sizeof(float));
    float *result = malloc(test_size * sizeof(float));
    
    // Test different data patterns
    struct {
        const char *name;
        void (*init_func)(float*, float*, size_t);
    } patterns[] = {
        {"Sequential", generate_test_vectors},
        {"Random", NULL}, // Will implement inline
        {"Constant", NULL},
        {"Alternating", NULL}
    };
    
    for (int p = 0; p < 4; p++) {
        printf("\nTesting %s pattern:\n", patterns[p].name);
        
        // Initialize data based on pattern
        if (strcmp(patterns[p].name, "Sequential") == 0) {
            generate_test_vectors(a, b, test_size);
        } else if (strcmp(patterns[p].name, "Random") == 0) {
            srand(42);
            for (size_t i = 0; i < test_size; i++) {
                a[i] = (float)rand() / RAND_MAX * 100.0f;
                b[i] = (float)rand() / RAND_MAX * 100.0f;
            }
        } else if (strcmp(patterns[p].name, "Constant") == 0) {
            for (size_t i = 0; i < test_size; i++) {
                a[i] = 3.14159f;
                b[i] = 2.71828f;
            }
        } else if (strcmp(patterns[p].name, "Alternating") == 0) {
            for (size_t i = 0; i < test_size; i++) {
                a[i] = (i % 2) ? 1.0f : -1.0f;
                b[i] = (i % 2) ? -1.0f : 1.0f;
            }
        }
        
        // Benchmark CPU
        double cpu_start = get_time();
        cpu_vector_add(a, b, result, test_size);
        double cpu_time = get_time() - cpu_start;
        
        printf("  CPU: %.2f ms (%.2f GB/s)\n", 
               cpu_time * 1000, 
               (test_size * 3 * sizeof(float)) / (cpu_time * 1e9));
        
#ifdef USE_CUDA
        if (cuda_initialized) {
            // Benchmark GPU
            double gpu_start = get_time();
            gpu_vector_add(a, b, result, test_size);
            double gpu_time = get_time() - gpu_start;
            
            printf("  GPU: %.2f ms (%.2f GB/s, %.2fx speedup)\n",
                   gpu_time * 1000,
                   (test_size * 3 * sizeof(float)) / (gpu_time * 1e9),
                   cpu_time / gpu_time);
        }
#endif
    }
    
    free(a);
    free(b);
    free(result);
}

// Performance scaling analysis
void analyze_vector_scaling(void) {
    printf("\n=== Vector Scaling Analysis ===\n");
    
    // Test performance scaling with vector size
    printf("Performance scaling:\n");
    size_t base_size = 10000;
    
    for (int multiplier = 1; multiplier <= 64; multiplier *= 2) {
        size_t size = base_size * multiplier;
        
        float *a = malloc(size * sizeof(float));
        float *b = malloc(size * sizeof(float));
        float *result = malloc(size * sizeof(float));
        
        generate_test_vectors(a, b, size);
        
        // Multiple iterations for more accurate timing
        int iterations = 100 / multiplier + 1; // Fewer iterations for larger sizes
        
        double cpu_total_time = 0.0;
        for (int iter = 0; iter < iterations; iter++) {
            double start = get_time();
            cpu_vector_add(a, b, result, size);
            cpu_total_time += get_time() - start;
        }
        
        double cpu_avg_time = cpu_total_time / iterations;
        double cpu_throughput = size / cpu_avg_time / 1e6; // Million elements/sec
        
        printf("  Size %7zu: %.3f ms, %.1f Melem/s", 
               size, cpu_avg_time * 1000, cpu_throughput);
        
#ifdef USE_CUDA
        if (cuda_initialized) {
            double gpu_total_time = 0.0;
            for (int iter = 0; iter < iterations; iter++) {
                double start = get_time();
                gpu_vector_add(a, b, result, size);
                gpu_total_time += get_time() - start;
            }
            
            double gpu_avg_time = gpu_total_time / iterations;
            double gpu_throughput = size / gpu_avg_time / 1e6;
            
            printf(", GPU: %.3f ms, %.1f Melem/s (%.2fx)",
                   gpu_avg_time * 1000, gpu_throughput, cpu_avg_time / gpu_avg_time);
        }
#endif
        
        printf("\n");
        
        free(a);
        free(b);
        free(result);
    }
}

// Memory alignment and access pattern tests
void test_memory_patterns(void) {
    printf("\n=== Memory Access Pattern Tests ===\n");
    
    const size_t test_size = 1000000;
    
    // Test different memory alignments
    printf("Memory alignment effects:\n");
    
    for (int offset = 0; offset < 16; offset += 4) {
        // Allocate extra space for alignment testing
        float *base_a = malloc((test_size + 16) * sizeof(float));
        float *base_b = malloc((test_size + 16) * sizeof(float));
        float *base_result = malloc((test_size + 16) * sizeof(float));
        
        // Apply offset
        float *a = base_a + offset;
        float *b = base_b + offset;
        float *result = base_result + offset;
        
        generate_test_vectors(a, b, test_size);
        
        double start = get_time();
        cpu_vector_add(a, b, result, test_size);
        double cpu_time = get_time() - start;
        
        printf("  Offset %2d: %.2f ms\n", offset, cpu_time * 1000);
        
        free(base_a);
        free(base_b);
        free(base_result);
    }
}

// Cache behavior analysis
void analyze_cache_behavior(void) {
    printf("\n=== Cache Behavior Analysis ===\n");
    
    // Test different array sizes relative to cache sizes
    printf("Array size vs performance:\n");
    
    // Common cache sizes (approximate)
    size_t l1_cache = 32 * 1024 / sizeof(float);      // 32KB L1
    size_t l2_cache = 256 * 1024 / sizeof(float);     // 256KB L2  
    size_t l3_cache = 8 * 1024 * 1024 / sizeof(float); // 8MB L3
    
    size_t test_sizes[] = {
        l1_cache / 4,     // Fits in L1
        l1_cache,         // Exactly L1
        l2_cache / 4,     // Fits in L2
        l2_cache,         // Exactly L2
        l3_cache / 4,     // Fits in L3
        l3_cache,         // Exactly L3
        l3_cache * 4      // Exceeds L3
    };
    
    const char *labels[] = {
        "L1/4", "L1", "L2/4", "L2", "L3/4", "L3", "L3*4"
    };
    
    for (int i = 0; i < 7; i++) {
        size_t size = test_sizes[i];
        
        float *a = malloc(size * sizeof(float));
        float *b = malloc(size * sizeof(float));
        float *result = malloc(size * sizeof(float));
        
        generate_test_vectors(a, b, size);
        
        // Run multiple iterations and take average
        int iterations = 100;
        double total_time = 0.0;
        
        for (int iter = 0; iter < iterations; iter++) {
            double start = get_time();
            cpu_vector_add(a, b, result, size);
            total_time += get_time() - start;
        }
        
        double avg_time = total_time / iterations;
        double bandwidth = (size * 3 * sizeof(float)) / (avg_time * 1e9);
        
        printf("  %s (%7zu elements): %.3f ms, %.2f GB/s\n",
               labels[i], size, avg_time * 1000, bandwidth);
        
        free(a);
        free(b);
        free(result);
    }
}

// Main vector addition task test
int main_vector_test(void) {
    printf("=== Vector Addition Task Tests ===\n");
    
    benchmark_vector_sizes();
    test_vector_patterns();
    analyze_vector_scaling();
    test_memory_patterns();
    analyze_cache_behavior();
    
    return 0;
}

// Integration with scheduler system
Task* create_random_vector_task(int task_id, size_t min_size, size_t max_size) {
    size_t size = min_size + rand() % (max_size - min_size + 1);
    
    float *a = malloc(size * sizeof(float));
    float *b = malloc(size * sizeof(float));
    float *result = malloc(size * sizeof(float));
    
    // Initialize with random values
    for (size_t i = 0; i < size; i++) {
        a[i] = (float)rand() / RAND_MAX * 100.0f - 50.0f; // Range [-50, 50]
        b[i] = (float)rand() / RAND_MAX * 100.0f - 50.0f;
    }
    
    return create_vector_add_task(task_id, size, a, b, result);
}

// Batch processing for multiple vector tasks
void process_vector_batch(Task **tasks, int num_tasks) {
    printf("Processing batch of %d vector addition tasks\n", num_tasks);
    
    double total_cpu_time = 0.0;
    double total_gpu_time = 0.0;
    int cpu_count = 0;
    int gpu_count = 0;
    size_t total_elements = 0;
    
    for (int i = 0; i < num_tasks; i++) {
        Task *task = tasks[i];
        bool use_gpu = use_gpu_for(task);
        
        double start = get_time();
        
        if (use_gpu) {
#ifdef USE_CUDA
            gpu_vector_add((float*)task->input1, (float*)task->input2,
                          (float*)task->output, task->size);
            gpu_count++;
            total_gpu_time += get_time() - start;
#endif
        } else {
            cpu_vector_add((float*)task->input1, (float*)task->input2,
                          (float*)task->output, task->size);
            cpu_count++;
            total_cpu_time += get_time() - start;
        }
        
        total_elements += task->size;
    }
    
    printf("Batch results:\n");
    printf("  Total elements processed: %zu\n", total_elements);
    printf("  CPU tasks: %d (%.2f ms average)\n", 
           cpu_count, cpu_count > 0 ? (total_cpu_time * 1000 / cpu_count) : 0);
    printf("  GPU tasks: %d (%.2f ms average)\n",
           gpu_count, gpu_count > 0 ? (total_gpu_time * 1000 / gpu_count) : 0);
    printf("  Total time: %.2f ms\n", (total_cpu_time + total_gpu_time) * 1000);
    
    if (total_cpu_time > 0) {
        printf("  CPU throughput: %.1f Melem/s\n",
               (cpu_count > 0 ? total_elements / cpu_count : 0) / (total_cpu_time * 1e6));
    }
    
    if (total_gpu_time > 0) {
        printf("  GPU throughput: %.1f Melem/s\n",
               (gpu_count > 0 ? total_elements / gpu_count : 0) / (total_gpu_time * 1e6));
    }
}