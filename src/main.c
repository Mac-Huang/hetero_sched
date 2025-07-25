#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "scheduler.h"
#include "ml_predict.h"

// Global variables
TaskQueue *g_task_queue = NULL;
SchedulerStats g_stats;
bool g_shutdown = false;
bool g_ml_enabled = false;

// Test data generators
float* generate_random_vector(size_t size) {
    float *vec = malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        vec[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    return vec;
}

float* generate_random_matrix(size_t rows, size_t cols) {
    return generate_random_vector(rows * cols);
}

// Test workload generators
void generate_mixed_workload(int num_tasks) {
    printf("Generating %d mixed tasks...\n", num_tasks);
    
    for (int i = 0; i < num_tasks; i++) {
        Task *task = NULL;
        
        // Mix of different task types and sizes
        switch (i % 4) {
            case 0: {
                // Small vector addition (should use CPU)
                size_t size = 1000 + rand() % 10000;
                float *a = generate_random_vector(size);
                float *b = generate_random_vector(size);
                float *result = malloc(size * sizeof(float));
                task = create_vector_add_task(i, size, a, b, result);
                break;
            }
            case 1: {
                // Large vector addition (should use GPU)
                size_t size = 100000 + rand() % 900000;
                float *a = generate_random_vector(size);
                float *b = generate_random_vector(size);
                float *result = malloc(size * sizeof(float));
                task = create_vector_add_task(i, size, a, b, result);
                break;
            }
            case 2: {
                // Small matrix multiplication (should use CPU)
                size_t size = 32 + rand() % 64;
                float *a = generate_random_matrix(size, size);
                float *b = generate_random_matrix(size, size);
                float *result = malloc(size * size * sizeof(float));
                task = create_matrix_mult_task(i, size, size, a, b, result);
                break;
            }
            case 3: {
                // Large matrix multiplication (should use GPU)
                size_t size = 256 + rand() % 256;
                float *a = generate_random_matrix(size, size);
                float *b = generate_random_matrix(size, size);
                float *result = malloc(size * size * sizeof(float));
                task = create_matrix_mult_task(i, size, size, a, b, result);
                break;
            }
        }
        
        if (task) {
            enqueue_task(g_task_queue, task);
        }
        
        // Add some delay to simulate realistic workload arrival
        usleep(1000); // 1ms
    }
}

void run_benchmark_suite() {
    printf("\n=== Running Benchmark Suite ===\n");
    
    // Vector addition benchmarks
    printf("\nVector Addition Benchmarks:\n");
    size_t vec_sizes[] = {1000, 10000, 100000, 1000000};
    int num_vec_sizes = sizeof(vec_sizes) / sizeof(vec_sizes[0]);
    
    for (int i = 0; i < num_vec_sizes; i++) {
        size_t size = vec_sizes[i];
        float *a = generate_random_vector(size);
        float *b = generate_random_vector(size);
        float *result = malloc(size * sizeof(float));
        
        Task *task = create_vector_add_task(1000 + i, size, a, b, result);
        enqueue_task(g_task_queue, task);
        
        printf("Queued vector addition task: size %zu\n", size);
    }
    
    // Matrix multiplication benchmarks  
    printf("\nMatrix Multiplication Benchmarks:\n");
    size_t mat_sizes[] = {32, 64, 128, 256, 512};
    int num_mat_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);
    
    for (int i = 0; i < num_mat_sizes; i++) {
        size_t size = mat_sizes[i];
        float *a = generate_random_matrix(size, size);
        float *b = generate_random_matrix(size, size);
        float *result = malloc(size * size * sizeof(float));
        
        Task *task = create_matrix_mult_task(2000 + i, size, size, a, b, result);
        enqueue_task(g_task_queue, task);
        
        printf("Queued matrix multiplication task: %zux%zu\n", size, size);
    }
}

int main(int argc, char *argv[]) {
    printf("=== Heterogeneous Task Scheduler ===\n");
    printf("CPU cores: %d\n", (int)sysconf(_SC_NPROCESSORS_ONLN));
    
    // Seed random number generator
    srand(time(NULL));
    
    // Initialize scheduler components
    g_task_queue = create_task_queue();
    if (!g_task_queue) {
        fprintf(stderr, "Failed to create task queue\n");
        return 1;
    }
    
    init_stats(&g_stats);
    init_csv_logging();
    
    // Initialize ML predictor
    g_ml_enabled = init_ml_predictor();
    
#ifdef USE_CUDA
    if (!init_cuda()) {
        printf("CUDA initialization failed, running CPU-only mode\n");
    } else {
        printf("CUDA initialized successfully\n");
    }
#else
    printf("Running in CPU-only mode (CUDA not enabled)\n");
#endif
    
    // Create worker threads
    pthread_t cpu_thread, gpu_thread;
    
    printf("Starting worker threads...\n");
    if (pthread_create(&cpu_thread, NULL, cpu_worker, NULL) != 0) {
        fprintf(stderr, "Failed to create CPU worker thread\n");
        return 1;
    }
    
#ifdef USE_CUDA
    if (pthread_create(&gpu_thread, NULL, gpu_worker, NULL) != 0) {
        fprintf(stderr, "Failed to create GPU worker thread\n");
        return 1;
    }
#endif
    
    // Parse command line arguments
    int num_tasks = 50;
    bool run_benchmarks = false;
    bool force_static = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tasks") == 0 && i + 1 < argc) {
            num_tasks = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            run_benchmarks = true;
        } else if (strcmp(argv[i], "--static") == 0) {
            force_static = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --tasks N      Number of mixed tasks to generate (default: 50)\n");
            printf("  --benchmark    Run systematic benchmark suite\n");
            printf("  --static       Force static scheduling (disable ML)\n");
            printf("  --help         Show this help message\n");
            return 0;
        }
    }
    
    // Override ML if forced to static
    if (force_static) {
        g_ml_enabled = false;
        printf("ML scheduling disabled (using static rules)\n");
    } else if (g_ml_enabled) {
        printf("ML scheduling enabled\n");
    } else {
        printf("ML scheduling unavailable (using static rules)\n");
    }
    
    // Generate workload
    double start_time = get_time();
    
    if (run_benchmarks) {
        run_benchmark_suite();
    } else {
        generate_mixed_workload(num_tasks);
    }
    
    printf("\nAll tasks queued. Waiting for completion...\n");
    
    // Monitor progress
    int last_total = 0;
    while (true) {
        sleep(1);
        
        pthread_mutex_lock(&g_stats.stats_mutex);
        int current_total = g_stats.total_tasks;
        pthread_mutex_unlock(&g_stats.stats_mutex);
        
        if (current_total > last_total) {
            printf("Completed %d tasks...\n", current_total);
            last_total = current_total;
        }
        
        // Check if queue is empty and no more tasks are being processed
        pthread_mutex_lock(&g_task_queue->mutex);
        bool queue_empty = (g_task_queue->count == 0);
        pthread_mutex_unlock(&g_task_queue->mutex);
        
        if (queue_empty && current_total >= (run_benchmarks ? 9 : num_tasks)) {
            break;
        }
    }
    
    double total_time = get_time() - start_time;
    
    // Shutdown scheduler
    printf("\nShutting down scheduler...\n");
    g_shutdown = true;
    shutdown_queue(g_task_queue);
    
    // Wait for worker threads to finish
    pthread_join(cpu_thread, NULL);
#ifdef USE_CUDA
    pthread_join(gpu_thread, NULL);
    cleanup_cuda();
#endif
    
    // Print final statistics
    printf("\n=== Final Results ===\n");
    printf("Total execution time: %.3f seconds\n", total_time);
    print_stats(&g_stats);
    
    // Cleanup
    cleanup_ml_predictor();
    cleanup_csv_logging();
    destroy_task_queue(g_task_queue);
    
    printf("\nScheduler shutdown complete.\n");
    return 0;
}
