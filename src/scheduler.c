#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "scheduler.h"

// Global CSV logging state
static FILE *g_csv_file = NULL;
static pthread_mutex_t g_csv_mutex = PTHREAD_MUTEX_INITIALIZER;

// ============================================================================
// Timing Utilities
// ============================================================================

double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// ============================================================================
// Task Queue Implementation
// ============================================================================

TaskQueue* create_task_queue(void) {
    TaskQueue *queue = malloc(sizeof(TaskQueue));
    if (!queue) return NULL;
    
    queue->head = NULL;
    queue->tail = NULL;
    queue->count = 0;
    queue->shutdown = false;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue);
        return NULL;
    }
    
    if (pthread_cond_init(&queue->not_empty, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        free(queue);
        return NULL;
    }
    
    return queue;
}

void destroy_task_queue(TaskQueue *queue) {
    if (!queue) return;
    
    pthread_mutex_lock(&queue->mutex);
    
    // Free remaining tasks
    TaskNode *current = queue->head;
    while (current) {
        TaskNode *next = current->next;
        destroy_task(current->task);
        free(current);
        current = next;
    }
    
    pthread_mutex_unlock(&queue->mutex);
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    free(queue);
}

void enqueue_task(TaskQueue *queue, Task *task) {
    if (!queue || !task) return;
    
    TaskNode *node = malloc(sizeof(TaskNode));
    if (!node) {
        destroy_task(task);
        return;
    }
    
    node->task = task;
    node->next = NULL;
    
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->tail) {
        queue->tail->next = node;
        queue->tail = node;
    } else {
        queue->head = queue->tail = node;
    }
    
    queue->count++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
}

Task* dequeue_task(TaskQueue *queue) {
    if (!queue) return NULL;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->count == 0 && !queue->shutdown) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    if (queue->shutdown && queue->count == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return NULL;
    }
    
    TaskNode *node = queue->head;
    Task *task = node->task;
    
    queue->head = node->next;
    if (!queue->head) {
        queue->tail = NULL;
    }
    
    queue->count--;
    free(node);
    
    pthread_mutex_unlock(&queue->mutex);
    return task;
}

void shutdown_queue(TaskQueue *queue) {
    if (!queue) return;
    
    pthread_mutex_lock(&queue->mutex);
    queue->shutdown = true;
    pthread_cond_broadcast(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
}

// ============================================================================
// Task Creation and Management
// ============================================================================

Task* create_vector_add_task(int task_id, size_t size, float *a, float *b, float *result) {
    Task *task = malloc(sizeof(Task));
    if (!task) return NULL;
    
    task->type = TASK_VEC_ADD;
    task->size = size;
    task->rows = size;
    task->cols = 1;
    task->input1 = a;
    task->input2 = b;
    task->output = result;
    task->cpu_time = 0.0;
    task->gpu_time = 0.0;
    task->completed = false;
    task->task_id = task_id;
    
    return task;
}

Task* create_matrix_mult_task(int task_id, size_t rows, size_t cols, float *a, float *b, float *result) {
    Task *task = malloc(sizeof(Task));
    if (!task) return NULL;
    
    task->type = TASK_MATMUL;
    task->size = rows * cols;
    task->rows = rows;
    task->cols = cols;
    task->input1 = a;
    task->input2 = b;
    task->output = result;
    task->cpu_time = 0.0;
    task->gpu_time = 0.0;
    task->completed = false;
    task->task_id = task_id;
    
    return task;
}

void destroy_task(Task *task) {
    if (!task) return;
    
    free(task->input1);
    free(task->input2);
    free(task->output);
    free(task);
}

// ============================================================================
// Scheduling Policy
// ============================================================================

// Global ML enable flag
extern bool g_ml_enabled;

bool use_gpu_for(Task *task) {
#ifndef USE_CUDA
    return false; // No GPU available
#endif
    
    if (!task) return false;
    
    // Try ML prediction first if available
    extern bool ml_use_gpu_for(Task *task);
    if (g_ml_enabled) {
        return ml_use_gpu_for(task);
    }
    
    // Fallback to static rules
    switch (task->type) {
        case TASK_VEC_ADD:
            // Use GPU for large vector operations
            return task->size > 50000;
            
        case TASK_MATMUL:
            // Use GPU for matrices larger than 128x128
            return task->rows > 128 || task->cols > 128;
            
        case TASK_VEC_SCALE:
            return task->size > 100000;
            
        case TASK_RELU:
            return task->size > 10000;
            
        default:
            return false;
    }
}

bool gpu_available(void) {
#ifdef USE_CUDA
    return true;
#else
    return false;
#endif
}

// ============================================================================
// Worker Threads
// ============================================================================

void* cpu_worker(void *arg) {
    (void)arg; // Suppress unused parameter warning
    
    printf("CPU worker thread started\n");
    
    while (!g_shutdown) {
        Task *task = dequeue_task(g_task_queue);
        if (!task) continue;
        
        // Check if this task should use GPU
        if (use_gpu_for(task) && gpu_available()) {
            // Put task back in queue for GPU worker
            enqueue_task(g_task_queue, task);
            continue;
        }
        
        // Execute task on CPU
        double start_time = get_time();
        
        switch (task->type) {
            case TASK_VEC_ADD:
                cpu_vector_add((float*)task->input1, (float*)task->input2, 
                              (float*)task->output, task->size);
                break;
                
            case TASK_MATMUL:
                cpu_matrix_mult((float*)task->input1, (float*)task->input2, 
                               (float*)task->output, task->rows, task->cols);
                break;
                
            case TASK_VEC_SCALE:
                cpu_vector_scale((float*)task->input1, (float*)task->output, 
                                2.0f, task->size);
                break;
                
            case TASK_RELU:
                cpu_relu((float*)task->input1, (float*)task->output, task->size);
                break;
                
            default:
                printf("Unknown task type: %d\n", task->type);
                break;
        }
        
        double end_time = get_time();
        task->cpu_time = end_time - start_time;
        task->completed = true;
        
        log_task_completion(task, "CPU");
        log_task_to_csv(task, "CPU", task->cpu_time);
        update_stats(&g_stats, task, false);
        
        destroy_task(task);
    }
    
    printf("CPU worker thread shutting down\n");
    return NULL;
}

void* gpu_worker(void *arg) {
    (void)arg; // Suppress unused parameter warning
    
#ifndef USE_CUDA
    // No GPU support, this thread does nothing
    return NULL;
#endif
    
    printf("GPU worker thread started\n");
    
    while (!g_shutdown) {
        Task *task = dequeue_task(g_task_queue);
        if (!task) continue;
        
        // Check if this task should use CPU
        if (!use_gpu_for(task)) {
            // Put task back in queue for CPU worker
            enqueue_task(g_task_queue, task);
            continue;
        }
        
        // Execute task on GPU
        double start_time = get_time();
        
#ifdef USE_CUDA
        switch (task->type) {
            case TASK_VEC_ADD:
                gpu_vector_add((float*)task->input1, (float*)task->input2, 
                              (float*)task->output, task->size);
                break;
                
            case TASK_MATMUL:
                gpu_matrix_mult((float*)task->input1, (float*)task->input2, 
                               (float*)task->output, task->rows, task->cols);
                break;
                
            case TASK_VEC_SCALE:
                gpu_vector_scale((float*)task->input1, (float*)task->output, 
                                2.0f, task->size);
                break;
                
            case TASK_RELU:
                gpu_relu((float*)task->input1, (float*)task->output, task->size);
                break;
                
            default:
                printf("Unknown task type: %d\n", task->type);
                break;
        }
#endif
        
        double end_time = get_time();
        task->gpu_time = end_time - start_time;
        task->completed = true;
        
        log_task_completion(task, "GPU");
        log_task_to_csv(task, "GPU", task->gpu_time);
        update_stats(&g_stats, task, true);
        
        destroy_task(task);
    }
    
    printf("GPU worker thread shutting down\n");
    return NULL;
}

// ============================================================================
// Logging and Statistics
// ============================================================================

void log_task_completion(Task *task, const char *device) {
    const char *task_name;
    switch (task->type) {
        case TASK_VEC_ADD: task_name = "VEC_ADD"; break;
        case TASK_MATMUL: task_name = "MATMUL"; break;
        case TASK_VEC_SCALE: task_name = "VEC_SCALE"; break;
        case TASK_RELU: task_name = "RELU"; break;
        default: task_name = "UNKNOWN"; break;
    }
    
    double exec_time = strcmp(device, "GPU") == 0 ? task->gpu_time : task->cpu_time;
    
    printf("[%s] Task %d (%s) size=%zu completed in %.4f ms\n",
           device, task->task_id, task_name, task->size, exec_time * 1000);
}

void init_stats(SchedulerStats *stats) {
    if (!stats) return;
    
    stats->total_tasks = 0;
    stats->cpu_tasks = 0;
    stats->gpu_tasks = 0;
    stats->total_cpu_time = 0.0;
    stats->total_gpu_time = 0.0;
    
    pthread_mutex_init(&stats->stats_mutex, NULL);
}

void update_stats(SchedulerStats *stats, Task *task, bool used_gpu) {
    if (!stats || !task) return;
    
    pthread_mutex_lock(&stats->stats_mutex);
    
    stats->total_tasks++;
    
    if (used_gpu) {
        stats->gpu_tasks++;
        stats->total_gpu_time += task->gpu_time;
    } else {
        stats->cpu_tasks++;
        stats->total_cpu_time += task->cpu_time;
    }
    
    pthread_mutex_unlock(&stats->stats_mutex);
}

void print_stats(SchedulerStats *stats) {
    if (!stats) return;
    
    pthread_mutex_lock(&stats->stats_mutex);
    
    printf("\n=== Scheduler Statistics ===\n");
    printf("Total tasks completed: %d\n", stats->total_tasks);
    printf("CPU tasks: %d (%.1f%%)\n", stats->cpu_tasks, 
           stats->total_tasks > 0 ? (stats->cpu_tasks * 100.0 / stats->total_tasks) : 0);
    printf("GPU tasks: %d (%.1f%%)\n", stats->gpu_tasks,
           stats->total_tasks > 0 ? (stats->gpu_tasks * 100.0 / stats->total_tasks) : 0);
    printf("Total CPU time: %.3f seconds\n", stats->total_cpu_time);
    printf("Total GPU time: %.3f seconds\n", stats->total_gpu_time);
    
    if (stats->cpu_tasks > 0) {
        printf("Average CPU task time: %.3f ms\n", 
               (stats->total_cpu_time / stats->cpu_tasks) * 1000);
    }
    
    if (stats->gpu_tasks > 0) {
        printf("Average GPU task time: %.3f ms\n", 
               (stats->total_gpu_time / stats->gpu_tasks) * 1000);
    }
    
    double total_compute_time = stats->total_cpu_time + stats->total_gpu_time;
    printf("Total compute time: %.3f seconds\n", total_compute_time);
    
    pthread_mutex_unlock(&stats->stats_mutex);
}

// ============================================================================
// CSV Logging for ML Training
// ============================================================================

void init_csv_logging(void) {
    pthread_mutex_lock(&g_csv_mutex);
    
    // Create logs directory if it doesn't exist
    system("mkdir -p logs");
    
    g_csv_file = fopen("logs/task_log.csv", "w");
    if (!g_csv_file) {
        fprintf(stderr, "Failed to open CSV log file\n");
        pthread_mutex_unlock(&g_csv_mutex);
        return;
    }
    
    // Write CSV header
    fprintf(g_csv_file, "timestamp,task_id,task_type,task_size,rows,cols,device,execution_time_ms\n");
    fflush(g_csv_file);
    
    pthread_mutex_unlock(&g_csv_mutex);
    printf("CSV logging initialized: logs/task_log.csv\n");
}

void log_task_to_csv(Task *task, const char *device, double execution_time) {
    if (!g_csv_file || !task) return;
    
    pthread_mutex_lock(&g_csv_mutex);
    
    // Get current timestamp (epoch time)
    time_t timestamp = time(NULL);
    
    // Convert task type to string
    const char *task_type_str;
    switch (task->type) {
        case TASK_VEC_ADD: task_type_str = "VEC_ADD"; break;
        case TASK_MATMUL: task_type_str = "MATMUL"; break;
        case TASK_VEC_SCALE: task_type_str = "VEC_SCALE"; break;
        case TASK_RELU: task_type_str = "RELU"; break;
        default: task_type_str = "UNKNOWN"; break;
    }
    
    // Log task data: timestamp, task_id, task_type, task_size, rows, cols, device, execution_time_ms
    fprintf(g_csv_file, "%ld,%d,%s,%zu,%zu,%zu,%s,%.4f\n",
            timestamp,
            task->task_id,
            task_type_str,
            task->size,
            task->rows,
            task->cols,
            device,
            execution_time * 1000.0); // Convert to milliseconds
    
    fflush(g_csv_file); // Ensure data is written immediately
    
    pthread_mutex_unlock(&g_csv_mutex);
}

void cleanup_csv_logging(void) {
    pthread_mutex_lock(&g_csv_mutex);
    
    if (g_csv_file) {
        fclose(g_csv_file);
        g_csv_file = NULL;
        printf("CSV logging finalized\n");
    }
    
    pthread_mutex_unlock(&g_csv_mutex);
}