#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

// Forward declaration for ML integration
#ifndef ML_PREDICT_H
bool ml_use_gpu_for(Task *task);
#endif

// Task types supported by the scheduler
typedef enum {
    TASK_VEC_ADD,
    TASK_MATMUL,
    TASK_VEC_SCALE,
    TASK_RELU
} TaskType;

// Task structure containing all necessary information
typedef struct {
    TaskType type;
    size_t size;        // Primary dimension (vector length, matrix size)
    size_t rows, cols;  // For matrix operations
    void *input1;
    void *input2;
    void *output;
    double cpu_time;    // Measured execution time on CPU
    double gpu_time;    // Measured execution time on GPU
    bool completed;
    int task_id;
} Task;

// Task queue node for linked list implementation
typedef struct TaskNode {
    Task *task;
    struct TaskNode *next;
} TaskNode;

// Task queue with synchronization
typedef struct {
    TaskNode *head;
    TaskNode *tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    bool shutdown;
} TaskQueue;

// Scheduler statistics
typedef struct {
    int total_tasks;
    int cpu_tasks;
    int gpu_tasks;
    double total_cpu_time;
    double total_gpu_time;
    pthread_mutex_t stats_mutex;
} SchedulerStats;

// Function declarations

// Queue operations
TaskQueue* create_task_queue(void);
void destroy_task_queue(TaskQueue *queue);
void enqueue_task(TaskQueue *queue, Task *task);
Task* dequeue_task(TaskQueue *queue);
void shutdown_queue(TaskQueue *queue);

// Task creation helpers
Task* create_vector_add_task(int task_id, size_t size, float *a, float *b, float *result);
Task* create_matrix_mult_task(int task_id, size_t rows, size_t cols, float *a, float *b, float *result);
void destroy_task(Task *task);

// Scheduling decisions
bool use_gpu_for(Task *task);
bool gpu_available(void);

// Worker threads
void* cpu_worker(void *arg);
void* gpu_worker(void *arg);

// CPU kernel implementations
void cpu_vector_add(float *a, float *b, float *result, size_t size);
void cpu_matrix_mult(float *a, float *b, float *result, size_t rows, size_t cols);
void cpu_vector_scale(float *input, float *output, float scale, size_t size);
void cpu_relu(float *input, float *output, size_t size);

// GPU kernel implementations (CUDA)
#ifdef USE_CUDA
void gpu_vector_add(float *a, float *b, float *result, size_t size);
void gpu_matrix_mult(float *a, float *b, float *result, size_t rows, size_t cols);
void gpu_vector_scale(float *input, float *output, float scale, size_t size);
void gpu_relu(float *input, float *output, size_t size);
bool init_cuda(void);
void cleanup_cuda(void);
#endif

// Timing utilities
double get_time(void);
void log_task_completion(Task *task, const char *device);

// CSV Logging for ML Training
void init_csv_logging(void);
void log_task_to_csv(Task *task, const char *device, double execution_time);
void cleanup_csv_logging(void);

// Statistics
void init_stats(SchedulerStats *stats);
void update_stats(SchedulerStats *stats, Task *task, bool used_gpu);
void print_stats(SchedulerStats *stats);

// Global scheduler state
extern TaskQueue *g_task_queue;
extern SchedulerStats g_stats;
extern bool g_shutdown;

#endif // SCHEDULER_H