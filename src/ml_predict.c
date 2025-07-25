#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "ml_predict.h"

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

// Global state
static bool g_ml_initialized = false;
static char g_python_path[256] = "python";

// ============================================================================
// Initialization and Cleanup
// ============================================================================

bool init_ml_predictor(void) {
    printf("Initializing ML predictor...\n");
    
    // Check if Python is available
    if (system("python --version > /dev/null 2>&1") != 0) {
        printf("Warning: Python not found, falling back to static rules\n");
        g_ml_initialized = false;
        return false;
    }
    
    // Check if prediction script exists
    if (access("scripts/predict_task.py", F_OK) != 0) {
        printf("Warning: ML prediction script not found, falling back to static rules\n");
        g_ml_initialized = false;
        return false;
    }
    
    // Check if models exist
    if (access("models/model_route.pkl", F_OK) != 0) {
        printf("Warning: ML models not found, falling back to static rules\n");
        printf("Run 'python scripts/train_cost_model.py' to train models\n");
        g_ml_initialized = false;
        return false;
    }
    
    g_ml_initialized = true;
    printf("ML predictor initialized successfully\n");
    return true;
}

void cleanup_ml_predictor(void) {
    g_ml_initialized = false;
}

// ============================================================================
// Helper Functions
// ============================================================================

const char* device_to_string(Device device) {
    switch (device) {
        case DEVICE_CPU: return "CPU";
        case DEVICE_GPU: return "GPU";
        default: return "UNKNOWN";
    }
}

Device string_to_device(const char* device_str) {
    if (strcmp(device_str, "CPU") == 0) return DEVICE_CPU;
    if (strcmp(device_str, "GPU") == 0) return DEVICE_GPU;
    return DEVICE_UNKNOWN;
}

const char* task_type_to_string(TaskType type) {
    switch (type) {
        case TASK_VEC_ADD: return "VEC_ADD";
        case TASK_MATMUL: return "MATMUL";
        case TASK_VEC_SCALE: return "VEC_SCALE";
        case TASK_RELU: return "RELU";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// ML Prediction Interface
// ============================================================================

MLPrediction predict_task_complete(Task *task) {
    MLPrediction result = {0};
    result.ml_available = g_ml_initialized;
    
    if (!task) {
        result.predicted_device = DEVICE_CPU;
        result.cpu_runtime_ms = 1.0;
        result.gpu_runtime_ms = 1.0;
        return result;
    }
    
    if (!g_ml_initialized) {
        // Fallback to static rules
        bool use_gpu = use_gpu_for(task);
        result.predicted_device = use_gpu ? DEVICE_GPU : DEVICE_CPU;
        
        // Rough runtime estimates (fallback)
        if (task->type == TASK_VEC_ADD) {
            result.cpu_runtime_ms = task->size * 0.001;
            result.gpu_runtime_ms = task->size > 50000 ? task->size * 0.0003 : task->size * 0.002;
        } else if (task->type == TASK_MATMUL) {
            double ops = (double)task->rows * task->cols * task->rows;
            result.cpu_runtime_ms = ops * 0.01;
            result.gpu_runtime_ms = task->rows > 128 ? ops * 0.003 : ops * 0.02;
        } else {
            result.cpu_runtime_ms = task->size * 0.0005;
            result.gpu_runtime_ms = task->size * 0.0002;
        }
        
        return result;
    }
    
    // Build command to call Python ML predictor
    char command[512];
    snprintf(command, sizeof(command), 
             "python scripts/predict_task.py %s %zu %zu %zu 2>/dev/null",
             task_type_to_string(task->type),
             task->size,
             task->rows,
             task->cols);
    
    // Execute prediction
    FILE *fp = popen(command, "r");
    if (!fp) {
        printf("Warning: Failed to execute ML prediction, using fallback\n");
        bool use_gpu = use_gpu_for(task);
        result.predicted_device = use_gpu ? DEVICE_GPU : DEVICE_CPU;
        result.cpu_runtime_ms = 1.0;
        result.gpu_runtime_ms = 1.0;
        return result;
    }
    
    // Parse output: <device> <cpu_time> <gpu_time>
    char device_str[16];
    double cpu_time, gpu_time;
    
    if (fscanf(fp, "%15s %lf %lf", device_str, &cpu_time, &gpu_time) == 3) {
        result.predicted_device = string_to_device(device_str);
        result.cpu_runtime_ms = cpu_time;
        result.gpu_runtime_ms = gpu_time;
    } else {
        printf("Warning: Failed to parse ML prediction output\n");
        bool use_gpu = use_gpu_for(task);
        result.predicted_device = use_gpu ? DEVICE_GPU : DEVICE_CPU;
        result.cpu_runtime_ms = 1.0;
        result.gpu_runtime_ms = 1.0;
    }
    
    pclose(fp);
    return result;
}

Device predict_device(Task *task) {
    MLPrediction prediction = predict_task_complete(task);
    return prediction.predicted_device;
}

double predict_runtime(Task *task, Device device) {
    MLPrediction prediction = predict_task_complete(task);
    
    if (device == DEVICE_CPU) {
        return prediction.cpu_runtime_ms;
    } else if (device == DEVICE_GPU) {
        return prediction.gpu_runtime_ms;
    } else {
        return 1.0; // Default fallback
    }
}

// ============================================================================
// System Monitoring Functions
// ============================================================================

bool get_cpu_load(double *load1, double *load5, double *load15) {
#ifdef __linux__
    // Read from /proc/loadavg on Linux
    FILE *fp = fopen("/proc/loadavg", "r");
    if (!fp) return false;
    
    int result = fscanf(fp, "%lf %lf %lf", load1, load5, load15);
    fclose(fp);
    
    return (result == 3);
#elif defined(__APPLE__)
    // Use getloadavg on macOS
    double loads[3];
    if (getloadavg(loads, 3) == 3) {
        *load1 = loads[0];
        *load5 = loads[1]; 
        *load15 = loads[2];
        return true;
    }
    return false;
#else
    // Windows or other platforms - return dummy values
    *load1 = 0.5;
    *load5 = 0.5;
    *load15 = 0.5;
    return true;
#endif
}

bool get_gpu_memory_info(size_t *used_mb, size_t *total_mb, double *utilization) {
#ifdef USE_CUDA
    // Try nvidia-smi command to get GPU stats
    FILE *fp = popen("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!fp) {
        *used_mb = 0;
        *total_mb = 1024; // Assume 1GB as fallback
        *utilization = 0.0;
        return false;
    }
    
    unsigned int used, total, util;
    int result = fscanf(fp, "%u, %u, %u", &used, &total, &util);
    pclose(fp);
    
    if (result == 3) {
        *used_mb = used;
        *total_mb = total;
        *utilization = util;
        return true;
    }
#endif
    
    // Fallback values
    *used_mb = 0;
    *total_mb = 1024;
    *utilization = 0.0;
    return false;
}

bool get_system_state(SystemState *state) {
    if (!state) return false;
    
    // Initialize with defaults
    state->cpu_load_1min = 0.0;
    state->cpu_load_5min = 0.0;
    state->cpu_load_15min = 0.0;
    state->gpu_memory_used_mb = 0;
    state->gpu_memory_total_mb = 1024;
    state->gpu_utilization_percent = 0.0;
    
    // Get CPU load
    get_cpu_load(&state->cpu_load_1min, &state->cpu_load_5min, &state->cpu_load_15min);
    
    // Get GPU memory info
    get_gpu_memory_info(&state->gpu_memory_used_mb, &state->gpu_memory_total_mb, &state->gpu_utilization_percent);
    
    return true;
}

// ============================================================================
// Enhanced Scheduling Logic with ML
// ============================================================================

bool ml_use_gpu_for(Task *task) {
    if (!task) return false;
    
    // Get current system state
    SystemState sys_state;
    get_system_state(&sys_state);
    
    // Get base ML prediction
    Device predicted = predict_device(task);
    
    // Apply system-aware constraints
    bool use_gpu = (predicted == DEVICE_GPU);
    
    // Override GPU decision if system is under stress
    if (use_gpu) {
        // Check GPU memory pressure (don't use GPU if >90% full)
        double gpu_memory_usage = (double)sys_state.gpu_memory_used_mb / sys_state.gpu_memory_total_mb;
        if (gpu_memory_usage > 0.90) {
            use_gpu = false;
            printf("[SYS] GPU memory full (%.1f%%), routing to CPU\n", gpu_memory_usage * 100);
        }
        
        // Check GPU utilization (don't overload if >95% utilized)
        if (sys_state.gpu_utilization_percent > 95.0) {
            use_gpu = false;
            printf("[SYS] GPU overloaded (%.1f%%), routing to CPU\n", sys_state.gpu_utilization_percent);
        }
    }
    
    // Override CPU decision if CPU is heavily loaded
    if (!use_gpu) {
        // If CPU load is very high (>80% of cores), consider GPU even for small tasks
        if (sys_state.cpu_load_1min > 0.8 * sysconf(_SC_NPROCESSORS_ONLN)) {
            // But only if GPU has reasonable memory available
            double gpu_memory_usage = (double)sys_state.gpu_memory_used_mb / sys_state.gpu_memory_total_mb;
            if (gpu_memory_usage < 0.70 && sys_state.gpu_utilization_percent < 80.0) {
                use_gpu = true;
                printf("[SYS] CPU overloaded (%.2f), routing to GPU\n", sys_state.cpu_load_1min);
            }
        }
    }
    
    // Log the final decision with system context
    if (g_ml_initialized) {
        printf("[ML+SYS] Task %d (%s, size=%zu) -> %s [CPU_load=%.2f, GPU_mem=%.1f%%, GPU_util=%.1f%%]\n", 
               task->task_id, 
               task_type_to_string(task->type),
               task->size,
               use_gpu ? "GPU" : "CPU",
               sys_state.cpu_load_1min,
               (double)sys_state.gpu_memory_used_mb / sys_state.gpu_memory_total_mb * 100,
               sys_state.gpu_utilization_percent);
    }
    
    return use_gpu;
}