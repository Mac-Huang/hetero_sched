#ifndef ML_PREDICT_H
#define ML_PREDICT_H

#include "scheduler.h"

// Device prediction result
typedef enum {
    DEVICE_CPU,
    DEVICE_GPU,
    DEVICE_UNKNOWN
} Device;

// ML prediction result structure
typedef struct {
    Device predicted_device;
    double cpu_runtime_ms;
    double gpu_runtime_ms;
    bool ml_available;
} MLPrediction;

// ML prediction interface functions
bool init_ml_predictor(void);
void cleanup_ml_predictor(void);

Device predict_device(Task *task);
double predict_runtime(Task *task, Device device);
MLPrediction predict_task_complete(Task *task);

// System monitoring functions
typedef struct {
    double cpu_load_1min;
    double cpu_load_5min;
    double cpu_load_15min;
    size_t gpu_memory_used_mb;
    size_t gpu_memory_total_mb;
    double gpu_utilization_percent;
} SystemState;

bool get_system_state(SystemState *state);

// Utility functions
const char* device_to_string(Device device);
Device string_to_device(const char* device_str);

#endif // ML_PREDICT_H