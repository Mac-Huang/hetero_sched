#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include "scheduler.h"

// ============================================================================
// CUDA Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// CUDA Global State
// ============================================================================

static cublasHandle_t cublas_handle = NULL;
static cudaStream_t compute_stream = NULL;
static bool cuda_initialized = false;

// Device memory pools for efficient memory management
static float *d_buffer_a = NULL;
static float *d_buffer_b = NULL;
static float *d_buffer_c = NULL;
static size_t buffer_size = 0;
static const size_t MAX_BUFFER_SIZE = 1024 * 1024 * 1024; // 1GB

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void vector_add_kernel(const float *a, const float *b, float *result, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_scale_kernel(const float *input, float *output, float scale, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scale;
    }
}

__global__ void relu_kernel(const float *input, float *output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Optimized matrix multiplication kernel using shared memory
__global__ void matrix_mult_kernel(const float *a, const float *b, float *c, 
                                  size_t rows, size_t cols) {
    const int TILE_SIZE = 16;
    
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (cols + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        if (row < rows && tile * TILE_SIZE + tx < cols) {
            s_a[ty][tx] = a[row * cols + tile * TILE_SIZE + tx];
        } else {
            s_a[ty][tx] = 0.0f;
        }
        
        if (col < cols && tile * TILE_SIZE + ty < rows) {
            s_b[ty][tx] = b[(tile * TILE_SIZE + ty) * cols + col];
        } else {
            s_b[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_a[ty][k] * s_b[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < rows && col < cols) {
        c[row * cols + col] = sum;
    }
}

// ============================================================================
// Memory Management
// ============================================================================

bool ensure_device_buffers(size_t required_size) {
    if (required_size <= buffer_size) {
        return true; // Current buffers are sufficient
    }
    
    // Free existing buffers
    if (d_buffer_a) cudaFree(d_buffer_a);
    if (d_buffer_b) cudaFree(d_buffer_b);
    if (d_buffer_c) cudaFree(d_buffer_c);
    
    // Allocate new buffers with some extra space
    size_t new_size = required_size * 1.5; // 50% extra for future tasks
    if (new_size > MAX_BUFFER_SIZE) {
        new_size = MAX_BUFFER_SIZE;
    }
    
    cudaError_t err_a = cudaMalloc(&d_buffer_a, new_size * sizeof(float));
    cudaError_t err_b = cudaMalloc(&d_buffer_b, new_size * sizeof(float));
    cudaError_t err_c = cudaMalloc(&d_buffer_c, new_size * sizeof(float));
    
    if (err_a != cudaSuccess || err_b != cudaSuccess || err_c != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory buffers\n");
        return false;
    }
    
    buffer_size = new_size;
    printf("Allocated GPU buffers: %.1f MB each\n", (new_size * sizeof(float)) / (1024.0 * 1024.0));
    
    return true;
}

// ============================================================================
// GPU Kernel Implementations
// ============================================================================

void gpu_vector_add(float *a, float *b, float *result, size_t size) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized\n");
        return;
    }
    
    if (!ensure_device_buffers(size)) {
        fprintf(stderr, "Failed to allocate device memory for vector add\n");
        return;
    }
    
    size_t bytes = size * sizeof(float);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_a, a, bytes, cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_b, b, bytes, cudaMemcpyHostToDevice, compute_stream));
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    vector_add_kernel<<<blocks, threads_per_block, 0, compute_stream>>>(
        d_buffer_a, d_buffer_b, d_buffer_c, size);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(result, d_buffer_c, bytes, cudaMemcpyDeviceToHost, compute_stream));
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
}

void gpu_matrix_mult(float *a, float *b, float *result, size_t rows, size_t cols) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized\n");
        return;
    }
    
    size_t total_elements = rows * cols;
    
    if (!ensure_device_buffers(total_elements)) {
        fprintf(stderr, "Failed to allocate device memory for matrix mult\n");
        return;
    }
    
    size_t bytes = total_elements * sizeof(float);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_a, a, bytes, cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_b, b, bytes, cudaMemcpyHostToDevice, compute_stream));
    
    // Use cuBLAS for large matrices (more efficient)
    if (rows >= 256) {
        const float alpha = 1.0f, beta = 0.0f;
        
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                cols, rows, cols,
                                &alpha,
                                d_buffer_b, cols,
                                d_buffer_a, cols,
                                &beta,
                                d_buffer_c, cols));
    } else {
        // Use custom kernel for smaller matrices
        const int TILE_SIZE = 16;
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
        
        matrix_mult_kernel<<<blocks, threads, 0, compute_stream>>>(
            d_buffer_a, d_buffer_b, d_buffer_c, rows, cols);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(result, d_buffer_c, bytes, cudaMemcpyDeviceToHost, compute_stream));
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
}

void gpu_vector_scale(float *input, float *output, float scale, size_t size) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized\n");
        return;
    }
    
    if (!ensure_device_buffers(size)) {
        fprintf(stderr, "Failed to allocate device memory for vector scale\n");
        return;
    }
    
    size_t bytes = size * sizeof(float);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_a, input, bytes, cudaMemcpyHostToDevice, compute_stream));
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    vector_scale_kernel<<<blocks, threads_per_block, 0, compute_stream>>>(
        d_buffer_a, d_buffer_c, scale, size);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(output, d_buffer_c, bytes, cudaMemcpyDeviceToHost, compute_stream));
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
}

void gpu_relu(float *input, float *output, size_t size) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized\n");
        return;
    }
    
    if (!ensure_device_buffers(size)) {
        fprintf(stderr, "Failed to allocate device memory for ReLU\n");
        return;
    }
    
    size_t bytes = size * sizeof(float);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_a, input, bytes, cudaMemcpyHostToDevice, compute_stream));
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    relu_kernel<<<blocks, threads_per_block, 0, compute_stream>>>(
        d_buffer_a, d_buffer_c, size);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(output, d_buffer_c, bytes, cudaMemcpyDeviceToHost, compute_stream));
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
}

// ============================================================================
// CUDA Initialization and Cleanup
// ============================================================================

bool init_cuda(void) {
    // Check if CUDA devices are available
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return false;
    }
    
    // Select the first available device
    CUDA_CHECK(cudaSetDevice(0));
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Using CUDA device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    
    // Create CUDA stream for async operations
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    
    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, compute_stream));
    
    // Initialize device buffers with default size
    if (!ensure_device_buffers(1024 * 1024)) { // 1M floats initially
        return false;
    }
    
    cuda_initialized = true;
    printf("CUDA initialization complete\n");
    
    return true;
}

void cleanup_cuda(void) {
    if (!cuda_initialized) return;
    
    // Free device memory
    if (d_buffer_a) cudaFree(d_buffer_a);
    if (d_buffer_b) cudaFree(d_buffer_b);
    if (d_buffer_c) cudaFree(d_buffer_c);
    
    // Cleanup cuBLAS
    if (cublas_handle) cublasDestroy(cublas_handle);
    
    // Cleanup CUDA stream
    if (compute_stream) cudaStreamDestroy(compute_stream);
    
    // Reset device
    cudaDeviceReset();
    
    cuda_initialized = false;
    printf("CUDA cleanup complete\n");
}

// ============================================================================
// GPU Benchmarking Functions
// ============================================================================

double benchmark_gpu_vector_add(size_t size, int iterations) {
    if (!cuda_initialized) return -1.0;
    
    float *h_a = (float*)malloc(size * sizeof(float));
    float *h_b = (float*)malloc(size * sizeof(float));
    float *h_result = (float*)malloc(size * sizeof(float));
    
    // Initialize test data
    for (size_t i = 0; i < size; i++) {
        h_a[i] = (float)i * 0.5f;
        h_b[i] = (float)i * 0.3f;
    }
    
    // Warm up
    gpu_vector_add(h_a, h_b, h_result, size);
    
    double start_time = get_time();
    
    for (int i = 0; i < iterations; i++) {
        gpu_vector_add(h_a, h_b, h_result, size);
    }
    
    double end_time = get_time();
    
    free(h_a);
    free(h_b);
    free(h_result);
    
    return end_time - start_time;
}

double benchmark_gpu_matrix_mult(size_t size, int iterations) {
    if (!cuda_initialized) return -1.0;
    
    size_t total_elements = size * size;
    float *h_a = (float*)malloc(total_elements * sizeof(float));
    float *h_b = (float*)malloc(total_elements * sizeof(float));
    float *h_result = (float*)malloc(total_elements * sizeof(float));
    
    // Initialize test data
    for (size_t i = 0; i < total_elements; i++) {
        h_a[i] = (float)i * 0.01f;
        h_b[i] = (float)i * 0.02f;
    }
    
    // Warm up
    gpu_matrix_mult(h_a, h_b, h_result, size, size);
    
    double start_time = get_time();
    
    for (int i = 0; i < iterations; i++) {
        gpu_matrix_mult(h_a, h_b, h_result, size, size);
    }
    
    double end_time = get_time();
    
    free(h_a);
    free(h_b);
    free(h_result);
    
    return end_time - start_time;
}

#endif // USE_CUDA