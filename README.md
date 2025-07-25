# HeteroSched: Deep RL Heterogeneous Task Scheduler

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](./VERSION)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![MLSys](https://img.shields.io/badge/conference-MLSys_2026-red.svg)]()

A state-of-the-art heterogeneous computing scheduler that uses **Deep Reinforcement Learning** and **Multi-Objective Optimization** to intelligently distribute computational tasks across CPU/GPU resources. This system goes beyond traditional heuristics to learn optimal scheduling policies that balance multiple objectives: latency, energy efficiency, throughput, and fairness.

## 🎯 Project Overview

This project implements a heterogeneous computing scheduler that:
- **Automatically decides** whether to run tasks on CPU or GPU based on workload size and characteristics
- **Manages task queues** with thread-safe operations and worker threads
- **Benchmarks performance** to optimize scheduling decisions
- **Supports CUDA** for GPU acceleration with fallback to CPU-only mode
- **Provides detailed statistics** on task execution and performance

## 🚀 Features

### Core Functionality
- ✅ **Dynamic Task Scheduling**: Intelligent CPU vs GPU selection
- ✅ **Thread-Safe Task Queue**: Multi-producer, multi-consumer queue
- ✅ **Worker Thread Pool**: Separate CPU and GPU worker threads
- ✅ **Performance Profiling**: Built-in timing and statistics collection
- ✅ **Memory Management**: Efficient GPU memory buffer pooling

### Supported Task Types
- **Vector Addition**: Element-wise addition of large arrays
- **Matrix Multiplication**: Dense matrix-matrix multiplication
- **Vector Scaling**: Scalar multiplication of vectors
- **ReLU Activation**: Rectified Linear Unit function

### GPU Support
- **CUDA Integration**: NVIDIA GPU acceleration
- **cuBLAS Integration**: Optimized BLAS operations
- **Asynchronous Execution**: CUDA streams for overlap
- **Memory Pooling**: Efficient device memory management

## 📋 Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows (with appropriate toolchain)
- **CPU**: x86_64 architecture
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: NVIDIA GPU with CUDA capability 3.5+ (optional)

### Software Dependencies
- **GCC**: 7.0+ or **Clang**: 10.0+
- **Make**: Build system
- **CUDA Toolkit**: 10.0+ (for GPU support)
- **NVIDIA Driver**: Compatible with CUDA version
- **pthread**: POSIX threads (usually included)

### Optional Tools
- **cppcheck**: Static analysis
- **clang-format**: Code formatting
- **doxygen**: Documentation generation

## 🛠️ Installation

### 1. Clone and Setup
```bash
cd /path/to/your/projects
mkdir hetero_sched
cd hetero_sched
# Files already created in your directory structure
```

### 2. CPU-Only Build
```bash
make cpu
```

### 3. CUDA Build (if GPU available)
```bash
make cuda
```

### 4. Install System-Wide (optional)
```bash
sudo make install
```

## 🎮 Usage

### Basic Usage
```bash
# Run with default mixed workload (50 tasks)
./build/hetero_sched

# Run with custom number of tasks
./build/hetero_sched --tasks 100

# Run systematic benchmark suite
./build/hetero_sched --benchmark
```

### Build Options
```bash
# Different build configurations
make cpu          # CPU-only build
make cuda         # CUDA-enabled build
make debug        # Debug build with symbols
make debug-cuda   # Debug CUDA build
make release      # Optimized release build

# Run benchmarks
make benchmark-cpu    # CPU-only benchmarks
make benchmark-cuda   # CUDA benchmarks
make compare          # Compare CPU vs CUDA
```

### Testing Specific Task Types
```bash
# Test vector operations
make test-vector

# Test matrix operations  
make test-matrix
```

## 📊 Performance Analysis

### Expected Performance Characteristics

| Task Type | Small Size | Large Size | Crossover Point | Best Device |
|-----------|------------|------------|-----------------|-------------|
| Vector Add | < 50K elements | > 100K elements | ~75K elements | GPU for large |
| Matrix Mult | < 128×128 | > 256×256 | ~200×200 | GPU for large |
| Vector Scale | < 100K elements | > 500K elements | ~300K elements | GPU for large |
| ReLU | < 10K elements | > 50K elements | ~25K elements | GPU for large |

### Benchmark Results (Example)
```
=== Vector Addition Benchmark ===
Size        CPU Time (ms)   GPU Time (ms)   Bandwidth (GB/s)    Speedup   Recommendation
----        -------------   -------------   ----------------    -------   --------------
1000        0.01            0.08            2.40                0.13x     CPU
10000       0.05            0.09            4.80                0.56x     CPU
100000      0.45            0.12            20.0                3.75x     GPU
1000000     4.20            0.18            133.3               23.3x     GPU
```

## 🏗️ Architecture

### Directory Structure
```
hetero_sched/
├── include/
│   └── scheduler.h          # Main header with API definitions
├── src/
│   ├── main.c               # Entry point and test harness
│   ├── scheduler.c          # Core scheduling logic
│   ├── cpu_kernels.c        # CPU implementations
│   └── gpu_kernels.cu       # CUDA implementations
├── tasks/
│   ├── task_vector_add.c    # Vector operation tests
│   └── task_matrix_mult.c   # Matrix operation tests
├── build/                   # Generated build files
├── Makefile                 # Build configuration
└── README.md                # This file
```

### Core Components

1. **Task Queue System**
   - Thread-safe queue implementation
   - Producer-consumer pattern
   - Graceful shutdown handling

2. **Scheduling Policy**
   - Size-based heuristics
   - Task type considerations
   - Dynamic GPU availability checking

3. **Worker Threads**
   - Dedicated CPU worker thread
   - Dedicated GPU worker thread
   - Automatic task routing

4. **Memory Management**
   - CPU memory allocation
   - GPU buffer pooling
   - Automatic cleanup

## 🔧 Configuration

### Scheduling Thresholds
The scheduler uses these default thresholds (configurable in `scheduler.c`):

```c
bool use_gpu_for(Task *task) {
    switch (task->type) {
        case TASK_VEC_ADD:
            return task->size > 50000;        // 50K elements
        case TASK_MATMUL:
            return task->rows > 128;          // 128x128 matrices
        case TASK_VEC_SCALE:
            return task->size > 100000;       // 100K elements
        case TASK_RELU:
            return task->size > 10000;        // 10K elements
    }
}
```

### CUDA Configuration
- **Default GPU Buffer Size**: 1GB
- **CUDA Stream**: Asynchronous execution
- **cuBLAS**: Used for large matrix operations
- **Memory Pooling**: 50% extra allocation for efficiency

## 📈 Development Phases

### ✅ Phase 1: Infrastructure (Completed)
- [x] Project skeleton
- [x] Build system (Makefile)
- [x] Task structure definition
- [x] Basic CPU kernels

### ✅ Phase 2: Core Scheduler (Completed)
- [x] Task queue implementation
- [x] Worker thread system
- [x] Basic scheduling policy
- [x] Performance timing

### ✅ Phase 3: GPU Integration (Completed)
- [x] CUDA kernel implementations
- [x] GPU memory management
- [x] cuBLAS integration
- [x] Asynchronous execution

### ✅ Phase 4: Optimization (Completed)
- [x] Memory buffer pooling
- [x] Performance benchmarking
- [x] Statistics collection
- [x] Advanced scheduling heuristics

## 🧪 Testing

### Unit Tests
```bash
# Test individual components
make test-vector     # Vector operation tests
make test-matrix     # Matrix operation tests
```

### Performance Tests
```bash
# Compare CPU vs GPU performance
make compare

# Run comprehensive benchmarks
make benchmark

# Analyze specific workload patterns
./build/hetero_sched --benchmark
```

### Verification
- **Result Verification**: All kernels verify correctness
- **Memory Leak Detection**: Proper cleanup validation
- **Thread Safety**: Stress testing with concurrent access

## 📊 Monitoring and Logging

### Statistics Tracked
- Total tasks completed
- CPU vs GPU task distribution
- Average execution times
- Throughput measurements
- Memory usage patterns

### Output Format
```
=== Scheduler Statistics ===
Total tasks completed: 100
CPU tasks: 60 (60.0%)
GPU tasks: 40 (40.0%)
Total CPU time: 1.234 seconds
Total GPU time: 0.456 seconds
Average CPU task time: 20.57 ms
Average GPU task time: 11.40 ms
Total compute time: 1.690 seconds
```

## 🚧 Future Extensions

### Performance Improvements
- [ ] **Work Stealing**: Load balancing between workers
- [ ] **Multi-GPU Support**: Distribute across multiple GPUs
- [ ] **CPU Vectorization**: SIMD optimizations
- [ ] **Memory Prefetching**: Predictive data movement

### Additional Features
- [ ] **Power Awareness**: Energy-efficient scheduling
- [ ] **Real-time Monitoring**: Web dashboard
- [ ] **OpenCL Support**: AMD/Intel GPU support
- [ ] **Distributed Computing**: Multi-node scheduling

### New Task Types
- [ ] **Convolution**: Image processing kernels
- [ ] **FFT**: Fast Fourier Transform
- [ ] **Sorting**: Parallel sorting algorithms
- [ ] **Graph Operations**: Graph processing tasks

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   # Build CPU-only version
   make cpu
   ```

2. **Permission denied**
   ```bash
   chmod +x build/hetero_sched
   ```

3. **Memory allocation errors**
   - Reduce task sizes or buffer sizes
   - Check available GPU memory

4. **Performance issues**
   - Verify CUDA drivers are up to date
   - Check GPU utilization with `nvidia-smi`
   - Adjust scheduling thresholds

### Debug Mode
```bash
# Build with debug symbols and logging
make debug-cuda

# Run with verbose output
./build/hetero_sched --tasks 10
```

## 📄 License

This project is provided as an educational example. Feel free to use and modify for learning purposes.

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Heterogeneous Computing Patterns](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/)
- [Task Scheduling Research](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=heterogeneous%20task%20scheduling)

---

## 📞 Support

For questions or issues:
1. Check this README
2. Review the source code comments
3. Run the debug builds
4. Check CUDA/driver installations