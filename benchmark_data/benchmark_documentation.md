# HeteroSched Community Benchmark Framework

## Overview

The HeteroSched Community Benchmark Framework provides standardized evaluation 
for heterogeneous scheduling algorithms. Our goal is to foster reproducible 
research and enable fair comparison across different approaches.

## Getting Started

### 1. Installation

```bash
pip install heterosched-benchmark
git clone https://github.com/heterosched/community-benchmark.git
cd community-benchmark
```

### 2. Running Benchmarks

```python
from heterosched_benchmark import CommunityBenchmarkFramework

# Initialize framework
framework = CommunityBenchmarkFramework()

# Submit your algorithm
submission_id = framework.submit_algorithm(
    algorithm_info={
        'algorithm_name': 'MyScheduler',
        'algorithm_type': 'deep_rl',
        'category': 'hpc_scheduling',
        'submitter_name': 'Your Name',
        'submitter_email': 'your.email@example.com',
        'description': 'Novel RL-based scheduler',
        'authors': ['Your Name'],
        'institution': 'Your University'
    },
    algorithm_code=open('my_scheduler.py').read()
)

# Evaluate against benchmarks
results = framework.evaluate_submission(submission_id, 'hpc_standard')
print(f"Overall score: {results['overall_score']:.3f}")
```

## Benchmark Suites


### HPC Standard Benchmark

**Category:** hpc_scheduling  
**Description:** Standard HPC workload scheduling with job dependencies  
**Datasets:** 3 standardized workload traces  
**Metrics:** makespan, utilization, fairness, energy_efficiency  
**Time Limit:** 60 minutes  
**Memory Limit:** 8192 MB  

**Baseline Performance:**
- FIFO: 0.450
- SJF: 0.620
- Priority: 0.580
- Backfill: 0.710

### Cloud Orchestration Benchmark

**Category:** cloud_orchestration  
**Description:** Multi-tenant cloud resource allocation and scaling  
**Datasets:** 3 standardized workload traces  
**Metrics:** response_time, cost_efficiency, sla_compliance, resource_waste  
**Time Limit:** 30 minutes  
**Memory Limit:** 4096 MB  

**Baseline Performance:**
- Round_Robin: 0.520
- Greedy: 0.650
- Kubernetes: 0.730
- AutoScale: 0.680

### Edge Computing Benchmark

**Category:** edge_computing  
**Description:** Distributed edge task placement with latency constraints  
**Datasets:** 3 standardized workload traces  
**Metrics:** latency, bandwidth_usage, energy_consumption, availability  
**Time Limit:** 15 minutes  
**Memory Limit:** 2048 MB  

**Baseline Performance:**
- Nearest: 0.480
- Load_Balance: 0.610
- Latency_Aware: 0.690
- Adaptive: 0.720

### Multi-Objective Optimization Benchmark

**Category:** multi_objective  
**Description:** Pareto-optimal scheduling across multiple objectives  
**Datasets:** 3 standardized workload traces  
**Metrics:** pareto_coverage, hypervolume, diversity, convergence  
**Time Limit:** 120 minutes  
**Memory Limit:** 16384 MB  

**Baseline Performance:**
- NSGA_II: 0.580
- MOEA_D: 0.630
- SPEA2: 0.610
- Random: 0.250


## Submission Guidelines

### Algorithm Interface

Your algorithm must implement the following interface:

```python
class SchedulingAlgorithm:
    def __init__(self):
        # Initialize your algorithm
        pass
        
    def schedule(self, data):
        # Implement scheduling logic
        # data: input workload and system configuration
        # returns: scheduling decisions
        pass
```

### Submission Package

Your submission should include:
- `algorithm.py`: Main algorithm implementation
- `config.yaml`: Algorithm configuration and metadata
- `requirements.txt`: Python dependencies
- `README.md`: Algorithm description and usage

### Evaluation Process

1. **Automated Testing**: Your submission runs against standardized datasets
2. **Metric Calculation**: Performance metrics are computed automatically
3. **Verification**: Results are verified for correctness and reproducibility
4. **Leaderboard Update**: Rankings are updated in real-time

## Metrics and Scoring

### Scoring System

- **Overall Score**: Normalized performance relative to baseline algorithms
- **Category Ranking**: Ranking within specific benchmark categories
- **Reproducibility**: Verified reproducibility across multiple runs

### Evaluation Metrics


**Performance Metrics:**
- `makespan`: Total completion time for all jobs
- `utilization`: Average resource utilization
- `fairness`: Fair share allocation (Jain's fairness index)
- `energy_efficiency`: Performance per watt consumption

**Quality Metrics:**
- `response_time`: Average task response time
- `cost_efficiency`: Performance per dollar cost
- `sla_compliance`: Service level agreement compliance rate
- `latency`: Average end-to-end latency

**Multi-Objective Metrics:**
- `pareto_coverage`: Coverage of Pareto frontier
- `hypervolume`: Hypervolume indicator
- `diversity`: Solution diversity measure
- `convergence`: Convergence to optimal solutions


## Community Guidelines

### Code of Conduct

1. **Fair Play**: Submit only your own work or properly attributed collaborations
2. **Reproducibility**: Ensure your algorithm produces consistent results
3. **Documentation**: Provide clear documentation and citations
4. **Respect**: Maintain respectful discourse in community forums

### Best Practices

1. **Start Simple**: Begin with baseline implementations before optimization
2. **Profile Performance**: Understand your algorithm's computational requirements
3. **Test Locally**: Validate your submission before public submission
4. **Engage Community**: Participate in discussions and share insights

## Leaderboard

Visit our live leaderboard at: https://heterosched.github.io/leaderboard

### Current Leaders


**HPC Scheduling:**
1. DeepScheduler v2.1 - Stanford University (Score: 1.24)
2. MetaRL Scheduler - MIT (Score: 1.19)
3. AdaptiveHPC - CMU (Score: 1.15)

**Cloud Orchestration:**
1. CloudOptimizer Pro - Google Research (Score: 1.31)
2. AutoScale RL - Microsoft Research (Score: 1.28)
3. ElasticScheduler - Amazon (Score: 1.22)

**Edge Computing:**
1. EdgeRL - Berkeley (Score: 1.18)
2. FogScheduler - Georgia Tech (Score: 1.14)
3. LatencyAware - UIUC (Score: 1.11)


## API Reference

### Submission API

```python
# Submit algorithm
submission_id = framework.submit_algorithm(algorithm_info, algorithm_code)

# Get submission status
status = framework.get_submission_status(submission_id)

# Evaluate submission
results = framework.evaluate_submission(submission_id, benchmark_suite)
```

### Query API

```python
# Get leaderboard
leaderboard = framework.get_leaderboard(category='hpc_scheduling', limit=10)

# Get benchmark results
results = framework.get_benchmark_results(submission_id)

# Get algorithm details
details = framework.get_algorithm_details(submission_id)
```

## Contributing

We welcome contributions to the benchmark framework:

1. **New Benchmarks**: Propose additional benchmark suites
2. **Improved Metrics**: Suggest better evaluation metrics
3. **Bug Reports**: Report issues and inconsistencies
4. **Documentation**: Improve documentation and examples

### Development Setup

```bash
git clone https://github.com/heterosched/community-benchmark.git
cd community-benchmark
pip install -e .[dev]
pytest tests/
```

## Support

- **GitHub Issues**: https://github.com/heterosched/community-benchmark/issues
- **Community Forum**: https://heterosched.github.io/forum
- **Email**: support@heterosched.org

## Citation

If you use this benchmark framework in your research, please cite:

```bibtex
@misc{heterosched_benchmark,
    title={HeteroSched Community Benchmark Framework},
    author={HeteroSched Research Team},
    year={2024},
    url={https://github.com/heterosched/community-benchmark}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Last Updated:** 2025-07-27
