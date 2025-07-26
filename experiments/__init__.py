"""
Experiments Module for HeteroSched

This module implements comprehensive experimental frameworks for evaluating
heterogeneous scheduling algorithms, including ablation studies, statistical
analysis, benchmarking, and large-scale evaluation.

Key Components:
- Comprehensive ablation study framework for reward components
- Statistical significance testing with multiple comparisons
- Large-scale evaluation on realistic datacenter traces
- Benchmark suite for heterogeneous scheduling algorithms
- Sensitivity analysis for hyperparameter robustness

Authors: HeteroSched Research Team
"""

from .ablation_study import (
    RewardComponent,
    AblationConfiguration,
    RewardFunctionFactory,
    AblationExperiment,
    AblationStudyFramework
)

__all__ = [
    'RewardComponent',
    'AblationConfiguration', 
    'RewardFunctionFactory',
    'AblationExperiment',
    'AblationStudyFramework'
]