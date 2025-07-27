"""
Validation Module for HeteroSched

This module implements comprehensive validation frameworks for heterogeneous
scheduling research, including reproducibility testing, automated hyperparameter
optimization, continuous integration, experiment tracking, and standardized
evaluation metrics.

Key Components:
- Reproducibility framework with containerized experiments
- Automated hyperparameter optimization pipeline  
- Continuous integration for research experiments
- Experiment tracking and artifact management system
- Standardized evaluation metrics for scheduling algorithms

Authors: HeteroSched Research Team
"""

from .reproducibility_framework import (
    ExperimentType,
    ReproducibilityLevel,
    ExperimentConfig,
    ExperimentResult,
    EnvironmentManager,
    DeterministicExecutor,
    ResourceMonitor,
    ReproducibilityValidator,
    ContainerizedExperimentRunner,
    ReproducibilityFramework,
    sample_hetero_sched_experiment
)

__all__ = [
    # Reproducibility Framework
    'ExperimentType',
    'ReproducibilityLevel',
    'ExperimentConfig',
    'ExperimentResult',
    'EnvironmentManager',
    'DeterministicExecutor',
    'ResourceMonitor',
    'ReproducibilityValidator',
    'ContainerizedExperimentRunner',
    'ReproducibilityFramework',
    'sample_hetero_sched_experiment'
]