"""
Transfer Learning Module for HeteroSched

This module implements foundation models, meta-learning, and transfer learning
techniques for heterogeneous scheduling systems.

Key Components:
- Foundation model architecture with transformer-based design
- Pre-training data generation for diverse workloads
- Meta-learning algorithms for rapid adaptation
- Few-shot learning for new scheduling environments
- Domain adaptation techniques for deployment

Authors: HeteroSched Research Team
"""

from .foundation_model import (
    HeteroSchedFoundationModel,
    FoundationModelConfig, 
    FoundationModelTrainer,
    ResourceAwareAttention,
    MultiScaleTemporalEncoder,
    TaskEmbedding
)

from .pretraining_data import (
    PretrainingDataGenerator,
    WorkloadGenerator,
    SystemStateSimulator,
    WorkloadPattern,
    SchedulingScenario
)

__all__ = [
    'HeteroSchedFoundationModel',
    'FoundationModelConfig',
    'FoundationModelTrainer', 
    'ResourceAwareAttention',
    'MultiScaleTemporalEncoder',
    'TaskEmbedding',
    'PretrainingDataGenerator',
    'WorkloadGenerator',
    'SystemStateSimulator',
    'WorkloadPattern',
    'SchedulingScenario'
]