"""
RL Training Pipeline for HeteroSched

This module provides comprehensive training infrastructure for Deep RL agents
including experiment management, hyperparameter optimization, distributed training,
and advanced logging capabilities.

Components:
- TrainingManager: Main training orchestration
- ExperimentConfig: Comprehensive experiment configuration
- TensorBoard logging and visualization
- Model checkpoint management
- Performance evaluation and comparison
"""

from .trainer import TrainingManager, ExperimentConfig
from .evaluator import AgentEvaluator
from .utils import setup_tensorboard, save_experiment_config

__all__ = [
    'TrainingManager',
    'ExperimentConfig', 
    'AgentEvaluator',
    'setup_tensorboard',
    'save_experiment_config'
]