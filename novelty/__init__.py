"""
Novelty Module for HeteroSched

This module implements cutting-edge and novel techniques for heterogeneous
scheduling research, including attention-based RL, causal inference,
adversarial training, neural architecture search, and quantum-inspired
optimization methods.

Key Components:
- Attention-based RL for dynamic priority scheduling
- Causal inference for scheduling policy interpretability  
- Adversarial training for robust scheduling policies
- Neural architecture search for optimal RL agent design
- Quantum-inspired optimization for multi-objective RL

Authors: HeteroSched Research Team
"""

from .attention_based_rl import (
    AttentionType,
    PriorityLevel,
    Task,
    Resource,
    SchedulingState,
    MultiHeadAttention,
    TemporalAttention,
    HierarchicalAttention,
    TaskResourceAttention,
    AttentionBasedPolicyNetwork,
    DynamicPriorityScheduler,
    create_sample_scheduling_state
)

__all__ = [
    # Attention-Based RL
    'AttentionType',
    'PriorityLevel',
    'Task',
    'Resource',
    'SchedulingState',
    'MultiHeadAttention',
    'TemporalAttention',
    'HierarchicalAttention',
    'TaskResourceAttention',
    'AttentionBasedPolicyNetwork',
    'DynamicPriorityScheduler',
    'create_sample_scheduling_state'
]