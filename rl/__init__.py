"""
HeteroSched Deep Reinforcement Learning Module

This module contains the RL infrastructure for intelligent heterogeneous task scheduling:
- Gym-compatible environments
- Deep RL agents (DQN, PPO, SAC)
- Multi-objective reward functions
- Training and evaluation pipelines
"""

__version__ = "2.1.0"
__author__ = "HeteroSched Research Team"

from .environments.hetero_env import HeteroSchedEnv, make_hetero_env
# from .agents import DQNAgent, PPOAgent, SACAgent  # Will be implemented in next versions
# from .rewards import MultiObjectiveReward
# from .utils import RLLogger, ModelManager

__all__ = [
    'HeteroSchedEnv',
    'make_hetero_env',
    # 'DQNAgent', 'PPOAgent', 'SACAgent',
    # 'MultiObjectiveReward',
    # 'RLLogger', 'ModelManager'
]