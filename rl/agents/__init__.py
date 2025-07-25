"""
Deep RL Agents for HeteroSched

This module provides implementations of Deep Reinforcement Learning agents
optimized for heterogeneous task scheduling:

- DQN (Deep Q-Network): Value-based learning with experience replay
- PPO (Proximal Policy Optimization): Policy gradient with clipped objectives
- Multi-objective variants with Pareto-optimal action selection

All agents are designed to work with the HeteroSchedEnv environment and
support multi-objective optimization through sophisticated reward functions.
"""

from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from .base_agent import BaseAgent

__all__ = [
    'BaseAgent',
    'DQNAgent', 
    'PPOAgent'
]