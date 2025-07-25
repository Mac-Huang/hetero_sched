#!/usr/bin/env python3
"""
Base Agent Class for HeteroSched RL Agents

Provides common interface and functionality for all RL agents including
training utilities, model management, and performance tracking.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import os
import pickle
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for RL agents"""
    learning_rate: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.99  # Discount factor
    tau: float = 0.001   # Soft update coefficient
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    update_frequency: int = 4  # Steps between updates
    target_update_frequency: int = 1000  # Steps between target updates
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_frequency: int = 10000  # Steps between model saves
    log_frequency: int = 1000   # Steps between performance logs

@dataclass 
class TrainingMetrics:
    """Metrics for tracking training progress"""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    epsilon_values: List[float] = field(default_factory=list)
    objective_scores: Dict[str, List[float]] = field(default_factory=dict)
    training_steps: int = 0
    episodes_completed: int = 0

class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or AgentConfig()
        
        # Set device
        self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")
        
        # Training metrics
        self.metrics = TrainingMetrics()
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.config.buffer_size)
        
        # Training state
        self.training_step = 0
        self.epsilon = self.config.epsilon_start
        self.is_training = True
        
        # Model save directory
        self.save_dir = "models"
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} with state_dim={state_dim}, action_dim={action_dim}")
    
    @abstractmethod
    def _build_networks(self):
        """Build neural networks (implement in subclasses)"""
        pass
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action given current state"""
        pass
    
    @abstractmethod
    def update(self) -> Dict[str, float]:
        """Update agent parameters and return loss metrics"""
        pass
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def can_update(self) -> bool:
        """Check if agent has enough experience to update"""
        return len(self.memory) >= self.config.min_buffer_size
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
    
    def soft_update(self, target_net: nn.Module, source_net: nn.Module):
        """Soft update of target network parameters"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data + (1.0 - self.config.tau) * target_param.data
            )
    
    def hard_update(self, target_net: nn.Module, source_net: nn.Module):
        """Hard update of target network parameters"""
        target_net.load_state_dict(source_net.state_dict())
    
    def sample_batch(self, batch_size: int = None) -> Tuple[torch.Tensor, ...]:
        """Sample batch from experience replay buffer"""
        batch_size = batch_size or self.config.batch_size
        
        # Sample random batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Unpack batch
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def log_training_step(self, episode_reward: float, episode_length: int, 
                         loss: float, info: Dict = None):
        """Log training metrics"""
        self.metrics.episode_rewards.append(episode_reward)
        self.metrics.episode_lengths.append(episode_length)
        self.metrics.losses.append(loss)
        self.metrics.epsilon_values.append(self.epsilon)
        self.metrics.training_steps = self.training_step
        self.metrics.episodes_completed += 1
        
        # Log objective scores if available
        if info and 'objective_scores' in info:
            for obj, score in info['objective_scores'].items():
                if obj not in self.metrics.objective_scores:
                    self.metrics.objective_scores[obj] = []
                self.metrics.objective_scores[obj].append(score)
        
        # Periodic logging
        if self.metrics.episodes_completed % (self.config.log_frequency // 100) == 0:
            recent_rewards = self.metrics.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            logger.info(f"Episode {self.metrics.episodes_completed}: "
                       f"avg_reward={avg_reward:.3f}, epsilon={self.epsilon:.3f}, "
                       f"buffer_size={len(self.memory)}, loss={loss:.4f}")
    
    def save_model(self, filepath: str = None):
        """Save agent model and training state"""
        if filepath is None:
            filepath = os.path.join(self.save_dir, 
                                  f"{self.__class__.__name__}_step_{self.training_step}.pth")
        
        save_data = {
            'model_state_dict': self.get_model_state(),
            'config': self.config,
            'metrics': self.metrics,
            'training_step': self.training_step,
            'epsilon': self.epsilon
        }
        
        torch.save(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load agent model and training state"""
        save_data = torch.load(filepath, map_location=self.device)
        
        self.load_model_state(save_data['model_state_dict'])
        self.metrics = save_data['metrics']
        self.training_step = save_data['training_step']
        self.epsilon = save_data['epsilon']
        
        logger.info(f"Model loaded from {filepath}")
    
    @abstractmethod
    def get_model_state(self) -> Dict:
        """Get model state dict for saving"""
        pass
    
    @abstractmethod
    def load_model_state(self, state_dict: Dict):
        """Load model state dict"""
        pass
    
    def get_training_summary(self) -> Dict:
        """Get summary of training performance"""
        if len(self.metrics.episode_rewards) < 10:
            return {'status': 'insufficient_data'}
        
        recent_rewards = self.metrics.episode_rewards[-100:]
        recent_lengths = self.metrics.episode_lengths[-100:]
        
        summary = {
            'episodes_completed': self.metrics.episodes_completed,
            'training_steps': self.training_step,
            'current_epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'avg_episode_reward': np.mean(recent_rewards),
            'std_episode_reward': np.std(recent_rewards),
            'avg_episode_length': np.mean(recent_lengths),
            'best_episode_reward': max(self.metrics.episode_rewards),
            'recent_improvement': self._compute_improvement(recent_rewards)
        }
        
        # Add objective-specific performance
        for obj_name, scores in self.metrics.objective_scores.items():
            if len(scores) >= 10:
                recent_scores = scores[-100:]
                summary[f'{obj_name}_avg'] = np.mean(recent_scores)
                summary[f'{obj_name}_trend'] = self._compute_trend(recent_scores)
        
        return summary
    
    def _compute_improvement(self, rewards: List[float]) -> float:
        """Compute recent performance improvement"""
        if len(rewards) < 20:
            return 0.0
        
        mid_point = len(rewards) // 2
        early_avg = np.mean(rewards[:mid_point])
        recent_avg = np.mean(rewards[mid_point:])
        
        if early_avg != 0:
            return (recent_avg - early_avg) / abs(early_avg)
        return 0.0
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend direction (-1 to 1)"""
        if len(values) < 10:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return np.tanh(coeffs[0] * 10)  # Normalize slope
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress metrics"""
        if len(self.metrics.episode_rewards) < 10:
            logger.warning("Insufficient data for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.__class__.__name__} Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.metrics.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.metrics.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Training losses
        if self.metrics.losses:
            axes[1, 0].plot(self.metrics.losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
        
        # Epsilon decay
        axes[1, 1].plot(self.metrics.epsilon_values)
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def set_training_mode(self, training: bool):
        """Set agent training/evaluation mode"""
        self.is_training = training
        if hasattr(self, 'q_network'):
            self.q_network.train(training)
        if hasattr(self, 'policy_network'):
            self.policy_network.train(training)
        if hasattr(self, 'value_network'):
            self.value_network.train(training)