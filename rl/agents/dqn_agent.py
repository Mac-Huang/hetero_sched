#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Agent for HeteroSched

Implements DQN with experience replay, target networks, and multi-discrete action spaces.
Optimized for the heterogeneous task scheduling problem with sophisticated exploration
strategies and multi-objective reward handling.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import random
import logging

from .base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """Deep Q-Network for multi-discrete action spaces"""
    
    def __init__(self, state_dim: int, action_dims: List[int], hidden_dims: List[int] = None):
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dims = action_dims  # [device, priority_boost, batch_size]
        self.total_actions = sum(action_dims)
        
        # Default hidden layer dimensions
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Shared feature extraction layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # Use LayerNorm instead of BatchNorm for single samples
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate Q-value heads for each action dimension
        self.q_heads = nn.ModuleList([
            nn.Linear(input_dim, action_dim) for action_dim in action_dims
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning Q-values for each action dimension"""
        features = self.shared_layers(state)
        
        # Compute Q-values for each action dimension
        q_values = [head(features) for head in self.q_heads]
        
        return q_values

class DQNAgent(BaseAgent):
    """Deep Q-Network agent with multi-discrete action space support"""
    
    def __init__(self, state_dim: int, action_dims: List[int], config: AgentConfig = None):
        # action_dims should be [2, 5, 10] for [device, priority_boost, batch_size]
        super().__init__(state_dim, action_dims, config)
        
        self.action_dims = action_dims
        self.total_actions = sum(action_dims)
        
        # Build networks
        self._build_networks()
        
        # Initialize target network
        self.hard_update(self.target_network, self.q_network)
        
        logger.info(f"DQN Agent initialized with action_dims={action_dims}")
    
    def _build_networks(self):
        """Build Q-network and target network"""
        hidden_dims = [512, 256, 128]  # Larger networks for complex state space
        
        self.q_network = DQNNetwork(
            self.state_dim, 
            self.action_dims, 
            hidden_dims
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            self.state_dim, 
            self.action_dims, 
            hidden_dims
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10000, 
            gamma=0.95
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            # Random action for each dimension
            action = [
                random.randint(0, dim - 1) for dim in self.action_dims
            ]
            return np.array(action)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Select best action for each dimension
            action = [
                q_vals.argmax().item() for q_vals in q_values
            ]
            
            return np.array(action)
    
    def update(self) -> Dict[str, float]:
        """Update Q-network using DQN loss"""
        if not self.can_update():
            return {'loss': 0.0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.sample_batch()
        
        # Compute current Q-values
        q_values = self.q_network(states)
        
        # Extract Q-values for taken actions
        current_q_values = []
        for i, (q_vals, action_dim) in enumerate(zip(q_values, self.action_dims)):
            action_indices = actions[:, i]
            current_q = q_vals.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            current_q_values.append(current_q)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            
            # Double DQN: use main network to select actions, target network for values
            main_next_q_values = self.q_network(next_states)
            
            target_q_values = []
            for i, (target_q_vals, main_q_vals) in enumerate(zip(next_q_values, main_next_q_values)):
                # Select actions using main network
                next_actions = main_q_vals.argmax(dim=1)
                # Evaluate actions using target network
                target_q = target_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q_values.append(target_q)
        
        # Compute expected Q-values (Bellman equation)
        expected_q_values = []
        for target_q in target_q_values:
            expected_q = rewards + (self.config.gamma * target_q * ~dones)
            expected_q_values.append(expected_q)
        
        # Compute losses for each action dimension
        losses = []
        for current_q, expected_q in zip(current_q_values, expected_q_values):
            loss = F.smooth_l1_loss(current_q, expected_q)
            losses.append(loss)
        
        # Total loss is sum of losses for all action dimensions
        total_loss = sum(losses)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update training step counter
        self.training_step += 1
        
        # Soft update target network
        if self.training_step % self.config.target_update_frequency == 0:
            self.soft_update(self.target_network, self.q_network)
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Return loss metrics
        loss_dict = {
            'total_loss': total_loss.item(),
            'device_loss': losses[0].item() if len(losses) > 0 else 0.0,
            'priority_loss': losses[1].item() if len(losses) > 1 else 0.0,
            'batch_loss': losses[2].item() if len(losses) > 2 else 0.0,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return loss_dict
    
    def get_q_values(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get Q-values for all actions given state (for analysis)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            q_dict = {
                'device': q_values[0].cpu().numpy().flatten(),
                'priority': q_values[1].cpu().numpy().flatten(),
                'batch_size': q_values[2].cpu().numpy().flatten()
            }
            
            return q_dict
    
    def get_action_values_summary(self, states: List[np.ndarray]) -> Dict:
        """Analyze Q-value distributions across multiple states"""
        if not states:
            return {}
        
        all_q_values = {'device': [], 'priority': [], 'batch_size': []}
        
        for state in states:
            q_values = self.get_q_values(state)
            for key, values in q_values.items():
                all_q_values[key].extend(values)
        
        summary = {}
        for action_type, values in all_q_values.items():
            values = np.array(values)
            summary[action_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'preferred_action': np.argmax(np.bincount(np.argmax(
                    np.array([self.get_q_values(s)[action_type] for s in states]), axis=1
                )))
            }
        
        return summary
    
    def get_model_state(self) -> Dict:
        """Get model state dict for saving"""
        return {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
    
    def load_model_state(self, state_dict: Dict):
        """Load model state dict"""
        self.q_network.load_state_dict(state_dict['q_network'])
        self.target_network.load_state_dict(state_dict['target_network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
    
    def get_network_info(self) -> Dict:
        """Get information about the neural networks"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': count_parameters(self.q_network),
            'network_architecture': str(self.q_network),
            'action_dimensions': self.action_dims,
            'device': str(self.device)
        }