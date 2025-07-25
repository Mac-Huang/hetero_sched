#!/usr/bin/env python3
"""
Proximal Policy Optimization (PPO) Agent for HeteroSched

Implements PPO with clipped objective, multi-discrete action spaces, and 
generalized advantage estimation. Optimized for stable policy learning 
in the heterogeneous task scheduling environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque

from .base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """Policy network for multi-discrete action spaces"""
    
    def __init__(self, state_dim: int, action_dims: List[int], hidden_dims: List[int] = None):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Shared feature extraction
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate policy heads for each action dimension
        self.policy_heads = nn.ModuleList([
            nn.Linear(input_dim, action_dim) for action_dim in action_dims
        ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning logits for each action dimension"""
        features = self.shared_layers(state)
        
        # Compute logits for each action dimension
        logits = [head(features) for head in self.policy_heads]
        
        return logits
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities"""
        logits = self.forward(state)
        
        actions = []
        log_probs = []
        
        for logit in logits:
            dist = Categorical(logits=logit)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            actions.append(action)
            log_probs.append(log_prob)
        
        actions = torch.stack(actions, dim=1)  # [batch_size, num_action_dims]
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)  # [batch_size]
        
        return actions, log_probs
    
    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy for given actions"""
        logits = self.forward(state)
        
        log_probs = []
        entropies = []
        
        for i, logit in enumerate(logits):
            dist = Categorical(logits=logit)
            action = actions[:, i]
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        total_entropy = torch.stack(entropies, dim=1).sum(dim=1)
        
        return total_log_prob, total_entropy

class ValueNetwork(nn.Module):
    """Value function network"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = None):
        super(ValueNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))  # Single value output
        
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state values"""
        return self.network(state).squeeze(-1)

class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent"""
    
    def __init__(self, state_dim: int, action_dims: List[int], config: AgentConfig = None):
        super().__init__(state_dim, action_dims, config)
        
        self.action_dims = action_dims
        
        # PPO-specific config
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.gae_lambda = 0.95
        self.ppo_epochs = 4
        self.mini_batch_size = 64
        
        # Build networks
        self._build_networks()
        
        # PPO rollout storage
        self.rollout_buffer = []
        self.rollout_size = 2048  # Steps per policy update
        
        logger.info(f"PPO Agent initialized with action_dims={action_dims}")
    
    def _build_networks(self):
        """Build policy and value networks"""
        hidden_dims = [512, 256, 128]
        
        self.policy_network = PolicyNetwork(
            self.state_dim, 
            self.action_dims, 
            hidden_dims
        ).to(self.device)
        
        self.value_network = ValueNetwork(
            self.state_dim, 
            hidden_dims
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        
        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.policy_optimizer,
            step_size=10000,
            gamma=0.95
        )
        
        self.value_scheduler = optim.lr_scheduler.StepLR(
            self.value_optimizer,
            step_size=10000,
            gamma=0.95
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                # Sample from policy distribution
                actions, log_probs = self.policy_network.get_action_and_log_prob(state_tensor)
                value = self.value_network(state_tensor)
                
                # Store for rollout
                self.rollout_buffer.append({
                    'state': state,
                    'action': actions.cpu().numpy().flatten(),
                    'log_prob': log_probs.cpu().item(),
                    'value': value.cpu().item()
                })
                
                return actions.cpu().numpy().flatten()
            else:
                # Deterministic policy (argmax)
                logits = self.policy_network.forward(state_tensor)
                actions = [torch.argmax(logit, dim=1).item() for logit in logits]
                return np.array(actions)
    
    def store_transition_reward(self, reward: float, done: bool):
        """Store reward and done flag for current transition"""
        if self.rollout_buffer:
            self.rollout_buffer[-1]['reward'] = reward
            self.rollout_buffer[-1]['done'] = done
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO objective"""
        
        # Need sufficient rollout data
        if len(self.rollout_buffer) < self.rollout_size:
            return {'loss': 0.0}
        
        # Compute advantages and returns
        self._compute_advantages_and_returns()
        
        # Convert rollout to tensors
        states = torch.FloatTensor([exp['state'] for exp in self.rollout_buffer]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in self.rollout_buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in self.rollout_buffer]).to(self.device)
        advantages = torch.FloatTensor([exp['advantage'] for exp in self.rollout_buffer]).to(self.device)
        returns = torch.FloatTensor([exp['return'] for exp in self.rollout_buffer]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.ppo_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(self.rollout_buffer))
            
            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                log_probs, entropy = self.policy_network.evaluate_actions(batch_states, batch_actions)
                values = self.value_network(batch_states)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                policy_loss1 = ratio * batch_advantages
                policy_loss2 = clipped_ratio * batch_advantages
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coef * value_loss + 
                             self.entropy_coef * entropy_loss)
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
                self.value_optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # Update learning rates
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        # Clear rollout buffer
        self.rollout_buffer.clear()
        
        # Update training step
        self.training_step += 1
        
        # Return loss metrics
        num_updates = self.ppo_epochs * (len(states) // self.mini_batch_size)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'policy_lr': self.policy_scheduler.get_last_lr()[0],
            'value_lr': self.value_scheduler.get_last_lr()[0]
        }
    
    def _compute_advantages_and_returns(self):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        
        # Add bootstrap value for last state if episode didn't end
        if not self.rollout_buffer[-1]['done']:
            last_state = self.rollout_buffer[-1]['state']
            with torch.no_grad():
                last_value = self.value_network(
                    torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                ).item()
        else:
            last_value = 0.0
        
        # Compute advantages and returns using GAE
        advantages = []
        returns = []
        
        gae = 0.0
        
        for i in reversed(range(len(self.rollout_buffer))):
            exp = self.rollout_buffer[i]
            
            if i == len(self.rollout_buffer) - 1:
                next_value = last_value
            else:
                next_value = self.rollout_buffer[i + 1]['value']
            
            # TD error
            delta = exp['reward'] + self.config.gamma * next_value * (1 - exp['done']) - exp['value']
            
            # GAE advantage
            gae = delta + self.config.gamma * self.gae_lambda * (1 - exp['done']) * gae
            advantages.append(gae)
            
            # Return (advantage + value)
            returns.append(gae + exp['value'])
        
        # Reverse to get correct order
        advantages.reverse()
        returns.reverse()
        
        # Store in rollout buffer
        for i, (advantage, return_val) in enumerate(zip(advantages, returns)):
            self.rollout_buffer[i]['advantage'] = advantage
            self.rollout_buffer[i]['return'] = return_val
    
    def get_model_state(self) -> Dict:
        """Get model state dict for saving"""
        return {
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'policy_scheduler': self.policy_scheduler.state_dict(),
            'value_scheduler': self.value_scheduler.state_dict()
        }
    
    def load_model_state(self, state_dict: Dict):
        """Load model state dict"""
        self.policy_network.load_state_dict(state_dict['policy_network'])
        self.value_network.load_state_dict(state_dict['value_network'])
        self.policy_optimizer.load_state_dict(state_dict['policy_optimizer'])
        self.value_optimizer.load_state_dict(state_dict['value_optimizer'])
        self.policy_scheduler.load_state_dict(state_dict['policy_scheduler'])
        self.value_scheduler.load_state_dict(state_dict['value_scheduler'])
    
    def get_network_info(self) -> Dict:
        """Get information about the neural networks"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'policy_parameters': count_parameters(self.policy_network),
            'value_parameters': count_parameters(self.value_network),
            'total_parameters': count_parameters(self.policy_network) + count_parameters(self.value_network),
            'policy_architecture': str(self.policy_network),
            'value_architecture': str(self.value_network),
            'action_dimensions': self.action_dims,
            'device': str(self.device)
        }
    
    def get_action_probabilities(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get action probabilities for analysis"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.policy_network.forward(state_tensor)
            
            probs = {}
            action_names = ['device', 'priority', 'batch_size']
            
            for i, (logit, name) in enumerate(zip(logits, action_names)):
                prob = F.softmax(logit, dim=1).cpu().numpy().flatten()
                probs[name] = prob
            
            return probs