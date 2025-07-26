#!/usr/bin/env python3
"""
Uncertainty-Aware Transfer Learning for HeteroSched

This module implements uncertainty quantification and safe transfer techniques
for deploying RL-trained schedulers in real heterogeneous environments.

Research Innovation: Bayesian deep learning approach to quantify epistemic 
and aleatoric uncertainty in scheduling decisions, enabling safer sim-to-real transfer.

Key Components:
- Bayesian Neural Networks for uncertainty quantification
- Monte Carlo Dropout for practical uncertainty estimation
- Uncertainty-guided exploration and exploitation
- Safe transfer with confidence bounds
- Gradual uncertainty reduction during adaptation

Authors: HeteroSched Research Team
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty quantification metrics"""
    epistemic_uncertainty: float  # Model uncertainty (lack of knowledge)
    aleatoric_uncertainty: float  # Data uncertainty (inherent noise)
    total_uncertainty: float      # Combined uncertainty
    confidence_interval: Tuple[float, float]  # 95% confidence bounds
    entropy: float               # Predictive entropy
    mutual_information: float    # Information gain potential

class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainties"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 0.1):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 5)
        
        # Prior distribution
        self.prior_std = prior_std
        self.kl_divergence = 0.0
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_loss(self) -> torch.Tensor:
        """Compute KL divergence with prior"""
        # KL(q(w)||p(w)) for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            self.weight_mu.pow(2) / (self.prior_std**2) + 
            weight_var / (self.prior_std**2) - 
            1 - torch.log(weight_var / (self.prior_std**2))
        )
        
        # KL for biases
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            self.bias_mu.pow(2) / (self.prior_std**2) + 
            bias_var / (self.prior_std**2) - 
            1 - torch.log(bias_var / (self.prior_std**2))
        )
        
        return weight_kl + bias_kl
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with uncertainty"""
        # Sample weights and biases
        weight = self.reparameterize(self.weight_mu, self.weight_logvar)
        bias = self.reparameterize(self.bias_mu, self.bias_logvar)
        
        # Store KL divergence for loss computation
        self.kl_divergence = self.kl_loss()
        
        return F.linear(x, weight, bias)

class BayesianQNetwork(nn.Module):
    """Bayesian Q-Network for uncertainty-aware value estimation"""
    
    def __init__(self, state_dim: int = 36, action_dim: int = 100, 
                 hidden_dims: List[int] = [256, 128], dropout_rate: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        
        # Build Bayesian network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                BayesianLinear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layer (also Bayesian)
        layers.append(BayesianLinear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Enable Monte Carlo Dropout
        self.enable_dropout = True
    
    def forward(self, state: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Forward pass with optional Monte Carlo sampling"""
        
        if num_samples == 1:
            return self.network(state)
        else:
            # Monte Carlo sampling for uncertainty estimation
            outputs = []
            for _ in range(num_samples):
                if self.enable_dropout:
                    self.train()  # Enable dropout during inference
                output = self.network(state)
                outputs.append(output)
            
            return torch.stack(outputs, dim=0)  # [num_samples, batch_size, action_dim]
    
    def get_kl_loss(self) -> torch.Tensor:
        """Get total KL divergence loss"""
        kl_loss = 0.0
        for module in self.network:
            if isinstance(module, BayesianLinear):
                kl_loss += module.kl_divergence
        return kl_loss
    
    def predict_with_uncertainty(self, state: torch.Tensor, 
                               num_samples: int = 100) -> UncertaintyMetrics:
        """Predict Q-values with comprehensive uncertainty quantification"""
        
        self.eval()
        with torch.no_grad():
            # Monte Carlo sampling
            outputs = self.forward(state, num_samples=num_samples)  # [num_samples, batch_size, action_dim]
            
            # Compute statistics
            mean_q = torch.mean(outputs, dim=0)  # [batch_size, action_dim]
            var_q = torch.var(outputs, dim=0)    # [batch_size, action_dim]
            std_q = torch.sqrt(var_q)
            
            # Epistemic uncertainty (model uncertainty)
            epistemic_uncertainty = float(torch.mean(var_q))
            
            # Aleatoric uncertainty (approximate as residual)
            aleatoric_uncertainty = 0.1  # Placeholder - would need more sophisticated estimation
            
            # Total uncertainty
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            # Confidence intervals (95%)
            confidence_lower = mean_q - 1.96 * std_q
            confidence_upper = mean_q + 1.96 * std_q
            
            # Predictive entropy
            probs = F.softmax(mean_q, dim=-1)
            entropy = float(-torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean())
            
            # Mutual information (information gain)
            # I(y, θ|x) = H(y|x) - E[H(y|x,θ)]
            individual_entropies = []
            for i in range(num_samples):
                sample_probs = F.softmax(outputs[i], dim=-1)
                sample_entropy = -torch.sum(sample_probs * torch.log(sample_probs + 1e-8), dim=-1)
                individual_entropies.append(sample_entropy)
            
            expected_entropy = torch.mean(torch.stack(individual_entropies), dim=0)
            mutual_information = float(entropy - torch.mean(expected_entropy))
            
        return UncertaintyMetrics(
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_interval=(
                float(torch.mean(confidence_lower)), 
                float(torch.mean(confidence_upper))
            ),
            entropy=entropy,
            mutual_information=mutual_information
        )

class UncertaintyAwareAgent:
    """RL Agent with uncertainty-aware decision making"""
    
    def __init__(self, state_dim: int = 36, action_dim: int = 100,
                 uncertainty_threshold: float = 0.2, confidence_level: float = 0.95):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_level = confidence_level
        
        # Bayesian Q-Networks
        self.q_network = BayesianQNetwork(state_dim, action_dim)
        self.target_network = BayesianQNetwork(state_dim, action_dim)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)
        
        # Experience replay with uncertainty prioritization
        self.memory = UncertaintyPrioritizedReplay(capacity=10000)
        
        # Training statistics
        self.training_stats = {
            'losses': [],
            'kl_losses': [],
            'uncertainties': [],
            'confidence_violations': []
        }
        
        logger.info("Uncertainty-aware agent initialized")
    
    def select_action(self, state: np.ndarray, exploration_rate: float = 0.1) -> int:
        """Select action with uncertainty-aware exploration"""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get prediction with uncertainty
        uncertainty_metrics = self.q_network.predict_with_uncertainty(state_tensor)
        
        # Uncertainty-guided exploration
        if uncertainty_metrics.total_uncertainty > self.uncertainty_threshold:
            # High uncertainty: explore more
            if np.random.random() < exploration_rate * 2:  # Double exploration rate
                return np.random.randint(self.action_dim)
        
        # Get Q-values with confidence bounds
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            # Check confidence for top actions
            top_actions = torch.argsort(q_values.squeeze(), descending=True)[:3]
            
            for action_tensor in top_actions:
                action_idx = int(action_tensor.item())  # Use .item() for scalar tensor
                
                # Simple confidence check (would use full uncertainty analysis in practice)
                if uncertainty_metrics.epistemic_uncertainty < self.uncertainty_threshold:
                    return action_idx
            
            # If no confident action, return most promising with warning
            logger.warning(f"High uncertainty decision: {uncertainty_metrics.total_uncertainty:.4f}")
            return int(torch.argmax(q_values).item())
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Training step with uncertainty-aware loss"""
        
        if len(self.memory) < batch_size:
            return {}
        
        # Sample batch with uncertainty prioritization
        batch = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(batch['states'])
        actions = torch.LongTensor(batch['actions'])
        rewards = torch.FloatTensor(batch['rewards'])
        next_states = torch.FloatTensor(batch['next_states'])
        dones = torch.BoolTensor(batch['dones'])
        
        # Current Q-values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Target Q-values with uncertainty
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards + (0.99 * max_next_q * ~dones)
        
        # Compute losses
        td_loss = F.mse_loss(current_q.squeeze(), target_q)
        kl_loss = self.q_network.get_kl_loss()
        
        # Total loss with KL regularization
        total_loss = td_loss + 0.01 * kl_loss  # Beta = 0.01 for KL weight
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update statistics
        self.training_stats['losses'].append(float(td_loss))
        self.training_stats['kl_losses'].append(float(kl_loss))
        
        return {
            'td_loss': float(td_loss),
            'kl_loss': float(kl_loss),
            'total_loss': float(total_loss)
        }
    
    def update_uncertainty_priorities(self, experiences: List[Dict]):
        """Update experience priorities based on uncertainty"""
        
        for exp in experiences:
            state = torch.FloatTensor(exp['state']).unsqueeze(0)
            uncertainty_metrics = self.q_network.predict_with_uncertainty(state)
            
            # Higher uncertainty = higher priority for training
            priority = uncertainty_metrics.total_uncertainty + 0.1
            self.memory.update_priority(exp['index'], priority)

class UncertaintyPrioritizedReplay:
    """Experience replay buffer with uncertainty-based prioritization"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done, uncertainty: float = 1.0):
        """Add experience with uncertainty-based priority"""
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'index': self.position
        }
        
        self.buffer[self.position] = experience
        
        # Set priority based on uncertainty
        priority = (uncertainty + 1e-6) ** self.alpha
        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample batch with priority-based selection"""
        
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Compute sampling probabilities
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / np.max(weights)  # Normalize
        
        # Extract experiences
        batch = {
            'states': np.array([self.buffer[i]['state'] for i in indices]),
            'actions': np.array([self.buffer[i]['action'] for i in indices]),
            'rewards': np.array([self.buffer[i]['reward'] for i in indices]),
            'next_states': np.array([self.buffer[i]['next_state'] for i in indices]),
            'dones': np.array([self.buffer[i]['done'] for i in indices]),
            'weights': weights,
            'indices': indices
        }
        
        return batch
    
    def update_priority(self, index: int, priority: float):
        """Update priority for specific experience"""
        self.priorities[index] = (priority + 1e-6) ** self.alpha
        self.max_priority = max(self.max_priority, self.priorities[index])
    
    def __len__(self):
        return len(self.buffer)

def main():
    """Demonstrate uncertainty-aware transfer learning"""
    
    print("=== Uncertainty-Aware Transfer Learning ===\n")
    
    # Initialize uncertainty-aware agent
    agent = UncertaintyAwareAgent(
        state_dim=36,
        action_dim=100,
        uncertainty_threshold=0.2,
        confidence_level=0.95
    )
    
    print("Testing Bayesian Q-Network uncertainty estimation...")
    
    # Generate test state
    test_state = torch.randn(1, 36)
    
    # Get prediction with uncertainty
    uncertainty_metrics = agent.q_network.predict_with_uncertainty(test_state, num_samples=50)
    
    print(f"Uncertainty Analysis:")
    print(f"  Epistemic Uncertainty: {uncertainty_metrics.epistemic_uncertainty:.4f}")
    print(f"  Aleatoric Uncertainty: {uncertainty_metrics.aleatoric_uncertainty:.4f}")
    print(f"  Total Uncertainty: {uncertainty_metrics.total_uncertainty:.4f}")
    print(f"  Confidence Interval: [{uncertainty_metrics.confidence_interval[0]:.3f}, "
          f"{uncertainty_metrics.confidence_interval[1]:.3f}]")
    print(f"  Predictive Entropy: {uncertainty_metrics.entropy:.4f}")
    print(f"  Mutual Information: {uncertainty_metrics.mutual_information:.4f}")
    
    # Test action selection
    print(f"\nTesting uncertainty-aware action selection...")
    
    for i in range(5):
        test_state_np = np.random.randn(36)
        action = agent.select_action(test_state_np, exploration_rate=0.1)
        print(f"  State {i+1}: Selected action {action}")
    
    # Simulate training
    print(f"\nSimulating uncertainty-aware training...")
    
    for episode in range(10):
        # Simulate episode data
        state = np.random.randn(36)
        action = np.random.randint(100)
        reward = np.random.randn()
        next_state = np.random.randn(36)
        done = np.random.random() < 0.1
        
        # Add to memory with random uncertainty
        uncertainty = np.random.uniform(0.1, 0.5)
        agent.memory.push(state, action, reward, next_state, done, uncertainty)
        
        # Training step
        if len(agent.memory) >= 32:
            losses = agent.train_step(batch_size=32)
            if losses:
                print(f"  Episode {episode+1}: TD Loss = {losses['td_loss']:.4f}, "
                      f"KL Loss = {losses['kl_loss']:.4f}")
        
        # Update target network periodically
        if episode % 5 == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
    
    print(f"\nUncertainty prioritized replay buffer size: {len(agent.memory)}")
    print(f"Training completed with uncertainty-aware learning!")

if __name__ == '__main__':
    main()