#!/usr/bin/env python3
"""
Standalone test for domain adaptation framework
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

# Simple mock classes for testing
class MockAgent:
    def select_action(self, obs):
        return np.random.randint(0, 100)
    
    def store_transition(self, *args):
        pass
    
    def train(self):
        pass

class MockEnv:
    def reset(self):
        return np.random.randn(36)
    
    def step(self, action):
        obs = np.random.randn(36)
        reward = np.random.randn()
        done = np.random.random() < 0.1
        info = {'safety_violation': False, 'real_execution': False}
        return obs, reward, done, info

# Import the domain adaptation components
from domain_adaptation import (
    DomainDiscriminator, FeatureExtractor, UncertaintyEstimator,
    AdversarialDomainAdapter, DomainGapMetrics, AdaptationConfig
)

def test_domain_adaptation():
    """Test domain adaptation functionality"""
    print("=== Testing Domain Adaptation Framework ===\n")
    
    # Configuration
    config = AdaptationConfig(
        adaptation_method="adversarial",
        adaptation_learning_rate=1e-4,
        fine_tuning_steps=100
    )
    
    # Mock environments and agent
    sim_env = MockEnv()
    real_env = MockEnv()
    base_agent = MockAgent()
    
    print("1. Testing Domain Gap Computation...")
    
    # Create adapter
    adapter = AdversarialDomainAdapter(base_agent, config)
    
    # Generate test data
    sim_states = np.random.randn(500, 36) * 0.5 + 0.3  # Simulation distribution
    real_states = np.random.randn(300, 36) * 0.3 + 0.7  # Real distribution (shifted)
    
    # Compute domain gap
    gap_metrics = adapter.compute_domain_gap(sim_states, real_states)
    
    print(f"   Domain Gap Metrics:")
    print(f"   - Wasserstein Distance: {gap_metrics.state_distribution_distance:.4f}")
    print(f"   - Performance Degradation: {gap_metrics.performance_degradation:.4f}")
    print(f"   - Uncertainty Level: {gap_metrics.uncertainty_level:.4f}")
    print(f"   - Adaptation Confidence: {gap_metrics.adaptation_confidence:.4f}")
    
    print("\n2. Testing Neural Network Components...")
    
    # Test feature extractor
    test_state = torch.randn(10, 36)
    features = adapter.feature_extractor(test_state)
    print(f"   Feature Extractor Output Shape: {features.shape}")
    
    # Test domain discriminator
    domain_pred = adapter.domain_discriminator(features)
    print(f"   Domain Discriminator Output Shape: {domain_pred.shape}")
    print(f"   Domain Predictions: {domain_pred.flatten()[:5].detach().numpy()}")
    
    # Test uncertainty estimator
    uncertainty_mu, uncertainty_var = adapter.uncertainty_estimator(test_state)
    print(f"   Uncertainty Estimator - Mean: {uncertainty_mu.mean():.4f}, Var: {uncertainty_var.mean():.4f}")
    
    print("\n3. Testing Adaptation Process (Mini Version)...")
    
    # Run mini adaptation
    adaptation_results = adapter.adapt_agent(sim_env, real_env, num_steps=50)
    
    print(f"   Adaptation Results:")
    print(f"   - Gap Improvement: {adaptation_results['improvement']:.4f}")
    print(f"   - Best Validation Performance: {adaptation_results['best_validation_performance']:.4f}")
    print(f"   - Total Steps: {adaptation_results['total_steps']}")
    print(f"   - Final Domain Loss: {adaptation_results['training_history']['domain_loss'][-1]:.4f}")
    
    print("\n[SUCCESS] Domain Adaptation Test Completed Successfully!")

if __name__ == '__main__':
    test_domain_adaptation()