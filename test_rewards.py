#!/usr/bin/env python3
"""Test the multi-objective reward system"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from rl.environments.hetero_env import make_hetero_env

def test_reward_strategies():
    """Test different reward strategies"""
    
    strategies = ['weighted_sum', 'adaptive', 'pareto_optimal']
    
    for strategy in strategies:
        print(f"\n=== Testing {strategy} strategy ===")
        
        config = {
            'max_episode_steps': 20,
            'reward_strategy': strategy,
            'reward_weights': {
                'latency': 0.4,
                'energy': 0.2,
                'throughput': 0.2,
                'fairness': 0.1,
                'stability': 0.1
            }
        }
        
        env = make_hetero_env('default')
        env.config.update(config)
        env.reward_function = env.reward_function.__class__(env.reward_function.config)
        
        # Test with different actions
        obs = env.reset()
        total_reward = 0
        
        for step in range(5):
            # Try different action types
            if step < 2:
                action = [0, 0, 0]  # CPU, no boost, small batch
            else:
                action = [1, 2, 3]  # GPU, medium boost, medium batch
                
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if hasattr(env, 'last_reward_info'):
                reward_info = env.last_reward_info
                print(f"  Step {step}: action={action}, reward={reward:.3f}")
                print(f"    Strategy: {reward_info['strategy']}")
                print(f"    Objectives: {reward_info['objective_scores']}")
                
            if done:
                break
        
        print(f"  Total reward: {total_reward:.3f}")
        env.close()

def test_adaptive_weights():
    """Test adaptive weight adjustment"""
    print("\n=== Testing Adaptive Weight Adjustment ===")
    
    config = {
        'max_episode_steps': 50,
        'reward_strategy': 'adaptive',
        'reward_weights': {
            'latency': 0.5,  # Start with high latency focus
            'energy': 0.2,
            'throughput': 0.2,
            'fairness': 0.05,
            'stability': 0.05
        }
    }
    
    env = make_hetero_env('default')
    env.config.update(config)
    
    obs = env.reset()
    
    # Track weight evolution
    initial_weights = None
    for step in range(25):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if hasattr(env, 'last_reward_info'):
            reward_info = env.last_reward_info
            if 'weights' in reward_info:
                weights = reward_info['weights']
                if initial_weights is None:
                    initial_weights = weights.copy()
                    print(f"Initial weights: {weights}")
                elif step == 24:  # Show final weights
                    print(f"Final weights:   {weights}")
                    print("Weight changes:")
                    for obj in weights:
                        change = weights[obj] - initial_weights[obj]
                        print(f"  {obj}: {change:+.3f}")
        
        if done:
            break
    
    env.close()

def test_constraint_violations():
    """Test constrained reward with violations"""
    print("\n=== Testing Constrained Rewards ===")
    
    config = {
        'max_episode_steps': 10,
        'reward_strategy': 'constrained',
        'constraints': {
            'stability': 0.5,  # Require stability > 0.5
            'fairness': 0.2    # Require fairness > 0.2
        }
    }
    
    env = make_hetero_env('default')
    env.config.update(config)
    
    obs = env.reset()
    
    for step in range(5):
        # Use actions that might violate constraints
        action = [1, 4, 9]  # GPU, high priority, large batch (might cause violations)
        
        obs, reward, done, info = env.step(action)
        
        if hasattr(env, 'last_reward_info'):
            reward_info = env.last_reward_info
            if 'info' in reward_info and 'violations' in reward_info['info']:
                violations = reward_info['info']['violations']
                penalty = reward_info['info']['total_penalty']
                print(f"Step {step}: reward={reward:.3f}, violations={violations}, penalty={penalty:.3f}")
        
        if done:
            break
    
    env.close()

if __name__ == '__main__':
    test_reward_strategies()
    test_adaptive_weights()
    test_constraint_violations()