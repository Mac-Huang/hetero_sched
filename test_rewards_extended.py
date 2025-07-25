#!/usr/bin/env python3
"""Extended test of the multi-objective reward system"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from rl.environments.hetero_env import make_hetero_env

def test_extended_episode():
    """Test longer episode to see reward strategies in action"""
    
    print("=== Extended Episode Test ===")
    
    config = {
        'max_episode_steps': 100,
        'reward_strategy': 'adaptive',
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
    
    obs = env.reset()
    total_reward = 0
    episode_rewards = []
    
    print("Running 50 steps with adaptive rewards...")
    
    for step in range(50):
        # Use more intelligent action selection
        if step < 10:
            action = [0, 1, 1]  # CPU, low boost, small batch
        elif step < 30:
            action = [1, 2, 3]  # GPU, medium boost, medium batch
        else:
            action = [0, 0, 2]  # CPU, no boost, small batch (energy efficient)
            
        obs, reward, done, info = env.step(action)
        total_reward += reward
        episode_rewards.append(reward)
        
        # Print detailed info every 10 steps
        if step % 10 == 0 and hasattr(env, 'last_reward_info'):
            reward_info = env.last_reward_info
            print(f"\nStep {step}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Objectives: {reward_info['objective_scores']}")
            if 'adaptive_weights' in reward_info['info']:
                print(f"  Adaptive weights: {reward_info['info']['adaptive_weights']}")
        
        if done:
            print(f"Episode terminated early at step {step}")
            break
    
    print(f"\nFinal Results:")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Average reward: {total_reward/len(episode_rewards):.3f}")
    print(f"  Tasks completed: {env.tasks_completed}")
    print(f"  Final metrics: {env.metrics}")
    
    env.close()

def test_all_strategies_comparison():
    """Compare all reward strategies on same episode"""
    
    print("\n=== Strategy Comparison ===")
    
    strategies = ['weighted_sum', 'pareto_optimal', 'adaptive', 'hierarchical']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy}...")
        
        config = {
            'max_episode_steps': 30,
            'reward_strategy': strategy,
            'reward_weights': {
                'latency': 0.3,
                'energy': 0.2,
                'throughput': 0.25,
                'fairness': 0.15,
                'stability': 0.1
            }
        }
        
        env = make_hetero_env('default')
        env.config.update(config)
        env.reward_function = env.reward_function.__class__(env.reward_function.config)
        
        obs = env.reset()
        total_reward = 0
        
        # Use same action sequence for fair comparison
        actions = [
            [0, 0, 1], [1, 1, 2], [0, 2, 1], [1, 0, 3],
            [0, 1, 1], [1, 2, 2], [0, 0, 2], [1, 1, 1],
            [0, 3, 1], [1, 0, 2]
        ]
        
        for step in range(min(10, len(actions))):
            action = actions[step]
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        results[strategy] = {
            'total_reward': total_reward,
            'tasks_completed': env.tasks_completed,
            'violations': env.metrics['thermal_violations'] + env.metrics['slo_violations']
        }
        
        env.close()
    
    print("\nStrategy Comparison Results:")
    for strategy, result in results.items():
        print(f"  {strategy}: reward={result['total_reward']:.3f}, "
              f"tasks={result['tasks_completed']}, violations={result['violations']}")

if __name__ == '__main__':
    test_extended_episode()
    test_all_strategies_comparison()