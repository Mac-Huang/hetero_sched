#!/usr/bin/env python3
"""
Test script for HeteroSched RL environment

Validates the environment implementation and demonstrates basic usage.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rl.environments.hetero_env import make_hetero_env
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment_basic():
    """Test basic environment functionality"""
    logger.info("Testing basic environment functionality...")
    
    env = make_hetero_env('default')
    
    # Test reset
    obs = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # Test random actions
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        logger.info(f"Step {step}: action={action}, reward={reward:.3f}, done={done}")
        
        if done:
            logger.info(f"Episode finished early at step {step}")
            break
    
    logger.info(f"Total reward over {step+1} steps: {total_reward:.3f}")
    env.close()

def test_environment_deterministic():
    """Test environment with deterministic actions"""
    logger.info("Testing environment with deterministic actions...")
    
    env = make_hetero_env('short')
    obs = env.reset()
    
    # Test CPU-only policy
    cpu_rewards = []
    for _ in range(5):
        action = [0, 0, 0]  # CPU, no priority boost, batch size 1
        obs, reward, done, info = env.step(action)
        cpu_rewards.append(reward)
        if done:
            break
    
    env.reset()
    
    # Test GPU-only policy  
    gpu_rewards = []
    for _ in range(5):
        action = [1, 0, 0]  # GPU, no priority boost, batch size 1
        obs, reward, done, info = env.step(action)
        gpu_rewards.append(reward)
        if done:
            break
    
    logger.info(f"Average CPU-only reward: {np.mean(cpu_rewards):.3f}")
    logger.info(f"Average GPU-only reward: {np.mean(gpu_rewards):.3f}")
    
    env.close()

def test_state_components():
    """Test individual state components"""
    logger.info("Testing state components...")
    
    env = make_hetero_env('default')
    obs = env.reset()
    
    # Check observation dimensions
    expected_dims = env.task_features_dim + env.system_state_dim + env.queue_state_dim + env.performance_history_dim
    assert obs.shape[0] == expected_dims, f"Expected {expected_dims} features, got {obs.shape[0]}"
    
    # Check observation bounds
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Observations should be normalized to [0,1]"
    
    logger.info(f"State validation passed: {obs.shape[0]} features, all in [0,1] range")
    
    # Test state evolution
    initial_obs = obs.copy()
    action = [1, 2, 3]  # GPU, priority boost 2, batch size 4
    obs, reward, done, info = env.step(action)
    
    # State should change after step
    assert not np.array_equal(initial_obs, obs), "State should change after action"
    
    logger.info("State evolution test passed")
    env.close()

def test_action_effects():
    """Test that different actions produce different outcomes"""
    logger.info("Testing action effects...")
    
    env = make_hetero_env('default')
    
    # Test multiple identical starting conditions
    results = []
    for device in [0, 1]:  # CPU, GPU
        for priority in [0, 2, 4]:  # Low, medium, high priority boost
            for batch in [0, 4, 9]:  # Small, medium, large batch
                env.reset()
                action = [device, priority, batch]
                obs, reward, done, info = env.step(action)
                
                results.append({
                    'action': action,
                    'reward': reward,
                    'latency': info['task_result']['total_latency'],
                    'energy': info['task_result']['energy'],
                    'device': info['task_result']['device'].name
                })
    
    # Verify action diversity produces different outcomes
    rewards = [r['reward'] for r in results]
    latencies = [r['latency'] for r in results]
    
    logger.info(f"Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    logger.info(f"Latency range: [{min(latencies):.6f}, {max(latencies):.6f}]")
    
    # Should have some variation in outcomes
    assert len(set(np.round(rewards, 3))) > 1, "Different actions should produce different rewards"
    
    env.close()

def test_episode_completion():
    """Test full episode completion"""
    logger.info("Testing full episode completion...")
    
    env = make_hetero_env('short')  # Short episodes for faster testing
    obs = env.reset()
    
    episode_rewards = []
    step_count = 0
    
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        step_count += 1
        
        if done:
            logger.info(f"Episode completed after {step_count} steps")
            logger.info(f"Final metrics: {info['metrics']}")
            break
        
        if step_count > 1000:  # Safety limit
            logger.warning("Episode exceeded 1000 steps, breaking")
            break
    
    logger.info(f"Total episode reward: {sum(episode_rewards):.3f}")
    logger.info(f"Average step reward: {np.mean(episode_rewards):.3f}")
    
    env.close()

def benchmark_environment():
    """Benchmark environment performance"""
    logger.info("Benchmarking environment performance...")
    
    env = make_hetero_env('default')
    
    import time
    start_time = time.time()
    
    n_steps = 1000
    env.reset()
    
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if done:
            env.reset()
    
    end_time = time.time()
    total_time = end_time - start_time
    steps_per_second = n_steps / total_time
    
    logger.info(f"Benchmark: {steps_per_second:.1f} steps/second ({total_time:.3f}s for {n_steps} steps)")
    
    env.close()

def main():
    """Run all tests"""
    logger.info("Starting HeteroSched RL Environment Tests...")
    
    try:
        test_environment_basic()
        test_environment_deterministic()
        test_state_components()
        test_action_effects()
        test_episode_completion()
        benchmark_environment()
        
        logger.info("✅ All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())