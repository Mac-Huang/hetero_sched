#!/usr/bin/env python3
"""Test the RL training pipeline"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from rl.training.trainer import TrainingManager, ExperimentConfig

def test_training_pipeline():
    """Test the training pipeline with minimal configuration"""
    print("=== Testing RL Training Pipeline ===")
    
    # Create minimal training configuration
    config = ExperimentConfig(
        experiment_name="test_training",
        description="Test training pipeline",
        
        # Short training for testing
        total_episodes=50,
        max_episode_steps=100,
        
        # Small network/batch for speed
        batch_size=16,
        min_buffer_size=50,
        
        # Frequent logging for testing
        log_frequency=10,
        eval_frequency=20,
        save_frequency=25,
        
        # Quick evaluation
        early_stopping_patience=50,
        
        # Testing both agent types
        agent_type="DQN",  # Will test DQN first
        
        verbose=1
    )
    
    print(f"Testing {config.agent_type} agent...")
    print(f"Episodes: {config.total_episodes}")
    print(f"Device: {config.device}")
    
    # Create and run training manager
    trainer = TrainingManager(config)
    
    try:
        results = trainer.train()
        print("\nTraining Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        print(f"\n{config.agent_type} training test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n{config.agent_type} training test FAILED: {e}")
        return False

def test_both_agents():
    """Test both DQN and PPO agents"""
    print("=== Testing Both Agent Types ===\n")
    
    results = {}
    
    # Test DQN
    print("1. Testing DQN Agent:")
    dqn_config = ExperimentConfig(
        experiment_name="test_dqn",
        agent_type="DQN",
        total_episodes=30,
        max_episode_steps=50,
        batch_size=16,
        min_buffer_size=30,
        log_frequency=10,
        eval_frequency=15,
        verbose=1
    )
    
    dqn_trainer = TrainingManager(dqn_config)
    try:
        dqn_results = dqn_trainer.train()
        results['DQN'] = dqn_results
        print("   DQN test PASSED!\n")
    except Exception as e:
        print(f"   DQN test FAILED: {e}\n")
        results['DQN'] = None
    
    # Test PPO
    print("2. Testing PPO Agent:")
    ppo_config = ExperimentConfig(
        experiment_name="test_ppo", 
        agent_type="PPO",
        total_episodes=30,
        max_episode_steps=50,
        batch_size=16,
        log_frequency=10,
        eval_frequency=15,
        verbose=1
    )
    
    ppo_trainer = TrainingManager(ppo_config)
    try:
        ppo_results = ppo_trainer.train()
        results['PPO'] = ppo_results
        print("   PPO test PASSED!\n")
    except Exception as e:
        print(f"   PPO test FAILED: {e}\n")
        results['PPO'] = None
    
    # Summary
    print("=== Test Summary ===")
    for agent_type, result in results.items():
        status = "PASSED" if result is not None else "FAILED"
        print(f"{agent_type}: {status}")
        if result:
            print(f"  Best reward: {result.get('best_eval_reward', 'N/A')}")
            print(f"  Total episodes: {result.get('total_episodes', 'N/A')}")
    
    return results

if __name__ == '__main__':
    print("Starting RL Training Pipeline Tests...\n")
    
    # Test basic training pipeline
    basic_test_passed = test_training_pipeline()
    
    print("\n" + "="*50 + "\n")
    
    # Test both agent types
    agent_results = test_both_agents()
    
    print("\n" + "="*50 + "\n")
    
    # Final summary
    if basic_test_passed and all(r is not None for r in agent_results.values()):
        print("All training pipeline tests PASSED!")
    else:
        print("Some tests FAILED. Check the logs above.")
        
    print("\nTraining pipeline testing completed.")