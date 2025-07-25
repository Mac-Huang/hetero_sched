#!/usr/bin/env python3
"""Test DQN and PPO agents with HeteroSched environment"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import torch
from rl.environments.hetero_env import make_hetero_env
from rl.agents.dqn_agent import DQNAgent
from rl.agents.ppo_agent import PPOAgent
from rl.agents.base_agent import AgentConfig

def test_dqn_agent():
    """Test DQN agent basic functionality"""
    print("=== Testing DQN Agent ===")
    
    # Create environment
    env = make_hetero_env('short')  # Shorter episodes for testing
    
    # Create DQN agent
    config = AgentConfig(
        learning_rate=1e-4,
        batch_size=32,
        buffer_size=10000,
        min_buffer_size=100,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99
    )
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dims=[2, 5, 10],  # [device, priority_boost, batch_size]
        config=config
    )
    
    print(f"Agent network info: {agent.get_network_info()}")
    
    # Test episode
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    print("Running test episode...")
    for step in range(50):
        # Select action
        action = agent.select_action(obs, training=True)
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        
        # Store experience
        agent.store_experience(obs, action, reward, next_obs, done)
        
        total_reward += reward
        steps += 1
        
        # Update agent if enough experience
        if agent.can_update() and step % 4 == 0:
            loss_info = agent.update()
            if step % 20 == 0:
                print(f"  Step {step}: action={action}, reward={reward:.3f}, "
                      f"loss={loss_info.get('total_loss', 0):.4f}, epsilon={agent.epsilon:.3f}")
        
        obs = next_obs
        
        if done:
            break
    
    print(f"Episode completed: {steps} steps, total_reward={total_reward:.3f}")
    print(f"Buffer size: {len(agent.memory)}")
    
    # Test Q-value analysis
    if len(agent.memory) > 0:
        sample_states = [agent.memory[i][0] for i in range(0, min(5, len(agent.memory)), 1)]
        q_analysis = agent.get_action_values_summary(sample_states)
        print(f"Q-value analysis: {q_analysis}")
    
    env.close()
    print("DQN test completed successfully!\n")

def test_ppo_agent():
    """Test PPO agent basic functionality"""
    print("=== Testing PPO Agent ===")
    
    # Create environment
    env = make_hetero_env('short')
    
    # Create PPO agent
    config = AgentConfig(
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99
    )
    
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dims=[2, 5, 10],
        config=config
    )
    
    print(f"Agent network info: {agent.get_network_info()}")
    
    # Test episode
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    print("Running test episode...")
    for step in range(100):  # PPO needs more steps for rollout
        # Select action
        action = agent.select_action(obs, training=True)
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        
        # Store reward in PPO rollout buffer
        agent.store_transition_reward(reward, done)
        
        total_reward += reward
        steps += 1
        
        if step % 20 == 0:
            print(f"  Step {step}: action={action}, reward={reward:.3f}")
        
        obs = next_obs
        
        if done:
            obs = env.reset()  # Continue collecting experience
    
    # Test policy update (if enough rollout data)
    if len(agent.rollout_buffer) >= agent.mini_batch_size:
        print("Testing policy update...")
        loss_info = agent.update()
        print(f"PPO update completed: {loss_info}")
    
    # Test action probability analysis
    sample_obs = env.reset()
    action_probs = agent.get_action_probabilities(sample_obs)
    print(f"Action probabilities: {action_probs}")
    
    print(f"Episode statistics: {steps} steps, total_reward={total_reward:.3f}")
    print(f"Rollout buffer size: {len(agent.rollout_buffer)}")
    
    env.close()
    print("PPO test completed successfully!\n")

def test_agent_comparison():
    """Compare DQN and PPO on same environment"""
    print("=== Agent Comparison Test ===")
    
    env = make_hetero_env('short')
    
    # Test both agents on same initial state
    initial_obs = env.reset()
    
    # DQN agent
    dqn_config = AgentConfig(learning_rate=1e-4, epsilon_start=0.1)  # Lower epsilon for comparison
    dqn_agent = DQNAgent(env.observation_space.shape[0], [2, 5, 10], dqn_config)
    
    # PPO agent  
    ppo_config = AgentConfig(learning_rate=3e-4)
    ppo_agent = PPOAgent(env.observation_space.shape[0], [2, 5, 10], ppo_config)
    
    print("Comparing action selection on same state:")
    for i in range(5):
        obs = env.reset()
        
        dqn_action = dqn_agent.select_action(obs, training=False)  # Deterministic
        ppo_action = ppo_agent.select_action(obs, training=False)  # Deterministic
        
        print(f"  State {i}: DQN={dqn_action}, PPO={ppo_action}")
    
    env.close()
    print("Agent comparison completed!\n")

def test_model_saving_loading():
    """Test model save/load functionality"""
    print("=== Testing Model Save/Load ===")
    
    env = make_hetero_env('short')
    
    # Create and train DQN agent briefly
    agent = DQNAgent(env.observation_space.shape[0], [2, 5, 10])
    
    # Quick training to change model weights
    obs = env.reset()
    for _ in range(20):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store_experience(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs = env.reset()
    
    # Get initial action for comparison
    test_obs = env.reset()
    initial_action = agent.select_action(test_obs, training=False)
    
    # Save model
    save_path = "test_model.pth"
    agent.save_model(save_path)
    
    # Create new agent and load model
    new_agent = DQNAgent(env.observation_space.shape[0], [2, 5, 10])
    new_agent.load_model(save_path)
    
    # Test that loaded model gives same action
    loaded_action = new_agent.select_action(test_obs, training=False)
    
    print(f"Original action: {initial_action}")
    print(f"Loaded action: {loaded_action}")
    print(f"Actions match: {np.array_equal(initial_action, loaded_action)}")
    
    # Clean up
    os.remove(save_path)
    env.close()
    
    print("Model save/load test completed!\n")

if __name__ == '__main__':
    print("Starting RL Agent Tests...\n")
    
    # Check if CUDA is available
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    # Run tests
    test_dqn_agent()
    test_ppo_agent()
    test_agent_comparison()
    test_model_saving_loading()
    
    print("All agent tests completed successfully! 🎉")