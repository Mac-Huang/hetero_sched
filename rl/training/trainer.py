#!/usr/bin/env python3
"""
Main Training Manager for HeteroSched RL Agents

Comprehensive training pipeline with experiment management, logging,
evaluation, and hyperparameter optimization capabilities.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from rl.environments.hetero_env import make_hetero_env
from rl.agents.dqn_agent import DQNAgent
from rl.agents.ppo_agent import PPOAgent
from rl.agents.base_agent import AgentConfig

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Comprehensive experiment configuration"""
    
    # Experiment metadata
    experiment_name: str = "hetero_sched_rl"
    description: str = "Deep RL for heterogeneous task scheduling"
    tags: List[str] = field(default_factory=lambda: ["deep_rl", "scheduling", "multi_objective"])
    
    # Environment configuration
    env_config: str = "default"  # Environment configuration name
    max_episode_steps: int = 1000
    reward_strategy: str = "adaptive"
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'latency': 0.3, 'energy': 0.2, 'throughput': 0.25,
        'fairness': 0.15, 'stability': 0.1
    })
    
    # Agent configuration  
    agent_type: str = "DQN"  # "DQN" or "PPO"
    learning_rate: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.001
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training configuration
    total_episodes: int = 2000
    max_steps_per_episode: int = 1000
    update_frequency: int = 4
    target_update_frequency: int = 1000
    eval_frequency: int = 100
    save_frequency: int = 500
    log_frequency: int = 50
    
    # Performance thresholds
    early_stopping_patience: int = 200
    min_improvement_threshold: float = 0.01
    target_reward_threshold: float = 50.0
    
    # System configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 1
    seed: int = 42
    
    # Logging and output
    log_dir: str = "logs"
    model_dir: str = "models" 
    results_dir: str = "results"
    tensorboard_log: bool = True
    verbose: int = 1  # 0: quiet, 1: info, 2: debug

class TrainingManager:
    """Main training manager for RL experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.env = None
        self.agent = None
        self.writer = None
        
        # Training state
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.training_start_time = None
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.loss_history = []
        
        logger.info(f"Training Manager initialized: {config.experiment_name}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _setup_directories(self):
        """Create necessary directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            "experiments", f"{self.config.experiment_name}_{timestamp}"
        )
        
        # Create subdirectories
        self.log_dir = os.path.join(self.experiment_dir, self.config.log_dir)
        self.model_dir = os.path.join(self.experiment_dir, self.config.model_dir)
        self.results_dir = os.path.join(self.experiment_dir, self.config.results_dir)
        
        for directory in [self.log_dir, self.model_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Save experiment configuration
        config_path = os.path.join(self.experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.config.verbose >= 2 else logging.INFO
        if self.config.verbose == 0:
            log_level = logging.WARNING
        
        # File handler
        log_file = os.path.join(self.log_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # TensorBoard writer
        if self.config.tensorboard_log:
            self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def _create_environment(self):
        """Create training environment"""
        env_config = {
            'max_episode_steps': self.config.max_episode_steps,
            'reward_strategy': self.config.reward_strategy,
            'reward_weights': self.config.reward_weights
        }
        
        self.env = make_hetero_env(self.config.env_config)
        self.env.config.update(env_config)
        
        # Update reward function
        self.env.reward_function = self.env.reward_function.__class__(
            self.env.reward_function.config
        )
        
        logger.info(f"Environment created: {self.config.env_config}")
        logger.info(f"State dimension: {self.env.observation_space.shape[0]}")
        logger.info(f"Action space: {self.env.action_space}")
    
    def _create_agent(self):
        """Create RL agent"""
        agent_config = AgentConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            tau=self.config.tau,
            buffer_size=self.config.buffer_size,
            min_buffer_size=self.config.min_buffer_size,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay=self.config.epsilon_decay,
            device=self.config.device,
            update_frequency=self.config.update_frequency,
            target_update_frequency=self.config.target_update_frequency
        )
        
        state_dim = self.env.observation_space.shape[0]
        action_dims = [self.env.action_space.nvec[i] for i in range(len(self.env.action_space.nvec))]
        
        if self.config.agent_type == "DQN":
            self.agent = DQNAgent(state_dim, action_dims, agent_config)
        elif self.config.agent_type == "PPO":
            self.agent = PPOAgent(state_dim, action_dims, agent_config)
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")
        
        logger.info(f"Agent created: {self.config.agent_type}")
        logger.info(f"Agent info: {self.agent.get_network_info()}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting training...")
        self.training_start_time = time.time()
        
        # Initialize environment and agent
        self._create_environment()
        self._create_agent()
        
        try:
            for episode in range(self.config.total_episodes):
                self.current_episode = episode
                
                # Run episode
                episode_metrics = self._run_episode()
                
                # Log progress
                if episode % self.config.log_frequency == 0:
                    self._log_progress(episode_metrics)
                
                # Evaluate agent
                if episode % self.config.eval_frequency == 0:
                    eval_metrics = self._evaluate_agent()
                    self._log_evaluation(eval_metrics)
                    
                    # Check for improvement
                    if eval_metrics['mean_reward'] > self.best_reward + self.config.min_improvement_threshold:
                        self.best_reward = eval_metrics['mean_reward']
                        self.episodes_without_improvement = 0
                        
                        # Save best model
                        best_model_path = os.path.join(self.model_dir, "best_model.pth")
                        self.agent.save_model(best_model_path)
                    else:
                        self.episodes_without_improvement += self.config.eval_frequency
                
                # Save checkpoint
                if episode % self.config.save_frequency == 0:
                    checkpoint_path = os.path.join(self.model_dir, f"checkpoint_ep_{episode}.pth")
                    self.agent.save_model(checkpoint_path)
                
                # Early stopping check
                if (self.episodes_without_improvement >= self.config.early_stopping_patience or
                    self.best_reward >= self.config.target_reward_threshold):
                    logger.info(f"Training stopped early at episode {episode}")
                    break
            
            # Final evaluation and cleanup
            final_metrics = self._finalize_training()
            return final_metrics
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return self._finalize_training()
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    def _run_episode(self) -> Dict[str, float]:
        """Run a single training episode"""
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        losses = []
        
        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(obs, training=True)
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience
            if self.config.agent_type == "DQN":
                self.agent.store_experience(obs, action, reward, next_obs, done)
            elif self.config.agent_type == "PPO":
                self.agent.store_transition_reward(reward, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Update agent
            if self.agent.can_update() and step % self.config.update_frequency == 0:
                loss_info = self.agent.update()
                if loss_info and 'total_loss' in loss_info:
                    losses.append(loss_info['total_loss'])
                elif loss_info and 'policy_loss' in loss_info:
                    losses.append(loss_info['policy_loss'])
            
            obs = next_obs
            
            if done:
                break
        
        # Store episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        if losses:
            self.loss_history.extend(losses)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'epsilon': getattr(self.agent, 'epsilon', 0.0)
        }
    
    def _evaluate_agent(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent performance"""
        eval_rewards = []
        eval_lengths = []
        
        # Set agent to evaluation mode
        self.agent.set_training_mode(False)
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for _ in range(self.config.max_steps_per_episode):
                action = self.agent.select_action(obs, training=False)
                obs, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        # Set agent back to training mode
        self.agent.set_training_mode(True)
        
        metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths)
        }
        
        self.eval_rewards.append(metrics['mean_reward'])
        return metrics
    
    def _log_progress(self, episode_metrics: Dict[str, float]):
        """Log training progress"""
        episode = self.current_episode
        
        # Recent performance
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        recent_avg = np.mean(recent_rewards)
        
        # Log to console
        if self.config.verbose >= 1:
            logger.info(
                f"Episode {episode:4d} | "
                f"Reward: {episode_metrics['episode_reward']:7.2f} | "
                f"Avg10: {recent_avg:7.2f} | "
                f"Length: {episode_metrics['episode_length']:3d} | "
                f"Loss: {episode_metrics['avg_loss']:6.4f} | "
                f"Eps: {episode_metrics['epsilon']:.3f}"
            )
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Train/EpisodeReward', episode_metrics['episode_reward'], episode)
            self.writer.add_scalar('Train/EpisodeLength', episode_metrics['episode_length'], episode)
            self.writer.add_scalar('Train/RecentAvgReward', recent_avg, episode)
            self.writer.add_scalar('Train/Loss', episode_metrics['avg_loss'], episode)
            self.writer.add_scalar('Train/Epsilon', episode_metrics['epsilon'], episode)
    
    def _log_evaluation(self, eval_metrics: Dict[str, float]):
        """Log evaluation results"""
        episode = self.current_episode
        
        if self.config.verbose >= 1:
            logger.info(
                f"Eval {episode:4d}  | "
                f"Mean: {eval_metrics['mean_reward']:7.2f} | "
                f"Std: {eval_metrics['std_reward']:6.2f} | "
                f"Best: {self.best_reward:7.2f}"
            )
        
        if self.writer:
            self.writer.add_scalar('Eval/MeanReward', eval_metrics['mean_reward'], episode)
            self.writer.add_scalar('Eval/StdReward', eval_metrics['std_reward'], episode)
            self.writer.add_scalar('Eval/MinReward', eval_metrics['min_reward'], episode)
            self.writer.add_scalar('Eval/MaxReward', eval_metrics['max_reward'], episode)
            self.writer.add_scalar('Eval/MeanLength', eval_metrics['mean_length'], episode)
            self.writer.add_scalar('Eval/BestReward', self.best_reward, episode)
    
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and return summary"""
        training_time = time.time() - self.training_start_time
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, "final_model.pth")
        self.agent.save_model(final_model_path)
        
        # Generate training summary
        summary = {
            'experiment_name': self.config.experiment_name,
            'agent_type': self.config.agent_type,
            'total_episodes': self.current_episode + 1,
            'training_time_hours': training_time / 3600,
            'best_eval_reward': self.best_reward,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'total_steps': sum(self.episode_lengths),
            'convergence_episode': len(self.episode_rewards) - self.episodes_without_improvement if self.episodes_without_improvement < self.config.early_stopping_patience else None
        }
        
        # Save summary
        summary_path = os.path.join(self.results_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Close writers and cleanup
        if self.writer:
            self.writer.close()
        
        if self.env:
            self.env.close()
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {self.experiment_dir}")
        
        return summary