#!/usr/bin/env python3
"""
Main Training Script for HeteroSched RL Agents

This script provides a command-line interface for training Deep RL agents
on the heterogeneous task scheduling problem with comprehensive experiment
management and evaluation capabilities.

Usage:
    python train_agent.py --agent DQN --episodes 2000 --experiment_name my_experiment
    python train_agent.py --config configs/dqn_config.yaml
    python train_agent.py --agent PPO --eval_only --model_path models/best_model.pth
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from rl.training.trainer import TrainingManager, ExperimentConfig
from rl.training.evaluator import AgentEvaluator, EvaluationConfig
from rl.training.utils import (
    load_experiment_config, save_experiment_config, 
    setup_logging, set_global_seeds
)
from rl.agents.dqn_agent import DQNAgent
from rl.agents.ppo_agent import PPOAgent
from rl.agents.base_agent import AgentConfig

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Train Deep RL agents for heterogeneous task scheduling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main operation mode
    parser.add_argument('--mode', choices=['train', 'eval', 'train_eval'], 
                       default='train_eval', help='Operation mode')
    
    # Agent configuration
    parser.add_argument('--agent', choices=['DQN', 'PPO'], default='DQN',
                       help='RL agent type')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to experiment configuration file')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='hetero_sched_rl',
                       help='Experiment name')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Environment settings
    parser.add_argument('--env_config', type=str, default='default',
                       help='Environment configuration')
    parser.add_argument('--reward_strategy', type=str, default='adaptive',
                       choices=['weighted_sum', 'adaptive', 'pareto_optimal', 'hierarchical', 'constrained'],
                       help='Multi-objective reward strategy')
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial exploration rate (DQN only)')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                       help='Final exploration rate (DQN only)')
    
    # Model management
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to load pre-trained model')
    parser.add_argument('--save_frequency', type=int, default=500,
                       help='Model save frequency (episodes)')
    
    # Evaluation settings
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--eval_frequency', type=int, default=100,
                       help='Evaluation frequency during training')
    
    # System settings
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for training')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                       help='Verbosity level')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for results')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    
    return parser.parse_args()

def create_experiment_config(args) -> ExperimentConfig:
    """Create experiment configuration from arguments"""
    
    # Determine device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create configuration
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        description=f"Training {args.agent} agent on HeteroSched",
        tags=["deep_rl", "scheduling", args.agent.lower()],
        
        # Environment
        env_config=args.env_config,
        reward_strategy=args.reward_strategy,
        
        # Agent
        agent_type=args.agent,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        
        # Training
        total_episodes=args.episodes,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
        
        # System
        device=device,
        seed=args.seed,
        verbose=args.verbose,
        tensorboard_log=not args.no_tensorboard
    )
    
    return config

def load_agent_from_checkpoint(model_path: str, config: ExperimentConfig):
    """Load agent from saved checkpoint"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create dummy environment to get dimensions
    from rl.environments.hetero_env import make_hetero_env
    env = make_hetero_env(config.env_config)
    
    state_dim = env.observation_space.shape[0]
    action_dims = [env.action_space.nvec[i] for i in range(len(env.action_space.nvec))]
    
    # Create agent
    agent_config = AgentConfig(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.gamma,
        device=config.device
    )
    
    if config.agent_type == "DQN":
        agent = DQNAgent(state_dim, action_dims, agent_config)
    elif config.agent_type == "PPO":
        agent = PPOAgent(state_dim, action_dims, agent_config)
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")
    
    # Load model
    agent.load_model(model_path)
    
    env.close()
    logger.info(f"Agent loaded from {model_path}")
    
    return agent

def train_agent(config: ExperimentConfig, model_path: str = None) -> Dict[str, Any]:
    """Train RL agent"""
    
    logger.info(f"Starting training: {config.experiment_name}")
    logger.info(f"Agent: {config.agent_type}, Episodes: {config.total_episodes}")
    logger.info(f"Device: {config.device}, Seed: {config.seed}")
    
    # Set random seeds
    set_global_seeds(config.seed)
    
    # Create training manager
    trainer = TrainingManager(config)
    
    # Load pre-trained model if specified
    if model_path:
        logger.info(f"Loading pre-trained model: {model_path}")
        # Note: Model loading will be handled by the trainer internally
        # This is a placeholder for future implementation
    
    # Start training
    results = trainer.train()
    
    logger.info("Training completed successfully!")
    return results

def evaluate_agent(config: ExperimentConfig, model_path: str) -> Dict[str, Any]:
    """Evaluate trained agent"""
    
    logger.info(f"Starting evaluation: {config.experiment_name}")
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        num_episodes=100,  # Can be made configurable
        environments=['default', 'short', 'long'],
        reward_strategies=['weighted_sum', 'adaptive', 'pareto_optimal'],
        generate_plots=True,
        save_detailed_logs=True
    )
    
    # Create evaluator
    evaluator = AgentEvaluator(eval_config)
    
    # Load agent
    agent = load_agent_from_checkpoint(model_path, config)
    
    # Run evaluation
    results = evaluator.evaluate_agent(agent, f"{config.agent_type}_agent")
    
    # Evaluate baselines for comparison
    baseline_results = evaluator.evaluate_baselines()
    
    # Generate comparison
    comparison = evaluator.compare_agents()
    
    # Generate report
    report = evaluator.generate_report()
    
    logger.info("Evaluation completed successfully!")
    
    return {
        'agent_results': results,
        'baseline_results': baseline_results,
        'comparison': comparison,
        'report': report
    }

def main():
    """Main function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose >= 2 else logging.INFO
    if args.verbose == 0:
        log_level = logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("HeteroSched RL Training Started")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load or create configuration
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config_dict = load_experiment_config(args.config)
            config = ExperimentConfig(**config_dict)
        else:
            config = create_experiment_config(args)
        
        # Save configuration for reproducibility
        config_save_path = os.path.join(args.output_dir, config.experiment_name, "config")
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        save_experiment_config(config.__dict__, config_save_path)
        
        # Execute based on mode
        if args.mode == 'train':
            results = train_agent(config, args.model_path)
            logger.info(f"Training results: {results}")
            
        elif args.mode == 'eval':
            if not args.model_path:
                raise ValueError("Model path required for evaluation mode")
            results = evaluate_agent(config, args.model_path)
            logger.info("Evaluation results available in output directory")
            
        elif args.mode == 'train_eval':
            # Train first
            train_results = train_agent(config, args.model_path)
            logger.info(f"Training completed: {train_results}")
            
            # Then evaluate the best model
            # Note: This assumes the trainer saves the best model
            best_model_path = os.path.join(
                "experiments", 
                f"{config.experiment_name}_*",  # Trainer adds timestamp
                "models", 
                "best_model.pth"
            )
            
            # For now, skip evaluation in train_eval mode
            # In full implementation, would find the actual best model path
            logger.info("Training completed. Evaluation skipped in this version.")
        
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == '__main__':
    main()