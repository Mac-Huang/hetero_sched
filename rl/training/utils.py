#!/usr/bin/env python3
"""
Training Utilities for HeteroSched RL

Helper functions for experiment management, logging, visualization,
and hyperparameter optimization.
"""

import os
import json
import yaml
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def setup_tensorboard(log_dir: str, experiment_name: str) -> SummaryWriter:
    """Setup TensorBoard logging"""
    
    tensorboard_dir = os.path.join(log_dir, "tensorboard", experiment_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logging setup: {tensorboard_dir}")
    
    return writer

def save_experiment_config(config: Dict[str, Any], save_path: str):
    """Save experiment configuration to file"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as JSON
    json_path = save_path.replace('.yaml', '.json') if save_path.endswith('.yaml') else f"{save_path}.json"
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save as YAML for readability
    yaml_path = save_path.replace('.json', '.yaml') if save_path.endswith('.json') else f"{save_path}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Experiment config saved to {json_path} and {yaml_path}")

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from file"""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")
    
    logger.info(f"Experiment config loaded from {config_path}")
    return config

def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """Create timestamped experiment directory"""
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    # Create subdirectories
    subdirs = ['logs', 'models', 'results', 'plots', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    logger.info(f"Experiment directory created: {experiment_dir}")
    return experiment_dir

def setup_logging(log_file: str, level: int = logging.INFO):
    """Setup comprehensive logging configuration"""
    
    # Create log directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging setup complete: {log_file}")

def set_global_seeds(seed: int):
    """Set global random seeds for reproducibility"""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Global seeds set to {seed}")

def compute_moving_average(values: List[float], window: int = 100) -> List[float]:
    """Compute moving average of values"""
    
    if len(values) < window:
        return values
    
    moving_avg = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        avg = np.mean(values[start_idx:i+1])
        moving_avg.append(avg)
    
    return moving_avg

def plot_training_curves(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot training curves"""
    
    num_metrics = len(metrics)
    if num_metrics == 0:
        logger.warning("No metrics to plot")
        return
    
    # Determine subplot layout
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        episodes = range(len(values))
        ax.plot(episodes, values, alpha=0.7, label='Raw')
        
        # Add moving average if enough data points
        if len(values) > 10:
            moving_avg = compute_moving_average(values, window=min(50, len(values)//5))
            ax.plot(episodes, moving_avg, linewidth=2, label='Moving Avg')
        
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_reward_distribution(rewards: List[float], agent_name: str = "Agent", save_path: str = None):
    """Plot reward distribution analysis"""
    
    if len(rewards) < 10:
        logger.warning("Insufficient data for reward distribution plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{agent_name} Reward Analysis', fontsize=16)
    
    # 1. Histogram
    axes[0, 0].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    axes[0, 0].axvline(np.median(rewards), color='green', linestyle='--', label=f'Median: {np.median(rewards):.2f}')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series
    episodes = range(len(rewards))
    axes[0, 1].plot(episodes, rewards, alpha=0.6)
    moving_avg = compute_moving_average(rewards, window=min(100, len(rewards)//5))
    axes[0, 1].plot(episodes, moving_avg, linewidth=2, color='red', label='Moving Average')
    axes[0, 1].set_title('Reward Over Time')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot
    axes[1, 0].boxplot(rewards)
    axes[1, 0].set_title('Reward Box Plot')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_rewards = np.sort(rewards)
    cumulative_prob = np.arange(1, len(rewards) + 1) / len(rewards)
    axes[1, 1].plot(sorted_rewards, cumulative_prob, linewidth=2)
    axes[1, 1].axvline(np.percentile(rewards, 25), color='orange', linestyle='--', alpha=0.7, label='25th percentile')
    axes[1, 1].axvline(np.percentile(rewards, 75), color='orange', linestyle='--', alpha=0.7, label='75th percentile')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reward distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_hyperparameter_grid(base_config: Dict[str, Any], 
                              param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Create hyperparameter grid for parameter sweep"""
    
    from itertools import product
    
    # Get parameter names and values
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Generate all combinations
    configs = []
    for combination in product(*param_values):
        config = base_config.copy()
        
        # Update with current parameter combination
        for param_name, param_value in zip(param_names, combination):
            # Support nested parameter names (e.g., "agent_config.learning_rate")
            if '.' in param_name:
                keys = param_name.split('.')
                current_dict = config
                for key in keys[:-1]:
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]
                current_dict[keys[-1]] = param_value
            else:
                config[param_name] = param_value
        
        configs.append(config)
    
    logger.info(f"Generated {len(configs)} hyperparameter configurations")
    return configs

def analyze_hyperparameter_results(results: List[Dict[str, Any]], 
                                 metric_name: str = 'best_reward') -> Dict[str, Any]:
    """Analyze hyperparameter search results"""
    
    if not results:
        logger.warning("No results to analyze")
        return {}
    
    # Extract performance metric and parameters
    performances = []
    parameter_values = {}
    
    for result in results:
        performance = result.get(metric_name, 0.0)
        performances.append(performance)
        
        # Extract parameter values
        config = result.get('config', {})
        for param_name, param_value in config.items():
            if param_name not in parameter_values:
                parameter_values[param_name] = []
            parameter_values[param_name].append(param_value)
    
    # Find best configuration
    best_idx = np.argmax(performances)
    best_performance = performances[best_idx]
    best_config = results[best_idx].get('config', {})
    
    # Compute parameter importance (correlation with performance)
    parameter_importance = {}
    for param_name, values in parameter_values.items():
        # Skip non-numeric parameters
        try:
            numeric_values = [float(v) for v in values]
            correlation = np.corrcoef(numeric_values, performances)[0, 1]
            parameter_importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
        except (ValueError, TypeError):
            continue
    
    analysis = {
        'best_performance': best_performance,
        'best_config': best_config,
        'performance_stats': {
            'mean': np.mean(performances),
            'std': np.std(performances),
            'min': np.min(performances),
            'max': np.max(performances)
        },
        'parameter_importance': parameter_importance,
        'num_configurations': len(results)
    }
    
    logger.info(f"Hyperparameter analysis completed: best {metric_name} = {best_performance:.4f}")
    return analysis

def save_checkpoint(agent, optimizer, episode: int, metrics: Dict[str, Any], 
                   checkpoint_path: str):
    """Save training checkpoint"""
    
    checkpoint = {
        'episode': episode,
        'agent_state_dict': agent.get_model_state(),
        'metrics': metrics,
        'timestamp': time.time()
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load training checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    return checkpoint

def compute_performance_metrics(episode_rewards: List[float], 
                              episode_lengths: List[int]) -> Dict[str, float]:
    """Compute comprehensive performance metrics"""
    
    if not episode_rewards:
        return {}
    
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths) if episode_lengths else None
    
    metrics = {
        # Reward statistics
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'median_reward': float(np.median(rewards)),
        'q25_reward': float(np.percentile(rewards, 25)),
        'q75_reward': float(np.percentile(rewards, 75)),
        
        # Performance consistency
        'reward_cv': float(np.std(rewards) / np.mean(rewards)) if np.mean(rewards) != 0 else 0.0,
        
        # Learning progress (trend in recent episodes)
        'recent_improvement': 0.0,
        'learning_rate': 0.0
    }
    
    # Compute learning metrics if enough episodes
    if len(rewards) >= 20:
        # Recent improvement (last 25% vs first 25%)
        split_point = len(rewards) // 4
        early_performance = np.mean(rewards[:split_point])
        recent_performance = np.mean(rewards[-split_point:])
        
        if early_performance != 0:
            metrics['recent_improvement'] = float((recent_performance - early_performance) / abs(early_performance))
        
        # Learning rate (linear regression slope)
        episodes = np.arange(len(rewards))
        slope, _ = np.polyfit(episodes, rewards, 1)
        metrics['learning_rate'] = float(slope)
    
    # Episode length statistics
    if lengths is not None and len(lengths) > 0:
        metrics.update({
            'mean_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths)),
            'min_length': float(np.min(lengths)),
            'max_length': float(np.max(lengths))
        })
    
    return metrics

# Utility functions for model management
def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb