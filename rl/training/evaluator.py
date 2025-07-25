#!/usr/bin/env python3
"""
Agent Evaluation and Benchmarking for HeteroSched

Comprehensive evaluation framework for comparing RL agents against baselines,
analyzing multi-objective performance, and generating detailed reports.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from rl.environments.hetero_env import make_hetero_env

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation"""
    num_episodes: int = 100
    max_episode_steps: int = 1000
    environments: List[str] = None
    reward_strategies: List[str] = None
    metrics_to_track: List[str] = None
    generate_plots: bool = True
    save_detailed_logs: bool = True
    
    def __post_init__(self):
        if self.environments is None:
            self.environments = ['default', 'short', 'long']
        if self.reward_strategies is None:
            self.reward_strategies = ['weighted_sum', 'adaptive', 'pareto_optimal']
        if self.metrics_to_track is None:
            self.metrics_to_track = [
                'episode_reward', 'episode_length', 'latency_violations',
                'thermal_violations', 'slo_violations', 'energy_consumption',
                'throughput', 'fairness_score'
            ]

class BaselinePolicy:
    """Simple baseline policies for comparison"""
    
    @staticmethod
    def random_policy(observation):
        """Random action selection"""
        return np.array([np.random.randint(2), np.random.randint(5), np.random.randint(10)])
    
    @staticmethod
    def greedy_cpu_policy(observation):
        """Always select CPU with minimal resource usage"""
        return np.array([0, 0, 0])  # CPU, no boost, minimal batch
    
    @staticmethod
    def greedy_gpu_policy(observation):
        """Always select GPU with high throughput settings"""
        return np.array([1, 2, 5])  # GPU, medium boost, medium batch
    
    @staticmethod
    def load_balanced_policy(observation):
        """Simple load balancing based on system state"""
        # Extract CPU and GPU load from observation (simplified)
        # In real implementation, would parse observation properly
        cpu_load = observation[9]  # Approximate CPU load position
        gpu_load = observation[16]  # Approximate GPU utilization position
        
        if cpu_load < gpu_load:
            return np.array([0, 1, 2])  # CPU with moderate settings
        else:
            return np.array([1, 1, 3])  # GPU with moderate settings

class AgentEvaluator:
    """Comprehensive agent evaluation framework"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.results = {}
        self.detailed_logs = []
        
        # Setup baseline policies
        self.baselines = {
            'random': BaselinePolicy.random_policy,
            'greedy_cpu': BaselinePolicy.greedy_cpu_policy,
            'greedy_gpu': BaselinePolicy.greedy_gpu_policy,
            'load_balanced': BaselinePolicy.load_balanced_policy
        }
        
        logger.info("Agent evaluator initialized")
    
    def evaluate_agent(self, agent, agent_name: str, 
                      save_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Evaluate a single agent across multiple configurations"""
        
        logger.info(f"Starting evaluation of {agent_name}")
        os.makedirs(save_dir, exist_ok=True)
        
        agent_results = {}
        
        # Evaluate on different environments
        for env_config in self.config.environments:
            logger.info(f"Evaluating on environment: {env_config}")
            
            # Evaluate on different reward strategies
            for reward_strategy in self.config.reward_strategies:
                logger.info(f"Using reward strategy: {reward_strategy}")
                
                # Create environment
                env = self._create_environment(env_config, reward_strategy)
                
                # Run evaluation
                metrics = self._run_evaluation(agent, env, agent_name)
                
                # Store results
                key = f"{env_config}_{reward_strategy}"
                agent_results[key] = metrics
                
                env.close()
        
        # Store agent results
        self.results[agent_name] = agent_results
        
        # Save results
        self._save_agent_results(agent_name, agent_results, save_dir)
        
        logger.info(f"Evaluation of {agent_name} completed")
        return agent_results
    
    def evaluate_baselines(self, save_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Evaluate baseline policies"""
        
        logger.info("Starting baseline evaluation")
        baseline_results = {}
        
        for baseline_name, baseline_policy in self.baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            
            baseline_agent_results = {}
            
            # Evaluate on different environments
            for env_config in self.config.environments:
                for reward_strategy in self.config.reward_strategies:
                    # Create environment
                    env = self._create_environment(env_config, reward_strategy)
                    
                    # Run evaluation with baseline policy
                    metrics = self._run_baseline_evaluation(baseline_policy, env, baseline_name)
                    
                    # Store results
                    key = f"{env_config}_{reward_strategy}"
                    baseline_agent_results[key] = metrics
                    
                    env.close()
            
            baseline_results[baseline_name] = baseline_agent_results
            self.results[baseline_name] = baseline_agent_results
        
        # Save baseline results
        self._save_baseline_results(baseline_results, save_dir)
        
        logger.info("Baseline evaluation completed")
        return baseline_results
    
    def _create_environment(self, env_config: str, reward_strategy: str):
        """Create evaluation environment"""
        env = make_hetero_env(env_config)
        
        # Configure environment
        env.config.update({
            'max_episode_steps': self.config.max_episode_steps,
            'reward_strategy': reward_strategy,
            'reward_weights': {
                'latency': 0.3, 'energy': 0.2, 'throughput': 0.25,
                'fairness': 0.15, 'stability': 0.1
            }
        })
        
        # Update reward function
        env.reward_function = env.reward_function.__class__(env.reward_function.config)
        
        return env
    
    def _run_evaluation(self, agent, env, agent_name: str) -> Dict[str, Any]:
        """Run evaluation episodes for an agent"""
        
        # Set agent to evaluation mode
        agent.set_training_mode(False)
        
        episode_metrics = []
        detailed_episode_logs = []
        
        for episode in range(self.config.num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_log = {
                'agent': agent_name,
                'episode': episode,
                'actions': [],
                'rewards': [],
                'states': [],
                'violations': {'thermal': 0, 'slo': 0}
            }
            
            for step in range(self.config.max_episode_steps):
                # Select action
                action = agent.select_action(obs, training=False)
                
                # Environment step
                next_obs, reward, done, info = env.step(action)
                
                # Log details
                if self.config.save_detailed_logs:
                    episode_log['actions'].append(action.tolist())
                    episode_log['rewards'].append(float(reward))
                    episode_log['states'].append(obs.tolist())
                
                # Track violations
                if 'task_result' in info:
                    task_result = info['task_result']
                    if task_result.get('thermal_violation', False):
                        episode_log['violations']['thermal'] += 1
                    if task_result.get('slo_violation', False):
                        episode_log['violations']['slo'] += 1
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                if done:
                    break
            
            # Store episode metrics
            episode_data = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'thermal_violations': episode_log['violations']['thermal'],
                'slo_violations': episode_log['violations']['slo'],
                'total_violations': episode_log['violations']['thermal'] + episode_log['violations']['slo']
            }
            
            # Add environment metrics if available
            if hasattr(env, 'metrics'):
                episode_data.update({
                    'total_latency': env.metrics.get('total_latency', 0),
                    'total_energy': env.metrics.get('total_energy', 0),
                    'total_throughput': env.metrics.get('total_throughput', 0)
                })
            
            episode_metrics.append(episode_data)
            
            if self.config.save_detailed_logs:
                episode_log['final_metrics'] = episode_data
                detailed_episode_logs.append(episode_log)
        
        # Compute summary statistics
        summary_metrics = self._compute_summary_metrics(episode_metrics)
        
        if self.config.save_detailed_logs:
            self.detailed_logs.extend(detailed_episode_logs)
        
        return {
            'summary': summary_metrics,
            'episodes': episode_metrics,
            'detailed_logs': detailed_episode_logs if self.config.save_detailed_logs else None
        }
    
    def _run_baseline_evaluation(self, baseline_policy, env, baseline_name: str) -> Dict[str, Any]:
        """Run evaluation for baseline policy"""
        
        episode_metrics = []
        
        for episode in range(self.config.num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0
            violations = {'thermal': 0, 'slo': 0}
            
            for step in range(self.config.max_episode_steps):
                # Select action using baseline policy
                action = baseline_policy(obs)
                
                # Environment step
                next_obs, reward, done, info = env.step(action)
                
                # Track violations
                if 'task_result' in info:
                    task_result = info['task_result']
                    if task_result.get('thermal_violation', False):
                        violations['thermal'] += 1
                    if task_result.get('slo_violation', False):
                        violations['slo'] += 1
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                if done:
                    break
            
            # Store episode metrics
            episode_data = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'thermal_violations': violations['thermal'],
                'slo_violations': violations['slo'],
                'total_violations': violations['thermal'] + violations['slo']
            }
            
            episode_metrics.append(episode_data)
        
        # Compute summary statistics
        summary_metrics = self._compute_summary_metrics(episode_metrics)
        
        return {
            'summary': summary_metrics,
            'episodes': episode_metrics
        }
    
    def _compute_summary_metrics(self, episode_metrics: List[Dict]) -> Dict[str, float]:
        """Compute summary statistics from episode metrics"""
        
        if not episode_metrics:
            return {}
        
        # Convert to DataFrame for easier computation
        df = pd.DataFrame(episode_metrics)
        
        summary = {}
        
        # Basic statistics for each metric
        for column in df.columns:
            if df[column].dtype in [np.float64, np.int64]:
                summary[f'{column}_mean'] = float(df[column].mean())
                summary[f'{column}_std'] = float(df[column].std())
                summary[f'{column}_min'] = float(df[column].min())
                summary[f'{column}_max'] = float(df[column].max())
                summary[f'{column}_median'] = float(df[column].median())
        
        # Additional derived metrics
        summary['success_rate'] = float((df['total_violations'] == 0).mean())
        summary['violation_rate'] = float((df['total_violations'] > 0).mean())
        
        # Performance consistency (coefficient of variation for rewards)
        if 'episode_reward' in df.columns and df['episode_reward'].std() > 0:
            summary['reward_consistency'] = float(
                df['episode_reward'].mean() / df['episode_reward'].std()
            )
        else:
            summary['reward_consistency'] = 0.0
        
        return summary
    
    def _save_agent_results(self, agent_name: str, results: Dict, save_dir: str):
        """Save agent evaluation results"""
        
        # Save detailed results
        results_file = os.path.join(save_dir, f"{agent_name}_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save summary table
        summary_data = []
        for config_key, config_results in results.items():
            summary_row = {'configuration': config_key}
            summary_row.update(config_results['summary'])
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(save_dir, f"{agent_name}_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Agent {agent_name} results saved to {save_dir}")
    
    def _save_baseline_results(self, baseline_results: Dict, save_dir: str):
        """Save baseline evaluation results"""
        
        baseline_file = os.path.join(save_dir, "baseline_results.json")
        with open(baseline_file, 'w') as f:
            serializable_results = self._make_json_serializable(baseline_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Baseline results saved to {save_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def compare_agents(self, save_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Generate comprehensive comparison of all evaluated agents"""
        
        if not self.results:
            logger.warning("No results available for comparison")
            return {}
        
        logger.info("Generating agent comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for agent_name, agent_results in self.results.items():
            for config_key, config_results in agent_results.items():
                row = {
                    'agent': agent_name,
                    'configuration': config_key,
                }
                row.update(config_results['summary'])
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_file = os.path.join(save_dir, "agent_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        
        # Generate ranking
        ranking = self._generate_agent_ranking(comparison_df)
        
        # Save ranking
        ranking_file = os.path.join(save_dir, "agent_ranking.json")
        with open(ranking_file, 'w') as f:
            json.dump(ranking, f, indent=2)
        
        # Generate plots if requested
        if self.config.generate_plots:
            self._generate_comparison_plots(comparison_df, save_dir)
        
        logger.info(f"Agent comparison saved to {save_dir}")
        
        return {
            'comparison_df': comparison_df,
            'ranking': ranking
        }
    
    def _generate_agent_ranking(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate agent ranking based on multiple criteria"""
        
        # Define scoring criteria (higher is better)
        criteria = {
            'episode_reward_mean': 1.0,      # Primary criterion
            'success_rate': 0.8,             # High importance
            'reward_consistency': 0.6,       # Medium importance
            'episode_length_mean': -0.3,     # Lower is better (negative weight)
            'violation_rate': -0.8           # Lower is better (negative weight)
        }
        
        # Normalize scores for each criterion
        normalized_df = comparison_df.copy()
        
        for criterion, weight in criteria.items():
            if criterion in comparison_df.columns:
                col_data = comparison_df[criterion]
                if weight > 0:  # Higher is better
                    normalized_df[f'{criterion}_norm'] = (col_data - col_data.min()) / (col_data.max() - col_data.min() + 1e-8)
                else:  # Lower is better
                    normalized_df[f'{criterion}_norm'] = (col_data.max() - col_data) / (col_data.max() - col_data.min() + 1e-8)
        
        # Compute weighted scores
        normalized_df['composite_score'] = 0
        for criterion, weight in criteria.items():
            if f'{criterion}_norm' in normalized_df.columns:
                normalized_df['composite_score'] += abs(weight) * normalized_df[f'{criterion}_norm']
        
        # Aggregate by agent (average across configurations)
        agent_scores = normalized_df.groupby('agent')['composite_score'].mean().sort_values(ascending=False)
        
        ranking = {
            'overall_ranking': agent_scores.to_dict(),
            'criteria_weights': criteria,
            'top_agent': agent_scores.index[0] if len(agent_scores) > 0 else None
        }
        
        return ranking
    
    def _generate_comparison_plots(self, comparison_df: pd.DataFrame, save_dir: str):
        """Generate comparison plots"""
        
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Reward comparison
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=comparison_df, x='agent', y='episode_reward_mean')
        plt.title('Episode Reward Comparison Across Agents')
        plt.xlabel('Agent')
        plt.ylabel('Mean Episode Reward')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'reward_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Success rate comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(data=comparison_df, x='agent', y='success_rate')
        plt.title('Success Rate Comparison (Zero Violations)')
        plt.xlabel('Agent')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'success_rate_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance vs consistency scatter
        if 'reward_consistency' in comparison_df.columns:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=comparison_df, x='episode_reward_mean', y='reward_consistency', 
                           hue='agent', style='configuration', s=100)
            plt.title('Performance vs Consistency Trade-off')
            plt.xlabel('Mean Episode Reward')
            plt.ylabel('Reward Consistency')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'performance_consistency.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Violation analysis
        violation_cols = [col for col in comparison_df.columns if 'violation' in col and 'rate' not in col]
        if violation_cols:
            fig, axes = plt.subplots(1, len(violation_cols), figsize=(6*len(violation_cols), 6))
            if len(violation_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(violation_cols):
                sns.boxplot(data=comparison_df, x='agent', y=col, ax=axes[i])
                axes[i].set_title(f'{col.replace("_", " ").title()}')
                axes[i].set_xlabel('Agent')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'violation_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Comparison plots saved to {plots_dir}")
    
    def generate_report(self, save_dir: str = "evaluation_results") -> str:
        """Generate comprehensive evaluation report"""
        
        if not self.results:
            logger.warning("No results available for report generation")
            return ""
        
        report_lines = []
        report_lines.append("# HeteroSched Agent Evaluation Report")
        report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Evaluation configuration
        report_lines.append("## Evaluation Configuration")
        report_lines.append(f"- Number of episodes per configuration: {self.config.num_episodes}")
        report_lines.append(f"- Max steps per episode: {self.config.max_episode_steps}")
        report_lines.append(f"- Environments tested: {', '.join(self.config.environments)}")
        report_lines.append(f"- Reward strategies: {', '.join(self.config.reward_strategies)}\n")
        
        # Agent performance summary
        report_lines.append("## Agent Performance Summary")
        
        # Get overall rankings
        comparison_results = self.compare_agents(save_dir)
        if 'ranking' in comparison_results:
            ranking = comparison_results['ranking']['overall_ranking']
            report_lines.append("### Overall Ranking (by composite score):")
            for i, (agent, score) in enumerate(ranking.items(), 1):
                report_lines.append(f"{i}. {agent}: {score:.4f}")
        
        report_lines.append("")
        
        # Detailed agent analysis
        report_lines.append("## Detailed Agent Analysis")
        
        for agent_name, agent_results in self.results.items():
            report_lines.append(f"### {agent_name}")
            
            # Compute agent-wide statistics
            all_rewards = []
            all_violations = []
            
            for config_results in agent_results.values():
                episodes = config_results['episodes']
                all_rewards.extend([ep['episode_reward'] for ep in episodes])
                all_violations.extend([ep['total_violations'] for ep in episodes])
            
            if all_rewards:
                report_lines.append(f"- Mean reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
                report_lines.append(f"- Success rate: {(np.array(all_violations) == 0).mean():.2%}")
                report_lines.append(f"- Configurations tested: {len(agent_results)}")
                
                # Best configuration
                best_config = max(agent_results.items(), 
                                key=lambda x: x[1]['summary']['episode_reward_mean'])
                report_lines.append(f"- Best configuration: {best_config[0]} "
                                  f"(reward: {best_config[1]['summary']['episode_reward_mean']:.3f})")
        
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        if 'ranking' in comparison_results and comparison_results['ranking']['top_agent']:
            top_agent = comparison_results['ranking']['top_agent']
            report_lines.append(f"- **Recommended agent**: {top_agent}")
            report_lines.append("- This agent demonstrates the best overall performance across")
            report_lines.append("  multiple evaluation criteria including reward, stability, and violation rates.")
        
        # Generate report file
        report_content = "\n".join(report_lines)
        report_file = os.path.join(save_dir, "evaluation_report.md")
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report saved to {report_file}")
        return report_content