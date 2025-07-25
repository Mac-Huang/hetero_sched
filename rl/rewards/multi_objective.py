#!/usr/bin/env python3
"""
Multi-Objective Reward Functions for HeteroSched RL

Implements various approaches to combining multiple objectives:
- Weighted scalarization
- Pareto-optimal rewards
- Adaptive weight adjustment
- Hierarchical objectives
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math

from .metrics import (
    ObjectiveMetric, TaskResult,
    LatencyMetric, EnergyMetric, ThroughputMetric, 
    FairnessMetric, StabilityMetric, PerformanceMetric
)

class RewardStrategy(Enum):
    WEIGHTED_SUM = "weighted_sum"
    PARETO_OPTIMAL = "pareto_optimal"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    CONSTRAINED = "constrained"

@dataclass
class RewardConfig:
    """Configuration for multi-objective reward function"""
    strategy: RewardStrategy = RewardStrategy.WEIGHTED_SUM
    weights: Dict[str, float] = field(default_factory=lambda: {
        'latency': 0.3,
        'energy': 0.2,
        'throughput': 0.25,
        'fairness': 0.15,
        'stability': 0.1
    })
    constraints: Dict[str, float] = field(default_factory=dict)
    adaptation_rate: float = 0.01
    pareto_alpha: float = 0.5

class MultiObjectiveReward:
    """Main multi-objective reward function coordinator"""
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        
        # Initialize objective metrics
        self.metrics = {
            'latency': LatencyMetric(),
            'energy': EnergyMetric(),
            'throughput': ThroughputMetric(),
            'fairness': FairnessMetric(),
            'stability': StabilityMetric(),
            'performance': PerformanceMetric()
        }
        
        # Initialize reward strategy
        self.reward_computer = self._create_reward_computer()
        
        # Performance tracking
        self.episode_rewards = []
        self.objective_history = {name: [] for name in self.metrics.keys()}
        
    def _create_reward_computer(self) -> 'RewardComputer':
        """Factory method to create appropriate reward computer"""
        if self.config.strategy == RewardStrategy.WEIGHTED_SUM:
            return ScalarizedReward(self.config)
        elif self.config.strategy == RewardStrategy.PARETO_OPTIMAL:
            return ParetoOptimalReward(self.config)
        elif self.config.strategy == RewardStrategy.ADAPTIVE:
            return AdaptiveReward(self.config)
        elif self.config.strategy == RewardStrategy.HIERARCHICAL:
            return HierarchicalReward(self.config)
        elif self.config.strategy == RewardStrategy.CONSTRAINED:
            return ConstrainedReward(self.config)
        else:
            raise ValueError(f"Unknown reward strategy: {self.config.strategy}")
    
    def compute_reward(self, task_result: TaskResult, system_state: Dict) -> Dict:
        """Compute multi-objective reward and return detailed breakdown"""
        
        # Compute individual objective scores
        objective_scores = {}
        for name, metric in self.metrics.items():
            score = metric.compute(task_result, system_state)
            objective_scores[name] = score
            self.objective_history[name].append(score)
        
        # Compute combined reward using selected strategy
        combined_reward, reward_info = self.reward_computer.compute(
            objective_scores, task_result, system_state
        )
        
        # Track episode performance
        self.episode_rewards.append(combined_reward)
        
        # Prepare detailed return information
        result = {
            'total_reward': combined_reward,
            'objective_scores': objective_scores,
            'strategy': self.config.strategy.value,
            'weights': self.config.weights.copy(),
            'info': reward_info
        }
        
        return result
    
    def get_performance_summary(self) -> Dict:
        """Get summary of reward performance over recent episodes"""
        if len(self.episode_rewards) < 10:
            return {'status': 'insufficient_data'}
        
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        
        summary = {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'trend': self._compute_trend(recent_rewards),
            'objective_performance': {}
        }
        
        # Objective-specific performance
        for name, history in self.objective_history.items():
            if len(history) >= 10:
                recent_scores = history[-100:]
                summary['objective_performance'][name] = {
                    'mean': np.mean(recent_scores),
                    'std': np.std(recent_scores),
                    'trend': self._compute_trend(recent_scores)
                }
        
        return summary
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend direction (-1 to 1)"""
        if len(values) < 10:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return np.tanh(coeffs[0] * 10)  # Normalize slope to [-1, 1]
    
    def adapt_weights(self, performance_feedback: Dict):
        """Adapt reward weights based on performance feedback"""
        if hasattr(self.reward_computer, 'adapt_weights'):
            self.reward_computer.adapt_weights(performance_feedback)

class RewardComputer(ABC):
    """Abstract base class for reward computation strategies"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    @abstractmethod
    def compute(self, objective_scores: Dict[str, float], 
                task_result: TaskResult, system_state: Dict) -> Tuple[float, Dict]:
        """Compute combined reward and return info dict"""
        pass

class ScalarizedReward(RewardComputer):
    """Weighted sum scalarization of multiple objectives"""
    
    def compute(self, objective_scores: Dict[str, float], 
                task_result: TaskResult, system_state: Dict) -> Tuple[float, Dict]:
        
        total_reward = 0.0
        weighted_scores = {}
        
        # Apply weights to objective scores
        for objective, score in objective_scores.items():
            weight = self.config.weights.get(objective, 0.0)
            weighted_score = weight * score
            weighted_scores[objective] = weighted_score
            total_reward += weighted_score
        
        info = {
            'method': 'weighted_sum',
            'weighted_scores': weighted_scores,
            'total_weight': sum(self.config.weights.values())
        }
        
        return total_reward, info

class ParetoOptimalReward(RewardComputer):
    """Pareto-optimal reward based on dominance relationships"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.historical_points = []
        self.max_history = 200
    
    def compute(self, objective_scores: Dict[str, float], 
                task_result: TaskResult, system_state: Dict) -> Tuple[float, Dict]:
        
        current_point = list(objective_scores.values())
        self.historical_points.append(current_point)
        
        # Maintain history size
        if len(self.historical_points) > self.max_history:
            self.historical_points.pop(0)
        
        # Compute Pareto dominance score
        pareto_score = self._compute_pareto_score(current_point)
        
        # Combine with weighted sum for stability
        weighted_sum = sum(self.config.weights.get(obj, 0.0) * score 
                          for obj, score in objective_scores.items())
        
        # Balanced combination
        alpha = self.config.pareto_alpha
        total_reward = alpha * pareto_score + (1 - alpha) * weighted_sum
        
        info = {
            'method': 'pareto_optimal',
            'pareto_score': pareto_score,
            'weighted_sum': weighted_sum,
            'dominance_count': self._count_dominated_points(current_point),
            'non_dominated_count': len(self._get_pareto_front())
        }
        
        return total_reward, info
    
    def _compute_pareto_score(self, point: List[float]) -> float:
        """Compute Pareto dominance score for current point"""
        if len(self.historical_points) < 10:
            return 0.0
        
        dominated_count = self._count_dominated_points(point)
        total_points = len(self.historical_points)
        
        # Score based on dominance ratio
        dominance_ratio = dominated_count / total_points
        return dominance_ratio * 2.0 - 1.0  # Scale to [-1, 1]
    
    def _count_dominated_points(self, point: List[float]) -> int:
        """Count how many historical points are dominated by current point"""
        dominated = 0
        for historical_point in self.historical_points:
            if self._dominates(point, historical_point):
                dominated += 1
        return dominated
    
    def _dominates(self, point1: List[float], point2: List[float]) -> bool:
        """Check if point1 dominates point2 (assuming maximization)"""
        if len(point1) != len(point2):
            return False
        
        all_greater_equal = all(a >= b for a, b in zip(point1, point2))
        any_strictly_greater = any(a > b for a, b in zip(point1, point2))
        
        return all_greater_equal and any_strictly_greater
    
    def _get_pareto_front(self) -> List[List[float]]:
        """Get current Pareto front from historical points"""
        if len(self.historical_points) < 2:
            return self.historical_points.copy()
        
        pareto_front = []
        for i, point1 in enumerate(self.historical_points):
            is_dominated = False
            for j, point2 in enumerate(self.historical_points):
                if i != j and self._dominates(point2, point1):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(point1)
        
        return pareto_front

class AdaptiveReward(RewardComputer):
    """Adaptive reward that adjusts weights based on performance"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.adaptive_weights = self.config.weights.copy()
        self.objective_performance = {obj: [] for obj in self.config.weights.keys()}
        self.adaptation_counter = 0
        
    def compute(self, objective_scores: Dict[str, float], 
                task_result: TaskResult, system_state: Dict) -> Tuple[float, Dict]:
        
        # Track objective performance
        for objective, score in objective_scores.items():
            if objective in self.objective_performance:
                self.objective_performance[objective].append(score)
                # Keep recent history
                if len(self.objective_performance[objective]) > 100:
                    self.objective_performance[objective].pop(0)
        
        # Adapt weights periodically
        self.adaptation_counter += 1
        if self.adaptation_counter % 50 == 0:  # Adapt every 50 steps
            self._adapt_weights()
        
        # Compute weighted reward with adaptive weights
        total_reward = sum(self.adaptive_weights.get(obj, 0.0) * score 
                          for obj, score in objective_scores.items())
        
        info = {
            'method': 'adaptive',
            'adaptive_weights': self.adaptive_weights.copy(),
            'original_weights': self.config.weights.copy(),
            'adaptation_step': self.adaptation_counter
        }
        
        return total_reward, info
    
    def _adapt_weights(self):
        """Adapt weights based on objective performance trends"""
        for objective in self.adaptive_weights.keys():
            if objective in self.objective_performance:
                performance = self.objective_performance[objective]
                if len(performance) >= 20:
                    # Compute recent trend
                    trend = self._compute_performance_trend(performance[-20:])
                    
                    # Adjust weight based on trend
                    # If performance is declining, increase weight
                    # If performance is improving, slightly decrease weight
                    adjustment = -trend * self.config.adaptation_rate
                    self.adaptive_weights[objective] += adjustment
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            for objective in self.adaptive_weights:
                self.adaptive_weights[objective] /= total_weight
        
        # Ensure minimum weight for stability
        min_weight = 0.05
        for objective in self.adaptive_weights:
            self.adaptive_weights[objective] = max(min_weight, self.adaptive_weights[objective])
    
    def _compute_performance_trend(self, values: List[float]) -> float:
        """Compute performance trend (-1 to 1)"""
        if len(values) < 5:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return np.tanh(coeffs[0] * 5)  # Normalize slope
    
    def adapt_weights(self, performance_feedback: Dict):
        """External weight adaptation based on system feedback"""
        for objective, feedback in performance_feedback.items():
            if objective in self.adaptive_weights:
                # Adjust based on external feedback
                if feedback > 0:  # Good performance, maintain weight
                    continue
                else:  # Poor performance, increase weight
                    self.adaptive_weights[objective] *= 1.1
        
        # Re-normalize
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            for objective in self.adaptive_weights:
                self.adaptive_weights[objective] /= total_weight

class HierarchicalReward(RewardComputer):
    """Hierarchical reward with primary and secondary objectives"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        # Define hierarchy: primary objectives must be satisfied first
        self.primary_objectives = ['stability', 'fairness']
        self.secondary_objectives = ['latency', 'energy', 'throughput', 'performance']
        self.primary_threshold = 0.0  # Minimum acceptable primary score
    
    def compute(self, objective_scores: Dict[str, float], 
                task_result: TaskResult, system_state: Dict) -> Tuple[float, Dict]:
        
        # Check primary objectives first
        primary_scores = {obj: objective_scores.get(obj, 0.0) 
                         for obj in self.primary_objectives}
        primary_satisfied = all(score >= self.primary_threshold 
                               for score in primary_scores.values())
        
        if not primary_satisfied:
            # Focus on primary objectives only
            primary_penalty = sum(min(0, score - self.primary_threshold) 
                                 for score in primary_scores.values())
            total_reward = primary_penalty * 5.0  # Heavy penalty
            
            info = {
                'method': 'hierarchical',
                'phase': 'primary_violation',
                'primary_scores': primary_scores,
                'penalty': primary_penalty
            }
        else:
            # Primary objectives satisfied, optimize secondary objectives
            secondary_scores = {obj: objective_scores.get(obj, 0.0) 
                               for obj in self.secondary_objectives}
            
            # Weighted sum of secondary objectives
            secondary_reward = sum(self.config.weights.get(obj, 0.0) * score 
                                  for obj, score in secondary_scores.items())
            
            # Bonus for satisfying primary objectives
            primary_bonus = sum(primary_scores.values()) * 0.2
            
            total_reward = secondary_reward + primary_bonus
            
            info = {
                'method': 'hierarchical',
                'phase': 'secondary_optimization',
                'primary_scores': primary_scores,
                'secondary_scores': secondary_scores,
                'primary_bonus': primary_bonus
            }
        
        return total_reward, info

class ConstrainedReward(RewardComputer):
    """Constrained optimization with penalty for constraint violations"""
    
    def compute(self, objective_scores: Dict[str, float], 
                task_result: TaskResult, system_state: Dict) -> Tuple[float, Dict]:
        
        # Base reward as weighted sum
        base_reward = sum(self.config.weights.get(obj, 0.0) * score 
                         for obj, score in objective_scores.items())
        
        # Check constraint violations
        violations = {}
        total_penalty = 0.0
        
        for constraint, threshold in self.config.constraints.items():
            if constraint in objective_scores:
                score = objective_scores[constraint]
                if score < threshold:
                    violation = threshold - score
                    violations[constraint] = violation
                    total_penalty += violation * 2.0  # Penalty factor
        
        # Apply penalties
        constrained_reward = base_reward - total_penalty
        
        info = {
            'method': 'constrained',
            'base_reward': base_reward,
            'violations': violations,
            'total_penalty': total_penalty,
            'constraints': self.config.constraints.copy()
        }
        
        return constrained_reward, info

# Factory function for easy reward creation
def create_reward_function(strategy: str = 'weighted_sum', **kwargs) -> MultiObjectiveReward:
    """Create a multi-objective reward function with specified strategy"""
    
    strategy_enum = RewardStrategy(strategy)
    config = RewardConfig(strategy=strategy_enum, **kwargs)
    
    return MultiObjectiveReward(config)