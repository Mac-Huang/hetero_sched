#!/usr/bin/env python3
"""
Experience Replay Prioritization for Rare Events in Heterogeneous Scheduling

This module implements advanced experience replay prioritization specifically
designed to handle rare but critical events in heterogeneous scheduling systems.
The system identifies, prioritizes, and learns from rare scheduling scenarios
such as system failures, deadline violations, and resource emergencies.

Research Innovation: First experience replay system specifically designed for
rare event learning in heterogeneous scheduling with multi-objective
prioritization and event-aware sampling.

Key Components:
- Rare event detection and classification
- Multi-objective prioritization with temporal decay
- Hindsight experience replay for counterfactual learning
- Curiosity-driven exploration for rare event discovery
- Event importance estimation using information theory
- Balanced sampling for rare vs. common event learning

Authors: HeteroSched Research Team
"""

import os
import sys
import time
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import math
import random
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events in scheduling systems"""
    NORMAL_EXECUTION = "normal_execution"
    DEADLINE_VIOLATION = "deadline_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SYSTEM_FAILURE = "system_failure"
    LOAD_SPIKE = "load_spike"
    NETWORK_PARTITION = "network_partition"
    PREEMPTION_CASCADE = "preemption_cascade"
    THERMAL_THROTTLING = "thermal_throttling"
    PRIORITY_INVERSION = "priority_inversion"
    RESOURCE_CONTENTION = "resource_contention"

class PriorityType(Enum):
    """Types of priority calculation methods"""
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"
    RANK_BASED = "rank_based"
    CURIOSITY_DRIVEN = "curiosity_driven"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE = "adaptive"

@dataclass
class SchedulingExperience:
    """Experience tuple for scheduling with metadata"""
    experience_id: str
    state: np.ndarray
    action: Dict[str, Any]
    reward: Union[float, np.ndarray]  # Can be multi-objective
    next_state: np.ndarray
    done: bool
    timestamp: float
    
    # Event classification
    event_type: EventType
    event_severity: float  # [0, 1] where 1 is most severe
    event_rarity: float   # [0, 1] where 1 is most rare
    
    # Multi-objective metadata
    objective_rewards: Dict[str, float]
    constraint_violations: List[str]
    
    # Temporal context
    episode_step: int
    episode_id: str
    
    # System context
    system_load: float
    resource_utilization: Dict[str, float]
    pending_tasks: int
    
    # Priority scores (updated by replay system)
    td_error: float = 0.0
    priority_score: float = 1.0
    sampling_weight: float = 1.0
    last_sampled: float = 0.0
    sample_count: int = 0

class RareEventDetector:
    """Detect and classify rare events in scheduling experiences"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Event statistics
        self.event_counts = defaultdict(int)
        self.total_events = 0
        self.event_features = defaultdict(list)
        
        # Anomaly detection
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.0)  # Standard deviations
        self.feature_stats = defaultdict(lambda: {'mean': 0.0, 'std': 1.0, 'count': 0})
        
        # Clustering for pattern detection
        self.dbscan = DBSCAN(eps=0.5, min_samples=3)
        self.scaler = StandardScaler()
        
        # Rarity thresholds
        self.rarity_thresholds = {
            EventType.NORMAL_EXECUTION: 0.1,
            EventType.DEADLINE_VIOLATION: 0.6,
            EventType.RESOURCE_EXHAUSTION: 0.7,
            EventType.SYSTEM_FAILURE: 0.9,
            EventType.LOAD_SPIKE: 0.5,
            EventType.NETWORK_PARTITION: 0.8,
            EventType.PREEMPTION_CASCADE: 0.7,
            EventType.THERMAL_THROTTLING: 0.6,
            EventType.PRIORITY_INVERSION: 0.6,
            EventType.RESOURCE_CONTENTION: 0.4
        }
    
    def detect_event_type(self, experience: SchedulingExperience) -> Tuple[EventType, float]:
        """Detect event type and severity from experience"""
        
        # Check for explicit failure indicators
        if any('failure' in violation for violation in experience.constraint_violations):
            return EventType.SYSTEM_FAILURE, 0.9
        
        # Check deadline violations
        if any('deadline' in violation for violation in experience.constraint_violations):
            severity = min(0.8, abs(experience.reward) / 10.0) if isinstance(experience.reward, float) else 0.6
            return EventType.DEADLINE_VIOLATION, severity
        
        # Check resource exhaustion
        max_utilization = max(experience.resource_utilization.values()) if experience.resource_utilization else 0.0
        if max_utilization > 0.95:
            severity = (max_utilization - 0.95) / 0.05
            return EventType.RESOURCE_EXHAUSTION, severity
        
        # Check load spikes
        if experience.system_load > 0.9:
            severity = (experience.system_load - 0.9) / 0.1
            return EventType.LOAD_SPIKE, severity
        
        # Check for resource contention (high pending tasks)
        if experience.pending_tasks > 50:
            severity = min(1.0, (experience.pending_tasks - 50) / 100.0)
            return EventType.RESOURCE_CONTENTION, severity
        
        # Check for anomalous patterns
        if self._is_anomalous_experience(experience):
            return self._classify_anomaly(experience)
        
        # Default to normal execution
        return EventType.NORMAL_EXECUTION, 0.1
    
    def calculate_event_rarity(self, event_type: EventType) -> float:
        """Calculate rarity score for an event type"""
        if self.total_events == 0:
            return self.rarity_thresholds.get(event_type, 0.5)
        
        event_frequency = self.event_counts[event_type] / self.total_events
        
        # Convert frequency to rarity (inverse relationship)
        if event_frequency == 0:
            return 1.0
        
        # Use logarithmic scale for rarity
        rarity = min(1.0, -np.log10(event_frequency + 1e-6) / 4.0)  # Normalize to [0, 1]
        
        return max(rarity, self.rarity_thresholds.get(event_type, 0.1))
    
    def update_event_statistics(self, experience: SchedulingExperience):
        """Update event statistics and patterns"""
        self.event_counts[experience.event_type] += 1
        self.total_events += 1
        
        # Update feature statistics for anomaly detection
        features = self._extract_features(experience)
        for feature_name, value in features.items():
            stats = self.feature_stats[feature_name]
            
            # Online mean and variance update
            stats['count'] += 1
            delta = value - stats['mean']
            stats['mean'] += delta / stats['count']
            
            if stats['count'] > 1:
                stats['std'] = np.sqrt(
                    ((stats['count'] - 2) * stats['std']**2 + delta**2) / (stats['count'] - 1)
                )
            
            self.event_features[feature_name].append(value)
            
            # Keep limited history
            if len(self.event_features[feature_name]) > 10000:
                self.event_features[feature_name] = self.event_features[feature_name][-10000:]
    
    def _extract_features(self, experience: SchedulingExperience) -> Dict[str, float]:
        """Extract features for anomaly detection"""
        features = {
            'reward': experience.reward if isinstance(experience.reward, float) else np.mean(experience.reward),
            'system_load': experience.system_load,
            'pending_tasks': experience.pending_tasks,
            'episode_step': experience.episode_step,
            'constraint_violations_count': len(experience.constraint_violations)
        }
        
        # Add resource utilization features
        if experience.resource_utilization:
            features.update({
                f'utilization_{resource}': util
                for resource, util in experience.resource_utilization.items()
            })
        
        # Add objective-specific features
        if experience.objective_rewards:
            features.update({
                f'objective_{obj}': reward
                for obj, reward in experience.objective_rewards.items()
            })
        
        return features
    
    def _is_anomalous_experience(self, experience: SchedulingExperience) -> bool:
        """Check if experience is anomalous based on feature statistics"""
        if self.total_events < 100:  # Need sufficient data
            return False
        
        features = self._extract_features(experience)
        anomaly_scores = []
        
        for feature_name, value in features.items():
            if feature_name in self.feature_stats:
                stats = self.feature_stats[feature_name]
                if stats['std'] > 0:
                    z_score = abs(value - stats['mean']) / stats['std']
                    anomaly_scores.append(z_score)
        
        if anomaly_scores:
            max_z_score = max(anomaly_scores)
            return max_z_score > self.anomaly_threshold
        
        return False
    
    def _classify_anomaly(self, experience: SchedulingExperience) -> Tuple[EventType, float]:
        """Classify anomalous experience into specific event type"""
        features = self._extract_features(experience)
        
        # Heuristic classification based on feature patterns
        if features.get('system_load', 0) > 0.8:
            return EventType.LOAD_SPIKE, 0.7
        elif features.get('constraint_violations_count', 0) > 2:
            return EventType.DEADLINE_VIOLATION, 0.6
        elif max(features.get(f'utilization_{res}', 0) for res in ['cpu', 'memory', 'gpu'] if f'utilization_{res}' in features) > 0.9:
            return EventType.RESOURCE_EXHAUSTION, 0.8
        else:
            return EventType.RESOURCE_CONTENTION, 0.5

class CuriosityDrivenExploration:
    """Curiosity-driven module for discovering rare events"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Intrinsic curiosity module
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.feature_encoder.parameters()) +
            list(self.inverse_model.parameters()) +
            list(self.forward_model.parameters()),
            lr=1e-4
        )
    
    def calculate_intrinsic_reward(self, state: np.ndarray, action: np.ndarray, 
                                 next_state: np.ndarray) -> float:
        """Calculate intrinsic curiosity reward"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            # Encode states
            state_features = self.feature_encoder(state_tensor)
            next_state_features = self.feature_encoder(next_state_tensor)
            
            # Predict next state features
            forward_input = torch.cat([state_features, action_tensor], dim=1)
            predicted_next_features = self.forward_model(forward_input)
            
            # Calculate prediction error (curiosity)
            prediction_error = F.mse_loss(predicted_next_features, next_state_features)
            
            return float(prediction_error)
    
    def update_curiosity_model(self, batch_states: torch.Tensor, batch_actions: torch.Tensor,
                             batch_next_states: torch.Tensor):
        """Update curiosity model with batch of experiences"""
        # Encode states
        state_features = self.feature_encoder(batch_states)
        next_state_features = self.feature_encoder(batch_next_states)
        
        # Inverse model loss (predict action from states)
        inverse_input = torch.cat([state_features, next_state_features], dim=1)
        predicted_actions = self.inverse_model(inverse_input)
        inverse_loss = F.mse_loss(predicted_actions, batch_actions)
        
        # Forward model loss (predict next state from state and action)
        forward_input = torch.cat([state_features, batch_actions], dim=1)
        predicted_next_features = self.forward_model(forward_input)
        forward_loss = F.mse_loss(predicted_next_features, next_state_features.detach())
        
        # Total loss
        total_loss = inverse_loss + forward_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {'inverse_loss': inverse_loss.item(), 'forward_loss': forward_loss.item()}

class HindsightExperienceReplay:
    """Hindsight experience replay for learning from failed attempts"""
    
    def __init__(self, goal_dim: int = 8, strategy: str = 'future'):
        self.goal_dim = goal_dim
        self.strategy = strategy  # 'future', 'final', 'episode', 'random'
        
    def generate_hindsight_experiences(self, episode_experiences: List[SchedulingExperience], 
                                     episode_goals: List[Dict[str, Any]]) -> List[SchedulingExperience]:
        """Generate hindsight experiences from episode"""
        hindsight_experiences = []
        
        if not episode_experiences or not episode_goals:
            return hindsight_experiences
        
        for i, experience in enumerate(episode_experiences):
            # Original experience
            hindsight_experiences.append(experience)
            
            # Generate hindsight goals
            if self.strategy == 'future':
                # Sample goals from future states in the episode
                future_indices = list(range(i + 1, len(episode_experiences)))
                if future_indices:
                    sampled_indices = random.sample(future_indices, min(4, len(future_indices)))
                    for j in sampled_indices:
                        hindsight_goal = self._extract_goal_from_state(episode_experiences[j].next_state)
                        hindsight_exp = self._create_hindsight_experience(experience, hindsight_goal, j >= len(episode_experiences) - 1)
                        hindsight_experiences.append(hindsight_exp)
            
            elif self.strategy == 'final':
                # Use final state as goal
                if i < len(episode_experiences) - 1:
                    final_goal = self._extract_goal_from_state(episode_experiences[-1].next_state)
                    hindsight_exp = self._create_hindsight_experience(experience, final_goal, i == len(episode_experiences) - 1)
                    hindsight_experiences.append(hindsight_exp)
        
        return hindsight_experiences
    
    def _extract_goal_from_state(self, state: np.ndarray) -> Dict[str, Any]:
        """Extract goal representation from state"""
        # Simple goal extraction - in practice would be more sophisticated
        goal_features = state[:self.goal_dim] if len(state) >= self.goal_dim else state
        
        return {
            'target_utilization': float(goal_features[0]) if len(goal_features) > 0 else 0.5,
            'target_load': float(goal_features[1]) if len(goal_features) > 1 else 0.5,
            'target_queue_length': int(goal_features[2] * 100) if len(goal_features) > 2 else 10
        }
    
    def _create_hindsight_experience(self, original_exp: SchedulingExperience, 
                                   hindsight_goal: Dict[str, Any], achieved: bool) -> SchedulingExperience:
        """Create hindsight experience with new goal"""
        # Calculate hindsight reward
        hindsight_reward = self._calculate_hindsight_reward(original_exp, hindsight_goal, achieved)
        
        # Create new experience
        hindsight_exp = SchedulingExperience(
            experience_id=f"{original_exp.experience_id}_hindsight_{random.randint(1000, 9999)}",
            state=original_exp.state.copy(),
            action=original_exp.action.copy(),
            reward=hindsight_reward,
            next_state=original_exp.next_state.copy(),
            done=achieved,
            timestamp=original_exp.timestamp,
            event_type=original_exp.event_type,
            event_severity=original_exp.event_severity,
            event_rarity=original_exp.event_rarity * 0.8,  # Slightly less rare for hindsight
            objective_rewards=original_exp.objective_rewards.copy(),
            constraint_violations=original_exp.constraint_violations.copy(),
            episode_step=original_exp.episode_step,
            episode_id=f"{original_exp.episode_id}_hindsight",
            system_load=original_exp.system_load,
            resource_utilization=original_exp.resource_utilization.copy(),
            pending_tasks=original_exp.pending_tasks
        )
        
        return hindsight_exp
    
    def _calculate_hindsight_reward(self, experience: SchedulingExperience, 
                                  goal: Dict[str, Any], achieved: bool) -> float:
        """Calculate reward for hindsight goal"""
        if achieved:
            return 1.0  # Goal achieved
        
        # Calculate partial reward based on goal progress
        progress_score = 0.0
        
        # Utilization progress
        if 'target_utilization' in goal:
            current_util = experience.resource_utilization.get('cpu', 0.5)
            target_util = goal['target_utilization']
            util_error = abs(current_util - target_util)
            progress_score += max(0, 1.0 - util_error)
        
        # Load progress
        if 'target_load' in goal:
            load_error = abs(experience.system_load - goal['target_load'])
            progress_score += max(0, 1.0 - load_error)
        
        # Queue length progress
        if 'target_queue_length' in goal:
            queue_error = abs(experience.pending_tasks - goal['target_queue_length']) / 100.0
            progress_score += max(0, 1.0 - queue_error)
        
        return progress_score / 3.0  # Average progress

class MultiObjectivePriorityCalculator:
    """Calculate priorities for multi-objective reinforcement learning"""
    
    def __init__(self, num_objectives: int = 3, config: Dict[str, Any] = None):
        self.num_objectives = num_objectives
        self.config = config or {}
        
        # Objective weights (can be learned or configured)
        self.objective_weights = np.ones(num_objectives) / num_objectives
        self.weight_adaptation_rate = self.config.get('weight_adaptation_rate', 0.01)
        
        # Priority calculation parameters
        self.td_error_weight = self.config.get('td_error_weight', 0.4)
        self.rarity_weight = self.config.get('rarity_weight', 0.3)
        self.severity_weight = self.config.get('severity_weight', 0.2)
        self.curiosity_weight = self.config.get('curiosity_weight', 0.1)
        
        # Temporal decay
        self.temporal_decay = self.config.get('temporal_decay', 0.99)
        
    def calculate_priority(self, experience: SchedulingExperience, 
                         td_errors: Optional[np.ndarray] = None,
                         curiosity_score: float = 0.0) -> float:
        """Calculate priority score for experience"""
        
        # TD error component
        if td_errors is not None:
            if isinstance(td_errors, np.ndarray) and len(td_errors) == self.num_objectives:
                # Multi-objective TD error
                weighted_td_error = np.sum(self.objective_weights * np.abs(td_errors))
            else:
                weighted_td_error = float(np.abs(td_errors))
        else:
            weighted_td_error = experience.td_error
        
        # Rarity component
        rarity_score = experience.event_rarity
        
        # Severity component
        severity_score = experience.event_severity
        
        # Temporal decay component
        time_since_creation = time.time() - experience.timestamp
        temporal_factor = self.temporal_decay ** (time_since_creation / 3600.0)  # Decay per hour
        
        # Combine components
        priority = (
            self.td_error_weight * weighted_td_error +
            self.rarity_weight * rarity_score +
            self.severity_weight * severity_score +
            self.curiosity_weight * curiosity_score
        ) * temporal_factor
        
        # Ensure minimum priority
        return max(0.01, priority)
    
    def update_objective_weights(self, performance_feedback: Dict[str, float]):
        """Update objective weights based on performance feedback"""
        # Simple adaptation based on objective performance
        for i, (objective, performance) in enumerate(performance_feedback.items()):
            if i < len(self.objective_weights):
                # Increase weight for poorly performing objectives
                weight_adjustment = (1.0 - performance) * self.weight_adaptation_rate
                self.objective_weights[i] += weight_adjustment
        
        # Normalize weights
        self.objective_weights = self.objective_weights / np.sum(self.objective_weights)

class PrioritizedExperienceReplay:
    """Main prioritized experience replay system for rare events"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4, 
                 config: Dict[str, Any] = None):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength
        self.beta = beta    # Importance sampling strength
        self.config = config or {}
        
        # Experience storage
        self.experiences: List[SchedulingExperience] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Component systems
        self.event_detector = RareEventDetector(config.get('event_detection', {}))
        self.curiosity_module = CuriosityDrivenExploration(
            state_dim=config.get('state_dim', 128),
            action_dim=config.get('action_dim', 64)
        )
        self.hindsight_replay = HindsightExperienceReplay(
            goal_dim=config.get('goal_dim', 8),
            strategy=config.get('hindsight_strategy', 'future')
        )
        self.priority_calculator = MultiObjectivePriorityCalculator(
            num_objectives=config.get('num_objectives', 3),
            config=config.get('priority_calculation', {})
        )
        
        # Rare event tracking
        self.rare_event_buffer = defaultdict(list)
        self.rare_event_threshold = config.get('rare_event_threshold', 0.6)
        self.max_rare_events_per_type = config.get('max_rare_events_per_type', 1000)
        
        # Sampling statistics
        self.sampling_stats = {
            'total_samples': 0,
            'rare_event_samples': 0,
            'event_type_samples': defaultdict(int),
            'priority_distribution': deque(maxlen=10000)
        }
        
        # Beta annealing
        self.beta_start = beta
        self.beta_end = config.get('beta_end', 1.0)
        self.beta_annealing_steps = config.get('beta_annealing_steps', 100000)
        self.training_step = 0
    
    def add_experience(self, experience: SchedulingExperience, td_error: Optional[np.ndarray] = None):
        """Add experience to replay buffer"""
        
        # Detect event type and rarity
        event_type, severity = self.event_detector.detect_event_type(experience)
        experience.event_type = event_type
        experience.event_severity = severity
        experience.event_rarity = self.event_detector.calculate_event_rarity(event_type)
        
        # Update TD error
        if td_error is not None:
            experience.td_error = float(np.mean(np.abs(td_error)))
        
        # Calculate curiosity score
        if hasattr(experience, 'state') and hasattr(experience, 'next_state'):
            action_array = self._action_to_array(experience.action)
            curiosity_score = self.curiosity_module.calculate_intrinsic_reward(
                experience.state, action_array, experience.next_state
            )
        else:
            curiosity_score = 0.0
        
        # Calculate priority
        priority = self.priority_calculator.calculate_priority(
            experience, td_error, curiosity_score
        )
        experience.priority_score = priority
        
        # Store experience
        if self.size < self.capacity:
            self.experiences.append(experience)
            self.size += 1
        else:
            self.experiences[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        
        # Special handling for rare events
        if experience.event_rarity >= self.rare_event_threshold:
            self._add_to_rare_event_buffer(experience)
        
        # Update event detector statistics
        self.event_detector.update_event_statistics(experience)
        
        logger.debug(f"Added experience: {event_type.value}, rarity={experience.event_rarity:.3f}, "
                    f"priority={priority:.3f}")
    
    def add_episode(self, episode_experiences: List[SchedulingExperience], 
                   episode_goals: List[Dict[str, Any]] = None):
        """Add full episode with hindsight experience generation"""
        
        # Generate hindsight experiences
        if episode_goals:
            hindsight_experiences = self.hindsight_replay.generate_hindsight_experiences(
                episode_experiences, episode_goals
            )
            all_experiences = episode_experiences + hindsight_experiences
        else:
            all_experiences = episode_experiences
        
        # Add all experiences
        for experience in all_experiences:
            self.add_experience(experience)
    
    def sample(self, batch_size: int, priority_type: PriorityType = PriorityType.ADAPTIVE) -> Tuple[List[SchedulingExperience], np.ndarray, np.ndarray]:
        """Sample batch of experiences based on priorities"""
        
        if self.size == 0:
            return [], np.array([]), np.array([])
        
        # Select sampling strategy
        if priority_type == PriorityType.ADAPTIVE:
            priority_type = self._select_adaptive_sampling_strategy()
        
        # Sample based on strategy
        if priority_type == PriorityType.UNIFORM:
            indices, weights = self._uniform_sampling(batch_size)
        elif priority_type == PriorityType.PROPORTIONAL:
            indices, weights = self._proportional_sampling(batch_size)
        elif priority_type == PriorityType.RANK_BASED:
            indices, weights = self._rank_based_sampling(batch_size)
        elif priority_type == PriorityType.CURIOSITY_DRIVEN:
            indices, weights = self._curiosity_driven_sampling(batch_size)
        elif priority_type == PriorityType.MULTI_OBJECTIVE:
            indices, weights = self._multi_objective_sampling(batch_size)
        else:
            indices, weights = self._proportional_sampling(batch_size)
        
        # Get experiences
        experiences = [self.experiences[idx] for idx in indices]
        
        # Update sampling statistics
        self._update_sampling_statistics(experiences)
        
        # Update beta (importance sampling annealing)
        self.training_step += 1
        current_beta = self._get_current_beta()
        
        # Adjust importance sampling weights
        is_weights = (self.size * weights) ** (-current_beta)
        is_weights = is_weights / is_weights.max()  # Normalize
        
        return experiences, indices, is_weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors"""
        for idx, td_error in zip(indices, td_errors):
            if idx < self.size:
                experience = self.experiences[idx]
                experience.td_error = float(np.mean(np.abs(td_error)))
                
                # Recalculate priority
                priority = self.priority_calculator.calculate_priority(experience)
                experience.priority_score = priority
                self.priorities[idx] = priority ** self.alpha
    
    def train_curiosity_model(self, batch_size: int = 32):
        """Train curiosity model on recent experiences"""
        if self.size < batch_size:
            return {}
        
        # Sample recent experiences
        recent_indices = list(range(max(0, self.size - 1000), self.size))
        sampled_indices = random.sample(recent_indices, min(batch_size, len(recent_indices)))
        
        # Prepare batch data
        states = []
        actions = []
        next_states = []
        
        for idx in sampled_indices:
            exp = self.experiences[idx]
            states.append(exp.state)
            actions.append(self._action_to_array(exp.action))
            next_states.append(exp.next_state)
        
        batch_states = torch.tensor(np.array(states), dtype=torch.float32)
        batch_actions = torch.tensor(np.array(actions), dtype=torch.float32)
        batch_next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        
        # Train curiosity model
        loss_info = self.curiosity_module.update_curiosity_model(
            batch_states, batch_actions, batch_next_states
        )
        
        return loss_info
    
    def get_rare_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about rare events in the buffer"""
        event_counts = defaultdict(int)
        total_rare_events = 0
        
        for experience in self.experiences[:self.size]:
            if experience.event_rarity >= self.rare_event_threshold:
                event_counts[experience.event_type] += 1
                total_rare_events += 1
        
        return {
            'total_rare_events': total_rare_events,
            'rare_event_ratio': total_rare_events / max(self.size, 1),
            'rare_events_by_type': dict(event_counts),
            'rare_event_threshold': self.rare_event_threshold,
            'buffer_size': self.size,
            'buffer_capacity': self.capacity
        }
    
    def _add_to_rare_event_buffer(self, experience: SchedulingExperience):
        """Add experience to rare event specific buffer"""
        event_type = experience.event_type
        
        # Add to specific event buffer
        self.rare_event_buffer[event_type].append(experience)
        
        # Maintain buffer size limits
        if len(self.rare_event_buffer[event_type]) > self.max_rare_events_per_type:
            # Remove oldest experiences
            self.rare_event_buffer[event_type] = self.rare_event_buffer[event_type][-self.max_rare_events_per_type:]
    
    def _action_to_array(self, action: Dict[str, Any]) -> np.ndarray:
        """Convert action dictionary to numpy array"""
        # Simple conversion - in practice would be more sophisticated
        action_features = [
            action.get('cpu_allocation', 0.0),
            action.get('memory_allocation', 0.0),
            action.get('gpu_allocation', 0.0),
            action.get('priority_adjustment', 0.0),
            1.0 if action.get('migrate', False) else 0.0,
            1.0 if action.get('preempt', False) else 0.0
        ]
        
        # Pad or truncate to fixed size
        target_size = 64  # Should match action_dim in config
        if len(action_features) < target_size:
            action_features.extend([0.0] * (target_size - len(action_features)))
        else:
            action_features = action_features[:target_size]
        
        return np.array(action_features, dtype=np.float32)
    
    def _select_adaptive_sampling_strategy(self) -> PriorityType:
        """Adaptively select sampling strategy based on buffer state"""
        rare_event_ratio = self.get_rare_event_statistics()['rare_event_ratio']
        
        if rare_event_ratio > 0.1:
            return PriorityType.MULTI_OBJECTIVE
        elif self.training_step < 10000:
            return PriorityType.CURIOSITY_DRIVEN
        else:
            return PriorityType.PROPORTIONAL
    
    def _uniform_sampling(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform random sampling"""
        indices = np.random.choice(self.size, batch_size, replace=True)
        weights = np.ones(batch_size) / self.size
        return indices, weights
    
    def _proportional_sampling(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Proportional priority sampling"""
        valid_priorities = self.priorities[:self.size]
        probs = valid_priorities / valid_priorities.sum()
        
        indices = np.random.choice(self.size, batch_size, replace=True, p=probs)
        weights = probs[indices]
        
        return indices, weights
    
    def _rank_based_sampling(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Rank-based priority sampling"""
        # Sort indices by priority
        sorted_indices = np.argsort(self.priorities[:self.size])[::-1]
        
        # Calculate rank-based probabilities
        ranks = np.arange(1, self.size + 1)
        probs = 1.0 / ranks
        probs = probs / probs.sum()
        
        # Sample based on ranks
        sampled_ranks = np.random.choice(self.size, batch_size, replace=True, p=probs)
        indices = sorted_indices[sampled_ranks]
        weights = probs[sampled_ranks]
        
        return indices, weights
    
    def _curiosity_driven_sampling(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Curiosity-driven sampling favoring high prediction error"""
        # Calculate curiosity scores for recent experiences
        curiosity_scores = np.zeros(self.size)
        
        for i in range(self.size):
            exp = self.experiences[i]
            if hasattr(exp, 'state') and hasattr(exp, 'next_state'):
                action_array = self._action_to_array(exp.action)
                curiosity_scores[i] = self.curiosity_module.calculate_intrinsic_reward(
                    exp.state, action_array, exp.next_state
                )
        
        # Combine curiosity with priority
        combined_scores = 0.6 * self.priorities[:self.size] + 0.4 * curiosity_scores
        probs = combined_scores / combined_scores.sum()
        
        indices = np.random.choice(self.size, batch_size, replace=True, p=probs)
        weights = probs[indices]
        
        return indices, weights
    
    def _multi_objective_sampling(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-objective sampling balancing different criteria"""
        # Ensure balanced sampling across event types
        event_type_counts = defaultdict(int)
        for exp in self.experiences[:self.size]:
            event_type_counts[exp.event_type] += 1
        
        # Calculate target samples per event type
        total_event_types = len(event_type_counts)
        if total_event_types == 0:
            return self._proportional_sampling(batch_size)
        
        min_samples_per_type = max(1, batch_size // (total_event_types * 2))
        remaining_samples = batch_size
        selected_indices = []
        
        # Sample minimum from each event type
        for event_type in event_type_counts:
            type_indices = [i for i in range(self.size) 
                          if self.experiences[i].event_type == event_type]
            
            if type_indices:
                type_samples = min(min_samples_per_type, len(type_indices), remaining_samples)
                if type_samples > 0:
                    # Priority-based sampling within event type
                    type_priorities = self.priorities[type_indices]
                    type_probs = type_priorities / type_priorities.sum()
                    
                    sampled = np.random.choice(type_indices, type_samples, 
                                             replace=True, p=type_probs)
                    selected_indices.extend(sampled)
                    remaining_samples -= type_samples
        
        # Fill remaining with proportional sampling
        if remaining_samples > 0:
            remaining_indices, _ = self._proportional_sampling(remaining_samples)
            selected_indices.extend(remaining_indices)
        
        # Calculate weights
        indices = np.array(selected_indices[:batch_size])
        weights = self.priorities[indices] / self.priorities[:self.size].sum()
        
        return indices, weights
    
    def _get_current_beta(self) -> float:
        """Get current beta value with annealing"""
        progress = min(1.0, self.training_step / self.beta_annealing_steps)
        return self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def _update_sampling_statistics(self, experiences: List[SchedulingExperience]):
        """Update sampling statistics"""
        self.sampling_stats['total_samples'] += len(experiences)
        
        for exp in experiences:
            if exp.event_rarity >= self.rare_event_threshold:
                self.sampling_stats['rare_event_samples'] += 1
            
            self.sampling_stats['event_type_samples'][exp.event_type] += 1
            self.sampling_stats['priority_distribution'].append(exp.priority_score)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        rare_stats = self.get_rare_event_statistics()
        
        stats = {
            'buffer_size': self.size,
            'buffer_capacity': self.capacity,
            'position': self.position,
            'training_step': self.training_step,
            'current_beta': self._get_current_beta(),
            
            # Rare event statistics
            'rare_events': rare_stats,
            
            # Sampling statistics
            'sampling_stats': {
                'total_samples': self.sampling_stats['total_samples'],
                'rare_event_samples': self.sampling_stats['rare_event_samples'],
                'rare_event_sample_ratio': (
                    self.sampling_stats['rare_event_samples'] / 
                    max(self.sampling_stats['total_samples'], 1)
                ),
                'event_type_distribution': dict(self.sampling_stats['event_type_samples']),
                'avg_priority': np.mean(self.sampling_stats['priority_distribution']) 
                              if self.sampling_stats['priority_distribution'] else 0.0
            },
            
            # Priority statistics
            'priority_stats': {
                'min_priority': float(np.min(self.priorities[:self.size])) if self.size > 0 else 0.0,
                'max_priority': float(np.max(self.priorities[:self.size])) if self.size > 0 else 0.0,
                'mean_priority': float(np.mean(self.priorities[:self.size])) if self.size > 0 else 0.0,
                'objective_weights': self.priority_calculator.objective_weights.tolist()
            }
        }
        
        return stats

def create_sample_experiences(num_experiences: int = 1000, 
                            rare_event_probability: float = 0.1) -> List[SchedulingExperience]:
    """Create sample scheduling experiences for testing"""
    experiences = []
    
    for i in range(num_experiences):
        # Random state and next state
        state = np.random.rand(128).astype(np.float32)
        next_state = np.random.rand(128).astype(np.float32)
        
        # Random action
        action = {
            'cpu_allocation': np.random.uniform(0.1, 4.0),
            'memory_allocation': np.random.uniform(0.5, 16.0),
            'gpu_allocation': np.random.uniform(0, 2),
            'priority_adjustment': np.random.uniform(-0.5, 0.5),
            'migrate': np.random.random() < 0.2,
            'preempt': np.random.random() < 0.1
        }
        
        # Determine if this should be a rare event
        is_rare_event = np.random.random() < rare_event_probability
        
        if is_rare_event:
            # Create rare event scenario
            if np.random.random() < 0.3:
                # System failure
                reward = -5.0
                constraint_violations = ['system_failure', 'resource_unavailable']
                system_load = 0.95
                pending_tasks = 100
            elif np.random.random() < 0.5:
                # Deadline violation
                reward = -2.0
                constraint_violations = ['deadline_missed']
                system_load = 0.8
                pending_tasks = 50
            else:
                # Resource exhaustion
                reward = -1.5
                constraint_violations = ['resource_exhausted']
                system_load = 0.9
                pending_tasks = 75
        else:
            # Normal execution
            reward = np.random.uniform(-0.5, 1.0)
            constraint_violations = []
            system_load = np.random.uniform(0.2, 0.7)
            pending_tasks = np.random.randint(5, 30)
        
        # Create experience
        experience = SchedulingExperience(
            experience_id=f"exp_{i}",
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=np.random.random() < 0.1,
            timestamp=time.time() - np.random.uniform(0, 3600),
            event_type=EventType.NORMAL_EXECUTION,  # Will be updated by detector
            event_severity=0.0,  # Will be updated by detector
            event_rarity=0.0,   # Will be updated by detector
            objective_rewards={
                'throughput': np.random.uniform(-1, 1),
                'latency': np.random.uniform(-1, 1),
                'energy': np.random.uniform(-1, 1)
            },
            constraint_violations=constraint_violations,
            episode_step=i % 100,
            episode_id=f"episode_{i // 100}",
            system_load=system_load,
            resource_utilization={
                'cpu': np.random.uniform(0.1, 0.9),
                'memory': np.random.uniform(0.2, 0.8),
                'gpu': np.random.uniform(0.0, 0.7)
            },
            pending_tasks=pending_tasks
        )
        
        experiences.append(experience)
    
    return experiences

async def main():
    """Demonstrate prioritized experience replay for rare events"""
    
    print("=== Experience Replay Prioritization for Rare Events ===\n")
    
    # Configuration
    config = {
        'state_dim': 128,
        'action_dim': 64,
        'goal_dim': 8,
        'num_objectives': 3,
        'rare_event_threshold': 0.6,
        'beta_end': 1.0,
        'beta_annealing_steps': 50000,
        'event_detection': {
            'anomaly_threshold': 2.0
        },
        'priority_calculation': {
            'td_error_weight': 0.4,
            'rarity_weight': 0.3,
            'severity_weight': 0.2,
            'curiosity_weight': 0.1
        }
    }
    
    # Create experience replay system
    print("1. Initializing Prioritized Experience Replay System...")
    replay_buffer = PrioritizedExperienceReplay(
        capacity=10000,
        alpha=0.6,
        beta=0.4,
        config=config
    )
    
    print(f"   Buffer capacity: {replay_buffer.capacity:,}")
    print(f"   Rare event threshold: {replay_buffer.rare_event_threshold}")
    print(f"   Priority calculation components: TD error, rarity, severity, curiosity")
    
    # Create sample experiences
    print("2. Creating Sample Experiences...")
    experiences = create_sample_experiences(num_experiences=5000, rare_event_probability=0.15)
    print(f"   Generated {len(experiences)} experiences")
    print(f"   Expected rare events: ~{len(experiences) * 0.15:.0f}")
    
    # Add experiences to buffer
    print("3. Adding Experiences to Buffer...")
    for i, experience in enumerate(experiences):
        # Simulate TD error
        td_error = np.random.uniform(0.1, 2.0, size=config['num_objectives'])
        replay_buffer.add_experience(experience, td_error)
        
        if (i + 1) % 1000 == 0:
            print(f"   Added {i + 1} experiences...")
    
    # Analyze rare events
    print("4. Rare Event Analysis...")
    rare_stats = replay_buffer.get_rare_event_statistics()
    print(f"   Total rare events detected: {rare_stats['total_rare_events']}")
    print(f"   Rare event ratio: {rare_stats['rare_event_ratio']:.3f}")
    print(f"   Rare events by type:")
    for event_type, count in rare_stats['rare_events_by_type'].items():
        print(f"     {event_type.value}: {count}")
    
    # Test different sampling strategies
    print("5. Testing Sampling Strategies...")
    
    sampling_strategies = [
        PriorityType.UNIFORM,
        PriorityType.PROPORTIONAL,
        PriorityType.RANK_BASED,
        PriorityType.CURIOSITY_DRIVEN,
        PriorityType.MULTI_OBJECTIVE
    ]
    
    strategy_results = {}
    batch_size = 64
    
    for strategy in sampling_strategies:
        sampled_experiences, indices, weights = replay_buffer.sample(batch_size, strategy)
        
        # Analyze sampled experiences
        rare_count = sum(1 for exp in sampled_experiences if exp.event_rarity >= config['rare_event_threshold'])
        avg_priority = np.mean([exp.priority_score for exp in sampled_experiences])
        event_type_counts = defaultdict(int)
        
        for exp in sampled_experiences:
            event_type_counts[exp.event_type] += 1
        
        strategy_results[strategy.value] = {
            'rare_events_sampled': rare_count,
            'rare_event_ratio': rare_count / batch_size,
            'avg_priority': avg_priority,
            'event_type_distribution': dict(event_type_counts)
        }
        
        print(f"   {strategy.value}: {rare_count}/{batch_size} rare events, "
              f"avg priority: {avg_priority:.3f}")
    
    # Test hindsight experience replay
    print("6. Testing Hindsight Experience Replay...")
    
    # Create episode data
    episode_experiences = experiences[:50]  # First 50 experiences as an episode
    episode_goals = [{'target_utilization': 0.7, 'target_load': 0.5} for _ in range(10)]
    
    # Add episode with hindsight
    initial_size = replay_buffer.size
    replay_buffer.add_episode(episode_experiences, episode_goals)
    final_size = replay_buffer.size
    
    hindsight_experiences_added = final_size - initial_size - len(episode_experiences)
    print(f"   Original episode experiences: {len(episode_experiences)}")
    print(f"   Hindsight experiences generated: {hindsight_experiences_added}")
    
    # Test curiosity model training
    print("7. Training Curiosity Model...")
    
    curiosity_losses = []
    for epoch in range(10):
        loss_info = replay_buffer.train_curiosity_model(batch_size=32)
        if loss_info:
            curiosity_losses.append(loss_info)
            if epoch % 3 == 0:
                print(f"   Epoch {epoch}: inverse_loss={loss_info['inverse_loss']:.4f}, "
                      f"forward_loss={loss_info['forward_loss']:.4f}")
    
    # Test adaptive sampling
    print("8. Testing Adaptive Sampling...")
    
    adaptive_samples = []
    for _ in range(10):
        sampled_experiences, _, _ = replay_buffer.sample(batch_size, PriorityType.ADAPTIVE)
        rare_count = sum(1 for exp in sampled_experiences if exp.event_rarity >= config['rare_event_threshold'])
        adaptive_samples.append(rare_count)
    
    avg_rare_in_adaptive = np.mean(adaptive_samples)
    print(f"   Average rare events in adaptive sampling: {avg_rare_in_adaptive:.1f}/{batch_size}")
    print(f"   Adaptive sampling efficiency: {avg_rare_in_adaptive / batch_size:.2f}")
    
    # Test priority updates
    print("9. Testing Priority Updates...")
    
    # Sample and update priorities
    sampled_experiences, indices, weights = replay_buffer.sample(batch_size)
    new_td_errors = np.random.uniform(0.1, 3.0, size=(len(indices), config['num_objectives']))
    
    old_priorities = [replay_buffer.experiences[idx].priority_score for idx in indices]
    replay_buffer.update_priorities(indices, new_td_errors)
    new_priorities = [replay_buffer.experiences[idx].priority_score for idx in indices]
    
    priority_change = np.mean(np.abs(np.array(new_priorities) - np.array(old_priorities)))
    print(f"   Average priority change after update: {priority_change:.3f}")
    
    # Performance analysis
    print("10. Performance Analysis...")
    
    # Buffer utilization
    buffer_stats = replay_buffer.get_statistics()
    print(f"   Buffer utilization: {buffer_stats['buffer_size']}/{buffer_stats['buffer_capacity']} "
          f"({buffer_stats['buffer_size']/buffer_stats['buffer_capacity']:.1%})")
    
    # Priority distribution
    priority_stats = buffer_stats['priority_stats']
    print(f"   Priority range: [{priority_stats['min_priority']:.3f}, {priority_stats['max_priority']:.3f}]")
    print(f"   Mean priority: {priority_stats['mean_priority']:.3f}")
    
    # Sampling efficiency
    sampling_stats = buffer_stats['sampling_stats']
    print(f"   Total samples drawn: {sampling_stats['total_samples']:,}")
    print(f"   Rare event sample ratio: {sampling_stats['rare_event_sample_ratio']:.3f}")
    
    # Event type distribution in sampling
    print(f"   Event type sampling distribution:")
    total_event_samples = sum(sampling_stats['event_type_distribution'].values())
    for event_type, count in sampling_stats['event_type_distribution'].items():
        percentage = count / total_event_samples * 100 if total_event_samples > 0 else 0
        print(f"     {event_type.value}: {count} ({percentage:.1f}%)")
    
    # Memory efficiency analysis
    print("11. Memory Efficiency Analysis...")
    
    import sys
    buffer_memory = sys.getsizeof(replay_buffer.experiences) + sys.getsizeof(replay_buffer.priorities)
    per_experience_memory = buffer_memory / max(replay_buffer.size, 1)
    
    print(f"   Buffer memory usage: {buffer_memory / 1024 / 1024:.2f} MB")
    print(f"   Memory per experience: {per_experience_memory:.0f} bytes")
    
    # Rare event discovery effectiveness
    print("12. Rare Event Discovery Effectiveness...")
    
    # Analyze how well the system discovers different types of rare events
    discovered_event_types = set()
    for exp in replay_buffer.experiences[:replay_buffer.size]:
        if exp.event_rarity >= config['rare_event_threshold']:
            discovered_event_types.add(exp.event_type)
    
    print(f"   Discovered rare event types: {len(discovered_event_types)}")
    print(f"   Event types discovered: {[et.value for et in discovered_event_types]}")
    
    # Calculate discovery efficiency
    total_possible_rare_events = len([et for et in EventType if et != EventType.NORMAL_EXECUTION])
    discovery_efficiency = len(discovered_event_types) / total_possible_rare_events
    print(f"   Discovery efficiency: {discovery_efficiency:.2f}")
    
    print(f"\n[SUCCESS] Prioritized Experience Replay for Rare Events R29 Complete!")
    print(f"\nKey Features Demonstrated:")
    print(f"+ Rare event detection and classification with automatic severity scoring")
    print(f"+ Multi-objective priority calculation with temporal decay")
    print(f"+ Hindsight experience replay for counterfactual learning")
    print(f"+ Curiosity-driven exploration for rare event discovery")
    print(f"+ Adaptive sampling strategies based on buffer composition")
    print(f"+ Information-theoretic event importance estimation")
    print(f"+ Balanced sampling ensuring rare event representation")
    print(f"+ Online learning and priority adaptation")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())