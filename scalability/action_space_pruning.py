#!/usr/bin/env python3
"""
Adaptive Action Space Pruning for Efficient Heterogeneous Scheduling

This module implements comprehensive action space pruning techniques to improve
computational efficiency in large-scale heterogeneous scheduling environments.
The system dynamically adapts the action space based on system state, resource
constraints, and learned action value distributions.

Research Innovation: First adaptive action space pruning system specifically
designed for heterogeneous scheduling with multi-objective optimization and
dynamic constraint-aware filtering.

Key Components:
- Dynamic action space reduction based on feasibility analysis
- Value-based action pruning using Q-value distributions
- Constraint-aware filtering for resource and deadline requirements
- Hierarchical action clustering for computational efficiency
- Online learning of action importance and effectiveness
- Multi-objective action ranking and selection

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
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import heapq
import networkx as nx

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class PruningStrategy(Enum):
    """Strategies for action space pruning"""
    FEASIBILITY = "feasibility"
    VALUE_BASED = "value_based"
    CONSTRAINT_AWARE = "constraint_aware"
    HIERARCHICAL = "hierarchical"
    LEARNED = "learned"
    HYBRID = "hybrid"

class ActionType(Enum):
    """Types of scheduling actions"""
    ASSIGN_TASK = "assign_task"
    MIGRATE_TASK = "migrate_task"
    SCALE_RESOURCE = "scale_resource"
    PREEMPT_TASK = "preempt_task"
    DEFER_TASK = "defer_task"
    NO_OPERATION = "no_operation"

@dataclass
class SchedulingAction:
    """Scheduling action with metadata"""
    action_id: str
    action_type: ActionType
    source_node: Optional[str]
    target_node: Optional[str]
    task_id: Optional[str]
    resource_changes: Dict[str, float]
    estimated_cost: float
    estimated_benefit: float
    feasibility_score: float
    constraints: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionPruningConfig:
    """Configuration for action space pruning"""
    max_actions: int = 100
    min_actions: int = 10
    feasibility_threshold: float = 0.5
    value_threshold: float = 0.1
    constraint_violation_penalty: float = 1000.0
    clustering_enabled: bool = True
    num_clusters: int = 20
    online_learning_rate: float = 0.01
    pruning_strategies: List[PruningStrategy] = field(default_factory=lambda: [PruningStrategy.HYBRID])

@dataclass
class SystemState:
    """Current state of the heterogeneous system"""
    nodes: Dict[str, Dict[str, float]]  # node_id -> resource utilization
    tasks: Dict[str, Dict[str, Any]]    # task_id -> task properties
    pending_tasks: List[str]
    constraints: Dict[str, Any]
    topology: Dict[str, List[str]]      # node_id -> neighbor nodes
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeasibilityAnalyzer:
    """Analyze action feasibility based on system constraints"""
    
    def __init__(self, config: ActionPruningConfig):
        self.config = config
        self.resource_types = ['cpu', 'memory', 'gpu', 'storage', 'bandwidth']
        
    def analyze_action_feasibility(self, action: SchedulingAction, state: SystemState) -> float:
        """Analyze feasibility of an action given current state"""
        
        if action.action_type == ActionType.NO_OPERATION:
            return 1.0
        
        feasibility_scores = []
        
        # Resource feasibility
        resource_score = self._check_resource_feasibility(action, state)
        feasibility_scores.append(resource_score)
        
        # Constraint feasibility
        constraint_score = self._check_constraint_feasibility(action, state)
        feasibility_scores.append(constraint_score)
        
        # Topology feasibility
        topology_score = self._check_topology_feasibility(action, state)
        feasibility_scores.append(topology_score)
        
        # Temporal feasibility
        temporal_score = self._check_temporal_feasibility(action, state)
        feasibility_scores.append(temporal_score)
        
        # Combined feasibility score
        return np.mean(feasibility_scores)
    
    def _check_resource_feasibility(self, action: SchedulingAction, state: SystemState) -> float:
        """Check if action is feasible from resource perspective"""
        if not action.target_node or action.target_node not in state.nodes:
            return 0.0
        
        target_resources = state.nodes[action.target_node]
        resource_scores = []
        
        for resource_type in self.resource_types:
            if resource_type in action.resource_changes:
                required = action.resource_changes[resource_type]
                available = target_resources.get(f'{resource_type}_available', 0.0)
                
                if required > 0:  # Resource consumption
                    score = min(1.0, available / max(required, 1e-6))
                else:  # Resource release
                    score = 1.0
                
                resource_scores.append(score)
        
        return np.mean(resource_scores) if resource_scores else 1.0
    
    def _check_constraint_feasibility(self, action: SchedulingAction, state: SystemState) -> float:
        """Check constraint violations"""
        constraint_scores = []
        
        # Check action-specific constraints
        for constraint_name, constraint_value in action.constraints.items():
            if constraint_name == 'deadline':
                deadline_score = self._check_deadline_constraint(action, constraint_value, state)
                constraint_scores.append(deadline_score)
            elif constraint_name == 'affinity':
                affinity_score = self._check_affinity_constraint(action, constraint_value, state)
                constraint_scores.append(affinity_score)
            elif constraint_name == 'anti_affinity':
                anti_affinity_score = self._check_anti_affinity_constraint(action, constraint_value, state)
                constraint_scores.append(anti_affinity_score)
        
        # Check system-level constraints
        system_constraint_score = self._check_system_constraints(action, state)
        constraint_scores.append(system_constraint_score)
        
        return np.mean(constraint_scores) if constraint_scores else 1.0
    
    def _check_topology_feasibility(self, action: SchedulingAction, state: SystemState) -> float:
        """Check topology-related feasibility"""
        if action.action_type in [ActionType.MIGRATE_TASK, ActionType.ASSIGN_TASK]:
            if action.source_node and action.target_node:
                # Check network connectivity and bandwidth
                if action.target_node in state.topology.get(action.source_node, []):
                    return 1.0
                else:
                    # Check if there's a path (might be slower)
                    return 0.7 if self._has_path(action.source_node, action.target_node, state.topology) else 0.3
        
        return 1.0
    
    def _check_temporal_feasibility(self, action: SchedulingAction, state: SystemState) -> float:
        """Check temporal constraints and timing feasibility"""
        if action.task_id and action.task_id in state.tasks:
            task = state.tasks[action.task_id]
            
            if 'deadline' in task:
                time_remaining = task['deadline'] - state.timestamp
                estimated_execution_time = task.get('estimated_duration', 0)
                
                if time_remaining > estimated_execution_time:
                    return 1.0
                elif time_remaining > 0:
                    return time_remaining / estimated_execution_time
                else:
                    return 0.0
        
        return 1.0
    
    def _check_deadline_constraint(self, action: SchedulingAction, deadline: float, state: SystemState) -> float:
        """Check deadline constraint"""
        time_remaining = deadline - state.timestamp
        if time_remaining <= 0:
            return 0.0
        
        # Estimate execution time based on action
        estimated_time = action.metadata.get('estimated_execution_time', 1.0)
        return min(1.0, time_remaining / estimated_time)
    
    def _check_affinity_constraint(self, action: SchedulingAction, affinity_rules: List[str], state: SystemState) -> float:
        """Check affinity constraints"""
        if not action.target_node:
            return 1.0
        
        target_node = state.nodes.get(action.target_node, {})
        
        for rule in affinity_rules:
            if rule in target_node.get('labels', []):
                return 1.0
        
        return 0.0 if affinity_rules else 1.0
    
    def _check_anti_affinity_constraint(self, action: SchedulingAction, anti_affinity_rules: List[str], state: SystemState) -> float:
        """Check anti-affinity constraints"""
        if not action.target_node:
            return 1.0
        
        target_node = state.nodes.get(action.target_node, {})
        
        for rule in anti_affinity_rules:
            if rule in target_node.get('labels', []):
                return 0.0
        
        return 1.0
    
    def _check_system_constraints(self, action: SchedulingAction, state: SystemState) -> float:
        """Check system-level constraints"""
        # Check overall system load
        system_load = state.metadata.get('system_load', 0.5)
        if system_load > 0.9 and action.action_type in [ActionType.ASSIGN_TASK, ActionType.SCALE_RESOURCE]:
            return 0.3
        
        # Check maintenance windows
        if state.metadata.get('maintenance_mode', False):
            if action.action_type in [ActionType.ASSIGN_TASK, ActionType.SCALE_RESOURCE]:
                return 0.1
        
        return 1.0
    
    def _has_path(self, source: str, target: str, topology: Dict[str, List[str]]) -> bool:
        """Check if there's a path between source and target nodes"""
        if source == target:
            return True
        
        visited = set()
        queue = [source]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == target:
                return True
            
            for neighbor in topology.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return False

class ValueBasedPruner:
    """Prune actions based on learned value estimates"""
    
    def __init__(self, config: ActionPruningConfig):
        self.config = config
        self.action_values = defaultdict(deque)  # action_type -> recent values
        self.action_success_rates = defaultdict(float)
        self.value_threshold = config.value_threshold
        
    def estimate_action_values(self, actions: List[SchedulingAction], state: SystemState) -> Dict[str, float]:
        """Estimate Q-values for actions"""
        values = {}
        
        for action in actions:
            # Base value from benefit-cost ratio
            base_value = action.estimated_benefit / max(action.estimated_cost, 1e-6)
            
            # Historical performance adjustment
            historical_value = self._get_historical_value(action)
            
            # State-dependent value adjustment
            state_value = self._get_state_dependent_value(action, state)
            
            # Combined value
            combined_value = 0.5 * base_value + 0.3 * historical_value + 0.2 * state_value
            values[action.action_id] = combined_value
        
        return values
    
    def prune_by_value(self, actions: List[SchedulingAction], state: SystemState, max_actions: int) -> List[SchedulingAction]:
        """Prune actions based on estimated values"""
        if len(actions) <= max_actions:
            return actions
        
        # Estimate values
        values = self.estimate_action_values(actions, state)
        
        # Sort by value and take top actions
        sorted_actions = sorted(actions, key=lambda a: values.get(a.action_id, 0), reverse=True)
        
        # Ensure we keep at least min_actions
        keep_count = max(self.config.min_actions, min(max_actions, len(sorted_actions)))
        
        return sorted_actions[:keep_count]
    
    def update_action_value(self, action_id: str, action_type: ActionType, reward: float, success: bool):
        """Update action value based on outcome"""
        # Update value history
        self.action_values[action_type].append(reward)
        
        # Keep limited history
        if len(self.action_values[action_type]) > 100:
            self.action_values[action_type].popleft()
        
        # Update success rate
        current_rate = self.action_success_rates[action_type]
        learning_rate = self.config.online_learning_rate
        
        new_rate = current_rate * (1 - learning_rate) + (1.0 if success else 0.0) * learning_rate
        self.action_success_rates[action_type] = new_rate
    
    def _get_historical_value(self, action: SchedulingAction) -> float:
        """Get historical value for action type"""
        action_type = action.action_type
        
        if action_type not in self.action_values or not self.action_values[action_type]:
            return 0.5  # Default neutral value
        
        recent_values = list(self.action_values[action_type])
        return np.mean(recent_values[-10:])  # Use recent 10 values
    
    def _get_state_dependent_value(self, action: SchedulingAction, state: SystemState) -> float:
        """Get state-dependent value adjustment"""
        value_adjustments = []
        
        # System load adjustment
        system_load = state.metadata.get('system_load', 0.5)
        if action.action_type == ActionType.ASSIGN_TASK:
            load_adjustment = 1.0 - system_load  # Prefer assignment when load is low
            value_adjustments.append(load_adjustment)
        elif action.action_type == ActionType.MIGRATE_TASK:
            load_adjustment = system_load  # Prefer migration when load is high
            value_adjustments.append(load_adjustment)
        
        # Queue length adjustment
        queue_length = len(state.pending_tasks)
        if queue_length > 10:
            urgency_adjustment = 1.2 if action.action_type == ActionType.ASSIGN_TASK else 0.8
            value_adjustments.append(urgency_adjustment)
        
        # Resource utilization adjustment
        if action.target_node and action.target_node in state.nodes:
            node_utilization = state.nodes[action.target_node].get('cpu_utilization', 0.5)
            util_adjustment = 1.0 - node_utilization  # Prefer less utilized nodes
            value_adjustments.append(util_adjustment)
        
        return np.mean(value_adjustments) if value_adjustments else 0.5

class HierarchicalActionClusterer:
    """Hierarchical clustering of actions for efficient pruning"""
    
    def __init__(self, config: ActionPruningConfig):
        self.config = config
        self.num_clusters = config.num_clusters
        self.scaler = StandardScaler()
        self.cluster_representatives = {}
        self.cluster_centers = None
        
    def cluster_actions(self, actions: List[SchedulingAction]) -> Dict[int, List[SchedulingAction]]:
        """Cluster actions into groups"""
        if len(actions) <= self.num_clusters:
            # Return each action as its own cluster
            return {i: [action] for i, action in enumerate(actions)}
        
        # Extract features for clustering
        features = self._extract_action_features(actions)
        
        if features.shape[0] == 0:
            return {0: actions}
        
        # Normalize features
        normalized_features = self.scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(self.num_clusters, len(actions)), random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_features)
        
        # Store cluster centers
        self.cluster_centers = kmeans.cluster_centers_
        
        # Group actions by cluster
        clusters = defaultdict(list)
        for action, label in zip(actions, cluster_labels):
            clusters[label].append(action)
        
        return dict(clusters)
    
    def select_cluster_representatives(self, clusters: Dict[int, List[SchedulingAction]]) -> List[SchedulingAction]:
        """Select representative actions from each cluster"""
        representatives = []
        
        for cluster_id, cluster_actions in clusters.items():
            if not cluster_actions:
                continue
            
            # Select representative based on multiple criteria
            representative = self._select_best_representative(cluster_actions)
            representatives.append(representative)
            
            # Store for future reference
            self.cluster_representatives[cluster_id] = representative
        
        return representatives
    
    def _extract_action_features(self, actions: List[SchedulingAction]) -> np.ndarray:
        """Extract numerical features from actions for clustering"""
        features = []
        
        for action in actions:
            feature_vector = [
                # Action type (one-hot encoded)
                1.0 if action.action_type == ActionType.ASSIGN_TASK else 0.0,
                1.0 if action.action_type == ActionType.MIGRATE_TASK else 0.0,
                1.0 if action.action_type == ActionType.SCALE_RESOURCE else 0.0,
                1.0 if action.action_type == ActionType.PREEMPT_TASK else 0.0,
                1.0 if action.action_type == ActionType.DEFER_TASK else 0.0,
                
                # Cost and benefit
                action.estimated_cost,
                action.estimated_benefit,
                action.estimated_benefit / max(action.estimated_cost, 1e-6),  # Benefit-cost ratio
                
                # Feasibility
                action.feasibility_score,
                
                # Resource changes
                action.resource_changes.get('cpu', 0.0),
                action.resource_changes.get('memory', 0.0),
                action.resource_changes.get('gpu', 0.0),
                action.resource_changes.get('storage', 0.0),
                action.resource_changes.get('bandwidth', 0.0),
                
                # Constraint complexity
                len(action.constraints),
                
                # Metadata features
                action.metadata.get('priority', 0.5),
                action.metadata.get('urgency', 0.5),
                action.metadata.get('estimated_execution_time', 1.0)
            ]
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _select_best_representative(self, cluster_actions: List[SchedulingAction]) -> SchedulingAction:
        """Select the best representative from a cluster"""
        if len(cluster_actions) == 1:
            return cluster_actions[0]
        
        # Score each action based on multiple criteria
        scores = []
        
        for action in cluster_actions:
            score = (
                0.4 * action.feasibility_score +
                0.3 * (action.estimated_benefit / max(action.estimated_cost, 1e-6)) +
                0.2 * action.metadata.get('priority', 0.5) +
                0.1 * action.metadata.get('urgency', 0.5)
            )
            scores.append(score)
        
        # Return action with highest score
        best_index = np.argmax(scores)
        return cluster_actions[best_index]

class LearnedActionPruner:
    """Learn action pruning patterns from historical data"""
    
    def __init__(self, config: ActionPruningConfig):
        self.config = config
        self.action_importance_model = self._build_importance_model()
        self.training_data = []
        
    def _build_importance_model(self) -> nn.Module:
        """Build neural network to predict action importance"""
        model = nn.Sequential(
            nn.Linear(18, 64),  # Input: action features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),   # Output: importance score
            nn.Sigmoid()
        )
        return model
    
    def predict_action_importance(self, actions: List[SchedulingAction], state: SystemState) -> Dict[str, float]:
        """Predict importance scores for actions"""
        if not actions:
            return {}
        
        # Extract features
        features = self._extract_learning_features(actions, state)
        
        # Predict importance
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32)
            importance_scores = self.action_importance_model(features_tensor)
            importance_scores = importance_scores.squeeze().numpy()
        
        # Return as dictionary
        if len(actions) == 1:
            importance_scores = [importance_scores]
        
        return {action.action_id: float(score) for action, score in zip(actions, importance_scores)}
    
    def add_training_example(self, action: SchedulingAction, state: SystemState, outcome: float):
        """Add training example for learning"""
        features = self._extract_learning_features([action], state)[0]
        self.training_data.append((features, outcome))
        
        # Keep limited training data
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-10000:]
    
    def train_model(self, epochs: int = 100, batch_size: int = 32):
        """Train the importance prediction model"""
        if len(self.training_data) < batch_size:
            return
        
        # Prepare training data
        features, targets = zip(*self.training_data)
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
        # Training setup
        optimizer = torch.optim.Adam(self.action_importance_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(features))
            features_shuffled = features[indices]
            targets_shuffled = targets[indices]
            
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(features), batch_size):
                batch_features = features_shuffled[i:i+batch_size]
                batch_targets = targets_shuffled[i:i+batch_size]
                
                # Forward pass
                predictions = self.action_importance_model(batch_features)
                loss = criterion(predictions, batch_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if epoch % 20 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                logger.info(f"Training epoch {epoch}, avg loss: {avg_loss:.4f}")
    
    def _extract_learning_features(self, actions: List[SchedulingAction], state: SystemState) -> np.ndarray:
        """Extract features for learning importance prediction"""
        features = []
        
        for action in actions:
            feature_vector = [
                # Basic action properties
                action.estimated_cost,
                action.estimated_benefit,
                action.feasibility_score,
                
                # Action type encoding
                1.0 if action.action_type == ActionType.ASSIGN_TASK else 0.0,
                1.0 if action.action_type == ActionType.MIGRATE_TASK else 0.0,
                1.0 if action.action_type == ActionType.SCALE_RESOURCE else 0.0,
                
                # Resource requirements
                action.resource_changes.get('cpu', 0.0),
                action.resource_changes.get('memory', 0.0),
                action.resource_changes.get('gpu', 0.0),
                
                # System state context
                len(state.pending_tasks),
                state.metadata.get('system_load', 0.5),
                len(state.nodes),
                
                # Temporal context
                state.timestamp % (24 * 3600),  # Time of day
                
                # Constraint complexity
                len(action.constraints),
                
                # Target node utilization (if available)
                state.nodes.get(action.target_node, {}).get('cpu_utilization', 0.5) if action.target_node else 0.5,
                
                # Queue pressure
                min(1.0, len(state.pending_tasks) / 100.0),
                
                # Priority and urgency
                action.metadata.get('priority', 0.5),
                action.metadata.get('urgency', 0.5)
            ]
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)

class AdaptiveActionSpacePruner:
    """Main adaptive action space pruning system"""
    
    def __init__(self, config: ActionPruningConfig = None):
        self.config = config or ActionPruningConfig()
        
        # Initialize pruning components
        self.feasibility_analyzer = FeasibilityAnalyzer(self.config)
        self.value_pruner = ValueBasedPruner(self.config)
        self.hierarchical_clusterer = HierarchicalActionClusterer(self.config)
        self.learned_pruner = LearnedActionPruner(self.config)
        
        # Statistics
        self.pruning_stats = {
            'total_actions_considered': 0,
            'total_actions_pruned': 0,
            'pruning_time_history': deque(maxlen=1000),
            'strategy_usage': defaultdict(int),
            'effectiveness_scores': defaultdict(list)
        }
        
        # Dynamic thresholds
        self.dynamic_thresholds = {
            'feasibility': self.config.feasibility_threshold,
            'value': self.config.value_threshold
        }
        
    def prune_action_space(self, actions: List[SchedulingAction], state: SystemState, 
                          strategy: PruningStrategy = None) -> Tuple[List[SchedulingAction], Dict[str, Any]]:
        """Main action space pruning method"""
        start_time = time.time()
        
        if not actions:
            return [], {'pruning_time': 0.0, 'strategy': 'none', 'original_size': 0, 'pruned_size': 0}
        
        # Select strategy
        if strategy is None:
            strategy = self._select_best_strategy(actions, state)
        
        # Update statistics
        self.pruning_stats['total_actions_considered'] += len(actions)
        self.pruning_stats['strategy_usage'][strategy] += 1
        
        # Apply pruning strategy
        if strategy == PruningStrategy.FEASIBILITY:
            pruned_actions = self._feasibility_pruning(actions, state)
        elif strategy == PruningStrategy.VALUE_BASED:
            pruned_actions = self._value_based_pruning(actions, state)
        elif strategy == PruningStrategy.CONSTRAINT_AWARE:
            pruned_actions = self._constraint_aware_pruning(actions, state)
        elif strategy == PruningStrategy.HIERARCHICAL:
            pruned_actions = self._hierarchical_pruning(actions, state)
        elif strategy == PruningStrategy.LEARNED:
            pruned_actions = self._learned_pruning(actions, state)
        else:  # HYBRID
            pruned_actions = self._hybrid_pruning(actions, state)
        
        # Update statistics
        pruning_time = time.time() - start_time
        self.pruning_stats['total_actions_pruned'] += len(actions) - len(pruned_actions)
        self.pruning_stats['pruning_time_history'].append(pruning_time)
        
        # Prepare metadata
        metadata = {
            'pruning_time': pruning_time,
            'strategy': strategy.value,
            'original_size': len(actions),
            'pruned_size': len(pruned_actions),
            'pruning_ratio': 1.0 - len(pruned_actions) / len(actions),
            'dynamic_thresholds': self.dynamic_thresholds.copy()
        }
        
        return pruned_actions, metadata
    
    def _select_best_strategy(self, actions: List[SchedulingAction], state: SystemState) -> PruningStrategy:
        """Adaptively select the best pruning strategy"""
        # Factors to consider
        action_count = len(actions)
        system_load = state.metadata.get('system_load', 0.5)
        queue_length = len(state.pending_tasks)
        
        # Strategy selection logic
        if action_count < 20:
            return PruningStrategy.FEASIBILITY  # Simple for small action spaces
        elif system_load > 0.8:
            return PruningStrategy.HIERARCHICAL  # Fast for high load
        elif queue_length > 50:
            return PruningStrategy.VALUE_BASED  # Focus on high-value actions
        elif len(self.learned_pruner.training_data) > 1000:
            return PruningStrategy.LEARNED  # Use learned model if enough data
        else:
            return PruningStrategy.HYBRID  # Default to hybrid approach
    
    def _feasibility_pruning(self, actions: List[SchedulingAction], state: SystemState) -> List[SchedulingAction]:
        """Prune based on feasibility analysis"""
        feasible_actions = []
        
        for action in actions:
            feasibility = self.feasibility_analyzer.analyze_action_feasibility(action, state)
            action.feasibility_score = feasibility
            
            if feasibility >= self.dynamic_thresholds['feasibility']:
                feasible_actions.append(action)
        
        # Ensure minimum actions
        if len(feasible_actions) < self.config.min_actions:
            # Sort by feasibility and take top min_actions
            sorted_actions = sorted(actions, key=lambda a: a.feasibility_score, reverse=True)
            feasible_actions = sorted_actions[:self.config.min_actions]
        
        return feasible_actions[:self.config.max_actions]
    
    def _value_based_pruning(self, actions: List[SchedulingAction], state: SystemState) -> List[SchedulingAction]:
        """Prune based on estimated action values"""
        return self.value_pruner.prune_by_value(actions, state, self.config.max_actions)
    
    def _constraint_aware_pruning(self, actions: List[SchedulingAction], state: SystemState) -> List[SchedulingAction]:
        """Prune based on constraint satisfaction"""
        valid_actions = []
        
        for action in actions:
            # Check hard constraints
            if self._satisfies_hard_constraints(action, state):
                valid_actions.append(action)
        
        # If too many actions, apply soft constraint ranking
        if len(valid_actions) > self.config.max_actions:
            constraint_scores = []
            for action in valid_actions:
                score = self._calculate_constraint_score(action, state)
                constraint_scores.append((score, action))
            
            # Sort by constraint score and take top actions
            constraint_scores.sort(reverse=True)
            valid_actions = [action for _, action in constraint_scores[:self.config.max_actions]]
        
        return valid_actions
    
    def _hierarchical_pruning(self, actions: List[SchedulingAction], state: SystemState) -> List[SchedulingAction]:
        """Prune using hierarchical clustering"""
        if not self.config.clustering_enabled:
            return actions[:self.config.max_actions]
        
        # Cluster actions
        clusters = self.hierarchical_clusterer.cluster_actions(actions)
        
        # Select representatives
        representatives = self.hierarchical_clusterer.select_cluster_representatives(clusters)
        
        # Limit to max_actions
        return representatives[:self.config.max_actions]
    
    def _learned_pruning(self, actions: List[SchedulingAction], state: SystemState) -> List[SchedulingAction]:
        """Prune using learned importance model"""
        if len(self.learned_pruner.training_data) < 100:
            # Fall back to value-based pruning if insufficient training data
            return self._value_based_pruning(actions, state)
        
        # Predict importance scores
        importance_scores = self.learned_pruner.predict_action_importance(actions, state)
        
        # Sort by importance and take top actions
        sorted_actions = sorted(actions, key=lambda a: importance_scores.get(a.action_id, 0), reverse=True)
        
        return sorted_actions[:self.config.max_actions]
    
    def _hybrid_pruning(self, actions: List[SchedulingAction], state: SystemState) -> List[SchedulingAction]:
        """Hybrid pruning combining multiple strategies"""
        # Step 1: Feasibility filtering
        feasible_actions = []
        for action in actions:
            feasibility = self.feasibility_analyzer.analyze_action_feasibility(action, state)
            action.feasibility_score = feasibility
            
            if feasibility >= 0.3:  # Lower threshold for hybrid
                feasible_actions.append(action)
        
        if not feasible_actions:
            feasible_actions = actions  # Keep all if none are feasible
        
        # Step 2: Value-based filtering
        if len(feasible_actions) > self.config.max_actions * 2:
            value_filtered = self.value_pruner.prune_by_value(
                feasible_actions, state, self.config.max_actions * 2
            )
        else:
            value_filtered = feasible_actions
        
        # Step 3: Constraint-aware final selection
        if len(value_filtered) > self.config.max_actions:
            final_actions = []
            constraint_scores = []
            
            for action in value_filtered:
                score = (
                    0.4 * action.feasibility_score +
                    0.3 * (action.estimated_benefit / max(action.estimated_cost, 1e-6)) +
                    0.3 * self._calculate_constraint_score(action, state)
                )
                constraint_scores.append((score, action))
            
            constraint_scores.sort(reverse=True)
            final_actions = [action for _, action in constraint_scores[:self.config.max_actions]]
        else:
            final_actions = value_filtered
        
        return final_actions
    
    def _satisfies_hard_constraints(self, action: SchedulingAction, state: SystemState) -> bool:
        """Check if action satisfies hard constraints"""
        # Resource constraints
        if action.target_node and action.target_node in state.nodes:
            target_resources = state.nodes[action.target_node]
            
            for resource, required in action.resource_changes.items():
                if required > 0:  # Resource consumption
                    available = target_resources.get(f'{resource}_available', 0)
                    if required > available:
                        return False
        
        # Deadline constraints
        if 'deadline' in action.constraints:
            deadline = action.constraints['deadline']
            if deadline <= state.timestamp:
                return False
        
        return True
    
    def _calculate_constraint_score(self, action: SchedulingAction, state: SystemState) -> float:
        """Calculate soft constraint satisfaction score"""
        scores = []
        
        # Resource efficiency score
        if action.target_node and action.target_node in state.nodes:
            target_resources = state.nodes[action.target_node]
            utilization = target_resources.get('cpu_utilization', 0.5)
            efficiency_score = 1.0 - abs(utilization - 0.7)  # Prefer ~70% utilization
            scores.append(efficiency_score)
        
        # Affinity score
        if 'affinity' in action.constraints:
            affinity_score = 1.0 if action.target_node else 0.5
            scores.append(affinity_score)
        
        # Priority score
        priority = action.metadata.get('priority', 0.5)
        scores.append(priority)
        
        return np.mean(scores) if scores else 0.5
    
    def update_from_outcome(self, action: SchedulingAction, outcome: Dict[str, Any]):
        """Update pruning system based on action outcome"""
        success = outcome.get('success', False)
        reward = outcome.get('reward', 0.0)
        execution_time = outcome.get('execution_time', 0.0)
        
        # Update value-based pruner
        self.value_pruner.update_action_value(
            action.action_id, action.action_type, reward, success
        )
        
        # Update learned pruner (if we have the state context)
        if 'state' in outcome:
            self.learned_pruner.add_training_example(
                action, outcome['state'], reward
            )
        
        # Update dynamic thresholds
        self._update_dynamic_thresholds(success, reward)
    
    def _update_dynamic_thresholds(self, success: bool, reward: float):
        """Update dynamic thresholds based on outcomes"""
        learning_rate = 0.01
        
        if success and reward > 0.5:
            # Good outcome - can be more selective
            self.dynamic_thresholds['feasibility'] = min(
                0.9, self.dynamic_thresholds['feasibility'] + learning_rate
            )
            self.dynamic_thresholds['value'] = min(
                0.5, self.dynamic_thresholds['value'] + learning_rate
            )
        elif not success or reward < 0.1:
            # Poor outcome - be less selective
            self.dynamic_thresholds['feasibility'] = max(
                0.1, self.dynamic_thresholds['feasibility'] - learning_rate
            )
            self.dynamic_thresholds['value'] = max(
                0.01, self.dynamic_thresholds['value'] - learning_rate
            )
    
    def train_learned_components(self):
        """Train learned components of the pruning system"""
        if len(self.learned_pruner.training_data) > 100:
            self.learned_pruner.train_model(epochs=50, batch_size=16)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pruning statistics"""
        total_considered = self.pruning_stats['total_actions_considered']
        total_pruned = self.pruning_stats['total_actions_pruned']
        
        stats = {
            'total_actions_considered': total_considered,
            'total_actions_pruned': total_pruned,
            'overall_pruning_ratio': total_pruned / max(total_considered, 1),
            'avg_pruning_time': np.mean(self.pruning_stats['pruning_time_history']) if self.pruning_stats['pruning_time_history'] else 0.0,
            'strategy_usage': dict(self.pruning_stats['strategy_usage']),
            'dynamic_thresholds': self.dynamic_thresholds.copy(),
            'training_data_size': len(self.learned_pruner.training_data),
            'config': {
                'max_actions': self.config.max_actions,
                'min_actions': self.config.min_actions,
                'clustering_enabled': self.config.clustering_enabled,
                'num_clusters': self.config.num_clusters
            }
        }
        
        return stats

def create_sample_actions(num_actions: int = 50, state: SystemState = None) -> List[SchedulingAction]:
    """Create sample scheduling actions for testing"""
    actions = []
    action_types = list(ActionType)
    
    nodes = list(state.nodes.keys()) if state else [f'node_{i}' for i in range(10)]
    tasks = list(state.tasks.keys()) if state else [f'task_{i}' for i in range(20)]
    
    for i in range(num_actions):
        action_type = np.random.choice(action_types)
        
        # Generate resource changes based on action type
        if action_type == ActionType.ASSIGN_TASK:
            resource_changes = {
                'cpu': np.random.uniform(0.5, 4.0),
                'memory': np.random.uniform(1.0, 8.0),
                'gpu': np.random.uniform(0, 2),
            }
        elif action_type == ActionType.MIGRATE_TASK:
            resource_changes = {
                'cpu': np.random.uniform(-2.0, 2.0),
                'memory': np.random.uniform(-4.0, 4.0),
                'bandwidth': np.random.uniform(10, 100)
            }
        elif action_type == ActionType.SCALE_RESOURCE:
            resource_changes = {
                'cpu': np.random.uniform(-1.0, 2.0),
                'memory': np.random.uniform(-2.0, 4.0)
            }
        else:
            resource_changes = {}
        
        # Generate cost and benefit
        estimated_cost = np.random.uniform(0.1, 10.0)
        estimated_benefit = np.random.uniform(0.5, 15.0)
        
        # Generate constraints
        constraints = {}
        if np.random.random() < 0.3:
            constraints['deadline'] = time.time() + np.random.uniform(60, 3600)
        if np.random.random() < 0.2:
            constraints['affinity'] = [f'label_{np.random.randint(0, 5)}']
        
        action = SchedulingAction(
            action_id=f'action_{i}',
            action_type=action_type,
            source_node=np.random.choice(nodes) if action_type == ActionType.MIGRATE_TASK else None,
            target_node=np.random.choice(nodes),
            task_id=np.random.choice(tasks) if action_type in [ActionType.ASSIGN_TASK, ActionType.MIGRATE_TASK] else None,
            resource_changes=resource_changes,
            estimated_cost=estimated_cost,
            estimated_benefit=estimated_benefit,
            feasibility_score=np.random.uniform(0.1, 1.0),
            constraints=constraints,
            metadata={
                'priority': np.random.uniform(0.1, 1.0),
                'urgency': np.random.uniform(0.1, 1.0),
                'estimated_execution_time': np.random.uniform(1.0, 60.0)
            }
        )
        
        actions.append(action)
    
    return actions

def create_sample_system_state() -> SystemState:
    """Create sample system state for testing"""
    nodes = {}
    for i in range(15):
        nodes[f'node_{i}'] = {
            'cpu_utilization': np.random.uniform(0.2, 0.9),
            'memory_utilization': np.random.uniform(0.3, 0.8),
            'gpu_utilization': np.random.uniform(0.0, 0.7),
            'cpu_available': np.random.uniform(1.0, 8.0),
            'memory_available': np.random.uniform(2.0, 32.0),
            'gpu_available': np.random.uniform(0, 4),
            'storage_available': np.random.uniform(10.0, 1000.0),
            'bandwidth_available': np.random.uniform(100, 10000),
            'labels': [f'label_{np.random.randint(0, 5)}' for _ in range(np.random.randint(0, 3))]
        }
    
    tasks = {}
    pending_tasks = []
    for i in range(25):
        task_id = f'task_{i}'
        tasks[task_id] = {
            'cpu_requirement': np.random.uniform(0.5, 4.0),
            'memory_requirement': np.random.uniform(1.0, 16.0),
            'gpu_requirement': np.random.uniform(0, 2),
            'estimated_duration': np.random.uniform(5.0, 300.0),
            'deadline': time.time() + np.random.uniform(60, 3600),
            'priority': np.random.uniform(0.1, 1.0)
        }
        if i < 15:  # Some tasks are pending
            pending_tasks.append(task_id)
    
    # Create topology (simple mesh)
    topology = {}
    for i in range(15):
        neighbors = []
        for j in range(15):
            if i != j and np.random.random() < 0.3:
                neighbors.append(f'node_{j}')
        topology[f'node_{i}'] = neighbors
    
    return SystemState(
        nodes=nodes,
        tasks=tasks,
        pending_tasks=pending_tasks,
        constraints={'system_load_limit': 0.9},
        topology=topology,
        timestamp=time.time(),
        metadata={
            'system_load': np.random.uniform(0.3, 0.8),
            'maintenance_mode': False
        }
    )

async def main():
    """Demonstrate adaptive action space pruning system"""
    
    print("=== Adaptive Action Space Pruning for Efficient Scheduling ===\n")
    
    # Configuration
    config = ActionPruningConfig(
        max_actions=20,
        min_actions=5,
        feasibility_threshold=0.6,
        value_threshold=0.2,
        clustering_enabled=True,
        num_clusters=10,
        pruning_strategies=[PruningStrategy.HYBRID]
    )
    
    # Create pruning system
    print("1. Initializing Adaptive Action Space Pruner...")
    pruner = AdaptiveActionSpacePruner(config)
    print(f"   Max actions: {config.max_actions}")
    print(f"   Min actions: {config.min_actions}")
    print(f"   Clustering enabled: {config.clustering_enabled}")
    
    # Create sample system state
    print("2. Creating Sample System State...")
    state = create_sample_system_state()
    print(f"   Nodes: {len(state.nodes)}")
    print(f"   Tasks: {len(state.tasks)} total, {len(state.pending_tasks)} pending")
    print(f"   System load: {state.metadata['system_load']:.2f}")
    
    # Test different pruning strategies
    print("3. Testing Different Pruning Strategies...")
    
    strategies = [
        PruningStrategy.FEASIBILITY,
        PruningStrategy.VALUE_BASED,
        PruningStrategy.CONSTRAINT_AWARE,
        PruningStrategy.HIERARCHICAL,
        PruningStrategy.HYBRID
    ]
    
    # Create large action space
    large_action_set = create_sample_actions(200, state)
    print(f"   Original action space size: {len(large_action_set)}")
    
    strategy_results = {}
    for strategy in strategies:
        pruned_actions, metadata = pruner.prune_action_space(large_action_set, state, strategy)
        strategy_results[strategy.value] = {
            'pruned_size': len(pruned_actions),
            'pruning_time': metadata['pruning_time'],
            'pruning_ratio': metadata['pruning_ratio']
        }
        
        print(f"   {strategy.value}: {len(pruned_actions)} actions, "
              f"{metadata['pruning_time']:.4f}s, "
              f"{metadata['pruning_ratio']:.2f} pruning ratio")
    
    # Test adaptive strategy selection
    print("4. Testing Adaptive Strategy Selection...")
    
    # Test under different system conditions
    conditions = [
        {'system_load': 0.3, 'pending_tasks': 5, 'description': 'Low load, few tasks'},
        {'system_load': 0.8, 'pending_tasks': 50, 'description': 'High load, many tasks'},
        {'system_load': 0.6, 'pending_tasks': 25, 'description': 'Medium load, medium tasks'}
    ]
    
    for condition in conditions:
        # Modify state
        test_state = SystemState(
            nodes=state.nodes,
            tasks=state.tasks,
            pending_tasks=[f'task_{i}' for i in range(condition['pending_tasks'])],
            constraints=state.constraints,
            topology=state.topology,
            timestamp=state.timestamp,
            metadata={'system_load': condition['system_load']}
        )
        
        # Test pruning
        pruned_actions, metadata = pruner.prune_action_space(large_action_set, test_state)
        print(f"   {condition['description']}: {metadata['strategy']} strategy, "
              f"{len(pruned_actions)} actions")
    
    # Test learning and adaptation
    print("5. Testing Learning and Adaptation...")
    
    # Simulate training episodes
    for episode in range(50):
        # Create episode-specific actions and state
        episode_actions = create_sample_actions(100, state)
        
        # Prune actions
        pruned_actions, _ = pruner.prune_action_space(episode_actions, state)
        
        # Simulate outcomes and feedback
        for action in pruned_actions[:5]:  # Simulate executing top 5 actions
            # Simulate outcome
            success = np.random.random() > 0.3
            reward = np.random.uniform(0, 1) if success else np.random.uniform(-0.5, 0.2)
            
            outcome = {
                'success': success,
                'reward': reward,
                'execution_time': np.random.uniform(1, 30),
                'state': state
            }
            
            # Update pruner
            pruner.update_from_outcome(action, outcome)
        
        # Train learned components periodically
        if episode % 10 == 0:
            pruner.train_learned_components()
    
    print(f"   Completed {50} training episodes")
    
    # Test scalability
    print("6. Scalability Testing...")
    
    action_sizes = [50, 100, 500, 1000, 2000]
    
    for size in action_sizes:
        test_actions = create_sample_actions(size, state)
        
        start_time = time.time()
        pruned_actions, metadata = pruner.prune_action_space(test_actions, state)
        pruning_time = time.time() - start_time
        
        print(f"   {size} actions -> {len(pruned_actions)} actions in {pruning_time:.4f}s")
    
    # Test clustering effectiveness
    print("7. Clustering Analysis...")
    
    if config.clustering_enabled:
        test_actions = create_sample_actions(100, state)
        clusters = pruner.hierarchical_clusterer.cluster_actions(test_actions)
        representatives = pruner.hierarchical_clusterer.select_cluster_representatives(clusters)
        
        print(f"   {len(test_actions)} actions clustered into {len(clusters)} clusters")
        print(f"   {len(representatives)} representative actions selected")
        
        # Analyze cluster sizes
        cluster_sizes = [len(actions) for actions in clusters.values()]
        print(f"   Cluster sizes: avg={np.mean(cluster_sizes):.1f}, "
              f"std={np.std(cluster_sizes):.1f}")
    
    # Test constraint satisfaction
    print("8. Constraint Satisfaction Analysis...")
    
    # Create actions with varying constraint complexity
    constrained_actions = []
    for i in range(50):
        action = create_sample_actions(1, state)[0]
        
        # Add various constraints
        if i % 3 == 0:
            action.constraints['deadline'] = time.time() + np.random.uniform(10, 600)
        if i % 4 == 0:
            action.constraints['affinity'] = [f'label_{np.random.randint(0, 3)}']
        if i % 5 == 0:
            action.constraints['anti_affinity'] = [f'label_{np.random.randint(0, 3)}']
        
        constrained_actions.append(action)
    
    # Test constraint-aware pruning
    constraint_pruned, metadata = pruner.prune_action_space(
        constrained_actions, state, PruningStrategy.CONSTRAINT_AWARE
    )
    
    print(f"   {len(constrained_actions)} constrained actions -> {len(constraint_pruned)} feasible")
    
    # Analyze constraint satisfaction
    satisfied_constraints = 0
    total_constraints = 0
    
    for action in constraint_pruned:
        for constraint_name, constraint_value in action.constraints.items():
            total_constraints += 1
            if pruner._satisfies_hard_constraints(action, state):
                satisfied_constraints += 1
    
    if total_constraints > 0:
        satisfaction_rate = satisfied_constraints / total_constraints
        print(f"   Constraint satisfaction rate: {satisfaction_rate:.2f}")
    
    # Performance statistics
    print("9. Performance Statistics...")
    stats = pruner.get_statistics()
    
    print(f"   Total actions considered: {stats['total_actions_considered']:,}")
    print(f"   Total actions pruned: {stats['total_actions_pruned']:,}")
    print(f"   Overall pruning ratio: {stats['overall_pruning_ratio']:.3f}")
    print(f"   Average pruning time: {stats['avg_pruning_time']:.4f}s")
    print(f"   Strategy usage: {stats['strategy_usage']}")
    print(f"   Training data size: {stats['training_data_size']}")
    
    # Dynamic threshold analysis
    print("10. Dynamic Threshold Analysis...")
    print(f"   Current feasibility threshold: {stats['dynamic_thresholds']['feasibility']:.3f}")
    print(f"   Current value threshold: {stats['dynamic_thresholds']['value']:.3f}")
    print(f"   Original feasibility threshold: {config.feasibility_threshold:.3f}")
    print(f"   Original value threshold: {config.value_threshold:.3f}")
    
    # Efficiency analysis
    original_computation = len(large_action_set) ** 2  # Quadratic complexity assumption
    pruned_computation = config.max_actions ** 2
    efficiency_gain = original_computation / pruned_computation
    
    print(f"   Computational efficiency gain: {efficiency_gain:.1f}x")
    print(f"   Memory reduction: {1 - config.max_actions / len(large_action_set):.2f}")
    
    print(f"\n[SUCCESS] Adaptive Action Space Pruning R28 Complete!")
    print(f"\nKey Features Demonstrated:")
    print(f"+ Dynamic action space reduction with multiple pruning strategies")
    print(f"+ Feasibility analysis with resource and constraint checking")
    print(f"+ Value-based pruning using learned Q-value distributions")
    print(f"+ Hierarchical clustering for computational efficiency")
    print(f"+ Online learning of action importance and effectiveness")
    print(f"+ Adaptive strategy selection based on system conditions")
    print(f"+ Constraint-aware filtering with soft/hard constraint handling")
    print(f"+ Dynamic threshold adjustment based on performance feedback")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())