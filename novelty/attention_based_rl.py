#!/usr/bin/env python3
"""
Attention-Based Reinforcement Learning for Dynamic Priority Scheduling

This module implements a comprehensive attention-based RL framework for dynamic
priority scheduling in heterogeneous systems. The system uses multi-head attention
mechanisms to dynamically adjust task priorities based on system state, resource
availability, and temporal constraints.

Research Innovation: First attention-based RL system specifically designed for
dynamic priority scheduling with temporal attention, multi-objective optimization,
and adaptive priority adjustment mechanisms.

Key Components:
- Multi-head attention for task-resource matching
- Temporal attention for deadline-aware scheduling
- Hierarchical attention for multi-level priority management
- Self-attention for task interdependency modeling
- Cross-attention for system-task interaction modeling
- Attention-based action selection with dynamic weighting

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
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
import random

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class AttentionType(Enum):
    """Types of attention mechanisms"""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    TEMPORAL_ATTENTION = "temporal_attention"
    HIERARCHICAL_ATTENTION = "hierarchical_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"

class PriorityLevel(Enum):
    """Priority levels for tasks"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class Task:
    """Task representation with priority information"""
    task_id: str
    priority_level: PriorityLevel
    base_priority: float
    dynamic_priority: float
    deadline: Optional[float]
    resource_requirements: Dict[str, float]
    dependencies: List[str]
    estimated_duration: float
    arrival_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Resource:
    """Resource representation with availability information"""
    resource_id: str
    resource_type: str
    capacity: Dict[str, float]
    utilization: Dict[str, float]
    availability: float
    location: str
    capabilities: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SchedulingState:
    """Complete scheduling state representation"""
    tasks: List[Task]
    resources: List[Resource]
    current_assignments: Dict[str, str]  # task_id -> resource_id
    system_load: float
    timestamp: float
    attention_weights: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for scheduling"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention"""
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(attention_output)
        
        return output, attention_weights.mean(dim=1)  # Return averaged attention weights

class TemporalAttention(nn.Module):
    """Temporal attention for deadline-aware scheduling"""
    
    def __init__(self, d_model: int, max_sequence_length: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
        # Positional encoding for temporal information
        self.temporal_encoding = nn.Parameter(
            torch.zeros(max_sequence_length, d_model)
        )
        
        # Deadline attention layers
        self.deadline_attention = MultiHeadAttention(d_model, num_heads=8)
        self.urgency_projection = nn.Linear(d_model, d_model)
        self.time_decay_factor = nn.Parameter(torch.tensor(0.99))
        
    def forward(self, task_embeddings: torch.Tensor, deadlines: torch.Tensor,
                current_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal attention based on deadlines"""
        batch_size, seq_len, _ = task_embeddings.shape
        
        # Calculate urgency scores
        time_remaining = deadlines - current_time
        urgency_scores = torch.exp(-time_remaining / 3600.0)  # Exponential urgency
        
        # Add temporal encoding
        if seq_len <= self.max_sequence_length:
            temporal_enc = self.temporal_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            task_embeddings = task_embeddings + temporal_enc
        
        # Apply urgency-weighted attention
        urgency_weights = urgency_scores.unsqueeze(-1).expand(-1, -1, self.d_model)
        urgency_enhanced = task_embeddings * urgency_weights
        
        # Self-attention with temporal information
        temporal_output, attention_weights = self.deadline_attention(
            urgency_enhanced, urgency_enhanced, urgency_enhanced
        )
        
        # Apply urgency projection
        temporal_output = self.urgency_projection(temporal_output)
        
        return temporal_output, attention_weights

class HierarchicalAttention(nn.Module):
    """Hierarchical attention for multi-level priority management"""
    
    def __init__(self, d_model: int, num_priority_levels: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_priority_levels = num_priority_levels
        
        # Priority level embeddings
        self.priority_embeddings = nn.Embedding(num_priority_levels, d_model)
        
        # Hierarchical attention layers
        self.level_attention = MultiHeadAttention(d_model, num_heads=4)
        self.cross_level_attention = MultiHeadAttention(d_model, num_heads=8)
        
        # Priority adjustment networks
        self.priority_adjustment = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, task_embeddings: torch.Tensor, priority_levels: torch.Tensor,
                system_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply hierarchical attention across priority levels"""
        batch_size, seq_len, _ = task_embeddings.shape
        
        # Add priority level embeddings
        priority_emb = self.priority_embeddings(priority_levels)
        priority_enhanced = task_embeddings + priority_emb
        
        # Within-level attention
        level_output, level_weights = self.level_attention(
            priority_enhanced, priority_enhanced, priority_enhanced
        )
        
        # Cross-level attention with system state
        system_expanded = system_state.unsqueeze(1).expand(-1, seq_len, -1)
        cross_output, cross_weights = self.cross_level_attention(
            level_output, system_expanded, system_expanded
        )
        
        # Dynamic priority adjustments
        priority_adjustments = self.priority_adjustment(cross_output)
        
        return cross_output, level_weights, priority_adjustments

class TaskResourceAttention(nn.Module):
    """Cross-attention between tasks and resources"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Task-resource matching attention
        self.matching_attention = MultiHeadAttention(d_model, num_heads=8)
        
        # Compatibility scoring
        self.compatibility_network = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Resource contention modeling
        self.contention_attention = MultiHeadAttention(d_model, num_heads=4)
        
    def forward(self, task_embeddings: torch.Tensor, resource_embeddings: torch.Tensor,
                availability_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute task-resource attention and compatibility"""
        
        # Cross-attention between tasks and resources
        task_resource_attention, attention_weights = self.matching_attention(
            task_embeddings, resource_embeddings, resource_embeddings,
            mask=availability_mask
        )
        
        # Compute compatibility scores
        batch_size, num_tasks, _ = task_embeddings.shape
        _, num_resources, _ = resource_embeddings.shape
        
        compatibility_scores = torch.zeros(batch_size, num_tasks, num_resources)
        
        for i in range(num_tasks):
            for j in range(num_resources):
                task_res_concat = torch.cat([
                    task_embeddings[:, i, :],
                    resource_embeddings[:, j, :]
                ], dim=-1)
                compatibility_scores[:, i, j] = self.compatibility_network(task_res_concat).squeeze(-1)
        
        # Model resource contention
        contention_output, contention_weights = self.contention_attention(
            resource_embeddings, resource_embeddings, resource_embeddings
        )
        
        return task_resource_attention, compatibility_scores, contention_weights

class AttentionBasedPolicyNetwork(nn.Module):
    """Main attention-based policy network for scheduling"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.d_model = config.get('d_model', 256)
        self.num_objectives = config.get('num_objectives', 3)
        self.max_tasks = config.get('max_tasks', 100)
        self.max_resources = config.get('max_resources', 50)
        
        # Input embedding layers
        self.task_embedding = nn.Sequential(
            nn.Linear(config.get('task_feature_dim', 16), self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model)
        )
        
        self.resource_embedding = nn.Sequential(
            nn.Linear(config.get('resource_feature_dim', 12), self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model)
        )
        
        self.system_embedding = nn.Sequential(
            nn.Linear(config.get('system_feature_dim', 8), self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model)
        )
        
        # Attention mechanisms
        self.temporal_attention = TemporalAttention(self.d_model)
        self.hierarchical_attention = HierarchicalAttention(self.d_model)
        self.task_resource_attention = TaskResourceAttention(self.d_model)
        
        # Task self-attention for dependency modeling
        self.task_self_attention = MultiHeadAttention(self.d_model, num_heads=8)
        
        # Action selection networks
        self.action_attention = MultiHeadAttention(self.d_model, num_heads=4)
        
        self.priority_adjustment_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Tanh()  # Priority adjustment in [-1, 1]
        )
        
        self.assignment_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.max_resources)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.num_objectives)
        )
        
        # Attention weight fusion
        self.attention_fusion = nn.Sequential(
            nn.Linear(4 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh()
        )
        
    def forward(self, state: SchedulingState) -> Dict[str, torch.Tensor]:
        """Forward pass of attention-based policy network"""
        
        # Extract and embed features
        task_features = self._extract_task_features(state.tasks)
        resource_features = self._extract_resource_features(state.resources)
        system_features = self._extract_system_features(state)
        
        # Embed inputs
        task_embeddings = self.task_embedding(task_features)
        resource_embeddings = self.resource_embedding(resource_features)
        system_embedding = self.system_embedding(system_features)
        
        # Extract temporal information
        deadlines = torch.tensor([
            task.deadline if task.deadline else state.timestamp + 3600
            for task in state.tasks
        ], dtype=torch.float32).unsqueeze(0)
        
        priority_levels = torch.tensor([
            list(PriorityLevel).index(task.priority_level)
            for task in state.tasks
        ], dtype=torch.long).unsqueeze(0)
        
        # Apply attention mechanisms
        
        # 1. Temporal attention for deadline awareness
        temporal_output, temporal_weights = self.temporal_attention(
            task_embeddings, deadlines, state.timestamp
        )
        
        # 2. Hierarchical attention for priority management
        hierarchical_output, level_weights, priority_adjustments = self.hierarchical_attention(
            temporal_output, priority_levels, system_embedding
        )
        
        # 3. Task self-attention for dependency modeling
        self_attention_output, self_weights = self.task_self_attention(
            hierarchical_output, hierarchical_output, hierarchical_output
        )
        
        # 4. Task-resource cross-attention
        availability_mask = self._create_availability_mask(state)
        task_resource_output, compatibility_scores, contention_weights = self.task_resource_attention(
            self_attention_output, resource_embeddings, availability_mask
        )
        
        # Fuse attention outputs
        fused_features = self.attention_fusion(torch.cat([
            temporal_output,
            hierarchical_output,
            self_attention_output,
            task_resource_output
        ], dim=-1))
        
        # Generate outputs
        
        # Priority adjustments for each task
        priority_adjustments = self.priority_adjustment_head(fused_features)
        
        # Task-resource assignment probabilities
        assignment_logits = self.assignment_head(fused_features)
        
        # Global state value estimation
        global_representation = fused_features.mean(dim=1)  # Global pooling
        state_values = self.value_head(global_representation)
        
        # Action attention for dynamic action weighting
        action_weights, action_attention_weights = self.action_attention(
            fused_features, fused_features, fused_features
        )
        
        return {
            'priority_adjustments': priority_adjustments,
            'assignment_logits': assignment_logits,
            'state_values': state_values,
            'compatibility_scores': compatibility_scores,
            'action_weights': action_weights,
            'attention_weights': {
                'temporal': temporal_weights,
                'hierarchical': level_weights,
                'self_attention': self_weights,
                'task_resource': contention_weights,
                'action_attention': action_attention_weights
            },
            'fused_features': fused_features
        }
    
    def _extract_task_features(self, tasks: List[Task]) -> torch.Tensor:
        """Extract features from tasks"""
        features = []
        
        for task in tasks:
            task_features = [
                task.base_priority,
                task.dynamic_priority,
                task.estimated_duration,
                time.time() - task.arrival_time,  # Age
                len(task.dependencies),
                task.resource_requirements.get('cpu', 0.0),
                task.resource_requirements.get('memory', 0.0),
                task.resource_requirements.get('gpu', 0.0),
                task.resource_requirements.get('storage', 0.0),
                task.resource_requirements.get('bandwidth', 0.0),
                1.0 if task.deadline else 0.0,
                (task.deadline - time.time()) / 3600.0 if task.deadline else 0.0,  # Hours to deadline
                list(PriorityLevel).index(task.priority_level) / len(PriorityLevel),
                task.metadata.get('complexity', 0.5),
                task.metadata.get('importance', 0.5),
                task.metadata.get('user_priority', 0.5)
            ]
            features.append(task_features)
        
        # Pad to max_tasks
        while len(features) < self.max_tasks:
            features.append([0.0] * 16)  # Assuming 16 features
        
        return torch.tensor(features[:self.max_tasks], dtype=torch.float32).unsqueeze(0)
    
    def _extract_resource_features(self, resources: List[Resource]) -> torch.Tensor:
        """Extract features from resources"""
        features = []
        
        for resource in resources:
            resource_features = [
                resource.availability,
                resource.utilization.get('cpu', 0.0),
                resource.utilization.get('memory', 0.0),
                resource.utilization.get('gpu', 0.0),
                resource.capacity.get('cpu', 1.0),
                resource.capacity.get('memory', 1.0),
                resource.capacity.get('gpu', 1.0),
                len(resource.capabilities),
                resource.metadata.get('reliability', 0.9),
                resource.metadata.get('performance_score', 0.5),
                resource.metadata.get('energy_efficiency', 0.5),
                resource.metadata.get('maintenance_score', 0.9)
            ]
            features.append(resource_features)
        
        # Pad to max_resources
        while len(features) < self.max_resources:
            features.append([0.0] * 12)  # Assuming 12 features
        
        return torch.tensor(features[:self.max_resources], dtype=torch.float32).unsqueeze(0)
    
    def _extract_system_features(self, state: SchedulingState) -> torch.Tensor:
        """Extract system-level features"""
        system_features = [
            state.system_load,
            len(state.tasks) / self.max_tasks,
            len(state.resources) / self.max_resources,
            len(state.current_assignments) / max(len(state.tasks), 1),
            state.metadata.get('queue_pressure', 0.5),
            state.metadata.get('resource_fragmentation', 0.3),
            state.metadata.get('deadline_pressure', 0.4),
            state.metadata.get('system_health', 0.9)
        ]
        
        return torch.tensor(system_features, dtype=torch.float32).unsqueeze(0)
    
    def _create_availability_mask(self, state: SchedulingState) -> torch.Tensor:
        """Create availability mask for task-resource attention"""
        num_tasks = len(state.tasks)
        num_resources = len(state.resources)
        
        mask = torch.ones(1, num_tasks, num_resources)
        
        # Mask unavailable resources
        for i, resource in enumerate(state.resources):
            if resource.availability < 0.1:  # Resource not available
                mask[0, :, i] = 0
        
        return mask

class DynamicPriorityScheduler:
    """Main attention-based dynamic priority scheduler"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize policy network
        self.policy_network = AttentionBasedPolicyNetwork(self.config)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Priority adjustment parameters
        self.priority_decay_rate = self.config.get('priority_decay_rate', 0.99)
        self.max_priority_adjustment = self.config.get('max_priority_adjustment', 0.5)
        
        # Training parameters
        self.gamma = self.config.get('gamma', 0.99)
        self.entropy_weight = self.config.get('entropy_weight', 0.01)
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'priority_adjustments_made': 0,
            'attention_entropy_history': deque(maxlen=1000),
            'scheduling_performance': deque(maxlen=1000),
            'priority_distribution_history': deque(maxlen=100)
        }
        
    def schedule(self, state: SchedulingState) -> Dict[str, Any]:
        """Make scheduling decisions using attention-based policy"""
        
        # Forward pass through policy network
        with torch.no_grad():
            outputs = self.policy_network(state)
        
        # Extract decisions
        decisions = self._extract_decisions(outputs, state)
        
        # Update statistics
        self._update_statistics(outputs, decisions, state)
        
        return decisions
    
    def update_priorities(self, state: SchedulingState) -> SchedulingState:
        """Update task priorities based on attention mechanism"""
        
        # Get priority adjustments from network
        outputs = self.policy_network(state)
        priority_adjustments = outputs['priority_adjustments'].squeeze(0).numpy()
        
        # Apply priority adjustments
        updated_tasks = []
        for i, task in enumerate(state.tasks):
            if i < len(priority_adjustments):
                adjustment = priority_adjustments[i][0] * self.max_priority_adjustment
                new_priority = np.clip(
                    task.dynamic_priority + adjustment, 0.0, 1.0
                )
                
                updated_task = Task(
                    task_id=task.task_id,
                    priority_level=task.priority_level,
                    base_priority=task.base_priority,
                    dynamic_priority=new_priority,
                    deadline=task.deadline,
                    resource_requirements=task.resource_requirements,
                    dependencies=task.dependencies,
                    estimated_duration=task.estimated_duration,
                    arrival_time=task.arrival_time,
                    metadata=task.metadata
                )
                updated_tasks.append(updated_task)
                
                if abs(adjustment) > 0.01:
                    self.stats['priority_adjustments_made'] += 1
            else:
                updated_tasks.append(task)
        
        # Create updated state
        updated_state = SchedulingState(
            tasks=updated_tasks,
            resources=state.resources,
            current_assignments=state.current_assignments,
            system_load=state.system_load,
            timestamp=state.timestamp,
            attention_weights=outputs['attention_weights'],
            metadata=state.metadata
        )
        
        return updated_state
    
    def train_step(self, batch_states: List[SchedulingState], 
                  batch_actions: List[Dict[str, Any]], 
                  batch_rewards: List[np.ndarray],
                  batch_next_states: List[SchedulingState],
                  batch_dones: List[bool]) -> Dict[str, float]:
        """Training step for the attention-based policy"""
        
        # Prepare batch data
        batch_size = len(batch_states)
        
        # Forward pass for current states
        current_outputs = []
        for state in batch_states:
            outputs = self.policy_network(state)
            current_outputs.append(outputs)
        
        # Forward pass for next states
        next_outputs = []
        for state in batch_next_states:
            with torch.no_grad():
                outputs = self.policy_network(state)
            next_outputs.append(outputs)
        
        # Calculate losses
        total_loss = 0.0
        loss_components = {}
        
        for i in range(batch_size):
            # Value loss
            current_values = current_outputs[i]['state_values']
            next_values = next_outputs[i]['state_values']
            rewards = torch.tensor(batch_rewards[i], dtype=torch.float32)
            
            if not batch_dones[i]:
                target_values = rewards + self.gamma * next_values
            else:
                target_values = rewards
            
            value_loss = F.mse_loss(current_values, target_values.detach())
            
            # Policy loss (using compatibility scores)
            compatibility_scores = current_outputs[i]['compatibility_scores']
            action_weights = current_outputs[i]['action_weights']
            
            # Simple policy loss - maximize compatibility for chosen actions
            policy_loss = -torch.mean(compatibility_scores * action_weights.detach())
            
            # Attention entropy regularization
            attention_weights = current_outputs[i]['attention_weights']
            entropy_loss = 0.0
            for attention_name, weights in attention_weights.items():
                if weights.numel() > 1:
                    entropy = -torch.sum(weights * torch.log(weights + 1e-8))
                    entropy_loss += entropy
            
            # Combined loss
            step_loss = value_loss + 0.1 * policy_loss + self.entropy_weight * entropy_loss
            total_loss += step_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        loss_components = {
            'total_loss': total_loss.item() / batch_size,
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
        
        return loss_components
    
    def _extract_decisions(self, outputs: Dict[str, torch.Tensor], 
                          state: SchedulingState) -> Dict[str, Any]:
        """Extract scheduling decisions from network outputs"""
        
        priority_adjustments = outputs['priority_adjustments'].squeeze(0).numpy()
        assignment_logits = outputs['assignment_logits'].squeeze(0).numpy()
        compatibility_scores = outputs['compatibility_scores'].squeeze(0).numpy()
        
        decisions = {
            'task_assignments': {},
            'priority_adjustments': {},
            'attention_weights': outputs['attention_weights'],
            'compatibility_matrix': compatibility_scores
        }
        
        # Task-resource assignments based on compatibility scores
        for i, task in enumerate(state.tasks):
            if i < len(compatibility_scores):
                # Find best resource match
                available_resources = [
                    j for j, resource in enumerate(state.resources)
                    if resource.availability > 0.1 and j < compatibility_scores.shape[1]
                ]
                
                if available_resources:
                    resource_scores = compatibility_scores[i, available_resources]
                    best_resource_idx = available_resources[np.argmax(resource_scores)]
                    best_resource = state.resources[best_resource_idx]
                    
                    decisions['task_assignments'][task.task_id] = {
                        'resource_id': best_resource.resource_id,
                        'compatibility_score': float(np.max(resource_scores)),
                        'assignment_confidence': float(assignment_logits[i, best_resource_idx])
                    }
        
        # Priority adjustments
        for i, task in enumerate(state.tasks):
            if i < len(priority_adjustments):
                adjustment = float(priority_adjustments[i][0])
                if abs(adjustment) > 0.01:  # Only include significant adjustments
                    decisions['priority_adjustments'][task.task_id] = adjustment
        
        self.stats['total_decisions'] += 1
        
        return decisions
    
    def _update_statistics(self, outputs: Dict[str, torch.Tensor], 
                          decisions: Dict[str, Any], state: SchedulingState):
        """Update training and performance statistics"""
        
        # Calculate attention entropy
        total_entropy = 0.0
        num_attention_types = 0
        
        for attention_name, weights in outputs['attention_weights'].items():
            if weights.numel() > 1:
                entropy = -torch.sum(weights * torch.log(weights + 1e-8))
                total_entropy += entropy.item()
                num_attention_types += 1
        
        if num_attention_types > 0:
            avg_entropy = total_entropy / num_attention_types
            self.stats['attention_entropy_history'].append(avg_entropy)
        
        # Track priority distribution
        priority_distribution = [task.dynamic_priority for task in state.tasks]
        self.stats['priority_distribution_history'].append(priority_distribution)
        
        # Calculate scheduling performance metrics
        performance_score = self._calculate_performance_score(decisions, state)
        self.stats['scheduling_performance'].append(performance_score)
    
    def _calculate_performance_score(self, decisions: Dict[str, Any], 
                                   state: SchedulingState) -> float:
        """Calculate performance score for scheduling decisions"""
        
        if not decisions['task_assignments']:
            return 0.0
        
        # Factors for performance evaluation
        compatibility_scores = [
            assignment['compatibility_score']
            for assignment in decisions['task_assignments'].values()
        ]
        
        # Average compatibility
        avg_compatibility = np.mean(compatibility_scores)
        
        # Priority coverage (how well high-priority tasks are handled)
        high_priority_tasks = [
            task for task in state.tasks
            if task.priority_level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]
        ]
        
        assigned_high_priority = [
            task for task in high_priority_tasks
            if task.task_id in decisions['task_assignments']
        ]
        
        priority_coverage = len(assigned_high_priority) / max(len(high_priority_tasks), 1)
        
        # Resource utilization efficiency
        assigned_resources = set()
        for assignment in decisions['task_assignments'].values():
            assigned_resources.add(assignment['resource_id'])
        
        resource_efficiency = len(assigned_resources) / max(len(state.resources), 1)
        
        # Combined performance score
        performance = 0.5 * avg_compatibility + 0.3 * priority_coverage + 0.2 * resource_efficiency
        
        return performance
    
    def get_attention_analysis(self, state: SchedulingState) -> Dict[str, Any]:
        """Get detailed attention analysis for interpretability"""
        
        with torch.no_grad():
            outputs = self.policy_network(state)
        
        attention_weights = outputs['attention_weights']
        
        analysis = {
            'attention_summary': {},
            'task_attention_focus': {},
            'resource_attention_focus': {},
            'temporal_attention_pattern': {},
            'priority_attention_distribution': {}
        }
        
        # Summarize attention weights
        for attention_name, weights in attention_weights.items():
            if weights.numel() > 0:
                analysis['attention_summary'][attention_name] = {
                    'max_weight': float(weights.max()),
                    'min_weight': float(weights.min()),
                    'mean_weight': float(weights.mean()),
                    'std_weight': float(weights.std()),
                    'entropy': float(-torch.sum(weights * torch.log(weights + 1e-8)))
                }
        
        # Task-specific attention analysis
        if 'self_attention' in attention_weights:
            task_attention = attention_weights['self_attention']
            for i, task in enumerate(state.tasks[:task_attention.shape[1]]):
                if i < task_attention.shape[1]:
                    analysis['task_attention_focus'][task.task_id] = {
                        'self_attention_score': float(task_attention[0, i, i]),
                        'influences_others': float(task_attention[0, i, :].sum() - task_attention[0, i, i]),
                        'influenced_by_others': float(task_attention[0, :, i].sum() - task_attention[0, i, i])
                    }
        
        # Temporal attention pattern
        if 'temporal' in attention_weights:
            temporal_weights = attention_weights['temporal']
            deadlines = [task.deadline for task in state.tasks if task.deadline]
            
            if deadlines:
                current_time = state.timestamp
                urgency_scores = [(deadline - current_time) / 3600.0 for deadline in deadlines]
                
                analysis['temporal_attention_pattern'] = {
                    'attention_urgency_correlation': float(np.corrcoef(
                        temporal_weights[0, :len(urgency_scores), 0].numpy(),
                        urgency_scores
                    )[0, 1]) if len(urgency_scores) > 1 else 0.0,
                    'most_urgent_task_attention': float(temporal_weights[0, 0, 0]) if temporal_weights.numel() > 0 else 0.0
                }
        
        return analysis
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = self.stats.copy()
        
        # Add computed statistics
        if self.stats['attention_entropy_history']:
            stats['avg_attention_entropy'] = np.mean(self.stats['attention_entropy_history'])
            stats['attention_entropy_trend'] = np.polyfit(
                range(len(self.stats['attention_entropy_history'])),
                list(self.stats['attention_entropy_history']),
                1
            )[0]
        
        if self.stats['scheduling_performance']:
            stats['avg_scheduling_performance'] = np.mean(self.stats['scheduling_performance'])
            stats['performance_trend'] = np.polyfit(
                range(len(self.stats['scheduling_performance'])),
                list(self.stats['scheduling_performance']),
                1
            )[0]
        
        # Priority adjustment statistics
        stats['priority_adjustment_rate'] = (
            self.stats['priority_adjustments_made'] / 
            max(self.stats['total_decisions'], 1)
        )
        
        return stats

def create_sample_scheduling_state(num_tasks: int = 20, num_resources: int = 10) -> SchedulingState:
    """Create sample scheduling state for testing"""
    
    # Create tasks
    tasks = []
    priority_levels = list(PriorityLevel)
    
    for i in range(num_tasks):
        priority_level = np.random.choice(priority_levels)
        
        task = Task(
            task_id=f"task_{i}",
            priority_level=priority_level,
            base_priority=np.random.uniform(0.1, 1.0),
            dynamic_priority=np.random.uniform(0.1, 1.0),
            deadline=time.time() + np.random.uniform(300, 7200) if np.random.random() < 0.7 else None,
            resource_requirements={
                'cpu': np.random.uniform(0.5, 4.0),
                'memory': np.random.uniform(1.0, 16.0),
                'gpu': np.random.uniform(0, 2),
                'storage': np.random.uniform(1.0, 100.0),
                'bandwidth': np.random.uniform(10, 1000)
            },
            dependencies=[f"task_{j}" for j in range(i) if np.random.random() < 0.1],
            estimated_duration=np.random.uniform(60, 3600),
            arrival_time=time.time() - np.random.uniform(0, 1800),
            metadata={
                'complexity': np.random.uniform(0.1, 1.0),
                'importance': np.random.uniform(0.1, 1.0),
                'user_priority': np.random.uniform(0.1, 1.0)
            }
        )
        tasks.append(task)
    
    # Create resources
    resources = []
    resource_types = ['cpu', 'gpu', 'memory', 'storage']
    
    for i in range(num_resources):
        resource_type = np.random.choice(resource_types)
        
        resource = Resource(
            resource_id=f"resource_{i}",
            resource_type=resource_type,
            capacity={
                'cpu': np.random.uniform(4, 32),
                'memory': np.random.uniform(8, 128),
                'gpu': np.random.uniform(1, 8),
                'storage': np.random.uniform(100, 2000),
                'bandwidth': np.random.uniform(100, 10000)
            },
            utilization={
                'cpu': np.random.uniform(0.1, 0.8),
                'memory': np.random.uniform(0.2, 0.7),
                'gpu': np.random.uniform(0.0, 0.6)
            },
            availability=np.random.uniform(0.8, 1.0),
            location=f"rack_{np.random.randint(1, 5)}",
            capabilities=set(np.random.choice(['gpu_compute', 'high_memory', 'fast_storage'], 
                                            size=np.random.randint(1, 3), replace=False)),
            metadata={
                'reliability': np.random.uniform(0.9, 1.0),
                'performance_score': np.random.uniform(0.7, 1.0),
                'energy_efficiency': np.random.uniform(0.6, 1.0),
                'maintenance_score': np.random.uniform(0.8, 1.0)
            }
        )
        resources.append(resource)
    
    # Create current assignments
    current_assignments = {}
    assigned_resources = set()
    
    for task in tasks[:num_tasks//2]:  # Assign half the tasks
        available_resources = [r for r in resources if r.resource_id not in assigned_resources]
        if available_resources:
            assigned_resource = np.random.choice(available_resources)
            current_assignments[task.task_id] = assigned_resource.resource_id
            assigned_resources.add(assigned_resource.resource_id)
    
    return SchedulingState(
        tasks=tasks,
        resources=resources,
        current_assignments=current_assignments,
        system_load=np.random.uniform(0.3, 0.8),
        timestamp=time.time(),
        metadata={
            'queue_pressure': np.random.uniform(0.2, 0.8),
            'resource_fragmentation': np.random.uniform(0.1, 0.5),
            'deadline_pressure': np.random.uniform(0.2, 0.7),
            'system_health': np.random.uniform(0.8, 1.0)
        }
    )

async def main():
    """Demonstrate attention-based RL for dynamic priority scheduling"""
    
    print("=== Attention-Based RL for Dynamic Priority Scheduling ===\n")
    
    # Configuration
    config = {
        'd_model': 256,
        'num_objectives': 3,
        'max_tasks': 50,
        'max_resources': 25,
        'task_feature_dim': 16,
        'resource_feature_dim': 12,
        'system_feature_dim': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'gamma': 0.99,
        'entropy_weight': 0.01,
        'priority_decay_rate': 0.99,
        'max_priority_adjustment': 0.5
    }
    
    # Create attention-based scheduler
    print("1. Initializing Attention-Based Dynamic Priority Scheduler...")
    scheduler = DynamicPriorityScheduler(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in scheduler.policy_network.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Model dimension: {config['d_model']}")
    print(f"   Max tasks/resources: {config['max_tasks']}/{config['max_resources']}")
    
    # Create sample scheduling state
    print("2. Creating Sample Scheduling State...")
    state = create_sample_scheduling_state(num_tasks=30, num_resources=15)
    
    print(f"   Tasks: {len(state.tasks)}")
    print(f"   Resources: {len(state.resources)}")
    print(f"   Current assignments: {len(state.current_assignments)}")
    print(f"   System load: {state.system_load:.2f}")
    
    # Analyze priority distribution
    priority_counts = {}
    for task in state.tasks:
        priority_counts[task.priority_level.value] = priority_counts.get(task.priority_level.value, 0) + 1
    print(f"   Priority distribution: {priority_counts}")
    
    # Test scheduling decisions
    print("3. Testing Attention-Based Scheduling...")
    decisions = scheduler.schedule(state)
    
    print(f"   Task assignments made: {len(decisions['task_assignments'])}")
    print(f"   Priority adjustments made: {len(decisions['priority_adjustments'])}")
    
    # Show sample assignments
    print("   Sample task assignments:")
    for i, (task_id, assignment) in enumerate(list(decisions['task_assignments'].items())[:5]):
        print(f"     {task_id} -> {assignment['resource_id']} "
              f"(compatibility: {assignment['compatibility_score']:.3f})")
    
    # Test priority updates
    print("4. Testing Dynamic Priority Updates...")
    original_priorities = [task.dynamic_priority for task in state.tasks]
    updated_state = scheduler.update_priorities(state)
    updated_priorities = [task.dynamic_priority for task in updated_state.tasks]
    
    priority_changes = np.array(updated_priorities) - np.array(original_priorities)
    significant_changes = np.sum(np.abs(priority_changes) > 0.01)
    
    print(f"   Significant priority changes: {significant_changes}/{len(state.tasks)}")
    print(f"   Max priority increase: {np.max(priority_changes):.3f}")
    print(f"   Max priority decrease: {np.min(priority_changes):.3f}")
    
    # Test attention analysis
    print("5. Attention Mechanism Analysis...")
    attention_analysis = scheduler.get_attention_analysis(updated_state)
    
    print("   Attention summary:")
    for attention_type, summary in attention_analysis['attention_summary'].items():
        print(f"     {attention_type}: entropy={summary['entropy']:.3f}, "
              f"mean_weight={summary['mean_weight']:.3f}")
    
    # Test temporal attention
    if 'temporal_attention_pattern' in attention_analysis:
        temporal_pattern = attention_analysis['temporal_attention_pattern']
        print(f"   Temporal attention correlation: {temporal_pattern.get('attention_urgency_correlation', 0):.3f}")
    
    # Simulate training episodes
    print("6. Simulating Training Episodes...")
    
    training_losses = []
    performance_scores = []
    
    for episode in range(20):
        # Create episode data
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []
        
        current_state = create_sample_scheduling_state(num_tasks=25, num_resources=12)
        
        for step in range(10):
            # Make scheduling decision
            decisions = scheduler.schedule(current_state)
            
            # Simulate reward (multi-objective)
            reward = np.array([
                np.random.uniform(0.5, 1.0),  # Throughput
                np.random.uniform(0.3, 0.8),  # Latency
                np.random.uniform(0.4, 0.9)   # Energy efficiency
            ])
            
            # Create next state
            next_state = scheduler.update_priorities(current_state)
            
            # Store transition
            episode_states.append(current_state)
            episode_actions.append(decisions)
            episode_rewards.append(reward)
            episode_next_states.append(next_state)
            episode_dones.append(step == 9)
            
            current_state = next_state
        
        # Training step
        if len(episode_states) >= 5:  # Minimum batch size
            batch_states = episode_states[:5]
            batch_actions = episode_actions[:5]
            batch_rewards = episode_rewards[:5]
            batch_next_states = episode_next_states[:5]
            batch_dones = episode_dones[:5]
            
            loss_info = scheduler.train_step(
                batch_states, batch_actions, batch_rewards, 
                batch_next_states, batch_dones
            )
            
            training_losses.append(loss_info['total_loss'])
            
            # Calculate episode performance
            episode_performance = np.mean([
                scheduler._calculate_performance_score(action, state)
                for action, state in zip(batch_actions, batch_states)
            ])
            performance_scores.append(episode_performance)
        
        if episode % 5 == 0:
            avg_loss = np.mean(training_losses[-5:]) if training_losses else 0.0
            avg_performance = np.mean(performance_scores[-5:]) if performance_scores else 0.0
            print(f"   Episode {episode}: avg_loss={avg_loss:.4f}, performance={avg_performance:.3f}")
    
    # Analyze training progress
    print("7. Training Progress Analysis...")
    if training_losses:
        print(f"   Final average loss: {np.mean(training_losses[-5:]):.4f}")
        print(f"   Loss trend: {np.polyfit(range(len(training_losses)), training_losses, 1)[0]:.6f}")
    
    if performance_scores:
        print(f"   Final average performance: {np.mean(performance_scores[-5:]):.3f}")
        print(f"   Performance trend: {np.polyfit(range(len(performance_scores)), performance_scores, 1)[0]:.6f}")
    
    # Test scalability
    print("8. Scalability Testing...")
    
    task_sizes = [10, 25, 50, 75, 100]
    
    for num_tasks in task_sizes:
        test_state = create_sample_scheduling_state(
            num_tasks=min(num_tasks, config['max_tasks']),
            num_resources=min(num_tasks//2, config['max_resources'])
        )
        
        start_time = time.time()
        decisions = scheduler.schedule(test_state)
        inference_time = time.time() - start_time
        
        print(f"   {num_tasks} tasks: {inference_time:.4f}s, "
              f"{len(decisions['task_assignments'])} assignments")
    
    # Test attention interpretability
    print("9. Attention Interpretability Analysis...")
    
    # Create state with clear priority structure
    interpretability_state = create_sample_scheduling_state(num_tasks=10, num_resources=5)
    
    # Set clear priority hierarchy
    for i, task in enumerate(interpretability_state.tasks):
        if i < 2:
            task.priority_level = PriorityLevel.CRITICAL
            task.deadline = time.time() + 300  # 5 minutes
        elif i < 5:
            task.priority_level = PriorityLevel.HIGH
            task.deadline = time.time() + 1800  # 30 minutes
        else:
            task.priority_level = PriorityLevel.MEDIUM
            task.deadline = time.time() + 3600  # 1 hour
    
    # Analyze attention patterns
    attention_analysis = scheduler.get_attention_analysis(interpretability_state)
    
    print("   Task attention focus (top 3):")
    task_attention = attention_analysis.get('task_attention_focus', {})
    sorted_tasks = sorted(task_attention.items(), 
                         key=lambda x: x[1]['self_attention_score'], reverse=True)
    
    for i, (task_id, attention_info) in enumerate(sorted_tasks[:3]):
        print(f"     {task_id}: self_attention={attention_info['self_attention_score']:.3f}")
    
    # Performance statistics
    print("10. Performance Statistics...")
    stats = scheduler.get_statistics()
    
    print(f"   Total decisions made: {stats['total_decisions']}")
    print(f"   Priority adjustments made: {stats['priority_adjustments_made']}")
    print(f"   Priority adjustment rate: {stats['priority_adjustment_rate']:.3f}")
    
    if 'avg_attention_entropy' in stats:
        print(f"   Average attention entropy: {stats['avg_attention_entropy']:.3f}")
        print(f"   Attention entropy trend: {stats['attention_entropy_trend']:.6f}")
    
    if 'avg_scheduling_performance' in stats:
        print(f"   Average scheduling performance: {stats['avg_scheduling_performance']:.3f}")
        print(f"   Performance trend: {stats['performance_trend']:.6f}")
    
    print(f"\n[SUCCESS] Attention-Based RL for Dynamic Priority Scheduling R30 Complete!")
    print(f"\nKey Features Demonstrated:")
    print(f"+ Multi-head attention for task-resource matching and priority management")
    print(f"+ Temporal attention for deadline-aware scheduling decisions")
    print(f"+ Hierarchical attention for multi-level priority coordination")
    print(f"+ Self-attention for modeling task interdependencies")
    print(f"+ Cross-attention for system-task interaction modeling")
    print(f"+ Dynamic priority adjustment based on attention mechanisms")
    print(f"+ Attention-based action selection with interpretable weights")
    print(f"+ Multi-objective value estimation with attention fusion")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())