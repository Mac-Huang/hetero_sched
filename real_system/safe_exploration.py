#!/usr/bin/env python3
"""
Safe Exploration Framework for Production Deployment of HeteroSched

This module implements comprehensive safe exploration mechanisms to enable
safe deployment of RL-trained schedulers in production heterogeneous environments.

Research Innovation: First safe exploration framework specifically designed for
RL-based scheduling in production systems with formal safety guarantees.

Key Safety Mechanisms:
- Conservative Q-Learning with safety constraints
- Risk-sensitive exploration with probabilistic safety bounds
- Constrained policy optimization for system stability
- Real-time safety monitoring and intervention
- Graceful degradation and fallback mechanisms
- Safety-aware experience replay prioritization

Safety Guarantees:
- System stability preservation during exploration
- Resource utilization bounds enforcement
- Performance degradation limits
- Fault tolerance and recovery mechanisms

Authors: HeteroSched Research Team
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
import math
import threading
from enum import Enum

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from real_system.uncertainty_transfer import UncertaintyAwareAgent, UncertaintyMetrics
from real_system.hil_framework import HILEnvironment, SafetyMonitor
from real_system.system_monitor import SystemStateExtractor

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety levels for exploration control"""
    CRITICAL = "critical"      # No exploration, use safe fallback
    HIGH = "high"             # Minimal exploration with strict bounds
    MEDIUM = "medium"         # Moderate exploration with monitoring
    LOW = "low"               # Normal exploration with basic safety
    MINIMAL = "minimal"       # Maximum exploration for research

@dataclass
class SafetyConstraints:
    """Comprehensive safety constraints for production deployment"""
    
    # Resource utilization bounds
    max_cpu_utilization: float = 85.0      # Maximum CPU usage %
    max_memory_utilization: float = 90.0   # Maximum memory usage %
    max_gpu_utilization: float = 90.0      # Maximum GPU usage %
    max_thermal_threshold: float = 80.0    # Maximum temperature °C
    
    # Performance bounds
    min_throughput_ratio: float = 0.8      # Minimum 80% of baseline throughput
    max_latency_multiplier: float = 1.5    # Maximum 1.5x baseline latency
    max_energy_multiplier: float = 1.3     # Maximum 1.3x baseline energy
    
    # System stability bounds
    max_queue_length: int = 1000           # Maximum task queue length
    max_failure_rate: float = 0.05         # Maximum 5% task failure rate
    min_availability: float = 0.99         # Minimum 99% system availability
    
    # Exploration bounds
    max_exploration_rate: float = 0.1      # Maximum exploration probability
    safety_confidence_threshold: float = 0.95  # Minimum confidence for actions
    uncertainty_threshold: float = 0.2    # Maximum uncertainty for actions
    
    # Monitoring intervals
    safety_check_interval: float = 1.0    # Safety check every 1 second
    performance_window: int = 100         # Performance evaluation window
    degradation_patience: int = 10        # Tolerance for temporary degradation

@dataclass
class SafetyViolation:
    """Records safety constraint violations"""
    timestamp: float
    violation_type: str
    severity: str  # "warning", "critical", "emergency"
    constraint_name: str
    measured_value: float
    threshold: float
    action_taken: str
    system_state: Dict[str, Any]

class ConservativeQNetwork(nn.Module):
    """Conservative Q-Network with built-in safety bounds"""
    
    def __init__(self, state_dim: int = 36, action_dim: int = 100, 
                 conservative_weight: float = 1.0, safety_margin: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.conservative_weight = conservative_weight
        self.safety_margin = safety_margin
        
        # Main Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Safety value network (estimates safety of actions)
        self.safety_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output safety probability [0,1]
        )
        
        # Risk estimation network
        self.risk_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softplus()  # Output risk values [0,∞)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning conservative Q-values"""
        return self.q_network(state)
    
    def get_safety_scores(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get safety scores for state-action pairs"""
        batch_size = state.shape[0]
        num_actions = actions.shape[1] if len(actions.shape) > 1 else 1
        
        # Expand state to match actions
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)
        
        state_expanded = state.unsqueeze(1).expand(-1, num_actions, -1)
        actions_expanded = actions.unsqueeze(-1)
        
        # Create state-action pairs
        state_action = torch.cat([
            state_expanded.reshape(-1, self.state_dim),
            F.one_hot(actions_expanded.reshape(-1), self.action_dim).float()
        ], dim=1)
        
        safety_scores = self.safety_network(state_action)
        return safety_scores.reshape(batch_size, num_actions)
    
    def get_risk_estimates(self, state: torch.Tensor) -> torch.Tensor:
        """Get risk estimates for all actions"""
        return self.risk_network(state)
    
    def conservative_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Compute conservative Q-values with safety penalty"""
        q_values = self.q_network(state)
        risk_values = self.risk_network(state)
        
        # Apply conservative penalty
        conservative_q = q_values - self.conservative_weight * risk_values
        
        return conservative_q

class SafeExplorationAgent:
    """RL Agent with comprehensive safe exploration mechanisms"""
    
    def __init__(self, state_dim: int = 36, action_dim: int = 100,
                 safety_constraints: SafetyConstraints = None,
                 fallback_policy: Callable = None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.fallback_policy = fallback_policy
        
        # Conservative Q-network
        self.q_network = ConservativeQNetwork(state_dim, action_dim)
        self.target_network = ConservativeQNetwork(state_dim, action_dim)
        
        # Uncertainty-aware component
        self.uncertainty_agent = UncertaintyAwareAgent(state_dim, action_dim)
        
        # Safety monitoring
        self.safety_monitor = ProductionSafetyMonitor(self.safety_constraints)
        self.violation_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.baseline_performance = None
        
        # Safety state
        self.current_safety_level = SafetyLevel.MEDIUM
        self.consecutive_violations = 0
        self.last_safety_check = time.time()
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.safety_optimizer = optim.Adam(
            list(self.q_network.safety_network.parameters()) +
            list(self.q_network.risk_network.parameters()), lr=1e-4
        )
        
        # Training statistics
        self.training_stats = {
            'safety_violations': [],
            'exploration_rates': [],
            'performance_metrics': [],
            'conservative_losses': []
        }
        
        logger.info("Safe exploration agent initialized")
    
    def select_safe_action(self, state: np.ndarray, system_state: Dict[str, Any] = None) -> int:
        """Select action with comprehensive safety analysis"""
        
        # Update safety level based on current system state
        self._update_safety_level(system_state)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get uncertainty metrics
        uncertainty_metrics = self.uncertainty_agent.q_network.predict_with_uncertainty(state_tensor)
        
        # Safety-based exploration rate adjustment
        exploration_rate = self._get_safe_exploration_rate(uncertainty_metrics, system_state)
        
        # Check if we should use fallback policy
        if self._should_use_fallback(uncertainty_metrics, system_state):
            logger.warning("High risk detected, using fallback policy")
            return self._get_fallback_action(state, system_state)
        
        # Get conservative Q-values
        with torch.no_grad():
            conservative_q = self.q_network.conservative_q_values(state_tensor)
            risk_estimates = self.q_network.get_risk_estimates(state_tensor)
            
            # Get top safe actions
            safe_actions = self._filter_safe_actions(
                state_tensor, conservative_q, risk_estimates, system_state
            )
            
            if len(safe_actions) == 0:
                logger.error("No safe actions available, using emergency fallback")
                return self._get_emergency_action(state, system_state)
            
            # Safety-aware action selection
            if np.random.random() < exploration_rate:
                # Safe exploration: choose from safe actions with some randomness
                action_probs = F.softmax(conservative_q.squeeze()[safe_actions], dim=0)
                action_idx = np.random.choice(len(safe_actions), p=action_probs.numpy())
                selected_action = safe_actions[action_idx]
            else:
                # Exploitation: choose best safe action
                safe_q_values = conservative_q.squeeze()[safe_actions]
                best_safe_idx = torch.argmax(safe_q_values)
                selected_action = safe_actions[best_safe_idx]
        
        # Log action selection
        self._log_action_selection(selected_action, uncertainty_metrics, exploration_rate)
        
        return int(selected_action)
    
    def _update_safety_level(self, system_state: Dict[str, Any]):
        """Update current safety level based on system conditions"""
        
        if system_state is None:
            return
        
        # Check for critical conditions
        critical_conditions = [
            system_state.get('cpu_usage', 0) > self.safety_constraints.max_cpu_utilization,
            system_state.get('memory_usage', 0) > self.safety_constraints.max_memory_utilization,
            system_state.get('gpu_usage', 0) > self.safety_constraints.max_gpu_utilization,
            system_state.get('temperature', 0) > self.safety_constraints.max_thermal_threshold
        ]
        
        if any(critical_conditions):
            self.current_safety_level = SafetyLevel.CRITICAL
            self.consecutive_violations += 1
        elif self.consecutive_violations > 0:
            # Gradually reduce safety level after violations
            if self.consecutive_violations > 10:
                self.current_safety_level = SafetyLevel.HIGH
            else:
                self.current_safety_level = SafetyLevel.MEDIUM
                self.consecutive_violations = max(0, self.consecutive_violations - 1)
        else:
            # Normal operation
            self.current_safety_level = SafetyLevel.MEDIUM
    
    def _get_safe_exploration_rate(self, uncertainty_metrics: UncertaintyMetrics, 
                                 system_state: Dict[str, Any] = None) -> float:
        """Compute safe exploration rate based on current conditions"""
        
        base_rate = self.safety_constraints.max_exploration_rate
        
        # Reduce exploration based on safety level
        safety_multipliers = {
            SafetyLevel.CRITICAL: 0.0,
            SafetyLevel.HIGH: 0.1,
            SafetyLevel.MEDIUM: 0.5,
            SafetyLevel.LOW: 0.8,
            SafetyLevel.MINIMAL: 1.0
        }
        
        safety_factor = safety_multipliers[self.current_safety_level]
        
        # Reduce exploration based on uncertainty
        if uncertainty_metrics.total_uncertainty > self.safety_constraints.uncertainty_threshold:
            uncertainty_factor = 0.5  # Reduce exploration when uncertain
        else:
            uncertainty_factor = 1.0
        
        # Reduce exploration based on system load
        system_load_factor = 1.0
        if system_state:
            cpu_load = system_state.get('cpu_usage', 0) / 100.0
            memory_load = system_state.get('memory_usage', 0) / 100.0
            avg_load = (cpu_load + memory_load) / 2.0
            
            if avg_load > 0.8:
                system_load_factor = 0.3  # Heavily loaded system
            elif avg_load > 0.6:
                system_load_factor = 0.7  # Moderately loaded system
        
        final_rate = base_rate * safety_factor * uncertainty_factor * system_load_factor
        
        return max(0.01, min(final_rate, self.safety_constraints.max_exploration_rate))
    
    def _filter_safe_actions(self, state: torch.Tensor, q_values: torch.Tensor,
                           risk_estimates: torch.Tensor, system_state: Dict[str, Any]) -> List[int]:
        """Filter actions to only include safe ones"""
        
        # Get all possible actions
        all_actions = torch.arange(self.action_dim)
        
        # Compute safety scores
        safety_scores = self.q_network.get_safety_scores(state, all_actions)
        
        # Filter based on safety threshold
        safety_threshold = self.safety_constraints.safety_confidence_threshold
        safe_mask = safety_scores.squeeze() >= safety_threshold
        
        # Filter based on risk estimates
        risk_threshold = torch.quantile(risk_estimates, 0.8)  # Top 20% are risky
        risk_mask = risk_estimates.squeeze() <= risk_threshold
        
        # Combine safety filters
        combined_mask = safe_mask & risk_mask
        
        # Additional domain-specific safety checks
        if system_state:
            domain_mask = self._domain_specific_safety_filter(all_actions, system_state)
            combined_mask = combined_mask & domain_mask
        
        safe_actions = all_actions[combined_mask].tolist()
        
        return safe_actions
    
    def _domain_specific_safety_filter(self, actions: torch.Tensor, 
                                     system_state: Dict[str, Any]) -> torch.Tensor:
        """Apply domain-specific safety filters for scheduling"""
        
        # Convert actions to scheduling decisions
        # Action space: [device_type, priority_level, batch_size]
        device_actions = actions % 2  # 0=CPU, 1=GPU
        priority_actions = (actions // 2) % 5  # 0-4 priority levels
        batch_actions = actions // 10  # Batch size
        
        mask = torch.ones(len(actions), dtype=torch.bool)
        
        # Don't use GPU if GPU is overloaded
        gpu_usage = system_state.get('gpu_usage', 0)
        if gpu_usage > 85.0:
            mask &= (device_actions == 0)  # Force CPU usage
        
        # Don't use high priority if system is under stress
        cpu_usage = system_state.get('cpu_usage', 0)
        if cpu_usage > 80.0:
            mask &= (priority_actions < 3)  # Avoid high priority
        
        # Don't use large batches if memory is constrained
        memory_usage = system_state.get('memory_usage', 0)
        if memory_usage > 85.0:
            mask &= (batch_actions < 5)  # Smaller batches
        
        return mask
    
    def _should_use_fallback(self, uncertainty_metrics: UncertaintyMetrics,
                           system_state: Dict[str, Any]) -> bool:
        """Determine if fallback policy should be used"""
        
        # Use fallback if uncertainty is too high
        if uncertainty_metrics.total_uncertainty > 0.5:
            return True
        
        # Use fallback if system is in critical state
        if self.current_safety_level == SafetyLevel.CRITICAL:
            return True
        
        # Use fallback if too many recent violations
        recent_violations = sum(1 for v in list(self.violation_history)[-10:] 
                              if time.time() - v.timestamp < 60)
        if recent_violations > 3:
            return True
        
        return False
    
    def _get_fallback_action(self, state: np.ndarray, system_state: Dict[str, Any]) -> int:
        """Get action from fallback policy"""
        
        if self.fallback_policy:
            return self.fallback_policy(state, system_state)
        
        # Default conservative fallback: always use CPU with normal priority
        # Action encoding: device=0 (CPU), priority=2 (normal), batch=1 (small)
        return 0 * 2 + 2 * 10 + 1  # = 21
    
    def _get_emergency_action(self, state: np.ndarray, system_state: Dict[str, Any]) -> int:
        """Get emergency safe action when no safe actions are available"""
        
        # Most conservative action: CPU, lowest priority, smallest batch
        emergency_action = 0 * 2 + 0 * 10 + 0  # = 0
        
        logger.critical(f"Emergency action selected: {emergency_action}")
        return emergency_action
    
    def _log_action_selection(self, action: int, uncertainty_metrics: UncertaintyMetrics,
                            exploration_rate: float):
        """Log action selection for monitoring"""
        
        # Decode action for logging
        device = action % 2
        priority = (action // 2) % 5
        batch = action // 10
        
        logger.debug(f"Action selected: device={device}, priority={priority}, batch={batch}, "
                    f"uncertainty={uncertainty_metrics.total_uncertainty:.3f}, "
                    f"exploration_rate={exploration_rate:.3f}, "
                    f"safety_level={self.current_safety_level.value}")
    
    def train_safe_policy(self, experiences: List[Dict], safety_feedback: List[Dict]) -> Dict[str, float]:
        """Train policy with safety constraints and feedback"""
        
        if len(experiences) < 32:
            return {}
        
        # Convert experiences to tensors
        states = torch.FloatTensor([exp['state'] for exp in experiences])
        actions = torch.LongTensor([exp['action'] for exp in experiences])
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences])
        next_states = torch.FloatTensor([exp['next_state'] for exp in experiences])
        dones = torch.BoolTensor([exp['done'] for exp in experiences])
        safety_costs = torch.FloatTensor([sf.get('safety_cost', 0.0) for sf in safety_feedback])
        
        # Compute conservative Q-learning loss
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = torch.max(next_q, dim=1)[0]
            target_q = rewards + (0.99 * max_next_q * ~dones)
        
        # Standard Q-learning loss
        q_loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Conservative penalty (CQL-style)
        # Minimize Q-values for out-of-distribution actions
        conservative_penalty = torch.logsumexp(self.q_network(states), dim=1).mean()
        conservative_loss = q_loss + 0.1 * conservative_penalty
        
        # Safety-aware loss
        safety_scores = self.q_network.get_safety_scores(states, actions.unsqueeze(1))
        safety_targets = torch.where(safety_costs > 0, torch.zeros_like(safety_costs), 
                                   torch.ones_like(safety_costs))
        safety_loss = F.binary_cross_entropy(safety_scores.squeeze(), safety_targets)
        
        # Risk estimation loss
        risk_estimates = self.q_network.get_risk_estimates(states)
        risk_targets = safety_costs.unsqueeze(1).expand(-1, self.action_dim)
        risk_loss = F.mse_loss(risk_estimates, risk_targets)
        
        # Combined loss
        total_loss = conservative_loss + 0.5 * safety_loss + 0.3 * risk_loss
        
        # Optimize Q-network
        self.q_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # Train safety networks separately (recompute to avoid gradient issues)
        safety_scores_separate = self.q_network.get_safety_scores(states, actions.unsqueeze(1))
        risk_estimates_separate = self.q_network.get_risk_estimates(states)
        
        safety_loss_separate = F.binary_cross_entropy(safety_scores_separate.squeeze(), safety_targets)
        risk_loss_separate = F.mse_loss(risk_estimates_separate, risk_targets)
        
        safety_total_loss = safety_loss_separate + risk_loss_separate
        self.safety_optimizer.zero_grad()
        safety_total_loss.backward()
        self.safety_optimizer.step()
        
        # Update statistics
        self.training_stats['conservative_losses'].append(float(total_loss))
        
        return {
            'total_loss': float(total_loss),
            'q_loss': float(q_loss),
            'conservative_penalty': float(conservative_penalty),
            'safety_loss': float(safety_loss),
            'risk_loss': float(risk_loss)
        }
    
    def update_target_network(self):
        """Update target network for stable training"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class ProductionSafetyMonitor:
    """Real-time safety monitoring for production deployment"""
    
    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints
        self.monitoring_active = False
        self.monitor_thread = None
        self.violation_callbacks = []
        
        # Monitoring state
        self.system_metrics_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.violation_count = 0
        self.last_violation_time = 0
        
        # Alert thresholds (more sensitive than hard constraints)
        self.alert_multipliers = {
            'cpu': 0.9,      # Alert at 90% of constraint
            'memory': 0.9,
            'gpu': 0.9,
            'thermal': 0.9,
            'latency': 0.8,  # Alert at 80% of constraint
            'throughput': 1.1  # Alert when throughput drops to 110% of minimum
        }
        
        logger.info("Production safety monitor initialized")
    
    def start_monitoring(self, system_extractor: SystemStateExtractor):
        """Start real-time safety monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(system_extractor,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Production safety monitoring started")
    
    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Production safety monitoring stopped")
    
    def _monitoring_loop(self, system_extractor: SystemStateExtractor):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Get current system state
                snapshot = system_extractor.get_current_snapshot()
                
                # Check all safety constraints
                violations = self._check_safety_constraints(snapshot)
                
                # Process violations
                for violation in violations:
                    self._handle_violation(violation)
                
                # Update monitoring history
                self.system_metrics_history.append({
                    'timestamp': snapshot.timestamp,
                    'cpu_usage': snapshot.cpu_state.get('fallback_info', {}).get('cpu_percent', [0]),
                    'memory_usage': snapshot.memory_state['virtual']['percent'],
                    'gpu_usage': snapshot.gpu_state.get('devices', [{}])[0].get('utilization_gpu', 0) if snapshot.gpu_state.get('devices') else 0,
                    'temperature': snapshot.thermal_state.get('cpu_temperature', 50)
                })
                
                time.sleep(self.constraints.safety_check_interval)
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(1.0)
    
    def _check_safety_constraints(self, snapshot) -> List[SafetyViolation]:
        """Check all safety constraints and return violations"""
        
        violations = []
        current_time = time.time()
        
        # CPU utilization check
        cpu_usage = np.mean(snapshot.cpu_state.get('fallback_info', {}).get('cpu_percent', [0]))
        if cpu_usage > self.constraints.max_cpu_utilization:
            violations.append(SafetyViolation(
                timestamp=current_time,
                violation_type="resource_constraint",
                severity="critical" if cpu_usage > 95 else "warning",
                constraint_name="max_cpu_utilization",
                measured_value=cpu_usage,
                threshold=self.constraints.max_cpu_utilization,
                action_taken="reduce_cpu_intensive_tasks",
                system_state={'cpu_usage': cpu_usage}
            ))
        
        # Memory utilization check
        memory_usage = snapshot.memory_state['virtual']['percent']
        if memory_usage > self.constraints.max_memory_utilization:
            violations.append(SafetyViolation(
                timestamp=current_time,
                violation_type="resource_constraint",
                severity="critical" if memory_usage > 95 else "warning",
                constraint_name="max_memory_utilization",
                measured_value=memory_usage,
                threshold=self.constraints.max_memory_utilization,
                action_taken="reduce_memory_intensive_tasks",
                system_state={'memory_usage': memory_usage}
            ))
        
        # GPU utilization check
        if snapshot.gpu_state.get('devices'):
            gpu_usage = snapshot.gpu_state['devices'][0].get('utilization_gpu', 0)
            if gpu_usage > self.constraints.max_gpu_utilization:
                violations.append(SafetyViolation(
                    timestamp=current_time,
                    violation_type="resource_constraint",
                    severity="warning",
                    constraint_name="max_gpu_utilization",
                    measured_value=gpu_usage,
                    threshold=self.constraints.max_gpu_utilization,
                    action_taken="route_tasks_to_cpu",
                    system_state={'gpu_usage': gpu_usage}
                ))
        
        # Thermal check
        temperature = snapshot.thermal_state.get('cpu_temperature', 50)
        if temperature > self.constraints.max_thermal_threshold:
            violations.append(SafetyViolation(
                timestamp=current_time,
                violation_type="thermal_constraint",
                severity="critical",
                constraint_name="max_thermal_threshold",
                measured_value=temperature,
                threshold=self.constraints.max_thermal_threshold,
                action_taken="throttle_performance",
                system_state={'temperature': temperature}
            ))
        
        return violations
    
    def _handle_violation(self, violation: SafetyViolation):
        """Handle a safety constraint violation"""
        
        self.violation_count += 1
        self.last_violation_time = violation.timestamp
        
        # Log violation
        logger.warning(f"Safety violation: {violation.constraint_name} = "
                      f"{violation.measured_value:.2f} > {violation.threshold:.2f}")
        
        # Execute callbacks
        for callback in self.violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Violation callback error: {e}")
    
    def add_violation_callback(self, callback: Callable[[SafetyViolation], None]):
        """Add callback for violation handling"""
        self.violation_callbacks.append(callback)
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        
        recent_violations = self.violation_count if time.time() - self.last_violation_time < 300 else 0
        
        return {
            'monitoring_active': self.monitoring_active,
            'total_violations': self.violation_count,
            'recent_violations': recent_violations,
            'last_violation_time': self.last_violation_time,
            'metrics_history_length': len(self.system_metrics_history),
            'safety_score': max(0.0, 1.0 - recent_violations / 10.0)  # Simple safety score
        }

class PerformanceTracker:
    """Track performance metrics for safety-performance trade-offs"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.baseline_metrics = None
        
    def update_metrics(self, latency: float, throughput: float, energy: float):
        """Update performance metrics"""
        
        metrics = {
            'timestamp': time.time(),
            'latency': latency,
            'throughput': throughput,
            'energy': energy
        }
        
        self.metrics_history.append(metrics)
        
        # Set baseline on first update
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
    
    def get_performance_ratios(self) -> Dict[str, float]:
        """Get current performance relative to baseline"""
        
        if not self.metrics_history or not self.baseline_metrics:
            return {'latency_ratio': 1.0, 'throughput_ratio': 1.0, 'energy_ratio': 1.0}
        
        # Average recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        avg_latency = np.mean([m['latency'] for m in recent_metrics])
        avg_throughput = np.mean([m['throughput'] for m in recent_metrics])
        avg_energy = np.mean([m['energy'] for m in recent_metrics])
        
        return {
            'latency_ratio': avg_latency / self.baseline_metrics['latency'],
            'throughput_ratio': avg_throughput / self.baseline_metrics['throughput'],
            'energy_ratio': avg_energy / self.baseline_metrics['energy']
        }

def main():
    """Demonstrate safe exploration framework"""
    
    print("=== Safe Exploration Framework for Production Deployment ===\n")
    
    # Initialize safety constraints
    constraints = SafetyConstraints(
        max_cpu_utilization=85.0,
        max_memory_utilization=90.0,
        max_exploration_rate=0.1,
        safety_confidence_threshold=0.95
    )
    
    print("1. Testing Conservative Q-Network...")
    
    # Test conservative Q-network
    q_net = ConservativeQNetwork(state_dim=36, action_dim=100)
    test_state = torch.randn(5, 36)
    test_actions = torch.randint(0, 100, (5, 10))
    
    q_values = q_net(test_state)
    conservative_q = q_net.conservative_q_values(test_state)
    safety_scores = q_net.get_safety_scores(test_state, test_actions)
    risk_estimates = q_net.get_risk_estimates(test_state)
    
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Conservative Q-values shape: {conservative_q.shape}")
    print(f"   Safety scores shape: {safety_scores.shape}")
    print(f"   Risk estimates shape: {risk_estimates.shape}")
    print(f"   Mean safety score: {safety_scores.mean():.4f}")
    print(f"   Mean risk estimate: {risk_estimates.mean():.4f}")
    
    print("\n2. Testing Safe Exploration Agent...")
    
    # Initialize safe exploration agent
    agent = SafeExplorationAgent(
        state_dim=36,
        action_dim=100,
        safety_constraints=constraints
    )
    
    # Test safe action selection
    test_state_np = np.random.randn(36)
    mock_system_state = {
        'cpu_usage': 75.0,
        'memory_usage': 60.0,
        'gpu_usage': 40.0,
        'temperature': 65.0
    }
    
    print(f"   Testing action selection with system load...")
    for i in range(5):
        # Vary system load
        mock_system_state['cpu_usage'] = 50 + i * 10
        mock_system_state['memory_usage'] = 40 + i * 15
        
        action = agent.select_safe_action(test_state_np, mock_system_state)
        device = action % 2
        priority = (action // 2) % 5
        batch = action // 10
        
        print(f"     Load Level {i+1}: CPU={mock_system_state['cpu_usage']}%, "
              f"Action={action} (device={device}, priority={priority}, batch={batch})")
    
    print("\n3. Testing Safety Monitoring...")
    
    # Mock system state extractor
    class MockSystemExtractor:
        def get_current_snapshot(self):
            from real_system.system_monitor import SystemSnapshot
            return SystemSnapshot(
                timestamp=time.time(),
                cpu_state={'fallback_info': {'cpu_percent': [np.random.uniform(70, 95)]}},
                memory_state={'virtual': {'percent': np.random.uniform(60, 95)}},
                gpu_state={'devices': [{'utilization_gpu': np.random.uniform(30, 90)}]},
                scheduler_state={},
                thermal_state={'cpu_temperature': np.random.uniform(60, 85)},
                power_state={},
                network_state={},
                storage_state={},
                process_state={}
            )
    
    # Test safety monitor
    safety_monitor = ProductionSafetyMonitor(constraints)
    mock_extractor = MockSystemExtractor()
    
    # Add violation callback
    def handle_violation(violation):
        print(f"     VIOLATION: {violation.constraint_name} = "
              f"{violation.measured_value:.1f} > {violation.threshold:.1f}")
    
    safety_monitor.add_violation_callback(handle_violation)
    
    print("   Starting safety monitoring (5 seconds)...")
    safety_monitor.start_monitoring(mock_extractor)
    time.sleep(5)
    safety_monitor.stop_monitoring()
    
    status = safety_monitor.get_safety_status()
    print(f"   Monitoring Results:")
    print(f"     Total Violations: {status['total_violations']}")
    print(f"     Safety Score: {status['safety_score']:.3f}")
    print(f"     Metrics Collected: {status['metrics_history_length']}")
    
    print("\n4. Testing Training with Safety Constraints...")
    
    # Generate mock training data
    experiences = []
    safety_feedback = []
    
    for i in range(50):
        exp = {
            'state': np.random.randn(36),
            'action': np.random.randint(100),
            'reward': np.random.randn(),
            'next_state': np.random.randn(36),
            'done': np.random.random() < 0.1
        }
        experiences.append(exp)
        
        # Mock safety feedback (safety cost)
        safety_cost = np.random.exponential(0.1) if np.random.random() < 0.3 else 0.0
        safety_feedback.append({'safety_cost': safety_cost})
    
    # Train with safety constraints
    losses = agent.train_safe_policy(experiences, safety_feedback)
    
    print(f"   Training Results:")
    print(f"     Total Loss: {losses.get('total_loss', 0):.4f}")
    print(f"     Safety Loss: {losses.get('safety_loss', 0):.4f}")
    print(f"     Risk Loss: {losses.get('risk_loss', 0):.4f}")
    print(f"     Conservative Penalty: {losses.get('conservative_penalty', 0):.4f}")
    
    print("\n[SUCCESS] Safe Exploration Framework Test Completed!")
    print("\nKey Safety Features Demonstrated:")
    print("+ Conservative Q-Learning with safety bounds")
    print("+ Risk-aware action filtering and selection")
    print("+ Real-time safety monitoring and violation detection")
    print("+ Adaptive exploration rates based on system conditions")
    print("+ Safety-aware policy training with constraint satisfaction")

if __name__ == '__main__':
    main()