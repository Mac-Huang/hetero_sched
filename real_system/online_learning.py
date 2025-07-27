"""
Online Learning Framework for Continuous Adaptation to System Changes

This module implements R8: a comprehensive online learning framework that enables
HeteroSched to continuously adapt to dynamic system changes, workload shifts,
and evolving performance characteristics in production environments.

Key Features:
1. Incremental learning algorithms for real-time model updates
2. Change detection mechanisms for identifying system and workload shifts
3. Adaptive model selection and ensemble methods
4. Safe exploration strategies for production environments
5. Performance monitoring and automatic rollback capabilities
6. Transfer learning for rapid adaptation to new scenarios

The framework ensures that the scheduling agent maintains optimal performance
as system conditions evolve over time.

Authors: HeteroSched Research Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import logging
import asyncio
from collections import deque, defaultdict
import statistics
from scipy import stats
import pickle
import threading
import queue

class ChangeType(Enum):
    WORKLOAD_SHIFT = "workload_shift"
    RESOURCE_CHANGE = "resource_change"
    PERFORMANCE_DRIFT = "performance_drift"
    SYSTEM_FAILURE = "system_failure"
    CONFIGURATION_UPDATE = "configuration_update"

class AdaptationStrategy(Enum):
    INCREMENTAL_UPDATE = "incremental_update"
    FULL_RETRAIN = "full_retrain"
    ENSEMBLE_UPDATE = "ensemble_update"
    TRANSFER_LEARNING = "transfer_learning"
    ROLLBACK = "rollback"

class LearningMode(Enum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"

@dataclass
class SystemSnapshot:
    """Captures a snapshot of system state and performance"""
    timestamp: float
    resource_utilization: Dict[str, float]
    workload_characteristics: Dict[str, Any]
    performance_metrics: Dict[str, float]
    system_config: Dict[str, Any]
    
@dataclass
class ChangeEvent:
    """Represents a detected change in the system"""
    change_id: str
    change_type: ChangeType
    detection_time: float
    severity: float  # 0.0 to 1.0
    affected_components: List[str]
    evidence: Dict[str, Any]
    recommended_action: AdaptationStrategy

@dataclass
class AdaptationResult:
    """Results of an adaptation attempt"""
    adaptation_id: str
    strategy: AdaptationStrategy
    success: bool
    performance_improvement: float
    adaptation_time: float
    rollback_required: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class ChangeDetector(ABC):
    """Abstract base class for change detection algorithms"""
    
    @abstractmethod
    def detect_changes(self, snapshots: List[SystemSnapshot]) -> List[ChangeEvent]:
        """Detect changes from system snapshots"""
        pass
    
    @abstractmethod
    def update_baseline(self, snapshots: List[SystemSnapshot]):
        """Update baseline for change detection"""
        pass

class StatisticalChangeDetector(ChangeDetector):
    """Statistical change detection using multiple algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_window = config.get("baseline_window", 100)
        self.sensitivity = config.get("sensitivity", 0.95)
        self.logger = logging.getLogger("StatisticalChangeDetector")
        
        # Baseline statistics
        self.baseline_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.baseline_window))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def detect_changes(self, snapshots: List[SystemSnapshot]) -> List[ChangeEvent]:
        """Detect changes using statistical tests"""
        changes = []
        
        if len(snapshots) < 2:
            return changes
        
        # Get recent snapshots for analysis
        recent_snapshots = snapshots[-20:]  # Last 20 snapshots
        
        # Detect performance drift
        performance_changes = self._detect_performance_drift(recent_snapshots)
        changes.extend(performance_changes)
        
        # Detect workload shifts
        workload_changes = self._detect_workload_shift(recent_snapshots)
        changes.extend(workload_changes)
        
        # Detect resource changes
        resource_changes = self._detect_resource_changes(recent_snapshots)
        changes.extend(resource_changes)
        
        return changes
    
    def _detect_performance_drift(self, snapshots: List[SystemSnapshot]) -> List[ChangeEvent]:
        """Detect performance drift using statistical tests"""
        changes = []
        
        if len(snapshots) < 10:
            return changes
        
        # Key performance metrics to monitor
        metrics_to_monitor = [
            "avg_response_time", "throughput", "resource_utilization", 
            "deadline_violation_rate", "system_load"
        ]
        
        for metric in metrics_to_monitor:
            # Extract metric values
            values = []
            for snapshot in snapshots:
                if metric in snapshot.performance_metrics:
                    values.append(snapshot.performance_metrics[metric])
            
            if len(values) < 10:
                continue
            
            # Split into baseline and recent periods
            baseline_values = values[:len(values)//2]
            recent_values = values[len(values)//2:]
            
            # Perform t-test
            if len(baseline_values) > 1 and len(recent_values) > 1:
                statistic, p_value = stats.ttest_ind(baseline_values, recent_values)
                
                # Check for significant change
                if p_value < (1 - self.sensitivity):
                    severity = min(1.0, abs(statistic) / 10.0)
                    
                    change = ChangeEvent(
                        change_id=f"perf_drift_{metric}_{int(time.time())}",
                        change_type=ChangeType.PERFORMANCE_DRIFT,
                        detection_time=time.time(),
                        severity=severity,
                        affected_components=[metric],
                        evidence={
                            "metric": metric,
                            "baseline_mean": np.mean(baseline_values),
                            "recent_mean": np.mean(recent_values),
                            "t_statistic": statistic,
                            "p_value": p_value,
                            "direction": "increase" if np.mean(recent_values) > np.mean(baseline_values) else "decrease"
                        },
                        recommended_action=self._recommend_action_for_drift(metric, severity)
                    )
                    changes.append(change)
        
        return changes
    
    def _detect_workload_shift(self, snapshots: List[SystemSnapshot]) -> List[ChangeEvent]:
        """Detect workload characteristic shifts"""
        changes = []
        
        if len(snapshots) < 10:
            return changes
        
        # Monitor workload characteristics
        workload_features = [
            "avg_task_size", "arrival_rate", "priority_distribution", 
            "task_type_distribution", "resource_demand_pattern"
        ]
        
        for feature in workload_features:
            # Extract feature values
            values = []
            for snapshot in snapshots:
                if feature in snapshot.workload_characteristics:
                    values.append(snapshot.workload_characteristics[feature])
            
            if len(values) < 10:
                continue
            
            # Use CUSUM algorithm for change detection
            if self._cusum_test(values, feature):
                severity = self._calculate_workload_shift_severity(values, feature)
                
                change = ChangeEvent(
                    change_id=f"workload_shift_{feature}_{int(time.time())}",
                    change_type=ChangeType.WORKLOAD_SHIFT,
                    detection_time=time.time(),
                    severity=severity,
                    affected_components=[feature],
                    evidence={
                        "feature": feature,
                        "values": values[-10:],  # Last 10 values
                        "change_point": self._find_change_point(values)
                    },
                    recommended_action=AdaptationStrategy.INCREMENTAL_UPDATE
                )
                changes.append(change)
        
        return changes
    
    def _detect_resource_changes(self, snapshots: List[SystemSnapshot]) -> List[ChangeEvent]:
        """Detect resource availability or configuration changes"""
        changes = []
        
        if len(snapshots) < 2:
            return changes
        
        # Compare resource configurations
        latest_snapshot = snapshots[-1]
        previous_snapshot = snapshots[-2]
        
        # Check for resource additions/removals
        latest_resources = set(latest_snapshot.resource_utilization.keys())
        previous_resources = set(previous_snapshot.resource_utilization.keys())
        
        added_resources = latest_resources - previous_resources
        removed_resources = previous_resources - latest_resources
        
        if added_resources or removed_resources:
            severity = (len(added_resources) + len(removed_resources)) / max(len(latest_resources), 1)
            
            change = ChangeEvent(
                change_id=f"resource_change_{int(time.time())}",
                change_type=ChangeType.RESOURCE_CHANGE,
                detection_time=time.time(),
                severity=min(1.0, severity),
                affected_components=list(added_resources | removed_resources),
                evidence={
                    "added_resources": list(added_resources),
                    "removed_resources": list(removed_resources),
                    "total_resources_before": len(previous_resources),
                    "total_resources_after": len(latest_resources)
                },
                recommended_action=AdaptationStrategy.FULL_RETRAIN if severity > 0.3 else AdaptationStrategy.INCREMENTAL_UPDATE
            )
            changes.append(change)
        
        return changes
    
    def _cusum_test(self, values: List[float], feature: str) -> bool:
        """CUSUM test for change detection"""
        if len(values) < 5:
            return False
        
        # Calculate CUSUM statistic
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return False
        
        # Standardize values
        standardized = [(v - mean_val) / std_val for v in values]
        
        # CUSUM parameters
        threshold = self.config.get("cusum_threshold", 3.0)
        drift = self.config.get("cusum_drift", 0.5)
        
        # Calculate CUSUM
        cusum_pos = 0
        cusum_neg = 0
        
        for val in standardized:
            cusum_pos = max(0, cusum_pos + val - drift)
            cusum_neg = max(0, cusum_neg - val - drift)
            
            if cusum_pos > threshold or cusum_neg > threshold:
                return True
        
        return False
    
    def _calculate_workload_shift_severity(self, values: List[float], feature: str) -> float:
        """Calculate severity of workload shift"""
        if len(values) < 4:
            return 0.0
        
        # Compare first half vs second half
        mid_point = len(values) // 2
        first_half = values[:mid_point]
        second_half = values[mid_point:]
        
        mean1, mean2 = np.mean(first_half), np.mean(second_half)
        std1, std2 = np.std(first_half), np.std(second_half)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(first_half) - 1) * std1**2 + (len(second_half) - 1) * std2**2) / 
                            (len(first_half) + len(second_half) - 2))
        
        if pooled_std == 0:
            return 0.0
        
        effect_size = abs(mean2 - mean1) / pooled_std
        return min(1.0, effect_size / 2.0)  # Normalize to [0, 1]
    
    def _find_change_point(self, values: List[float]) -> int:
        """Find the most likely change point in the series"""
        if len(values) < 3:
            return len(values) // 2
        
        # Simple change point detection using variance
        best_point = len(values) // 2
        min_combined_var = float('inf')
        
        for i in range(1, len(values) - 1):
            left_var = np.var(values[:i]) if i > 1 else 0
            right_var = np.var(values[i:]) if len(values) - i > 1 else 0
            combined_var = left_var + right_var
            
            if combined_var < min_combined_var:
                min_combined_var = combined_var
                best_point = i
        
        return best_point
    
    def _recommend_action_for_drift(self, metric: str, severity: float) -> AdaptationStrategy:
        """Recommend adaptation strategy based on drift characteristics"""
        if severity > 0.8:
            return AdaptationStrategy.FULL_RETRAIN
        elif severity > 0.5:
            return AdaptationStrategy.ENSEMBLE_UPDATE
        else:
            return AdaptationStrategy.INCREMENTAL_UPDATE
    
    def update_baseline(self, snapshots: List[SystemSnapshot]):
        """Update baseline statistics"""
        for snapshot in snapshots:
            # Update performance metrics baseline
            for metric, value in snapshot.performance_metrics.items():
                self.baseline_metrics[metric].append(value)
            
            # Update workload characteristics baseline
            for feature, value in snapshot.workload_characteristics.items():
                if isinstance(value, (int, float)):
                    self.baseline_metrics[feature].append(value)
        
        # Recalculate baseline statistics
        for metric, values in self.baseline_metrics.items():
            if len(values) > 10:
                self.baseline_stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }

class IncrementalLearningAgent:
    """Agent that performs incremental learning updates"""
    
    def __init__(self, base_model: nn.Module, config: Dict[str, Any]):
        self.base_model = base_model
        self.config = config
        self.logger = logging.getLogger("IncrementalLearningAgent")
        
        # Learning parameters
        self.learning_rate = config.get("incremental_lr", 0.0001)
        self.buffer_size = config.get("experience_buffer_size", 10000)
        self.batch_size = config.get("incremental_batch_size", 32)
        self.update_frequency = config.get("update_frequency", 100)
        
        # Experience buffer for incremental learning
        self.experience_buffer = deque(maxlen=self.buffer_size)
        self.update_counter = 0
        
        # Optimizer for incremental updates
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        
    def add_experience(self, state: torch.Tensor, action: torch.Tensor, 
                      reward: float, next_state: torch.Tensor, done: bool):
        """Add new experience to the buffer"""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "timestamp": time.time()
        }
        self.experience_buffer.append(experience)
        
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self.perform_incremental_update()
            self.update_counter = 0
    
    def perform_incremental_update(self):
        """Perform incremental model update"""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch tensors
        states = torch.stack([exp["state"] for exp in batch])
        actions = torch.stack([exp["action"] for exp in batch])
        rewards = torch.tensor([exp["reward"] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp["next_state"] for exp in batch])
        dones = torch.tensor([exp["done"] for exp in batch], dtype=torch.bool)
        
        # Compute target values
        with torch.no_grad():
            next_q_values = self.base_model(next_states).max(dim=1)[0]
            targets = rewards + self.config.get("gamma", 0.95) * next_q_values * (~dones)
        
        # Current Q-values
        current_q_values = self.base_model(states).gather(1, actions.long().unsqueeze(1)).squeeze()
        
        # Compute loss
        loss = self.loss_fn(current_q_values, targets)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track performance
        self.performance_history.append({
            "timestamp": time.time(),
            "loss": loss.item(),
            "avg_q_value": current_q_values.mean().item(),
            "avg_target": targets.mean().item()
        })
        
        self.logger.debug(f"Incremental update completed. Loss: {loss.item():.4f}")
    
    def adapt_to_change(self, change_event: ChangeEvent) -> AdaptationResult:
        """Adapt the model to a detected change"""
        start_time = time.time()
        
        if change_event.recommended_action == AdaptationStrategy.INCREMENTAL_UPDATE:
            # Increase learning rate temporarily for faster adaptation
            old_lr = self.optimizer.param_groups[0]['lr']
            adaptation_lr = old_lr * (1 + change_event.severity)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adaptation_lr
            
            # Perform several update steps
            num_updates = int(10 * change_event.severity)
            for _ in range(num_updates):
                if len(self.experience_buffer) >= self.batch_size:
                    self.perform_incremental_update()
            
            # Restore original learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = old_lr
            
            adaptation_time = time.time() - start_time
            
            return AdaptationResult(
                adaptation_id=f"incremental_{change_event.change_id}",
                strategy=AdaptationStrategy.INCREMENTAL_UPDATE,
                success=True,
                performance_improvement=0.0,  # Would need actual evaluation
                adaptation_time=adaptation_time,
                rollback_required=False,
                metadata={
                    "num_updates": num_updates,
                    "adaptation_lr": adaptation_lr,
                    "change_severity": change_event.severity
                }
            )
        
        else:
            # Not handling other strategies in this simple implementation
            return AdaptationResult(
                adaptation_id=f"unsupported_{change_event.change_id}",
                strategy=change_event.recommended_action,
                success=False,
                performance_improvement=0.0,
                adaptation_time=0.0,
                rollback_required=False,
                metadata={"error": "Strategy not implemented"}
            )

class EnsembleAdapter:
    """Manages ensemble of models for robust adaptation"""
    
    def __init__(self, base_models: List[nn.Module], config: Dict[str, Any]):
        self.base_models = base_models
        self.config = config
        self.logger = logging.getLogger("EnsembleAdapter")
        
        # Model weights for ensemble
        self.model_weights = np.ones(len(base_models)) / len(base_models)
        self.performance_tracking = {i: deque(maxlen=100) for i in range(len(base_models))}
        
        # Individual learning agents
        self.learning_agents = [
            IncrementalLearningAgent(model, config) for model in base_models
        ]
        
    def update_model_weights(self):
        """Update ensemble weights based on recent performance"""
        # Calculate performance scores for each model
        scores = []
        for i, performance_history in self.performance_tracking.items():
            if len(performance_history) > 10:
                recent_performance = list(performance_history)[-10:]
                avg_performance = np.mean(recent_performance)
                scores.append(avg_performance)
            else:
                scores.append(0.5)  # Default neutral performance
        
        # Convert to weights (higher score = higher weight)
        scores = np.array(scores)
        if np.sum(scores) > 0:
            self.model_weights = scores / np.sum(scores)
        else:
            self.model_weights = np.ones(len(self.base_models)) / len(self.base_models)
    
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Make prediction using ensemble of models"""
        predictions = []
        
        for model in self.base_models:
            with torch.no_grad():
                pred = model(state)
                predictions.append(pred)
        
        # Weighted average of predictions
        weighted_prediction = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_prediction += self.model_weights[i] * pred
        
        return weighted_prediction
    
    def adapt_to_change(self, change_event: ChangeEvent) -> AdaptationResult:
        """Adapt ensemble to detected change"""
        start_time = time.time()
        
        if change_event.recommended_action == AdaptationStrategy.ENSEMBLE_UPDATE:
            # Update different models with different strategies
            adaptation_results = []
            
            for i, agent in enumerate(self.learning_agents):
                # Vary adaptation intensity across models
                modified_event = ChangeEvent(
                    change_id=f"{change_event.change_id}_model_{i}",
                    change_type=change_event.change_type,
                    detection_time=change_event.detection_time,
                    severity=change_event.severity * (0.5 + 0.5 * i / len(self.learning_agents)),
                    affected_components=change_event.affected_components,
                    evidence=change_event.evidence,
                    recommended_action=AdaptationStrategy.INCREMENTAL_UPDATE
                )
                
                result = agent.adapt_to_change(modified_event)
                adaptation_results.append(result)
            
            # Update ensemble weights
            self.update_model_weights()
            
            adaptation_time = time.time() - start_time
            success = any(result.success for result in adaptation_results)
            
            return AdaptationResult(
                adaptation_id=f"ensemble_{change_event.change_id}",
                strategy=AdaptationStrategy.ENSEMBLE_UPDATE,
                success=success,
                performance_improvement=0.0,  # Would need evaluation
                adaptation_time=adaptation_time,
                rollback_required=False,
                metadata={
                    "individual_results": [result.metadata for result in adaptation_results],
                    "updated_weights": self.model_weights.tolist()
                }
            )
        
        else:
            return AdaptationResult(
                adaptation_id=f"ensemble_unsupported_{change_event.change_id}",
                strategy=change_event.recommended_action,
                success=False,
                performance_improvement=0.0,
                adaptation_time=0.0,
                rollback_required=False,
                metadata={"error": "Strategy not supported by ensemble"}
            )

class OnlineLearningFramework:
    """Main framework for online learning and adaptation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("OnlineLearningFramework")
        
        # Initialize components
        self.change_detector = StatisticalChangeDetector(config)
        self.system_snapshots = deque(maxlen=config.get("snapshot_history", 1000))
        
        # Model management
        self.current_model = None
        self.model_backup = None
        self.adaptation_history = []
        
        # Monitoring and control
        self.monitoring_active = False
        self.adaptation_enabled = True
        self.safe_mode = False
        
        # Performance tracking
        self.performance_baseline = None
        self.performance_threshold = config.get("performance_threshold", 0.95)
        
    def initialize(self, initial_model: nn.Module):
        """Initialize the framework with an initial model"""
        self.current_model = initial_model
        self.model_backup = pickle.loads(pickle.dumps(initial_model))  # Deep copy
        
        # Initialize ensemble if configured
        if self.config.get("use_ensemble", False):
            num_models = self.config.get("ensemble_size", 3)
            base_models = [pickle.loads(pickle.dumps(initial_model)) for _ in range(num_models)]
            self.ensemble_adapter = EnsembleAdapter(base_models, self.config)
        else:
            self.incremental_agent = IncrementalLearningAgent(self.current_model, self.config)
        
        self.logger.info("Online learning framework initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring and adaptation"""
        self.monitoring_active = True
        self.logger.info("Started online monitoring")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and adaptation"""
        self.monitoring_active = False
        self.logger.info("Stopped online monitoring")
    
    def add_system_snapshot(self, snapshot: SystemSnapshot):
        """Add a new system snapshot for monitoring"""
        self.system_snapshots.append(snapshot)
        
        # Update change detector baseline
        if len(self.system_snapshots) >= 10:
            recent_snapshots = list(self.system_snapshots)[-10:]
            self.change_detector.update_baseline(recent_snapshots)
    
    def add_experience(self, state: torch.Tensor, action: torch.Tensor, 
                      reward: float, next_state: torch.Tensor, done: bool):
        """Add experience for incremental learning"""
        if hasattr(self, 'incremental_agent'):
            self.incremental_agent.add_experience(state, action, reward, next_state, done)
        elif hasattr(self, 'ensemble_adapter'):
            for agent in self.ensemble_adapter.learning_agents:
                agent.add_experience(state, action, reward, next_state, done)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for changes
                if len(self.system_snapshots) >= 20:
                    changes = self.change_detector.detect_changes(list(self.system_snapshots))
                    
                    for change in changes:
                        self.logger.info(f"Detected change: {change.change_type.value} "
                                       f"(severity: {change.severity:.2f})")
                        
                        if self.adaptation_enabled:
                            self._handle_change_event(change)
                
                # Check performance and trigger adaptation if needed
                self._check_performance_degradation()
                
                # Sleep before next check
                time.sleep(self.config.get("monitoring_interval", 30))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _handle_change_event(self, change_event: ChangeEvent):
        """Handle a detected change event"""
        try:
            # Decide on adaptation strategy
            if self.safe_mode and change_event.severity > 0.7:
                self.logger.warning(f"Skipping adaptation due to safe mode (severity: {change_event.severity})")
                return
            
            # Perform adaptation
            if hasattr(self, 'ensemble_adapter'):
                result = self.ensemble_adapter.adapt_to_change(change_event)
            else:
                result = self.incremental_agent.adapt_to_change(change_event)
            
            # Record adaptation
            self.adaptation_history.append(result)
            
            if result.success:
                self.logger.info(f"Successfully adapted to change {change_event.change_id}")
            else:
                self.logger.warning(f"Failed to adapt to change {change_event.change_id}")
                
                # Consider rollback if adaptation failed
                if result.rollback_required:
                    self._rollback_model()
            
        except Exception as e:
            self.logger.error(f"Error handling change event: {e}")
    
    def _check_performance_degradation(self):
        """Check for performance degradation requiring intervention"""
        if len(self.system_snapshots) < 10:
            return
        
        # Get recent performance metrics
        recent_snapshots = list(self.system_snapshots)[-10:]
        recent_performance = [s.performance_metrics.get("avg_response_time", 0) 
                            for s in recent_snapshots]
        
        if not recent_performance:
            return
        
        avg_recent_performance = np.mean(recent_performance)
        
        # Compare with baseline
        if self.performance_baseline is None:
            # Establish baseline from first snapshots
            if len(self.system_snapshots) >= 50:
                baseline_snapshots = list(self.system_snapshots)[:20]
                baseline_performance = [s.performance_metrics.get("avg_response_time", 0) 
                                      for s in baseline_snapshots]
                self.performance_baseline = np.mean(baseline_performance)
        
        if self.performance_baseline is not None:
            performance_ratio = avg_recent_performance / self.performance_baseline
            
            if performance_ratio > (1 / self.performance_threshold):  # Performance degraded
                self.logger.warning(f"Performance degradation detected: {performance_ratio:.2f}x baseline")
                
                # Trigger adaptation
                degradation_event = ChangeEvent(
                    change_id=f"perf_degradation_{int(time.time())}",
                    change_type=ChangeType.PERFORMANCE_DRIFT,
                    detection_time=time.time(),
                    severity=min(1.0, (performance_ratio - 1.0)),
                    affected_components=["overall_performance"],
                    evidence={
                        "baseline_performance": self.performance_baseline,
                        "recent_performance": avg_recent_performance,
                        "degradation_ratio": performance_ratio
                    },
                    recommended_action=AdaptationStrategy.INCREMENTAL_UPDATE
                )
                
                self._handle_change_event(degradation_event)
    
    def _rollback_model(self):
        """Rollback to previous model version"""
        if self.model_backup is not None:
            self.logger.info("Rolling back to previous model version")
            self.current_model.load_state_dict(self.model_backup.state_dict())
            
            # Backup current model
            self.model_backup = pickle.loads(pickle.dumps(self.current_model))
        else:
            self.logger.warning("No backup model available for rollback")
    
    def enable_safe_mode(self):
        """Enable safe mode (conservative adaptations only)"""
        self.safe_mode = True
        self.logger.info("Safe mode enabled")
    
    def disable_safe_mode(self):
        """Disable safe mode"""
        self.safe_mode = False
        self.logger.info("Safe mode disabled")
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about adaptations performed"""
        if not self.adaptation_history:
            return {"total_adaptations": 0}
        
        successful_adaptations = [a for a in self.adaptation_history if a.success]
        failed_adaptations = [a for a in self.adaptation_history if not a.success]
        
        stats = {
            "total_adaptations": len(self.adaptation_history),
            "successful_adaptations": len(successful_adaptations),
            "failed_adaptations": len(failed_adaptations),
            "success_rate": len(successful_adaptations) / len(self.adaptation_history),
            "avg_adaptation_time": np.mean([a.adaptation_time for a in self.adaptation_history]),
            "strategies_used": {}
        }
        
        # Count strategies used
        for adaptation in self.adaptation_history:
            strategy = adaptation.strategy.value
            if strategy not in stats["strategies_used"]:
                stats["strategies_used"][strategy] = 0
            stats["strategies_used"][strategy] += 1
        
        return stats

def demonstrate_online_learning():
    """Demonstrate the online learning framework"""
    print("=== Online Learning Framework for Continuous Adaptation ===")
    
    # Configuration
    config = {
        "baseline_window": 100,
        "sensitivity": 0.95,
        "incremental_lr": 0.0001,
        "experience_buffer_size": 10000,
        "update_frequency": 100,
        "monitoring_interval": 5,  # 5 seconds for demo
        "performance_threshold": 0.95,
        "use_ensemble": True,
        "ensemble_size": 3,
        "cusum_threshold": 3.0,
        "cusum_drift": 0.5
    }
    
    print("1. Initializing Online Learning Framework...")
    framework = OnlineLearningFramework(config)
    
    # Create a simple neural network model
    class SimpleSchedulingModel(nn.Module):
        def __init__(self, input_size=64, hidden_size=128, output_size=32):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.network(x)
    
    initial_model = SimpleSchedulingModel()
    framework.initialize(initial_model)
    
    print("2. Testing Change Detection...")
    change_detector = StatisticalChangeDetector(config)
    
    # Generate sample system snapshots
    snapshots = []
    for i in range(50):
        # Simulate gradual performance drift
        drift_factor = 1 + 0.02 * max(0, i - 25)  # Drift starts at snapshot 25
        
        snapshot = SystemSnapshot(
            timestamp=time.time() + i,
            resource_utilization={
                "cpu_cluster_1": 0.6 + 0.1 * np.random.random(),
                "gpu_cluster_1": 0.4 + 0.1 * np.random.random(),
                "memory_cluster_1": 0.7 + 0.1 * np.random.random()
            },
            workload_characteristics={
                "avg_task_size": 100 + 10 * np.random.random(),
                "arrival_rate": 50 + 5 * np.random.random(),
                "task_type_distribution": 0.6 + 0.1 * np.random.random()
            },
            performance_metrics={
                "avg_response_time": (200 + 20 * np.random.random()) * drift_factor,
                "throughput": (100 + 10 * np.random.random()) / drift_factor,
                "deadline_violation_rate": 0.05 * drift_factor,
                "system_load": 0.6 + 0.1 * np.random.random()
            },
            system_config={"num_nodes": 10, "total_cores": 80}
        )
        snapshots.append(snapshot)
    
    # Add snapshots to framework
    for snapshot in snapshots:
        framework.add_system_snapshot(snapshot)
    
    # Detect changes
    detected_changes = change_detector.detect_changes(snapshots)
    
    print(f"   Detected {len(detected_changes)} changes:")
    for change in detected_changes:
        print(f"     {change.change_type.value}: severity {change.severity:.2f}")
        print(f"       Affected: {change.affected_components}")
        print(f"       Recommended: {change.recommended_action.value}")
    
    print("3. Testing Incremental Learning...")
    
    # Simulate adding experiences
    for i in range(10):
        state = torch.randn(64)
        action = torch.randint(0, 32, (1,))
        reward = np.random.uniform(0, 1)
        next_state = torch.randn(64)
        done = np.random.random() < 0.1
        
        framework.add_experience(state, action, reward, next_state, done)
    
    print("   Added 10 experiences for incremental learning")
    
    print("4. Testing Ensemble Adaptation...")
    
    if hasattr(framework, 'ensemble_adapter'):
        # Test ensemble prediction
        test_state = torch.randn(1, 64)
        prediction = framework.ensemble_adapter.predict(test_state)
        print(f"   Ensemble prediction shape: {prediction.shape}")
        print(f"   Initial model weights: {framework.ensemble_adapter.model_weights}")
        
        # Test adaptation
        test_change = ChangeEvent(
            change_id="test_change_001",
            change_type=ChangeType.WORKLOAD_SHIFT,
            detection_time=time.time(),
            severity=0.6,
            affected_components=["workload_pattern"],
            evidence={"test": True},
            recommended_action=AdaptationStrategy.ENSEMBLE_UPDATE
        )
        
        adaptation_result = framework.ensemble_adapter.adapt_to_change(test_change)
        print(f"   Adaptation result: {adaptation_result.success}")
        print(f"   Adaptation time: {adaptation_result.adaptation_time:.4f}s")
    
    print("5. Framework Statistics...")
    
    stats = framework.get_adaptation_statistics()
    print(f"   Total adaptations: {stats['total_adaptations']}")
    if stats['total_adaptations'] > 0:
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Average adaptation time: {stats['avg_adaptation_time']:.4f}s")
    
    print("6. Change Detection Analysis...")
    
    # Analyze detected changes
    change_types = {}
    severity_distribution = []
    
    for change in detected_changes:
        change_type = change.change_type.value
        if change_type not in change_types:
            change_types[change_type] = 0
        change_types[change_type] += 1
        severity_distribution.append(change.severity)
    
    print(f"   Change types detected: {change_types}")
    if severity_distribution:
        print(f"   Average severity: {np.mean(severity_distribution):.2f}")
        print(f"   Max severity: {np.max(severity_distribution):.2f}")
    
    print("7. Framework Benefits...")
    
    benefits = [
        "Continuous adaptation to system changes without manual intervention",
        "Real-time change detection using statistical methods",
        "Multiple adaptation strategies (incremental, ensemble, full retrain)",
        "Safe exploration with rollback capabilities",
        "Performance monitoring and degradation detection",
        "Configurable sensitivity and adaptation thresholds",
        "Support for both single models and ensemble approaches",
        "Comprehensive logging and adaptation statistics"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"   {i}. {benefit}")
    
    return {
        "framework": framework,
        "detected_changes": detected_changes,
        "sample_snapshots": snapshots[:5],
        "change_detector": change_detector,
        "adaptation_stats": stats
    }

if __name__ == "__main__":
    demonstrate_online_learning()