#!/usr/bin/env python3
"""
Individual Objective Metrics for Multi-Objective Reward Functions

Each metric class computes normalized scores for different objectives
that can be combined into multi-objective reward functions.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

@dataclass
class TaskResult:
    """Task execution result for metric computation"""
    execution_time: float
    queue_wait: float
    total_latency: float
    energy: float
    throughput: float
    device: str
    batch_size: int
    thermal_violation: bool
    slo_violation: bool
    priority: int
    task_size: int
    task_type: str

class ObjectiveMetric(ABC):
    """Abstract base class for objective metrics"""
    
    @abstractmethod
    def compute(self, task_result: TaskResult, system_state: Dict) -> float:
        """Compute normalized metric score in [-1, 1] range"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name"""
        pass

class LatencyMetric(ObjectiveMetric):
    """Minimize task latency objective"""
    
    def __init__(self, target_latency: float = 10.0, penalty_scale: float = 2.0):
        self.target_latency = target_latency  # Target latency in ms
        self.penalty_scale = penalty_scale    # Penalty scaling factor
        self.historical_latencies = []
        self.window_size = 100
    
    def compute(self, task_result: TaskResult, system_state: Dict) -> float:
        """Compute latency reward score"""
        latency = task_result.total_latency
        
        # Update historical data
        self.historical_latencies.append(latency)
        if len(self.historical_latencies) > self.window_size:
            self.historical_latencies.pop(0)
        
        # Adaptive target based on recent performance
        if len(self.historical_latencies) > 10:
            recent_median = np.median(self.historical_latencies[-20:])
            adaptive_target = max(self.target_latency, recent_median * 0.8)
        else:
            adaptive_target = self.target_latency
        
        # Normalized latency score
        if latency <= adaptive_target:
            # Reward for meeting target
            score = 1.0 - (latency / adaptive_target) * 0.5
        else:
            # Penalty for exceeding target (capped exponential to prevent overflow)
            excess_ratio = (latency - adaptive_target) / adaptive_target
            # Cap excess_ratio to prevent math overflow
            capped_ratio = min(excess_ratio, 10.0)
            score = -self.penalty_scale * (math.exp(capped_ratio) - 1)
        
        # Additional penalty for extreme latencies
        if latency > adaptive_target * 10:
            score -= 5.0
        
        return np.clip(score, -10.0, 1.0)
    
    def get_name(self) -> str:
        return "latency"

class EnergyMetric(ObjectiveMetric):
    """Minimize energy consumption objective"""
    
    def __init__(self, target_energy: float = 1.0, efficiency_weight: float = 0.5):
        self.target_energy = target_energy      # Target energy in Joules
        self.efficiency_weight = efficiency_weight
        self.energy_history = []
        self.window_size = 100
    
    def compute(self, task_result: TaskResult, system_state: Dict) -> float:
        """Compute energy efficiency reward score"""
        energy = task_result.energy
        
        # Track energy consumption history
        self.energy_history.append(energy)
        if len(self.energy_history) > self.window_size:
            self.energy_history.pop(0)
        
        # Adaptive energy target
        if len(self.energy_history) > 10:
            recent_median = np.median(self.energy_history[-20:])
            adaptive_target = max(self.target_energy, recent_median * 0.9)
        else:
            adaptive_target = self.target_energy
        
        # Energy efficiency score
        if energy <= adaptive_target:
            score = 1.0 - (energy / adaptive_target) * 0.6
        else:
            excess_ratio = (energy - adaptive_target) / adaptive_target
            score = -excess_ratio * 2.0
        
        # Bonus for energy efficiency relative to computation
        work_per_joule = task_result.task_size / max(energy, 0.001)
        if len(self.energy_history) > 5:
            avg_efficiency = np.mean([task_result.task_size / max(e, 0.001) for e in self.energy_history[-10:]])
            if work_per_joule > avg_efficiency * 1.2:
                score += 0.3  # Efficiency bonus
        
        return np.clip(score, -5.0, 1.5)
    
    def get_name(self) -> str:
        return "energy"

class ThroughputMetric(ObjectiveMetric):
    """Maximize system throughput objective"""
    
    def __init__(self, target_throughput: float = 100.0):
        self.target_throughput = target_throughput  # Tasks per second
        self.throughput_history = []
        self.window_size = 50
    
    def compute(self, task_result: TaskResult, system_state: Dict) -> float:
        """Compute throughput reward score"""
        throughput = task_result.throughput
        
        # Track throughput history
        self.throughput_history.append(throughput)
        if len(self.throughput_history) > self.window_size:
            self.throughput_history.pop(0)
        
        # Adaptive throughput target
        if len(self.throughput_history) > 10:
            recent_max = np.max(self.throughput_history[-20:])
            adaptive_target = min(self.target_throughput, recent_max * 1.1)
        else:
            adaptive_target = self.target_throughput
        
        # Throughput score
        if throughput >= adaptive_target:
            score = 1.0
        else:
            score = throughput / adaptive_target
        
        # Bonus for sustained high throughput
        if len(self.throughput_history) >= 10:
            recent_avg = np.mean(self.throughput_history[-10:])
            if recent_avg > adaptive_target * 0.8:
                score += 0.2  # Consistency bonus
        
        # Batch size efficiency bonus
        if task_result.batch_size > 1:
            batch_efficiency = min(0.3, task_result.batch_size * 0.05)
            score += batch_efficiency
        
        return np.clip(score, 0.0, 2.0)
    
    def get_name(self) -> str:
        return "throughput"

class FairnessMetric(ObjectiveMetric):
    """Ensure fairness across task priorities and types"""
    
    def __init__(self):
        self.priority_history = {}  # Priority -> [latencies]
        self.type_history = {}      # Type -> [latencies] 
        self.window_size = 100
    
    def compute(self, task_result: TaskResult, system_state: Dict) -> float:
        """Compute fairness reward score"""
        priority = task_result.priority
        task_type = task_result.task_type
        latency = task_result.total_latency
        
        # Track per-priority performance
        if priority not in self.priority_history:
            self.priority_history[priority] = []
        self.priority_history[priority].append(latency)
        
        # Track per-type performance
        if task_type not in self.type_history:
            self.type_history[task_type] = []
        self.type_history[task_type].append(latency)
        
        # Maintain window size
        for hist in self.priority_history.values():
            if len(hist) > self.window_size:
                hist.pop(0)
        for hist in self.type_history.values():
            if len(hist) > self.window_size:
                hist.pop(0)
        
        score = 0.0
        
        # Priority fairness: higher priority should have lower latency
        if len(self.priority_history) > 1:
            priority_fairness = self._compute_priority_fairness()
            score += priority_fairness * 0.6
        
        # Type fairness: similar tasks should have similar performance
        if len(self.type_history) > 1:
            type_fairness = self._compute_type_fairness()
            score += type_fairness * 0.4
        
        # SLO compliance bonus
        if not task_result.slo_violation:
            score += 0.3
        else:
            score -= 0.5
        
        return np.clip(score, -2.0, 1.0)
    
    def _compute_priority_fairness(self) -> float:
        """Compute priority-based fairness score"""
        priorities = sorted(self.priority_history.keys(), reverse=True)
        if len(priorities) < 2:
            return 0.0
        
        fairness_score = 0.0
        comparisons = 0
        
        for i in range(len(priorities) - 1):
            high_pri = priorities[i]
            low_pri = priorities[i + 1]
            
            if (len(self.priority_history[high_pri]) > 5 and 
                len(self.priority_history[low_pri]) > 5):
                
                high_latency = np.median(self.priority_history[high_pri][-10:])
                low_latency = np.median(self.priority_history[low_pri][-10:])
                
                if high_latency <= low_latency:
                    fairness_score += 1.0  # Correct priority ordering
                else:
                    fairness_score -= 0.5  # Priority inversion penalty
                
                comparisons += 1
        
        return fairness_score / max(comparisons, 1)
    
    def _compute_type_fairness(self) -> float:
        """Compute task-type fairness score"""
        type_medians = {}
        for task_type, latencies in self.type_history.items():
            if len(latencies) > 5:
                type_medians[task_type] = np.median(latencies[-10:])
        
        if len(type_medians) < 2:
            return 0.0
        
        # Coefficient of variation for type fairness
        medians = list(type_medians.values())
        cv = np.std(medians) / max(np.mean(medians), 0.001)
        
        # Lower CV means more fair
        fairness_score = max(0, 1.0 - cv)
        return fairness_score
    
    def get_name(self) -> str:
        return "fairness"

class StabilityMetric(ObjectiveMetric):
    """Ensure system stability and avoid thermal/resource violations"""
    
    def __init__(self):
        self.violation_history = []
        self.window_size = 50
    
    def compute(self, task_result: TaskResult, system_state: Dict) -> float:
        """Compute system stability reward score"""
        score = 1.0  # Start with perfect stability
        
        # Thermal violation penalty
        if task_result.thermal_violation:
            score -= 2.0
            self.violation_history.append('thermal')
        
        # SLO violation penalty
        if task_result.slo_violation:
            score -= 1.0
            self.violation_history.append('slo')
        
        # Maintain violation history
        if len(self.violation_history) > self.window_size:
            self.violation_history.pop(0)
        
        # Recent violation rate penalty
        if len(self.violation_history) > 10:
            recent_violations = len([v for v in self.violation_history[-20:] if v])
            violation_rate = recent_violations / 20.0
            score -= violation_rate * 3.0
        
        # System state penalties
        cpu_temp = system_state.get('cpu_temperature', 40)
        gpu_temp = system_state.get('gpu_temperature', 40)
        gpu_memory_util = system_state.get('gpu_memory_util', 0)
        
        # Temperature penalties
        if cpu_temp > 80:
            score -= (cpu_temp - 80) * 0.1
        if gpu_temp > 85:
            score -= (gpu_temp - 85) * 0.1
        
        # Memory pressure penalty
        if gpu_memory_util > 0.9:
            score -= (gpu_memory_util - 0.9) * 5.0
        
        # Stability bonus for consistent performance
        if (not task_result.thermal_violation and 
            not task_result.slo_violation and 
            len(self.violation_history) == 0):
            score += 0.2
        
        return np.clip(score, -5.0, 1.5)
    
    def get_name(self) -> str:
        return "stability"

class PerformanceMetric(ObjectiveMetric):
    """Overall performance metric combining execution efficiency"""
    
    def __init__(self):
        self.baseline_times = {
            'VEC_ADD': {'CPU': 0.001, 'GPU': 0.0003},
            'MATMUL': {'CPU': 0.01, 'GPU': 0.003},
            'VEC_SCALE': {'CPU': 0.0005, 'GPU': 0.0002},
            'RELU': {'CPU': 0.0002, 'GPU': 0.0001}
        }
    
    def compute(self, task_result: TaskResult, system_state: Dict) -> float:
        """Compute performance efficiency score"""
        task_type = task_result.task_type
        device = task_result.device
        actual_time = task_result.execution_time
        
        # Get baseline time for comparison
        if task_type in self.baseline_times and device in self.baseline_times[task_type]:
            baseline = self.baseline_times[task_type][device]
            expected_time = baseline * task_result.task_size
        else:
            expected_time = actual_time  # No baseline available
        
        # Performance ratio
        if expected_time > 0:
            performance_ratio = expected_time / actual_time
        else:
            performance_ratio = 1.0
        
        # Score based on performance ratio
        if performance_ratio >= 1.0:
            score = min(1.0, performance_ratio - 1.0)  # Bonus for beating baseline
        else:
            score = performance_ratio - 1.0  # Penalty for underperforming
        
        # Device utilization bonus
        optimal_device = self._get_optimal_device(task_result.task_type, task_result.task_size)
        if device == optimal_device:
            score += 0.2
        
        return np.clip(score, -2.0, 2.0)
    
    def _get_optimal_device(self, task_type: str, task_size: int) -> str:
        """Determine optimal device based on task characteristics"""
        if task_type == 'VEC_ADD':
            return 'GPU' if task_size > 50000 else 'CPU'
        elif task_type == 'MATMUL':
            return 'GPU' if task_size > 128*128 else 'CPU'
        elif task_type == 'VEC_SCALE':
            return 'GPU' if task_size > 100000 else 'CPU'
        elif task_type == 'RELU':
            return 'GPU' if task_size > 10000 else 'CPU'
        return 'CPU'
    
    def get_name(self) -> str:
        return "performance"