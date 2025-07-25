#!/usr/bin/env python3
"""
HeteroSched Deep RL Environment

Gym-compatible environment for training RL agents on heterogeneous task scheduling.
State space includes task characteristics, system state, and queue information.
Action space covers device selection, priority assignment, and batching decisions.

Author: HeteroSched Research Team
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Device(Enum):
    CPU = 0
    GPU = 1
    
class TaskType(Enum):
    VEC_ADD = 0
    MATMUL = 1
    VEC_SCALE = 2
    RELU = 3

@dataclass
class Task:
    """Task representation for RL environment"""
    task_id: int
    task_type: TaskType
    size: int
    rows: int
    cols: int
    priority: int = 1
    arrival_time: float = 0.0
    deadline: Optional[float] = None
    
    def to_features(self) -> np.ndarray:
        """Convert task to feature vector"""
        # One-hot encode task type
        task_type_onehot = np.zeros(4)
        task_type_onehot[self.task_type.value] = 1.0
        
        # Normalize size features (log scale and normalize to [0,1])
        log_size = np.log1p(self.size) / 20.0  # Assume max log size ~20
        log_rows = np.log1p(self.rows) / 20.0
        log_cols = np.log1p(self.cols) / 20.0
        
        # Compute derived features
        complexity = self.size * (self.rows if self.task_type == TaskType.MATMUL else 1)
        log_complexity = np.log1p(complexity) / 30.0  # Assume max log complexity ~30
        
        # Normalize priority
        norm_priority = self.priority / 5.0  # Assume max priority is 5
        
        # Clip all values to [0,1] range
        features = np.array([
            *task_type_onehot,           # 4 features
            log_size, log_rows, log_cols, log_complexity,  # 4 features
            norm_priority                # 1 feature
        ])
        
        return np.clip(features, 0.0, 1.0)  # Total: 9 features

@dataclass
class SystemState:
    """Current system state representation"""
    cpu_load_1min: float = 0.0
    cpu_load_5min: float = 0.0
    cpu_load_15min: float = 0.0
    cpu_temperature: float = 40.0  # Celsius
    cpu_power_draw: float = 50.0   # Watts
    
    gpu_memory_used_mb: int = 0
    gpu_memory_total_mb: int = 8192
    gpu_utilization: float = 0.0   # 0-100%
    gpu_temperature: float = 40.0  # Celsius
    gpu_power_draw: float = 100.0  # Watts
    
    pcie_bandwidth_utilization: float = 0.0  # 0-1
    total_power_consumption: float = 150.0   # Watts
    
    def to_features(self) -> np.ndarray:
        """Convert system state to feature vector"""
        # Normalize features to [0, 1] range with clipping
        cpu_load_norm = np.clip(np.array([self.cpu_load_1min, self.cpu_load_5min, self.cpu_load_15min]) / 8.0, 0, 1)  # Assume 8 cores
        cpu_temp_norm = np.clip((self.cpu_temperature - 20) / 80, 0, 1)  # 20-100°C range
        cpu_power_norm = np.clip(self.cpu_power_draw / 200, 0, 1)  # 0-200W range
        
        gpu_memory_norm = np.clip(self.gpu_memory_used_mb / self.gpu_memory_total_mb, 0, 1)
        gpu_util_norm = np.clip(self.gpu_utilization / 100, 0, 1)
        gpu_temp_norm = np.clip((self.gpu_temperature - 20) / 80, 0, 1)  # 20-100°C range
        gpu_power_norm = np.clip(self.gpu_power_draw / 300, 0, 1)  # 0-300W range
        
        pcie_norm = np.clip(self.pcie_bandwidth_utilization, 0, 1)
        total_power_norm = np.clip(self.total_power_consumption / 500, 0, 1)  # 0-500W range
        
        return np.array([
            *cpu_load_norm,      # 3 features
            cpu_temp_norm,       # 1 feature
            cpu_power_norm,      # 1 feature
            gpu_memory_norm,     # 1 feature
            gpu_util_norm,       # 1 feature
            gpu_temp_norm,       # 1 feature
            gpu_power_norm,      # 1 feature
            pcie_norm,           # 1 feature
            total_power_norm     # 1 feature
        ])  # Total: 11 features

@dataclass
class QueueState:
    """Task queue state representation"""
    cpu_queue_length: int = 0
    gpu_queue_length: int = 0
    cpu_queue_total_size: int = 0
    gpu_queue_total_size: int = 0
    cpu_avg_wait_time: float = 0.0
    gpu_avg_wait_time: float = 0.0
    high_priority_waiting: int = 0
    oldest_task_wait_time: float = 0.0
    
    def to_features(self) -> np.ndarray:
        """Convert queue state to feature vector"""
        # Normalize queue lengths (assume max 100 tasks)
        cpu_queue_norm = min(self.cpu_queue_length / 100, 1.0)
        gpu_queue_norm = min(self.gpu_queue_length / 100, 1.0)
        
        # Normalize total sizes (log scale)
        cpu_size_norm = np.log1p(self.cpu_queue_total_size) / 20  # Assume max log size ~20
        gpu_size_norm = np.log1p(self.gpu_queue_total_size) / 20
        
        # Normalize wait times (assume max 60 seconds)
        cpu_wait_norm = min(self.cpu_avg_wait_time / 60, 1.0)
        gpu_wait_norm = min(self.gpu_avg_wait_time / 60, 1.0)
        oldest_wait_norm = min(self.oldest_task_wait_time / 120, 1.0)  # Max 2 minutes
        
        # Normalize high priority count
        high_priority_norm = min(self.high_priority_waiting / 20, 1.0)  # Assume max 20
        
        return np.array([
            cpu_queue_norm,      # 1 feature
            gpu_queue_norm,      # 1 feature
            cpu_size_norm,       # 1 feature
            gpu_size_norm,       # 1 feature
            cpu_wait_norm,       # 1 feature
            gpu_wait_norm,       # 1 feature
            high_priority_norm,  # 1 feature
            oldest_wait_norm     # 1 feature
        ])  # Total: 8 features

class HeteroSchedEnv(gym.Env):
    """
    Gym environment for heterogeneous task scheduling with RL
    
    State Space: [current_task_features, system_state, queue_state, performance_history]
    Action Space: [device_selection, priority_boost, batch_decision]
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        self.config = config or {}
        self.max_episode_steps = self.config.get('max_episode_steps', 1000)
        self.simulation_mode = self.config.get('simulation_mode', True)
        
        # State space dimensions
        self.task_features_dim = 9
        self.system_state_dim = 11
        self.queue_state_dim = 8
        self.performance_history_dim = 8  # Recent performance metrics
        self.total_state_dim = (self.task_features_dim + self.system_state_dim + 
                               self.queue_state_dim + self.performance_history_dim)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.total_state_dim,), 
            dtype=np.float32
        )
        
        # Define action space
        # Action: [device (0=CPU, 1=GPU), priority_boost (0-4), batch_size (1-10)]
        self.action_space = spaces.MultiDiscrete([2, 5, 10])
        
        # Environment state
        self.current_task: Optional[Task] = None
        self.system_state = SystemState()
        self.queue_state = QueueState()
        self.performance_history = np.zeros(self.performance_history_dim)
        
        # Episode tracking
        self.episode_step = 0
        self.total_rewards = 0.0
        self.tasks_completed = 0
        self.tasks_queue = []
        
        # Performance tracking
        self.metrics = {
            'total_latency': 0.0,
            'total_energy': 0.0,
            'total_throughput': 0.0,
            'fairness_violations': 0,
            'thermal_violations': 0,
            'slo_violations': 0
        }
        
        logger.info(f"Initialized HeteroSchedEnv with state_dim={self.total_state_dim}, action_dim={self.action_space}")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.episode_step = 0
        self.total_rewards = 0.0
        self.tasks_completed = 0
        
        # Reset system state
        self.system_state = SystemState()
        self.queue_state = QueueState()
        self.performance_history = np.zeros(self.performance_history_dim)
        
        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = 0.0
        
        # Generate initial task
        self.current_task = self._generate_random_task()
        
        logger.debug(f"Environment reset, initial task: {self.current_task}")
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step"""
        self.episode_step += 1
        
        # Parse action
        device = Device(action[0])
        priority_boost = action[1]
        batch_size = action[2] + 1  # 1-10 range
        
        # Execute task with given action
        task_result = self._execute_task(self.current_task, device, priority_boost, batch_size)
        
        # Calculate reward
        reward = self._calculate_reward(task_result)
        self.total_rewards += reward
        
        # Update environment state
        self._update_system_state(task_result)
        self._update_queue_state(task_result)
        self._update_performance_history(task_result)
        
        # Generate next task
        self.current_task = self._generate_random_task()
        self.tasks_completed += 1
        
        # Check if episode is done
        done = (self.episode_step >= self.max_episode_steps or 
                self._check_termination_conditions())
        
        # Prepare info dict
        info = {
            'episode_step': self.episode_step,
            'tasks_completed': self.tasks_completed,
            'total_reward': self.total_rewards,
            'metrics': self.metrics.copy(),
            'task_result': task_result
        }
        
        if done:
            logger.info(f"Episode finished: {self.tasks_completed} tasks, total_reward={self.total_rewards:.3f}")
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct current observation vector"""
        if self.current_task is None:
            task_features = np.zeros(self.task_features_dim)
        else:
            task_features = self.current_task.to_features()
        
        system_features = self.system_state.to_features()
        queue_features = self.queue_state.to_features()
        
        observation = np.concatenate([
            task_features,           # 9 features
            system_features,         # 11 features
            queue_features,          # 8 features
            self.performance_history # 8 features
        ])
        
        return observation.astype(np.float32)
    
    def _generate_random_task(self) -> Task:
        """Generate a random task for training"""
        task_types = list(TaskType)
        task_type = np.random.choice(task_types)
        
        # Generate size based on task type
        if task_type == TaskType.VEC_ADD:
            size = np.random.randint(1000, 1000000)
            rows, cols = size, 1
        elif task_type == TaskType.MATMUL:
            dim = np.random.randint(32, 512)
            size = dim * dim
            rows, cols = dim, dim
        elif task_type == TaskType.VEC_SCALE:
            size = np.random.randint(5000, 500000)
            rows, cols = size, 1
        else:  # RELU
            size = np.random.randint(2000, 100000)
            rows, cols = size, 1
        
        # Random priority (1-5)
        priority = np.random.randint(1, 6)
        
        task = Task(
            task_id=self.tasks_completed + 1,
            task_type=task_type,
            size=size,
            rows=rows,
            cols=cols,
            priority=priority,
            arrival_time=self.episode_step
        )
        
        return task
    
    def _execute_task(self, task: Task, device: Device, priority_boost: int, batch_size: int) -> Dict:
        """Simulate task execution and return performance metrics"""
        # Baseline execution time prediction (simplified)
        if device == Device.CPU:
            if task.task_type == TaskType.VEC_ADD:
                base_time = task.size * 0.001  # 1 μs per element
            elif task.task_type == TaskType.MATMUL:
                base_time = task.rows * task.cols * task.rows * 0.01  # O(n³)
            elif task.task_type == TaskType.VEC_SCALE:
                base_time = task.size * 0.0005
            else:  # RELU
                base_time = task.size * 0.0002
        else:  # GPU
            # GPU is generally faster but has overhead for small tasks
            cpu_time = self._execute_task(task, Device.CPU, 0, 1)['execution_time']
            if task.size > 50000:
                base_time = cpu_time * 0.3  # 3x speedup for large tasks
            else:
                base_time = cpu_time * 1.5  # Overhead for small tasks
        
        # Apply system state effects
        system_slowdown = self._calculate_system_slowdown(device)
        execution_time = base_time * system_slowdown
        
        # Apply batching effects
        if batch_size > 1:
            batching_efficiency = 1.0 + (batch_size - 1) * 0.1  # 10% improvement per additional task
            execution_time /= batching_efficiency
        
        # Apply priority boost effects (reduces queue wait time)
        queue_wait = self._estimate_queue_wait(device, priority_boost)
        total_latency = queue_wait + execution_time
        
        # Calculate energy consumption
        if device == Device.CPU:
            power = 50 + (execution_time * 100)  # Base + dynamic power
        else:
            power = 100 + (execution_time * 200)  # GPU uses more power
        
        energy = power * execution_time / 1000  # Convert to Joules
        
        # Calculate throughput (tasks per second)
        throughput = 1.0 / total_latency if total_latency > 0 else 0.0
        
        # Check violations
        thermal_violation = self._check_thermal_violation(device, execution_time)
        slo_violation = self._check_slo_violation(task, total_latency)
        
        return {
            'execution_time': execution_time,
            'queue_wait': queue_wait,
            'total_latency': total_latency,
            'energy': energy,
            'throughput': throughput,
            'device': device,
            'batch_size': batch_size,
            'thermal_violation': thermal_violation,
            'slo_violation': slo_violation,
            'priority': task.priority + priority_boost
        }
    
    def _calculate_system_slowdown(self, device: Device) -> float:
        """Calculate performance impact of current system state"""
        if device == Device.CPU:
            # CPU slowdown based on load and temperature
            load_factor = 1.0 + (self.system_state.cpu_load_1min / 8.0) * 0.5
            temp_factor = 1.0 if self.system_state.cpu_temperature < 70 else 1.2
            return load_factor * temp_factor
        else:
            # GPU slowdown based on utilization and memory pressure
            util_factor = 1.0 + (self.system_state.gpu_utilization / 100) * 0.3
            memory_factor = 1.0 + (self.system_state.gpu_memory_used_mb / self.system_state.gpu_memory_total_mb) * 0.4
            temp_factor = 1.0 if self.system_state.gpu_temperature < 80 else 1.3
            return util_factor * memory_factor * temp_factor
    
    def _estimate_queue_wait(self, device: Device, priority_boost: int) -> float:
        """Estimate queue waiting time"""
        if device == Device.CPU:
            base_wait = self.queue_state.cpu_avg_wait_time
        else:
            base_wait = self.queue_state.gpu_avg_wait_time
        
        # Priority boost reduces wait time
        priority_factor = max(0.1, 1.0 - (priority_boost * 0.2))
        
        return base_wait * priority_factor
    
    def _check_thermal_violation(self, device: Device, execution_time: float) -> bool:
        """Check if task execution would cause thermal violation"""
        temp = self.system_state.cpu_temperature if device == Device.CPU else self.system_state.gpu_temperature
        temp_increase = execution_time * 2.0  # Simplified thermal model
        return (temp + temp_increase) > 85.0  # 85°C threshold
    
    def _check_slo_violation(self, task: Task, latency: float) -> bool:
        """Check if latency violates Service Level Objective"""
        # Simplified SLO: high priority tasks should complete within 10ms
        if task.priority >= 4:
            return latency > 10.0
        return False
    
    def _calculate_reward(self, task_result: Dict) -> float:
        """Multi-objective reward function"""
        # Normalized metrics (all scaled to similar ranges)
        latency_reward = -np.log1p(task_result['total_latency']) / 10  # Negative log latency
        energy_reward = -task_result['energy'] / 100  # Negative energy consumption
        throughput_reward = task_result['throughput'] * 10  # Positive throughput
        
        # Penalty terms
        thermal_penalty = -10.0 if task_result['thermal_violation'] else 0.0
        slo_penalty = -20.0 if task_result['slo_violation'] else 0.0
        
        # Fairness bonus (simplified)
        fairness_bonus = 1.0 if task_result['priority'] >= 3 else 0.0
        
        # Weighted combination
        reward = (0.4 * latency_reward +
                 0.2 * energy_reward +
                 0.3 * throughput_reward +
                 0.1 * fairness_bonus +
                 thermal_penalty +
                 slo_penalty)
        
        return reward
    
    def _update_system_state(self, task_result: Dict):
        """Update system state based on task execution"""
        # More realistic system state evolution
        if task_result['device'] == Device.CPU:
            self.system_state.cpu_load_1min = min(8.0, self.system_state.cpu_load_1min + 0.1)
            temp_increase = min(2.0, task_result['execution_time'] * 0.1)  # Max 2°C increase
            self.system_state.cpu_temperature = min(85, self.system_state.cpu_temperature + temp_increase)
            self.system_state.cpu_power_draw = min(200, self.system_state.cpu_power_draw + 5)
        else:
            self.system_state.gpu_utilization = min(100, self.system_state.gpu_utilization + 2)
            memory_increase = min(100, int(task_result['execution_time'] * 10))
            self.system_state.gpu_memory_used_mb = min(self.system_state.gpu_memory_total_mb, 
                                                       self.system_state.gpu_memory_used_mb + memory_increase)
            temp_increase = min(3.0, task_result['execution_time'] * 0.15)  # Max 3°C increase
            self.system_state.gpu_temperature = min(90, self.system_state.gpu_temperature + temp_increase)
            self.system_state.gpu_power_draw = min(300, self.system_state.gpu_power_draw + 10)
        
        # Gradual decay (cooling/unloading) - more realistic cooling
        self.system_state.cpu_temperature = max(40, self.system_state.cpu_temperature - 0.5)
        self.system_state.gpu_temperature = max(40, self.system_state.gpu_temperature - 0.5)
        self.system_state.cpu_load_1min = max(0, self.system_state.cpu_load_1min - 0.1)
        self.system_state.gpu_utilization = max(0, self.system_state.gpu_utilization - 1)
        self.system_state.gpu_memory_used_mb = max(0, self.system_state.gpu_memory_used_mb - 10)
    
    def _update_queue_state(self, task_result: Dict):
        """Update queue state based on task completion"""
        # Simplified queue evolution
        if task_result['device'] == Device.CPU:
            self.queue_state.cpu_queue_length = max(0, self.queue_state.cpu_queue_length - 1)
            self.queue_state.cpu_avg_wait_time = task_result['queue_wait']
        else:
            self.queue_state.gpu_queue_length = max(0, self.queue_state.gpu_queue_length - 1)
            self.queue_state.gpu_avg_wait_time = task_result['queue_wait']
        
        # Add some randomness to simulate dynamic workload
        if np.random.random() < 0.3:  # 30% chance of new task arrival
            target_queue = np.random.choice([Device.CPU, Device.GPU])
            if target_queue == Device.CPU:
                self.queue_state.cpu_queue_length += 1
            else:
                self.queue_state.gpu_queue_length += 1
    
    def _update_performance_history(self, task_result: Dict):
        """Update rolling performance history"""
        # Update metrics
        self.metrics['total_latency'] += task_result['total_latency']
        self.metrics['total_energy'] += task_result['energy']
        self.metrics['total_throughput'] += task_result['throughput']
        
        if task_result['thermal_violation']:
            self.metrics['thermal_violations'] += 1
        if task_result['slo_violation']:
            self.metrics['slo_violations'] += 1
        
        # Rolling average of recent performance (exponential moving average)
        alpha = 0.1  # Smoothing factor
        recent_metrics = np.array([
            task_result['total_latency'],
            task_result['energy'],
            task_result['throughput'],
            float(task_result['thermal_violation']),
            float(task_result['slo_violation']),
            task_result['queue_wait'],
            task_result['priority'],
            1.0 if task_result['device'] == Device.GPU else 0.0
        ])
        
        self.performance_history = (1 - alpha) * self.performance_history + alpha * recent_metrics
    
    def _check_termination_conditions(self) -> bool:
        """Check for early termination conditions"""
        # Terminate if system becomes unstable
        if (self.system_state.cpu_temperature > 90 or 
            self.system_state.gpu_temperature > 90):
            logger.warning("Terminating episode due to thermal runaway")
            return True
        
        # Terminate if too many violations
        violation_rate = (self.metrics['thermal_violations'] + self.metrics['slo_violations']) / max(1, self.tasks_completed)
        if violation_rate > 0.5:  # More than 50% violations
            logger.warning("Terminating episode due to high violation rate")
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render environment state (for debugging)"""
        if mode == 'human':
            print(f"\\nEpisode Step: {self.episode_step}")
            print(f"Current Task: {self.current_task}")
            print(f"System State: CPU_temp={self.system_state.cpu_temperature:.1f}°C, "
                  f"GPU_util={self.system_state.gpu_utilization:.1f}%")
            print(f"Queue State: CPU={self.queue_state.cpu_queue_length}, GPU={self.queue_state.gpu_queue_length}")
            print(f"Total Rewards: {self.total_rewards:.3f}")
            print(f"Tasks Completed: {self.tasks_completed}")
    
    def close(self):
        """Clean up environment"""
        logger.info("Environment closed")

# Factory function for different environment configurations
def make_hetero_env(config_name: str = 'default') -> HeteroSchedEnv:
    """Create HeteroSched environment with predefined configurations"""
    
    configs = {
        'default': {
            'max_episode_steps': 1000,
            'simulation_mode': True
        },
        'long': {
            'max_episode_steps': 5000,
            'simulation_mode': True
        },
        'short': {
            'max_episode_steps': 200,
            'simulation_mode': True
        },
        'evaluation': {
            'max_episode_steps': 2000,
            'simulation_mode': False  # More realistic simulation
        }
    }
    
    config = configs.get(config_name, configs['default'])
    return HeteroSchedEnv(config)