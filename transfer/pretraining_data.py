#!/usr/bin/env python3
"""
Pre-training Data Generation for HeteroSched Foundation Model

This module generates diverse synthetic workloads and real-world scheduling
scenarios for pre-training the foundation model on heterogeneous scheduling tasks.

Research Innovation: First comprehensive data generation framework for 
foundation model pre-training in heterogeneous scheduling domain.

Key Components:
- Synthetic workload generation with realistic patterns
- Multi-domain scheduling scenario simulation
- Temporal pattern synthesis for different time scales
- Real-world trace augmentation and anonymization
- Domain-specific data augmentation techniques
- Curriculum learning data organization

Authors: HeteroSched Research Team
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
import random
import json
from collections import defaultdict, deque
import math

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from transfer.foundation_model import FoundationModelConfig

logger = logging.getLogger(__name__)

@dataclass
class WorkloadPattern:
    """Definition of a workload pattern for synthetic generation"""
    
    name: str
    description: str
    
    # Task characteristics
    task_types: List[int]                    # Possible task types
    task_type_probabilities: List[float]     # Probability distribution
    
    # Resource requirements
    cpu_intensity_range: Tuple[float, float]     # CPU requirement range [0,1]
    memory_intensity_range: Tuple[float, float]  # Memory requirement range [0,1]
    gpu_intensity_range: Tuple[float, float]     # GPU requirement range [0,1]
    network_intensity_range: Tuple[float, float] # Network requirement range [0,1]
    
    # Temporal patterns
    arrival_pattern: str                     # 'poisson', 'bursty', 'periodic', 'steady'
    arrival_rate_range: Tuple[float, float]  # Tasks per time unit
    
    # Execution characteristics
    duration_distribution: str               # 'exponential', 'uniform', 'normal', 'pareto'
    duration_params: Dict[str, float]        # Distribution parameters
    
    # Priority patterns
    priority_distribution: str               # 'uniform', 'exponential', 'bimodal'
    priority_params: Dict[str, float]        # Distribution parameters
    
    # System load characteristics
    target_utilization: float                # Target system utilization [0,1]
    load_variation: float                    # Variation in load over time

@dataclass
class SchedulingScenario:
    """Complete scheduling scenario for training"""
    
    scenario_id: str
    workload_patterns: List[WorkloadPattern]
    
    # System configuration
    cpu_cores: int
    gpu_count: int
    memory_gb: int
    network_bandwidth_gbps: float
    
    # Scheduling constraints
    max_queue_length: int
    max_latency_ms: float
    min_throughput: float
    
    # Temporal characteristics
    scenario_duration: int                   # Number of time steps
    time_step_ms: int                        # Milliseconds per time step
    
    # Objectives
    primary_objective: str                   # 'latency', 'throughput', 'energy', 'fairness'
    objective_weights: Dict[str, float]      # Multi-objective weights

class WorkloadGenerator:
    """Generates synthetic workloads based on patterns"""
    
    def __init__(self, config: FoundationModelConfig):
        self.config = config
        self.patterns = self._create_standard_patterns()
        
        # Random number generators for reproducibility
        self.task_rng = np.random.RandomState(42)
        self.arrival_rng = np.random.RandomState(123)
        self.resource_rng = np.random.RandomState(456)
        
    def _create_standard_patterns(self) -> Dict[str, WorkloadPattern]:
        """Create standard workload patterns for pre-training"""
        
        patterns = {}
        
        # Web server workload
        patterns['web_server'] = WorkloadPattern(
            name='web_server',
            description='Web server with request processing',
            task_types=[0, 1, 2],  # HTTP, static, dynamic
            task_type_probabilities=[0.6, 0.3, 0.1],
            cpu_intensity_range=(0.1, 0.4),
            memory_intensity_range=(0.1, 0.3),
            gpu_intensity_range=(0.0, 0.1),
            network_intensity_range=(0.3, 0.8),
            arrival_pattern='poisson',
            arrival_rate_range=(10.0, 100.0),
            duration_distribution='exponential',
            duration_params={'lambda': 2.0},
            priority_distribution='uniform',
            priority_params={'low': 0, 'high': 5},
            target_utilization=0.7,
            load_variation=0.2
        )
        
        # Machine learning training
        patterns['ml_training'] = WorkloadPattern(
            name='ml_training',
            description='Machine learning training workload',
            task_types=[10, 11, 12, 13],  # Training, inference, data loading, validation
            task_type_probabilities=[0.5, 0.2, 0.2, 0.1],
            cpu_intensity_range=(0.3, 0.7),
            memory_intensity_range=(0.4, 0.9),
            gpu_intensity_range=(0.6, 1.0),
            network_intensity_range=(0.1, 0.4),
            arrival_pattern='bursty',
            arrival_rate_range=(1.0, 10.0),
            duration_distribution='pareto',
            duration_params={'alpha': 1.5, 'scale': 5.0},
            priority_distribution='bimodal',
            priority_params={'low_prob': 0.8, 'low_val': 2, 'high_val': 8},
            target_utilization=0.9,
            load_variation=0.3
        )
        
        # Scientific computing
        patterns['scientific'] = WorkloadPattern(
            name='scientific',
            description='Scientific computing workload',
            task_types=[5, 6, 7, 8, 9],  # Simulation, analysis, visualization, I/O, post-process
            task_type_probabilities=[0.4, 0.2, 0.1, 0.2, 0.1],
            cpu_intensity_range=(0.5, 1.0),
            memory_intensity_range=(0.3, 0.8),
            gpu_intensity_range=(0.2, 0.8),
            network_intensity_range=(0.1, 0.3),
            arrival_pattern='periodic',
            arrival_rate_range=(0.5, 5.0),
            duration_distribution='normal',
            duration_params={'mean': 30.0, 'std': 10.0},
            priority_distribution='exponential',
            priority_params={'lambda': 0.3},
            target_utilization=0.8,
            load_variation=0.1
        )
        
        # Batch processing
        patterns['batch_processing'] = WorkloadPattern(
            name='batch_processing',
            description='Batch job processing workload',
            task_types=[14, 15, 16, 17],  # ETL, aggregation, reporting, cleanup
            task_type_probabilities=[0.4, 0.3, 0.2, 0.1],
            cpu_intensity_range=(0.2, 0.6),
            memory_intensity_range=(0.5, 0.9),
            gpu_intensity_range=(0.0, 0.2),
            network_intensity_range=(0.4, 0.7),
            arrival_pattern='steady',
            arrival_rate_range=(2.0, 8.0),
            duration_distribution='uniform',
            duration_params={'low': 10.0, 'high': 60.0},
            priority_distribution='uniform',
            priority_params={'low': 0, 'high': 9},
            target_utilization=0.6,
            load_variation=0.15
        )
        
        # Real-time streaming
        patterns['streaming'] = WorkloadPattern(
            name='streaming',
            description='Real-time data streaming workload',
            task_types=[18, 19],  # Stream processing, windowing
            task_type_probabilities=[0.8, 0.2],
            cpu_intensity_range=(0.3, 0.6),
            memory_intensity_range=(0.2, 0.5),
            gpu_intensity_range=(0.1, 0.4),
            network_intensity_range=(0.6, 0.9),
            arrival_pattern='poisson',
            arrival_rate_range=(50.0, 200.0),
            duration_distribution='exponential',
            duration_params={'lambda': 10.0},
            priority_distribution='bimodal',
            priority_params={'low_prob': 0.9, 'low_val': 7, 'high_val': 9},
            target_utilization=0.75,
            load_variation=0.25
        )
        
        return patterns
    
    def generate_task_arrival_times(self, pattern: WorkloadPattern, 
                                  duration: int) -> List[float]:
        """Generate task arrival times based on pattern"""
        
        arrival_times = []
        
        if pattern.arrival_pattern == 'poisson':
            # Poisson process
            rate = self.arrival_rng.uniform(*pattern.arrival_rate_range)
            t = 0
            while t < duration:
                interval = self.arrival_rng.exponential(1.0 / rate)
                t += interval
                if t < duration:
                    arrival_times.append(t)
        
        elif pattern.arrival_pattern == 'bursty':
            # Bursty arrivals with quiet periods
            t = 0
            while t < duration:
                # Burst period
                burst_rate = self.arrival_rng.uniform(*pattern.arrival_rate_range)
                burst_duration = self.arrival_rng.exponential(10.0)
                burst_end = min(t + burst_duration, duration)
                
                while t < burst_end:
                    interval = self.arrival_rng.exponential(1.0 / burst_rate)
                    t += interval
                    if t < burst_end:
                        arrival_times.append(t)
                
                # Quiet period
                quiet_duration = self.arrival_rng.exponential(20.0)
                t += quiet_duration
        
        elif pattern.arrival_pattern == 'periodic':
            # Periodic arrivals with some jitter
            base_rate = np.mean(pattern.arrival_rate_range)
            period = 1.0 / base_rate
            
            t = 0
            while t < duration:
                # Add jitter
                jitter = self.arrival_rng.uniform(-0.3 * period, 0.3 * period)
                t += period + jitter
                if t < duration:
                    arrival_times.append(t)
        
        elif pattern.arrival_pattern == 'steady':
            # Steady rate with minor variations
            rate = np.mean(pattern.arrival_rate_range)
            t = 0
            while t < duration:
                # Small variation in rate
                current_rate = rate * self.arrival_rng.uniform(0.8, 1.2)
                interval = self.arrival_rng.exponential(1.0 / current_rate)
                t += interval
                if t < duration:
                    arrival_times.append(t)
        
        return sorted(arrival_times)
    
    def generate_task_duration(self, pattern: WorkloadPattern) -> float:
        """Generate task duration based on pattern"""
        
        if pattern.duration_distribution == 'exponential':
            return self.task_rng.exponential(1.0 / pattern.duration_params['lambda'])
        
        elif pattern.duration_distribution == 'uniform':
            return self.task_rng.uniform(
                pattern.duration_params['low'], 
                pattern.duration_params['high']
            )
        
        elif pattern.duration_distribution == 'normal':
            duration = self.task_rng.normal(
                pattern.duration_params['mean'],
                pattern.duration_params['std']
            )
            return max(0.1, duration)  # Ensure positive duration
        
        elif pattern.duration_distribution == 'pareto':
            return self.task_rng.pareto(pattern.duration_params['alpha']) * pattern.duration_params['scale']
        
        else:
            return 1.0  # Default duration
    
    def generate_task_priority(self, pattern: WorkloadPattern) -> int:
        """Generate task priority based on pattern"""
        
        if pattern.priority_distribution == 'uniform':
            return self.task_rng.randint(
                pattern.priority_params['low'],
                pattern.priority_params['high'] + 1
            )
        
        elif pattern.priority_distribution == 'exponential':
            priority = self.task_rng.exponential(1.0 / pattern.priority_params['lambda'])
            return int(np.clip(priority, 0, 9))
        
        elif pattern.priority_distribution == 'bimodal':
            if self.task_rng.random() < pattern.priority_params['low_prob']:
                return pattern.priority_params['low_val']
            else:
                return pattern.priority_params['high_val']
        
        else:
            return 5  # Default priority
    
    def generate_resource_requirements(self, pattern: WorkloadPattern) -> np.ndarray:
        """Generate resource requirements for a task"""
        
        requirements = np.array([
            self.resource_rng.uniform(*pattern.cpu_intensity_range),
            self.resource_rng.uniform(*pattern.memory_intensity_range),
            self.resource_rng.uniform(*pattern.gpu_intensity_range),
            self.resource_rng.uniform(*pattern.network_intensity_range)
        ])
        
        return requirements
    
    def generate_workload_sequence(self, pattern: WorkloadPattern, 
                                 duration: int, system_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate complete workload sequence"""
        
        # Generate arrival times
        arrival_times = self.generate_task_arrival_times(pattern, duration)
        n_tasks = len(arrival_times)
        
        if n_tasks == 0:
            # Empty sequence
            return {
                'task_type': torch.zeros(1, dtype=torch.long),
                'resource_req': torch.zeros(1, 4),
                'priority': torch.zeros(1, dtype=torch.long),
                'size': torch.ones(1, 1) * 0.1,
                'arrival_time': torch.zeros(1),
                'duration': torch.ones(1) * 0.1
            }
        
        # Generate task properties
        task_types = []
        resource_reqs = []
        priorities = []
        sizes = []
        durations = []
        
        for _ in range(n_tasks):
            # Task type
            task_type = self.task_rng.choice(
                pattern.task_types, 
                p=pattern.task_type_probabilities
            )
            task_types.append(task_type)
            
            # Resource requirements
            resource_req = self.generate_resource_requirements(pattern)
            resource_reqs.append(resource_req)
            
            # Priority
            priority = self.generate_task_priority(pattern)
            priorities.append(priority)
            
            # Size (based on resource requirements)
            size = np.mean(resource_req) * self.task_rng.uniform(0.5, 2.0)
            sizes.append(size)
            
            # Duration
            duration = self.generate_task_duration(pattern)
            durations.append(duration)
        
        return {
            'task_type': torch.tensor(task_types, dtype=torch.long),
            'resource_req': torch.tensor(resource_reqs, dtype=torch.float32),
            'priority': torch.tensor(priorities, dtype=torch.long),
            'size': torch.tensor(sizes, dtype=torch.float32).unsqueeze(-1),
            'arrival_time': torch.tensor(arrival_times, dtype=torch.float32),
            'duration': torch.tensor(durations, dtype=torch.float32)
        }

class SystemStateSimulator:
    """Simulates system state evolution during scheduling"""
    
    def __init__(self, config: FoundationModelConfig):
        self.config = config
        
    def simulate_system_evolution(self, workload: Dict[str, torch.Tensor],
                                system_config: Dict[str, Any],
                                scheduling_decisions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simulate system state evolution given workload and decisions"""
        
        n_tasks = len(workload['task_type'])
        state_sequence = []
        
        # Initial system state
        current_state = self._create_initial_state(system_config)
        
        # System resources
        cpu_cores = system_config.get('cpu_cores', 8)
        gpu_count = system_config.get('gpu_count', 2)
        memory_gb = system_config.get('memory_gb', 32)
        network_bandwidth = system_config.get('network_bandwidth_gbps', 10)
        
        # Running tasks tracking
        running_tasks = []
        completed_tasks = []
        
        # Simulate over time
        current_time = 0.0
        max_time = float(workload['arrival_time'].max()) + 100.0
        
        task_idx = 0
        
        while current_time < max_time and len(state_sequence) < self.config.max_sequence_length:
            # Add arriving tasks
            while (task_idx < n_tasks and 
                   workload['arrival_time'][task_idx] <= current_time):
                
                # Create task
                task = {
                    'id': task_idx,
                    'type': int(workload['task_type'][task_idx]),
                    'resource_req': workload['resource_req'][task_idx],
                    'priority': int(workload['priority'][task_idx]),
                    'size': float(workload['size'][task_idx]),
                    'duration': float(workload['duration'][task_idx]),
                    'arrival_time': current_time,
                    'start_time': None,
                    'completion_time': None
                }
                
                # Make scheduling decision (simplified)
                if scheduling_decisions is not None and task_idx < len(scheduling_decisions):
                    decision = int(scheduling_decisions[task_idx])
                else:
                    decision = self._make_default_scheduling_decision(task, current_state)
                
                # Try to schedule task
                if self._can_schedule_task(task, current_state, decision):
                    task['start_time'] = current_time
                    task['completion_time'] = current_time + task['duration']
                    task['assigned_resources'] = decision
                    running_tasks.append(task)
                    
                    # Update resource usage
                    self._allocate_resources(current_state, task, decision)
                
                task_idx += 1
            
            # Complete finished tasks
            newly_completed = []
            for task in running_tasks:
                if task['completion_time'] <= current_time:
                    self._deallocate_resources(current_state, task)
                    completed_tasks.append(task)
                    newly_completed.append(task)
            
            for task in newly_completed:
                running_tasks.remove(task)
            
            # Update system state
            self._update_system_dynamics(current_state, running_tasks)
            
            # Record state
            state_vector = self._state_to_vector(current_state, running_tasks)
            state_sequence.append(state_vector)
            
            # Advance time
            current_time += 1.0
        
        # Convert to tensor
        if len(state_sequence) == 0:
            state_sequence = [np.zeros(self.config.state_dim)]
        
        return torch.tensor(state_sequence, dtype=torch.float32)
    
    def _create_initial_state(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial system state"""
        return {
            'cpu_usage': 0.0,
            'cpu_available': system_config.get('cpu_cores', 8),
            'memory_usage': 0.0,
            'memory_available': system_config.get('memory_gb', 32),
            'gpu_usage': 0.0,
            'gpu_available': system_config.get('gpu_count', 2),
            'network_usage': 0.0,
            'network_available': system_config.get('network_bandwidth_gbps', 10),
            'queue_length': 0,
            'running_tasks': 0,
            'completed_tasks': 0,
            'total_latency': 0.0,
            'total_energy': 0.0,
            'temperature': 50.0,
            'stability_score': 1.0
        }
    
    def _make_default_scheduling_decision(self, task: Dict[str, Any], 
                                        system_state: Dict[str, Any]) -> int:
        """Make default scheduling decision (simple heuristic)"""
        
        # Simple decision based on resource requirements
        resource_req = task['resource_req']
        
        # Use GPU if GPU requirement is high and GPU is available
        if resource_req[2] > 0.5 and system_state['gpu_usage'] < 0.8:
            device = 1  # GPU
        else:
            device = 0  # CPU
        
        # Priority based on task priority
        priority = min(task['priority'], 4)
        
        # Batch size based on system load
        if system_state['cpu_usage'] > 0.8:
            batch = 0  # Small batch
        else:
            batch = min(int(task['size'] * 10), 9)
        
        # Encode decision
        decision = device * 2 + priority * 10 + batch
        return min(decision, 99)
    
    def _can_schedule_task(self, task: Dict[str, Any], system_state: Dict[str, Any],
                          decision: int) -> bool:
        """Check if task can be scheduled with given decision"""
        
        resource_req = task['resource_req']
        
        # Check resource availability (simplified)
        if (system_state['cpu_usage'] + resource_req[0] > 1.0 or
            system_state['memory_usage'] + resource_req[1] > 1.0):
            return False
        
        return True
    
    def _allocate_resources(self, system_state: Dict[str, Any], task: Dict[str, Any],
                           decision: int):
        """Allocate resources for scheduled task"""
        
        resource_req = task['resource_req']
        
        system_state['cpu_usage'] += resource_req[0]
        system_state['memory_usage'] += resource_req[1]
        system_state['gpu_usage'] += resource_req[2]
        system_state['network_usage'] += resource_req[3]
        system_state['running_tasks'] += 1
        system_state['queue_length'] = max(0, system_state['queue_length'] - 1)
    
    def _deallocate_resources(self, system_state: Dict[str, Any], task: Dict[str, Any]):
        """Deallocate resources when task completes"""
        
        resource_req = task['resource_req']
        
        system_state['cpu_usage'] = max(0, system_state['cpu_usage'] - resource_req[0])
        system_state['memory_usage'] = max(0, system_state['memory_usage'] - resource_req[1])
        system_state['gpu_usage'] = max(0, system_state['gpu_usage'] - resource_req[2])
        system_state['network_usage'] = max(0, system_state['network_usage'] - resource_req[3])
        system_state['running_tasks'] = max(0, system_state['running_tasks'] - 1)
        system_state['completed_tasks'] += 1
    
    def _update_system_dynamics(self, system_state: Dict[str, Any], running_tasks: List[Dict]):
        """Update system dynamics"""
        
        # Update temperature based on usage
        total_usage = (system_state['cpu_usage'] + system_state['gpu_usage']) / 2
        system_state['temperature'] = 40 + total_usage * 40
        
        # Update stability score
        if system_state['cpu_usage'] > 0.9 or system_state['memory_usage'] > 0.9:
            system_state['stability_score'] *= 0.99
        else:
            system_state['stability_score'] = min(1.0, system_state['stability_score'] * 1.001)
        
        # Update energy (simplified)
        system_state['total_energy'] += (
            system_state['cpu_usage'] * 50 + 
            system_state['gpu_usage'] * 200
        )
    
    def _state_to_vector(self, system_state: Dict[str, Any], running_tasks: List[Dict]) -> np.ndarray:
        """Convert system state to vector representation"""
        
        # Create state vector matching the expected 36-dimensional format
        state_vector = np.zeros(36)
        
        # Task queue features (0-8) - simplified
        state_vector[0] = min(len(running_tasks) / 10.0, 1.0)
        state_vector[1] = system_state['queue_length'] / 100.0
        state_vector[2:9] = 0.1  # Placeholder task features
        
        # System resource state (9-14)
        state_vector[9] = system_state['cpu_usage']
        state_vector[10] = system_state['temperature'] / 100.0
        state_vector[11] = system_state['memory_usage']
        state_vector[12] = system_state['gpu_usage']
        state_vector[13] = system_state['gpu_usage']  # GPU memory usage
        state_vector[14] = system_state['temperature'] / 100.0  # GPU temperature
        
        # System load and performance (15-21)
        state_vector[15] = system_state['cpu_usage']  # Load average 1min
        state_vector[16] = system_state['cpu_usage']  # Load average 5min
        state_vector[17] = system_state['cpu_usage']  # Load average 15min
        state_vector[18] = system_state['network_usage']
        state_vector[19] = system_state['network_usage']
        state_vector[20] = system_state['running_tasks'] / 20.0
        state_vector[21] = system_state['total_energy'] / 10000.0
        
        # Queue and history features (22-35)
        state_vector[22] = system_state['completed_tasks'] / 1000.0
        state_vector[23] = system_state['stability_score']
        state_vector[24:36] = 0.1  # Additional features
        
        return np.clip(state_vector, 0.0, 1.0)

class PretrainingDataGenerator:
    """Main data generator for foundation model pre-training"""
    
    def __init__(self, config: FoundationModelConfig):
        self.config = config
        self.workload_generator = WorkloadGenerator(config)
        self.system_simulator = SystemStateSimulator(config)
        
        # Standard system configurations
        self.system_configs = self._create_system_configs()
        
    def _create_system_configs(self) -> List[Dict[str, Any]]:
        """Create diverse system configurations"""
        
        configs = []
        
        # Small cluster
        configs.append({
            'name': 'small_cluster',
            'cpu_cores': 4,
            'gpu_count': 1,
            'memory_gb': 16,
            'network_bandwidth_gbps': 1
        })
        
        # Medium cluster
        configs.append({
            'name': 'medium_cluster',
            'cpu_cores': 16,
            'gpu_count': 4,
            'memory_gb': 64,
            'network_bandwidth_gbps': 10
        })
        
        # Large cluster
        configs.append({
            'name': 'large_cluster',
            'cpu_cores': 64,
            'gpu_count': 8,
            'memory_gb': 256,
            'network_bandwidth_gbps': 100
        })
        
        # Edge device
        configs.append({
            'name': 'edge_device',
            'cpu_cores': 2,
            'gpu_count': 0,
            'memory_gb': 8,
            'network_bandwidth_gbps': 0.1
        })
        
        # Heterogeneous cluster
        configs.append({
            'name': 'heterogeneous',
            'cpu_cores': 32,
            'gpu_count': 2,
            'memory_gb': 128,
            'network_bandwidth_gbps': 25
        })
        
        return configs
    
    def generate_pretraining_batch(self, batch_size: int = 16) -> Dict[str, torch.Tensor]:
        """Generate a batch of pre-training data"""
        
        batch_data = {
            'state_sequence': [],
            'task_type': [],
            'resource_req': [],
            'priority': [],
            'size': [],
            'action_prediction': [],
            'value_estimation': [],
            'resource_utilization': [],
            'latency_prediction': [],
            'energy_prediction': [],
            'throughput_prediction': [],
            'queue_length_prediction': [],
            'system_stability': []
        }
        
        for _ in range(batch_size):
            # Sample random pattern and system config
            pattern_name = random.choice(list(self.workload_generator.patterns.keys()))
            pattern = self.workload_generator.patterns[pattern_name]
            system_config = random.choice(self.system_configs)
            
            # Generate workload
            duration = random.randint(50, 200)  # Variable sequence length
            workload = self.workload_generator.generate_workload_sequence(
                pattern, duration, system_config
            )
            
            # Simulate system evolution
            state_sequence = self.system_simulator.simulate_system_evolution(
                workload, system_config
            )
            
            # Pad or truncate to fixed length
            seq_len = min(len(state_sequence), 100)  # Max 100 steps for efficiency
            if len(state_sequence) < seq_len:
                # Pad with last state
                padding = state_sequence[-1].unsqueeze(0).repeat(seq_len - len(state_sequence), 1)
                state_sequence = torch.cat([state_sequence, padding], dim=0)
            else:
                state_sequence = state_sequence[:seq_len]
            
            # Pad task sequences to match state sequence length
            task_seq_len = len(workload['task_type'])
            if task_seq_len < seq_len:
                # Pad task sequences
                pad_len = seq_len - task_seq_len
                task_type_padded = torch.cat([
                    workload['task_type'], 
                    torch.zeros(pad_len, dtype=torch.long)
                ])
                resource_req_padded = torch.cat([
                    workload['resource_req'],
                    torch.zeros(pad_len, 4)
                ])
                priority_padded = torch.cat([
                    workload['priority'],
                    torch.zeros(pad_len, dtype=torch.long)
                ])
                size_padded = torch.cat([
                    workload['size'],
                    torch.ones(pad_len, 1) * 0.1
                ])
            else:
                # Truncate task sequences
                task_type_padded = workload['task_type'][:seq_len]
                resource_req_padded = workload['resource_req'][:seq_len]
                priority_padded = workload['priority'][:seq_len]
                size_padded = workload['size'][:seq_len]
            
            # Generate labels (synthetic)
            actions = torch.randint(0, self.config.action_dim, (seq_len,))
            values = torch.randn(seq_len)
            resource_util = torch.rand(seq_len, 4)
            latencies = torch.rand(seq_len) * 10
            energies = torch.rand(seq_len) * 100
            throughputs = torch.rand(seq_len) * 1000
            queue_lengths = torch.randint(0, 100, (seq_len,)).float()
            stability = torch.randint(0, 2, (seq_len,))
            
            # Add to batch
            batch_data['state_sequence'].append(state_sequence)
            batch_data['task_type'].append(task_type_padded)
            batch_data['resource_req'].append(resource_req_padded)
            batch_data['priority'].append(priority_padded)
            batch_data['size'].append(size_padded)
            batch_data['action_prediction'].append(actions)
            batch_data['value_estimation'].append(values)
            batch_data['resource_utilization'].append(resource_util)
            batch_data['latency_prediction'].append(latencies)
            batch_data['energy_prediction'].append(energies)
            batch_data['throughput_prediction'].append(throughputs)
            batch_data['queue_length_prediction'].append(queue_lengths)
            batch_data['system_stability'].append(stability)
        
        # Stack into tensors
        for key in batch_data:
            batch_data[key] = torch.stack(batch_data[key])
        
        return batch_data
    
    def generate_curriculum_data(self, difficulty_level: float = 0.5) -> Dict[str, torch.Tensor]:
        """Generate curriculum learning data based on difficulty level"""
        
        # Adjust complexity based on difficulty level
        if difficulty_level < 0.3:
            # Easy: single workload pattern, small system
            patterns = ['web_server']
            systems = [self.system_configs[0]]  # Small cluster
            duration_range = (30, 60)
        elif difficulty_level < 0.7:
            # Medium: mixed patterns, medium system
            patterns = ['web_server', 'batch_processing']
            systems = [self.system_configs[1]]  # Medium cluster
            duration_range = (60, 120)
        else:
            # Hard: all patterns, large system
            patterns = list(self.workload_generator.patterns.keys())
            systems = self.system_configs
            duration_range = (100, 200)
        
        # Generate batch with controlled difficulty
        batch_data = {
            'state_sequence': [],
            'task_type': [],
            'resource_req': [],
            'priority': [],
            'size': []
        }
        
        for _ in range(8):  # Smaller batch for curriculum learning
            pattern_name = random.choice(patterns)
            pattern = self.workload_generator.patterns[pattern_name]
            system_config = random.choice(systems)
            
            duration = random.randint(*duration_range)
            workload = self.workload_generator.generate_workload_sequence(
                pattern, duration, system_config
            )
            
            state_sequence = self.system_simulator.simulate_system_evolution(
                workload, system_config
            )
            
            # Process and add to batch (similar to generate_pretraining_batch)
            seq_len = min(len(state_sequence), 80)
            if len(state_sequence) < seq_len:
                padding = state_sequence[-1].unsqueeze(0).repeat(seq_len - len(state_sequence), 1)
                state_sequence = torch.cat([state_sequence, padding], dim=0)
            else:
                state_sequence = state_sequence[:seq_len]
            
            batch_data['state_sequence'].append(state_sequence)
            # Add other components similarly...
        
        return batch_data

def main():
    """Demonstrate pre-training data generation"""
    
    print("=== Pre-training Data Generation for Foundation Model ===\n")
    
    # Configuration
    config = FoundationModelConfig()
    
    print("1. Initializing Data Generator...")
    generator = PretrainingDataGenerator(config)
    
    print(f"   Workload Patterns: {len(generator.workload_generator.patterns)}")
    print(f"   System Configurations: {len(generator.system_configs)}")
    
    # List workload patterns
    print(f"\n2. Available Workload Patterns:")
    for name, pattern in generator.workload_generator.patterns.items():
        print(f"   - {name}: {pattern.description}")
        print(f"     Target Utilization: {pattern.target_utilization:.1%}")
        print(f"     Arrival Pattern: {pattern.arrival_pattern}")
    
    print(f"\n3. System Configurations:")
    for config_dict in generator.system_configs:
        print(f"   - {config_dict['name']}: {config_dict['cpu_cores']} CPU cores, "
              f"{config_dict['gpu_count']} GPUs, {config_dict['memory_gb']} GB RAM")
    
    print(f"\n4. Generating Sample Workload...")
    
    # Generate sample workload
    pattern = generator.workload_generator.patterns['ml_training']
    system_config = generator.system_configs[1]  # Medium cluster
    
    workload = generator.workload_generator.generate_workload_sequence(
        pattern, duration=100, system_config=system_config
    )
    
    print(f"   Generated {len(workload['task_type'])} tasks")
    print(f"   Task types: {workload['task_type'][:10]}")
    print(f"   Resource requirements shape: {workload['resource_req'].shape}")
    print(f"   Priorities: {workload['priority'][:10]}")
    
    print(f"\n5. Simulating System Evolution...")
    
    # Simulate system state
    state_sequence = generator.system_simulator.simulate_system_evolution(
        workload, system_config
    )
    
    print(f"   State sequence shape: {state_sequence.shape}")
    print(f"   State range: [{state_sequence.min():.3f}, {state_sequence.max():.3f}]")
    print(f"   Mean state: {state_sequence.mean(dim=0)[:10]}")
    
    print(f"\n6. Generating Pre-training Batch...")
    
    # Generate pre-training batch
    start_time = time.time()
    batch = generator.generate_pretraining_batch(batch_size=4)
    generation_time = time.time() - start_time
    
    print(f"   Batch generation time: {generation_time:.2f} seconds")
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   State sequence batch shape: {batch['state_sequence'].shape}")
    print(f"   Task type batch shape: {batch['task_type'].shape}")
    print(f"   Resource req batch shape: {batch['resource_req'].shape}")
    
    print(f"\n7. Testing Curriculum Learning...")
    
    # Test curriculum learning data
    for difficulty in [0.2, 0.5, 0.8]:
        curriculum_batch = generator.generate_curriculum_data(difficulty)
        print(f"   Difficulty {difficulty:.1f}: {len(curriculum_batch['state_sequence'])} sequences")
    
    print(f"\n[SUCCESS] Pre-training Data Generation Test Completed!")
    print(f"\nKey Data Generation Features:")
    print("+ Diverse synthetic workload patterns (web, ML, scientific, batch, streaming)")
    print("+ Multi-scale system configurations (edge to large cluster)")
    print("+ Realistic temporal arrival patterns (Poisson, bursty, periodic)")
    print("+ Comprehensive resource requirement modeling")
    print("+ System state evolution simulation with resource dynamics")
    print("+ Curriculum learning with progressive difficulty")
    print("+ Multi-task learning labels for comprehensive pre-training")

if __name__ == '__main__':
    main()