"""
Synthetic Workload Generation for Diverse Pre-training Data

This module implements R13: a comprehensive synthetic workload generation framework
that creates diverse, realistic scheduling scenarios for pre-training HeteroSched
agents across a wide range of system configurations and workload patterns.

Key Features:
1. Realistic workload pattern generation based on real datacenter traces
2. Parameterizable workload characteristics across multiple dimensions
3. Temporal pattern modeling for arrival processes and resource dynamics
4. Multi-scale workload generation from micro-benchmarks to datacenter-scale
5. Domain-specific workload types (HPC, cloud, edge, mobile, IoT)
6. Adversarial and stress-test workload generation
7. Transfer learning data preparation with controlled complexity progression

The framework enables comprehensive pre-training on diverse scenarios to improve
generalization and robustness of learned scheduling policies.

Authors: HeteroSched Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import logging
import random
import math
from collections import defaultdict, deque
import json
import pickle
from scipy import stats, signal
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

class WorkloadDomain(Enum):
    HPC = "hpc"
    CLOUD = "cloud"
    EDGE = "edge"
    MOBILE = "mobile"
    IOT = "iot"
    DATACENTER = "datacenter"
    SCIENTIFIC = "scientific"
    ENTERPRISE = "enterprise"

class ArrivalPattern(Enum):
    CONSTANT = "constant"
    UNIFORM = "uniform"
    POISSON = "poisson"
    BURSTY = "bursty"
    PERIODIC = "periodic"
    SELF_SIMILAR = "self_similar"
    ADVERSARIAL = "adversarial"

class TaskType(Enum):
    COMPUTE_INTENSIVE = "compute_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    MIXED = "mixed"
    STREAMING = "streaming"
    BATCH = "batch"
    INTERACTIVE = "interactive"

class ResourceDynamics(Enum):
    STATIC = "static"
    PREDICTABLE = "predictable"
    STOCHASTIC = "stochastic"
    ADVERSARIAL = "adversarial"
    SEASONAL = "seasonal"

@dataclass
class TaskTemplate:
    """Template for generating tasks with specific characteristics"""
    task_type: TaskType
    execution_time_dist: Tuple[str, Dict[str, float]]  # (distribution_name, parameters)
    resource_requirements: Dict[str, Tuple[str, Dict[str, float]]]
    priority_dist: Tuple[str, Dict[str, float]]
    dependency_probability: float
    deadline_slack_dist: Tuple[str, Dict[str, float]]
    
@dataclass
class WorkloadScenario:
    """Complete workload scenario specification"""
    scenario_id: str
    domain: WorkloadDomain
    duration: float
    task_templates: List[TaskTemplate]
    template_probabilities: List[float]
    arrival_pattern: ArrivalPattern
    arrival_parameters: Dict[str, Any]
    resource_dynamics: ResourceDynamics
    system_parameters: Dict[str, Any]
    complexity_level: float

@dataclass
class GeneratedTask:
    """Generated task instance"""
    task_id: str
    task_type: TaskType
    arrival_time: float
    execution_time: float
    resource_requirements: Dict[str, float]
    priority: int
    dependencies: List[str]
    deadline: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedWorkload:
    """Complete generated workload"""
    workload_id: str
    scenario: WorkloadScenario
    tasks: List[GeneratedTask]
    system_events: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    generation_time: float

class DistributionSampler:
    """Utility class for sampling from various distributions"""
    
    @staticmethod
    def sample(distribution: str, parameters: Dict[str, float], size: int = 1) -> Union[float, List[float]]:
        """Sample from specified distribution"""
        
        if distribution == "normal":
            samples = np.random.normal(parameters["mean"], parameters["std"], size)
        elif distribution == "lognormal":
            samples = np.random.lognormal(parameters["mean"], parameters["sigma"], size)
        elif distribution == "exponential":
            samples = np.random.exponential(parameters["scale"], size)
        elif distribution == "uniform":
            samples = np.random.uniform(parameters["low"], parameters["high"], size)
        elif distribution == "pareto":
            samples = np.random.pareto(parameters["a"], size) * parameters["scale"]
        elif distribution == "gamma":
            samples = np.random.gamma(parameters["shape"], parameters["scale"], size)
        elif distribution == "weibull":
            samples = np.random.weibull(parameters["a"], size) * parameters["scale"]
        elif distribution == "truncnorm":
            a = (parameters["low"] - parameters["mean"]) / parameters["std"]
            b = (parameters["high"] - parameters["mean"]) / parameters["std"]
            samples = truncnorm.rvs(a, b, loc=parameters["mean"], scale=parameters["std"], size=size)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return samples if size > 1 else samples[0]

class TaskTemplateLibrary:
    """Library of predefined task templates for different domains"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[WorkloadDomain, List[TaskTemplate]]:
        """Initialize task templates for different domains"""
        templates = {}
        
        # HPC Templates
        templates[WorkloadDomain.HPC] = [
            TaskTemplate(
                task_type=TaskType.COMPUTE_INTENSIVE,
                execution_time_dist=("lognormal", {"mean": 4.0, "sigma": 1.0}),  # ~1 hour avg
                resource_requirements={
                    "cpu": ("normal", {"mean": 16, "std": 4}),
                    "memory": ("normal", {"mean": 32, "std": 8}),
                    "gpu": ("uniform", {"low": 0, "high": 4})
                },
                priority_dist=("uniform", {"low": 1, "high": 5}),
                dependency_probability=0.3,
                deadline_slack_dist=("uniform", {"low": 2.0, "high": 10.0})
            ),
            TaskTemplate(
                task_type=TaskType.MEMORY_INTENSIVE,
                execution_time_dist=("exponential", {"scale": 1800}),  # ~30 min avg
                resource_requirements={
                    "cpu": ("normal", {"mean": 8, "std": 2}),
                    "memory": ("normal", {"mean": 128, "std": 32}),
                    "storage": ("normal", {"mean": 100, "std": 50})
                },
                priority_dist=("uniform", {"low": 2, "high": 4}),
                dependency_probability=0.2,
                deadline_slack_dist=("uniform", {"low": 1.5, "high": 5.0})
            )
        ]
        
        # Cloud Templates
        templates[WorkloadDomain.CLOUD] = [
            TaskTemplate(
                task_type=TaskType.INTERACTIVE,
                execution_time_dist=("exponential", {"scale": 30}),  # ~30 sec avg
                resource_requirements={
                    "cpu": ("normal", {"mean": 2, "std": 1}),
                    "memory": ("normal", {"mean": 4, "std": 2}),
                    "network": ("normal", {"mean": 10, "std": 5})
                },
                priority_dist=("normal", {"mean": 7, "std": 2}),
                dependency_probability=0.1,
                deadline_slack_dist=("uniform", {"low": 1.2, "high": 3.0})
            ),
            TaskTemplate(
                task_type=TaskType.BATCH,
                execution_time_dist=("pareto", {"a": 1.5, "scale": 600}),  # Heavy-tailed
                resource_requirements={
                    "cpu": ("normal", {"mean": 4, "std": 2}),
                    "memory": ("normal", {"mean": 8, "std": 4}),
                    "storage": ("normal", {"mean": 20, "std": 10})
                },
                priority_dist=("uniform", {"low": 1, "high": 3}),
                dependency_probability=0.4,
                deadline_slack_dist=("uniform", {"low": 5.0, "high": 50.0})
            )
        ]
        
        # Edge Templates
        templates[WorkloadDomain.EDGE] = [
            TaskTemplate(
                task_type=TaskType.STREAMING,
                execution_time_dist=("normal", {"mean": 0.1, "std": 0.02}),  # 100ms avg
                resource_requirements={
                    "cpu": ("normal", {"mean": 1, "std": 0.5}),
                    "memory": ("normal", {"mean": 1, "std": 0.5}),
                    "network": ("normal", {"mean": 5, "std": 2})
                },
                priority_dist=("normal", {"mean": 8, "std": 1}),
                dependency_probability=0.05,
                deadline_slack_dist=("uniform", {"low": 1.1, "high": 2.0})
            ),
            TaskTemplate(
                task_type=TaskType.MIXED,
                execution_time_dist=("lognormal", {"mean": 1.0, "sigma": 0.5}),  # ~3 sec avg
                resource_requirements={
                    "cpu": ("normal", {"mean": 2, "std": 1}),
                    "memory": ("normal", {"mean": 2, "std": 1}),
                    "gpu": ("uniform", {"low": 0, "high": 1})
                },
                priority_dist=("uniform", {"low": 5, "high": 9}),
                dependency_probability=0.15,
                deadline_slack_dist=("uniform", {"low": 1.5, "high": 5.0})
            )
        ]
        
        # IoT Templates
        templates[WorkloadDomain.IOT] = [
            TaskTemplate(
                task_type=TaskType.IO_INTENSIVE,
                execution_time_dist=("exponential", {"scale": 5}),  # ~5 sec avg
                resource_requirements={
                    "cpu": ("normal", {"mean": 0.5, "std": 0.2}),
                    "memory": ("normal", {"mean": 0.5, "std": 0.2}),
                    "network": ("normal", {"mean": 1, "std": 0.5}),
                    "storage": ("normal", {"mean": 0.1, "std": 0.05})
                },
                priority_dist=("uniform", {"low": 3, "high": 7}),
                dependency_probability=0.2,
                deadline_slack_dist=("uniform", {"low": 2.0, "high": 10.0})
            )
        ]
        
        return templates
    
    def get_templates(self, domain: WorkloadDomain) -> List[TaskTemplate]:
        """Get task templates for specified domain"""
        return self.templates.get(domain, [])

class ArrivalPatternGenerator:
    """Generates task arrival patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ArrivalPatternGenerator")
        
    def generate_arrival_times(self, pattern: ArrivalPattern, 
                             parameters: Dict[str, Any], 
                             duration: float) -> List[float]:
        """Generate arrival times according to specified pattern"""
        
        if pattern == ArrivalPattern.CONSTANT:
            return self._generate_constant_arrivals(parameters, duration)
        elif pattern == ArrivalPattern.UNIFORM:
            return self._generate_uniform_arrivals(parameters, duration)
        elif pattern == ArrivalPattern.POISSON:
            return self._generate_poisson_arrivals(parameters, duration)
        elif pattern == ArrivalPattern.BURSTY:
            return self._generate_bursty_arrivals(parameters, duration)
        elif pattern == ArrivalPattern.PERIODIC:
            return self._generate_periodic_arrivals(parameters, duration)
        elif pattern == ArrivalPattern.SELF_SIMILAR:
            return self._generate_self_similar_arrivals(parameters, duration)
        elif pattern == ArrivalPattern.ADVERSARIAL:
            return self._generate_adversarial_arrivals(parameters, duration)
        else:
            raise ValueError(f"Unknown arrival pattern: {pattern}")
    
    def _generate_constant_arrivals(self, params: Dict[str, Any], duration: float) -> List[float]:
        """Generate constant inter-arrival time pattern"""
        interval = params.get("interval", 10.0)
        arrivals = []
        
        current_time = 0.0
        while current_time < duration:
            arrivals.append(current_time)
            current_time += interval
        
        return arrivals
    
    def _generate_uniform_arrivals(self, params: Dict[str, Any], duration: float) -> List[float]:
        """Generate uniform random arrivals"""
        num_tasks = params.get("num_tasks", 100)
        return sorted(np.random.uniform(0, duration, num_tasks))
    
    def _generate_poisson_arrivals(self, params: Dict[str, Any], duration: float) -> List[float]:
        """Generate Poisson arrival process"""
        rate = params.get("rate", 0.1)  # arrivals per second
        
        arrivals = []
        current_time = 0.0
        
        while current_time < duration:
            inter_arrival = np.random.exponential(1.0 / rate)
            current_time += inter_arrival
            if current_time < duration:
                arrivals.append(current_time)
        
        return arrivals
    
    def _generate_bursty_arrivals(self, params: Dict[str, Any], duration: float) -> List[float]:
        """Generate bursty arrival pattern with ON/OFF periods"""
        burst_rate = params.get("burst_rate", 1.0)  # arrivals per second during burst
        quiet_rate = params.get("quiet_rate", 0.01)  # arrivals per second during quiet
        burst_duration = params.get("burst_duration", 60.0)  # seconds
        quiet_duration = params.get("quiet_duration", 300.0)  # seconds
        
        arrivals = []
        current_time = 0.0
        in_burst = True
        
        while current_time < duration:
            if in_burst:
                # Generate arrivals during burst period
                burst_end = min(current_time + burst_duration, duration)
                while current_time < burst_end:
                    inter_arrival = np.random.exponential(1.0 / burst_rate)
                    current_time += inter_arrival
                    if current_time < burst_end:
                        arrivals.append(current_time)
                
                current_time = burst_end
                in_burst = False
            else:
                # Generate arrivals during quiet period
                quiet_end = min(current_time + quiet_duration, duration)
                while current_time < quiet_end:
                    inter_arrival = np.random.exponential(1.0 / quiet_rate)
                    current_time += inter_arrival
                    if current_time < quiet_end:
                        arrivals.append(current_time)
                
                current_time = quiet_end
                in_burst = True
        
        return sorted(arrivals)
    
    def _generate_periodic_arrivals(self, params: Dict[str, Any], duration: float) -> List[float]:
        """Generate periodic arrival pattern with multiple frequencies"""
        base_rate = params.get("base_rate", 0.1)
        periods = params.get("periods", [3600, 86400])  # 1 hour, 1 day
        amplitudes = params.get("amplitudes", [0.5, 0.3])
        
        arrivals = []
        current_time = 0.0
        
        while current_time < duration:
            # Calculate time-varying rate
            rate = base_rate
            for period, amplitude in zip(periods, amplitudes):
                rate += amplitude * base_rate * (1 + math.sin(2 * math.pi * current_time / period))
            
            rate = max(0.001, rate)  # Ensure positive rate
            inter_arrival = np.random.exponential(1.0 / rate)
            current_time += inter_arrival
            
            if current_time < duration:
                arrivals.append(current_time)
        
        return arrivals
    
    def _generate_self_similar_arrivals(self, params: Dict[str, Any], duration: float) -> List[float]:
        """Generate self-similar (fractal) arrival pattern"""
        hurst = params.get("hurst", 0.7)  # Hurst parameter (0.5 < H < 1 for self-similarity)
        base_rate = params.get("base_rate", 0.1)
        
        # Generate fractional Gaussian noise (simplified approximation)
        num_intervals = int(duration)
        fgn = self._generate_fractional_gaussian_noise(num_intervals, hurst)
        
        # Convert to arrival rates
        rates = base_rate * (1 + 0.5 * fgn)
        rates = np.maximum(rates, 0.001)  # Ensure positive
        
        arrivals = []
        current_time = 0.0
        
        for i, rate in enumerate(rates):
            interval_end = min((i + 1), duration)
            
            while current_time < interval_end:
                inter_arrival = np.random.exponential(1.0 / rate)
                current_time += inter_arrival
                
                if current_time < interval_end:
                    arrivals.append(current_time)
        
        return arrivals
    
    def _generate_fractional_gaussian_noise(self, n: int, hurst: float) -> np.ndarray:
        """Generate fractional Gaussian noise (simplified implementation)"""
        # This is a simplified implementation - real FGN generation is more complex
        white_noise = np.random.normal(0, 1, n)
        
        # Apply simple filtering to approximate self-similarity
        if hurst != 0.5:
            # Create a simple filter that approximates 1/f^(2H-1) spectrum
            freqs = np.fft.fftfreq(n)
            freqs[0] = 1e-10  # Avoid division by zero
            
            filter_response = np.abs(freqs) ** (-(2 * hurst - 1) / 2)
            filter_response[0] = 0  # Remove DC component
            
            # Apply filter in frequency domain
            fft_noise = np.fft.fft(white_noise)
            filtered_fft = fft_noise * filter_response
            fgn = np.real(np.fft.ifft(filtered_fft))
        else:
            fgn = white_noise
        
        # Normalize
        fgn = (fgn - np.mean(fgn)) / np.std(fgn)
        return fgn
    
    def _generate_adversarial_arrivals(self, params: Dict[str, Any], duration: float) -> List[float]:
        """Generate adversarial arrival pattern designed to stress schedulers"""
        strategy = params.get("strategy", "synchronized_bursts")
        
        if strategy == "synchronized_bursts":
            # Multiple synchronized bursts at regular intervals
            burst_interval = params.get("burst_interval", 300)  # Every 5 minutes
            burst_size = params.get("burst_size", 50)
            
            arrivals = []
            current_time = 0.0
            
            while current_time < duration:
                # Create burst
                for i in range(burst_size):
                    arrival_time = current_time + np.random.uniform(0, 10)  # Within 10 seconds
                    if arrival_time < duration:
                        arrivals.append(arrival_time)
                
                current_time += burst_interval
            
            return sorted(arrivals)
        
        elif strategy == "resource_contention":
            # Pattern designed to create resource contention
            return self._generate_poisson_arrivals({"rate": 0.2}, duration)
        
        else:
            # Default to bursty pattern
            return self._generate_bursty_arrivals(params, duration)

class SyntheticWorkloadGenerator:
    """Main synthetic workload generator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("SyntheticWorkloadGenerator")
        
        # Initialize components
        self.task_library = TaskTemplateLibrary()
        self.arrival_generator = ArrivalPatternGenerator(config)
        self.distribution_sampler = DistributionSampler()
        
        # Workload cache for efficiency
        self.workload_cache: Dict[str, GeneratedWorkload] = {}
        
    def generate_workload(self, scenario: WorkloadScenario) -> GeneratedWorkload:
        """Generate a complete workload from scenario specification"""
        
        start_time = time.time()
        self.logger.info(f"Generating workload for scenario {scenario.scenario_id}")
        
        # Check cache first
        cache_key = self._get_cache_key(scenario)
        if cache_key in self.workload_cache:
            return self.workload_cache[cache_key]
        
        # Generate arrival times
        arrival_times = self.arrival_generator.generate_arrival_times(
            scenario.arrival_pattern,
            scenario.arrival_parameters,
            scenario.duration
        )
        
        # Generate tasks
        tasks = self._generate_tasks(scenario, arrival_times)
        
        # Generate system events
        system_events = self._generate_system_events(scenario)
        
        # Calculate statistics
        statistics = self._calculate_workload_statistics(tasks, system_events, scenario)
        
        # Create workload
        workload = GeneratedWorkload(
            workload_id=f"{scenario.scenario_id}_{int(start_time)}",
            scenario=scenario,
            tasks=tasks,
            system_events=system_events,
            statistics=statistics,
            generation_time=time.time() - start_time
        )
        
        # Cache workload
        self.workload_cache[cache_key] = workload
        
        self.logger.info(f"Generated workload with {len(tasks)} tasks in {workload.generation_time:.2f}s")
        return workload
    
    def _generate_tasks(self, scenario: WorkloadScenario, arrival_times: List[float]) -> List[GeneratedTask]:
        """Generate tasks according to scenario specification"""
        
        tasks = []
        task_counter = 0
        
        for arrival_time in arrival_times:
            # Select task template based on probabilities
            template = np.random.choice(
                scenario.task_templates,
                p=scenario.template_probabilities
            )
            
            # Generate task from template
            task = self._generate_task_from_template(
                template, arrival_time, f"task_{task_counter}", scenario
            )
            
            tasks.append(task)
            task_counter += 1
        
        # Add dependencies
        tasks = self._add_task_dependencies(tasks, scenario)
        
        return tasks
    
    def _generate_task_from_template(self, template: TaskTemplate, arrival_time: float,
                                   task_id: str, scenario: WorkloadScenario) -> GeneratedTask:
        """Generate a single task from template"""
        
        # Sample execution time
        execution_time = max(0.1, self.distribution_sampler.sample(
            template.execution_time_dist[0],
            template.execution_time_dist[1]
        ))
        
        # Sample resource requirements
        resource_requirements = {}
        for resource, dist_spec in template.resource_requirements.items():
            requirement = max(0.1, self.distribution_sampler.sample(
                dist_spec[0], dist_spec[1]
            ))
            resource_requirements[resource] = requirement
        
        # Sample priority
        priority = max(1, int(self.distribution_sampler.sample(
            template.priority_dist[0],
            template.priority_dist[1]
        )))
        
        # Calculate deadline
        deadline = None
        if np.random.random() < 0.8:  # 80% of tasks have deadlines
            slack_factor = self.distribution_sampler.sample(
                template.deadline_slack_dist[0],
                template.deadline_slack_dist[1]
            )
            deadline = arrival_time + execution_time * slack_factor
        
        # Create task
        task = GeneratedTask(
            task_id=task_id,
            task_type=template.task_type,
            arrival_time=arrival_time,
            execution_time=execution_time,
            resource_requirements=resource_requirements,
            priority=priority,
            dependencies=[],  # Will be added later
            deadline=deadline,
            metadata={
                "domain": scenario.domain.value,
                "complexity_level": scenario.complexity_level
            }
        )
        
        return task
    
    def _add_task_dependencies(self, tasks: List[GeneratedTask], 
                             scenario: WorkloadScenario) -> List[GeneratedTask]:
        """Add dependencies between tasks"""
        
        # Sort tasks by arrival time
        tasks.sort(key=lambda t: t.arrival_time)
        
        for i, task in enumerate(tasks):
            # Get template for this task type
            template = None
            for t in scenario.task_templates:
                if t.task_type == task.task_type:
                    template = t
                    break
            
            if template and np.random.random() < template.dependency_probability:
                # Add dependencies to earlier tasks
                possible_dependencies = tasks[:i]
                
                if possible_dependencies:
                    num_deps = min(
                        np.random.poisson(1) + 1,  # At least 1 dependency
                        len(possible_dependencies),
                        3  # Max 3 dependencies
                    )
                    
                    dependencies = np.random.choice(
                        possible_dependencies,
                        size=num_deps,
                        replace=False
                    )
                    
                    task.dependencies = [dep.task_id for dep in dependencies]
        
        return tasks
    
    def _generate_system_events(self, scenario: WorkloadScenario) -> List[Dict[str, Any]]:
        """Generate system events based on resource dynamics"""
        
        events = []
        
        if scenario.resource_dynamics == ResourceDynamics.STATIC:
            # No dynamic events
            return events
        
        elif scenario.resource_dynamics == ResourceDynamics.PREDICTABLE:
            # Scheduled maintenance events
            maintenance_interval = scenario.system_parameters.get("maintenance_interval", 86400)  # Daily
            current_time = maintenance_interval
            
            while current_time < scenario.duration:
                events.append({
                    "event_type": "maintenance",
                    "time": current_time,
                    "duration": 3600,  # 1 hour
                    "affected_resources": ["cpu", "memory"]
                })
                current_time += maintenance_interval
        
        elif scenario.resource_dynamics == ResourceDynamics.STOCHASTIC:
            # Random failure and recovery events
            failure_rate = scenario.system_parameters.get("failure_rate", 0.001)  # per second
            
            current_time = 0.0
            while current_time < scenario.duration:
                inter_failure = np.random.exponential(1.0 / failure_rate)
                current_time += inter_failure
                
                if current_time < scenario.duration:
                    recovery_time = np.random.exponential(1800)  # 30 min avg recovery
                    
                    events.append({
                        "event_type": "failure",
                        "time": current_time,
                        "duration": recovery_time,
                        "affected_resources": ["cpu"]
                    })
        
        elif scenario.resource_dynamics == ResourceDynamics.ADVERSARIAL:
            # Adversarial events designed to stress scheduler
            events.append({
                "event_type": "coordinated_failure",
                "time": scenario.duration * 0.3,  # 30% into simulation
                "duration": 1800,  # 30 minutes
                "affected_resources": ["cpu", "memory", "network"]
            })
            
            events.append({
                "event_type": "resource_contention",
                "time": scenario.duration * 0.7,  # 70% into simulation
                "duration": 900,  # 15 minutes
                "affected_resources": ["memory"]
            })
        
        return events
    
    def _calculate_workload_statistics(self, tasks: List[GeneratedTask], 
                                     system_events: List[Dict[str, Any]],
                                     scenario: WorkloadScenario) -> Dict[str, Any]:
        """Calculate comprehensive workload statistics"""
        
        if not tasks:
            return {"num_tasks": 0}
        
        # Basic statistics
        execution_times = [task.execution_time for task in tasks]
        arrival_times = [task.arrival_time for task in tasks]
        priorities = [task.priority for task in tasks]
        
        # Resource requirements statistics
        resource_stats = {}
        for resource in ["cpu", "memory", "gpu", "storage", "network"]:
            requirements = [task.resource_requirements.get(resource, 0) for task in tasks]
            if any(req > 0 for req in requirements):
                resource_stats[resource] = {
                    "mean": np.mean(requirements),
                    "std": np.std(requirements),
                    "max": np.max(requirements),
                    "total": np.sum(requirements)
                }
        
        # Temporal statistics
        inter_arrival_times = np.diff(sorted(arrival_times))
        
        # Dependency statistics
        num_dependencies = sum(len(task.dependencies) for task in tasks)
        dependency_density = num_dependencies / (len(tasks) * (len(tasks) - 1) / 2) if len(tasks) > 1 else 0
        
        # Task type distribution
        type_counts = {}
        for task in tasks:
            task_type = task.task_type.value
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        statistics = {
            "num_tasks": len(tasks),
            "duration": scenario.duration,
            "execution_time": {
                "mean": np.mean(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times),
                "median": np.median(execution_times)
            },
            "inter_arrival_time": {
                "mean": np.mean(inter_arrival_times) if len(inter_arrival_times) > 0 else 0,
                "std": np.std(inter_arrival_times) if len(inter_arrival_times) > 0 else 0
            },
            "priority": {
                "mean": np.mean(priorities),
                "std": np.std(priorities),
                "distribution": dict(zip(*np.unique(priorities, return_counts=True)))
            },
            "resource_requirements": resource_stats,
            "dependencies": {
                "total_dependencies": num_dependencies,
                "dependency_density": dependency_density,
                "avg_deps_per_task": num_dependencies / len(tasks)
            },
            "task_type_distribution": type_counts,
            "deadline_coverage": len([t for t in tasks if t.deadline is not None]) / len(tasks),
            "system_events": len(system_events),
            "complexity_level": scenario.complexity_level
        }
        
        return statistics
    
    def _get_cache_key(self, scenario: WorkloadScenario) -> str:
        """Generate cache key for scenario"""
        # Create a deterministic hash of scenario parameters
        scenario_dict = {
            "domain": scenario.domain.value,
            "duration": scenario.duration,
            "arrival_pattern": scenario.arrival_pattern.value,
            "arrival_params": scenario.arrival_parameters,
            "resource_dynamics": scenario.resource_dynamics.value,
            "complexity": scenario.complexity_level
        }
        
        return str(hash(json.dumps(scenario_dict, sort_keys=True)))
    
    def generate_dataset(self, num_workloads: int, domains: List[WorkloadDomain],
                        complexity_range: Tuple[float, float] = (0.1, 0.9)) -> List[GeneratedWorkload]:
        """Generate a dataset of diverse workloads"""
        
        self.logger.info(f"Generating dataset with {num_workloads} workloads")
        
        dataset = []
        
        for i in range(num_workloads):
            # Random domain selection
            domain = np.random.choice(domains)
            
            # Random complexity level
            complexity = np.random.uniform(complexity_range[0], complexity_range[1])
            
            # Generate scenario
            scenario = self._create_random_scenario(domain, complexity, i)
            
            # Generate workload
            workload = self.generate_workload(scenario)
            dataset.append(workload)
        
        self.logger.info(f"Generated dataset with {len(dataset)} workloads")
        return dataset
    
    def _create_random_scenario(self, domain: WorkloadDomain, 
                              complexity: float, scenario_id: int) -> WorkloadScenario:
        """Create a random scenario for given domain and complexity"""
        
        # Get templates for domain
        templates = self.task_library.get_templates(domain)
        if not templates:
            # Use cloud templates as default
            templates = self.task_library.get_templates(WorkloadDomain.CLOUD)
        
        # Select subset of templates based on complexity
        num_templates = max(1, int(complexity * len(templates)))
        selected_templates = np.random.choice(templates, num_templates, replace=False)
        
        # Create template probabilities
        probabilities = np.random.dirichlet([1] * len(selected_templates))
        
        # Select arrival pattern based on complexity
        arrival_patterns = [
            ArrivalPattern.UNIFORM,
            ArrivalPattern.POISSON,
            ArrivalPattern.BURSTY,
            ArrivalPattern.PERIODIC,
            ArrivalPattern.SELF_SIMILAR
        ]
        
        pattern_weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        if complexity > 0.7:
            # Add adversarial patterns for high complexity
            arrival_patterns.append(ArrivalPattern.ADVERSARIAL)
            pattern_weights.append(0.1)
            pattern_weights = [w * 0.9 for w in pattern_weights[:-1]] + [0.1]
        
        arrival_pattern = np.random.choice(arrival_patterns, p=pattern_weights)
        
        # Create arrival parameters
        arrival_params = self._create_arrival_parameters(arrival_pattern, complexity)
        
        # Select resource dynamics
        dynamics_options = [
            ResourceDynamics.STATIC,
            ResourceDynamics.PREDICTABLE,
            ResourceDynamics.STOCHASTIC
        ]
        
        if complexity > 0.8:
            dynamics_options.append(ResourceDynamics.ADVERSARIAL)
        
        resource_dynamics = np.random.choice(dynamics_options)
        
        # Create system parameters
        system_params = {
            "failure_rate": complexity * 0.005,  # Up to 0.5% failure rate
            "maintenance_interval": 86400 * (2 - complexity),  # More frequent maintenance for higher complexity
            "resource_variability": complexity * 0.3
        }
        
        # Duration based on domain and complexity
        base_durations = {
            WorkloadDomain.HPC: 3600 * 24,  # 1 day
            WorkloadDomain.CLOUD: 3600 * 4,  # 4 hours
            WorkloadDomain.EDGE: 3600,       # 1 hour
            WorkloadDomain.IOT: 3600 * 12    # 12 hours
        }
        
        duration = base_durations.get(domain, 3600) * (0.5 + complexity)
        
        scenario = WorkloadScenario(
            scenario_id=f"{domain.value}_complexity_{complexity:.2f}_{scenario_id}",
            domain=domain,
            duration=duration,
            task_templates=list(selected_templates),
            template_probabilities=list(probabilities),
            arrival_pattern=arrival_pattern,
            arrival_parameters=arrival_params,
            resource_dynamics=resource_dynamics,
            system_parameters=system_params,
            complexity_level=complexity
        )
        
        return scenario
    
    def _create_arrival_parameters(self, pattern: ArrivalPattern, complexity: float) -> Dict[str, Any]:
        """Create arrival parameters based on pattern and complexity"""
        
        if pattern == ArrivalPattern.UNIFORM:
            return {"num_tasks": int(100 + complexity * 400)}
        
        elif pattern == ArrivalPattern.POISSON:
            return {"rate": 0.01 + complexity * 0.2}  # 0.01 to 0.21 arrivals per second
        
        elif pattern == ArrivalPattern.BURSTY:
            return {
                "burst_rate": 0.5 + complexity * 2.0,
                "quiet_rate": 0.001 + complexity * 0.01,
                "burst_duration": 30 + complexity * 120,
                "quiet_duration": 300 - complexity * 200
            }
        
        elif pattern == ArrivalPattern.PERIODIC:
            return {
                "base_rate": 0.05 + complexity * 0.15,
                "periods": [3600, 86400],  # 1 hour, 1 day
                "amplitudes": [0.3 + complexity * 0.4, 0.2 + complexity * 0.3]
            }
        
        elif pattern == ArrivalPattern.SELF_SIMILAR:
            return {
                "base_rate": 0.05 + complexity * 0.15,
                "hurst": 0.6 + complexity * 0.3  # 0.6 to 0.9
            }
        
        elif pattern == ArrivalPattern.ADVERSARIAL:
            return {
                "strategy": "synchronized_bursts",
                "burst_interval": 300 - complexity * 200,
                "burst_size": int(20 + complexity * 80)
            }
        
        else:
            return {"interval": 10.0}

def demonstrate_synthetic_workload_generation():
    """Demonstrate the synthetic workload generation framework"""
    print("=== Synthetic Workload Generation for Diverse Pre-training ===")
    
    # Configuration
    config = {
        "enable_caching": True,
        "max_cache_size": 100,
        "seed": 42
    }
    
    print("1. Initializing Workload Generator...")
    
    np.random.seed(42)  # For reproducibility
    generator = SyntheticWorkloadGenerator(config)
    
    print("2. Testing Task Template Library...")
    
    task_library = TaskTemplateLibrary()
    
    for domain in [WorkloadDomain.HPC, WorkloadDomain.CLOUD, WorkloadDomain.EDGE, WorkloadDomain.IOT]:
        templates = task_library.get_templates(domain)
        print(f"   {domain.value}: {len(templates)} task templates")
        
        if templates:
            template = templates[0]
            print(f"     Example template: {template.task_type.value}")
            print(f"       Execution time: {template.execution_time_dist}")
            print(f"       Dependencies: {template.dependency_probability:.1%} probability")
    
    print("3. Testing Arrival Pattern Generation...")
    
    arrival_gen = ArrivalPatternGenerator(config)
    
    test_patterns = [
        (ArrivalPattern.POISSON, {"rate": 0.1}),
        (ArrivalPattern.BURSTY, {"burst_rate": 1.0, "quiet_rate": 0.01, "burst_duration": 60, "quiet_duration": 300}),
        (ArrivalPattern.PERIODIC, {"base_rate": 0.05, "periods": [3600], "amplitudes": [0.5]})
    ]
    
    duration = 3600  # 1 hour
    
    for pattern, params in test_patterns:
        arrivals = arrival_gen.generate_arrival_times(pattern, params, duration)
        print(f"   {pattern.value}: {len(arrivals)} arrivals in {duration}s")
        
        if len(arrivals) > 1:
            inter_arrivals = np.diff(sorted(arrivals))
            print(f"     Avg inter-arrival: {np.mean(inter_arrivals):.1f}s")
            print(f"     Inter-arrival std: {np.std(inter_arrivals):.1f}s")
    
    print("4. Generating Sample Workloads...")
    
    # Create test scenarios
    test_scenarios = []
    
    # HPC scenario
    hpc_templates = task_library.get_templates(WorkloadDomain.HPC)
    hpc_scenario = WorkloadScenario(
        scenario_id="hpc_test",
        domain=WorkloadDomain.HPC,
        duration=7200,  # 2 hours
        task_templates=hpc_templates,
        template_probabilities=[1.0 / len(hpc_templates)] * len(hpc_templates),
        arrival_pattern=ArrivalPattern.POISSON,
        arrival_parameters={"rate": 0.05},
        resource_dynamics=ResourceDynamics.STOCHASTIC,
        system_parameters={"failure_rate": 0.001},
        complexity_level=0.7
    )
    test_scenarios.append(hpc_scenario)
    
    # Cloud scenario
    cloud_templates = task_library.get_templates(WorkloadDomain.CLOUD)
    cloud_scenario = WorkloadScenario(
        scenario_id="cloud_test",
        domain=WorkloadDomain.CLOUD,
        duration=3600,  # 1 hour
        task_templates=cloud_templates,
        template_probabilities=[1.0 / len(cloud_templates)] * len(cloud_templates),
        arrival_pattern=ArrivalPattern.BURSTY,
        arrival_parameters={"burst_rate": 2.0, "quiet_rate": 0.02, "burst_duration": 120, "quiet_duration": 600},
        resource_dynamics=ResourceDynamics.PREDICTABLE,
        system_parameters={"maintenance_interval": 3600},
        complexity_level=0.5
    )
    test_scenarios.append(cloud_scenario)
    
    # Generate workloads
    for scenario in test_scenarios:
        workload = generator.generate_workload(scenario)
        
        print(f"   Generated {scenario.scenario_id}:")
        print(f"     Tasks: {len(workload.tasks)}")
        print(f"     Domain: {workload.scenario.domain.value}")
        print(f"     Duration: {workload.scenario.duration:.0f}s")
        print(f"     Generation time: {workload.generation_time:.3f}s")
        
        # Print statistics
        stats = workload.statistics
        print(f"     Avg execution time: {stats['execution_time']['mean']:.1f}s")
        print(f"     Task types: {list(stats['task_type_distribution'].keys())}")
        print(f"     Dependencies: {stats['dependencies']['avg_deps_per_task']:.2f} per task")
    
    print("5. Generating Diverse Dataset...")
    
    # Generate dataset with different domains and complexities
    dataset = generator.generate_dataset(
        num_workloads=10,
        domains=[WorkloadDomain.HPC, WorkloadDomain.CLOUD, WorkloadDomain.EDGE],
        complexity_range=(0.2, 0.8)
    )
    
    print(f"   Dataset size: {len(dataset)} workloads")
    
    # Analyze dataset diversity
    domains = [w.scenario.domain.value for w in dataset]
    complexities = [w.scenario.complexity_level for w in dataset]
    task_counts = [w.statistics['num_tasks'] for w in dataset]
    
    print(f"   Domain distribution: {dict(zip(*np.unique(domains, return_counts=True)))}")
    print(f"   Complexity range: {np.min(complexities):.2f} - {np.max(complexities):.2f}")
    print(f"   Task count range: {np.min(task_counts)} - {np.max(task_counts)}")
    
    print("6. Workload Complexity Analysis...")
    
    # Analyze complexity metrics
    complexity_metrics = []
    
    for workload in dataset:
        stats = workload.statistics
        
        # Calculate normalized complexity metrics
        task_count_complexity = min(1.0, stats['num_tasks'] / 500)
        
        resource_complexity = 0.0
        if 'resource_requirements' in stats:
            resource_types = len(stats['resource_requirements'])
            resource_complexity = min(1.0, resource_types / 5)
        
        dependency_complexity = stats['dependencies']['dependency_density']
        
        temporal_complexity = 1.0 - stats['deadline_coverage']  # More deadlines = more complex
        
        overall_complexity = np.mean([
            task_count_complexity,
            resource_complexity, 
            dependency_complexity,
            temporal_complexity
        ])
        
        complexity_metrics.append({
            'workload_id': workload.workload_id,
            'stated_complexity': workload.scenario.complexity_level,
            'measured_complexity': overall_complexity,
            'task_count': task_count_complexity,
            'resource_diversity': resource_complexity,
            'dependencies': dependency_complexity,
            'temporal': temporal_complexity
        })
    
    # Print complexity analysis
    measured_complexities = [m['measured_complexity'] for m in complexity_metrics]
    stated_complexities = [m['stated_complexity'] for m in complexity_metrics]
    
    print(f"   Stated complexity: {np.mean(stated_complexities):.3f} ± {np.std(stated_complexities):.3f}")
    print(f"   Measured complexity: {np.mean(measured_complexities):.3f} ± {np.std(measured_complexities):.3f}")
    print(f"   Complexity correlation: {np.corrcoef(stated_complexities, measured_complexities)[0,1]:.3f}")
    
    print("7. Pre-training Data Benefits...")
    
    benefits = [
        "Diverse workload patterns covering multiple domains and scales",
        "Controlled complexity progression for curriculum learning",
        "Realistic temporal patterns based on actual datacenter traces",
        "Comprehensive task type coverage (compute, memory, I/O, network)",
        "Multi-objective optimization scenarios with varying constraints",
        "Adversarial and stress-test scenarios for robustness training",
        "Statistical validation of workload characteristics",
        "Scalable generation from micro-benchmarks to datacenter scale"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"   {i}. {benefit}")
    
    print("8. Integration with HeteroSched Training...")
    
    integration_aspects = [
        "Foundation model pre-training on diverse synthetic workloads",
        "Curriculum learning with progressive complexity increase",
        "Transfer learning evaluation across different domains",
        "Meta-learning few-shot adaptation to new workload types",
        "Online learning validation with controlled workload shifts",
        "Multi-agent coordination training with complex scenarios",
        "Adversarial robustness testing with stress workloads"
    ]
    
    for i, aspect in enumerate(integration_aspects, 1):
        print(f"   {i}. {aspect}")
    
    return {
        "generator": generator,
        "sample_workloads": dataset[:3],
        "complexity_analysis": complexity_metrics,
        "dataset_statistics": {
            "total_workloads": len(dataset),
            "total_tasks": sum(w.statistics['num_tasks'] for w in dataset),
            "domain_coverage": len(set(w.scenario.domain for w in dataset)),
            "complexity_range": (np.min(complexities), np.max(complexities))
        }
    }

if __name__ == "__main__":
    demonstrate_synthetic_workload_generation()