#!/usr/bin/env python3
"""
Hardware-in-the-Loop (HIL) Framework for HeteroSched

This module bridges the gap between simulation and reality by enabling
RL agents to interact with actual hardware schedulers, GPUs, and system
components while maintaining the RL training interface.

Research Innovation: First HIL framework for Deep RL in heterogeneous scheduling,
enabling safe sim-to-real transfer with continuous learning capabilities.

Authors: HeteroSched Research Team
"""

import os
import sys
import time
import subprocess
import threading
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import psutil
import GPUtil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rl.environments.hetero_env import HeteroSchedEnv

logger = logging.getLogger(__name__)

class SystemInterface(ABC):
    """Abstract interface for real system components"""
    
    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """Get current system state"""
        pass
    
    @abstractmethod
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scheduling action on real system"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if system is healthy and responsive"""
        pass

@dataclass
class RealTaskExecution:
    """Real task execution result from hardware"""
    task_id: str
    start_time: float
    end_time: float
    execution_time: float
    device_used: str
    memory_peak: int  # Peak memory usage in MB
    energy_consumed: float  # Energy in Joules (estimated)
    exit_code: int
    stdout: str
    stderr: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def was_successful(self) -> bool:
        return self.exit_code == 0
    
    @property
    def total_latency(self) -> float:
        return self.end_time - self.start_time

class LinuxSchedulerInterface(SystemInterface):
    """Interface to Linux CFS scheduler via cgroups and system calls"""
    
    def __init__(self, cgroup_path: str = "/sys/fs/cgroup"):
        self.cgroup_path = cgroup_path
        self.active_tasks = {}
        self.task_counter = 0
        
        # Create our own cgroup for controlled execution
        self.hetero_cgroup = os.path.join(cgroup_path, "hetero_sched")
        self._setup_cgroup()
        
        logger.info(f"Linux scheduler interface initialized with cgroup: {self.hetero_cgroup}")
    
    def _setup_cgroup(self):
        """Setup cgroup for controlled task execution"""
        try:
            if not os.path.exists(self.hetero_cgroup):
                os.makedirs(self.hetero_cgroup, exist_ok=True)
            
            # Enable CPU controller
            cpu_controller = os.path.join(self.hetero_cgroup, "cgroup.controllers")
            if os.path.exists(cpu_controller):
                with open(cpu_controller, 'w') as f:
                    f.write("cpu")
            
            logger.info("Cgroup setup completed")
            
        except PermissionError:
            logger.warning("Insufficient permissions for cgroup setup. Running without cgroup control.")
        except Exception as e:
            logger.error(f"Cgroup setup failed: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Extract current system state from Linux"""
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        # Load average (Windows doesn't have getloadavg)
        try:
            load_avg = os.getloadavg()
        except AttributeError:
            # Windows fallback: approximate with CPU usage
            current_load = np.mean(cpu_percent) / 100.0
            load_avg = (current_load, current_load, current_load)
        
        # Memory information  
        memory = psutil.virtual_memory()
        
        # Temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            cpu_temp = temps.get('coretemp', [{'current': 50.0}])[0]['current']
        except:
            cpu_temp = 50.0  # Default fallback
        
        # Process information
        process_count = len(psutil.pids())
        
        # Network and disk I/O
        net_io = psutil.net_io_counters()
        disk_io = psutil.disk_io_counters()
        
        system_state = {
            'timestamp': time.time(),
            'cpu': {
                'usage_percent': np.mean(cpu_percent),
                'usage_per_core': cpu_percent,
                'frequency_mhz': cpu_freq.current if cpu_freq else 2000,
                'load_avg_1min': load_avg[0],
                'load_avg_5min': load_avg[1], 
                'load_avg_15min': load_avg[2],
                'temperature_celsius': cpu_temp,
                'core_count': psutil.cpu_count()
            },
            'memory': {
                'total_mb': memory.total // (1024*1024),
                'available_mb': memory.available // (1024*1024),
                'used_mb': memory.used // (1024*1024),
                'usage_percent': memory.percent
            },
            'system': {
                'process_count': process_count,
                'uptime_seconds': time.time() - psutil.boot_time(),
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'disk_write_bytes': disk_io.write_bytes if disk_io else 0
            }
        }
        
        return system_state
    
    def execute_action(self, action: Dict[str, Any]) -> RealTaskExecution:
        """Execute task with specified scheduling parameters"""
        
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        # Parse action parameters
        device = action.get('device', 'cpu')  # 'cpu' or 'gpu'
        priority = action.get('priority', 0)   # -20 to 19 (nice values)
        cpu_limit = action.get('cpu_limit', 100)  # Percentage
        memory_limit = action.get('memory_limit', 1024)  # MB
        
        # Create task command (using a simple CPU-bound task for testing)
        if device == 'cpu':
            # CPU-intensive task: matrix multiplication
            cmd = [
                'python3', '-c',
                f'''
import numpy as np
import time
start = time.time()
# CPU-bound computation
for i in range(100):
    a = np.random.rand(100, 100)
    b = np.random.rand(100, 100)
    c = np.dot(a, b)
end = time.time()
print(f"Execution time: {{end - start:.4f}} seconds")
'''
            ]
        else:
            # GPU task (if available) - fallback to CPU
            cmd = [
                'python3', '-c',
                '''
import time
start = time.time()
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        a = torch.rand(1000, 1000, device=device)
        b = torch.rand(1000, 1000, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print("GPU computation completed")
    else:
        raise ImportError("CUDA not available")
except ImportError:
    # Fallback to CPU
    import numpy as np
    for i in range(200):
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100) 
        c = np.dot(a, b)
    print("CPU fallback computation completed")
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")
'''
            ]
        
        # Execute with system controls
        start_time = time.time()
        
        try:
            # Set process priority
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=lambda: os.nice(priority) if priority != 0 else None
            )
            
            # Monitor resource usage
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process.pid, task_id)
            )
            monitor_thread.start()
            
            # Wait for completion
            stdout, stderr = process.communicate()
            end_time = time.time()
            
            monitor_thread.join(timeout=1.0)
            
            # Collect results
            execution_result = RealTaskExecution(
                task_id=task_id,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                device_used=device,
                memory_peak=self.active_tasks.get(task_id, {}).get('peak_memory', 0),
                energy_consumed=self._estimate_energy_consumption(
                    end_time - start_time, device
                ),
                exit_code=process.returncode,
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore')
            )
            
            logger.info(f"Task {task_id} completed: {execution_result.execution_time:.3f}s on {device}")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return RealTaskExecution(
                task_id=task_id,
                start_time=start_time,
                end_time=time.time(),
                execution_time=0.0,
                device_used=device,
                memory_peak=0,
                energy_consumed=0.0,
                exit_code=-1,
                stdout="",
                stderr=str(e)
            )
    
    def _monitor_process(self, pid: int, task_id: str):
        """Monitor process resource usage during execution"""
        try:
            process = psutil.Process(pid)
            peak_memory = 0
            
            while process.is_running():
                try:
                    memory_info = process.memory_info()
                    peak_memory = max(peak_memory, memory_info.rss // (1024*1024))  # MB
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
            
            self.active_tasks[task_id] = {'peak_memory': peak_memory}
            
        except Exception as e:
            logger.warning(f"Process monitoring failed for {task_id}: {e}")
    
    def _estimate_energy_consumption(self, execution_time: float, device: str) -> float:
        """Estimate energy consumption based on execution time and device"""
        
        # Rough energy estimates (would need actual power measurement in production)
        if device == 'cpu':
            # Assume ~50W average CPU power during computation
            power_watts = 50.0
        else:
            # Assume ~200W average GPU power during computation  
            power_watts = 200.0
        
        energy_joules = power_watts * execution_time
        return energy_joules
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        try:
            # Check basic system responsiveness
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # System is healthy if not overloaded
            return cpu_usage < 95.0 and memory_usage < 95.0
            
        except Exception:
            return False

class GPUInterface(SystemInterface):
    """Interface to GPU scheduler and CUDA runtime"""
    
    def __init__(self):
        self.gpus = GPUtil.getGPUs()
        self.available = len(self.gpus) > 0
        
        if self.available:
            logger.info(f"GPU interface initialized: {len(self.gpus)} GPUs found")
        else:
            logger.warning("No GPUs found. GPU interface will use CPU fallback.")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current GPU state"""
        
        if not self.available:
            return {
                'gpu_count': 0,
                'available': False,
                'message': 'No GPUs detected'
            }
        
        gpu_states = []
        for i, gpu in enumerate(GPUtil.getGPUs()):
            gpu_state = {
                'id': i,
                'name': gpu.name,
                'memory_total_mb': gpu.memoryTotal,
                'memory_used_mb': gpu.memoryUsed,
                'memory_free_mb': gpu.memoryFree,
                'memory_util_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'gpu_util_percent': gpu.load * 100,
                'temperature_celsius': gpu.temperature
            }
            gpu_states.append(gpu_state)
        
        return {
            'gpu_count': len(self.gpus),
            'available': True,
            'gpus': gpu_states,
            'timestamp': time.time()
        }
    
    def execute_action(self, action: Dict[str, Any]) -> RealTaskExecution:
        """Execute GPU task"""
        
        # For now, delegate to CPU interface with GPU flag
        # In full implementation, would use CUDA runtime API
        linux_interface = LinuxSchedulerInterface()
        return linux_interface.execute_action({**action, 'device': 'gpu'})
    
    def is_healthy(self) -> bool:
        """Check GPU health"""
        if not self.available:
            return True  # No GPUs to check
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.temperature > 85.0:  # Temperature threshold
                    return False
                if gpu.memoryUtil > 0.95:  # Memory threshold
                    return False
            return True
        except Exception:
            return False

class HILEnvironment:
    """
    Hardware-in-the-Loop RL Environment
    
    Wraps real hardware interfaces to provide RL-compatible interaction
    while maintaining safety and monitoring capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize hardware interfaces
        self.linux_interface = LinuxSchedulerInterface()
        self.gpu_interface = GPUInterface()
        
        # Safety monitoring
        self.safety_monitor = SafetyMonitor()
        
        # Simulation fallback for training
        self.sim_env = HeteroSchedEnv(config)
        self.use_simulation = self.config.get('use_simulation_fallback', True)
        
        # HIL-specific tracking
        self.episode_step = 0
        self.real_executions = []
        self.safety_violations = []
        
        logger.info("HIL Environment initialized")
    
    def reset(self) -> np.ndarray:
        """Reset environment state"""
        self.episode_step = 0
        self.real_executions.clear()
        self.safety_violations.clear()
        
        if self.use_simulation:
            return self.sim_env.reset()
        else:
            # Get real system state for RL agent
            return self._get_real_system_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step in HIL environment"""
        
        # Convert RL action to system action
        system_action = self._rl_action_to_system_action(action)
        
        # Safety check
        if not self.safety_monitor.is_action_safe(system_action, self._get_current_system_state()):
            logger.warning(f"Unsafe action blocked: {system_action}")
            
            # Return penalty and use simulation fallback
            obs = self._get_real_system_observation()
            reward = -10.0  # Safety violation penalty
            done = False
            info = {'safety_violation': True, 'blocked_action': system_action}
            
            self.safety_violations.append({
                'step': self.episode_step,
                'action': system_action,
                'reason': 'safety_check_failed'
            })
            
            return obs, reward, done, info
        
        # Decide whether to use real hardware or simulation
        use_real_hardware = self._should_use_real_hardware()
        
        if use_real_hardware and not self.use_simulation:
            # Execute on real hardware
            execution_result = self.linux_interface.execute_action(system_action)
            self.real_executions.append(execution_result)
            
            # Convert real execution to RL feedback
            obs = self._get_real_system_observation()
            reward = self._compute_reward_from_real_execution(execution_result)
            done = self._check_episode_termination()
            info = {
                'real_execution': True,
                'task_id': execution_result.task_id,
                'execution_time': execution_result.execution_time,
                'device_used': execution_result.device_used
            }
            
        else:
            # Use simulation
            obs, reward, done, info = self.sim_env.step(action)
            info['real_execution'] = False
        
        self.episode_step += 1
        return obs, reward, done, info
    
    def _rl_action_to_system_action(self, rl_action: np.ndarray) -> Dict[str, Any]:
        """Convert RL action to system-executable action"""
        
        device_idx, priority_boost, batch_size = rl_action
        
        # Map to system parameters
        device = 'cpu' if device_idx == 0 else 'gpu'
        priority = int(priority_boost - 10)  # Convert to nice values (-20 to 19)
        cpu_limit = min(100, 20 + batch_size * 10)  # Scale batch to CPU limit
        memory_limit = min(4096, 512 + batch_size * 256)  # Scale to memory limit
        
        return {
            'device': device,
            'priority': priority,
            'cpu_limit': cpu_limit,
            'memory_limit': memory_limit,
            'batch_size': int(batch_size) + 1
        }
    
    def _get_real_system_observation(self) -> np.ndarray:
        """Get RL observation from real system state"""
        
        # Get system state from hardware interfaces
        linux_state = self.linux_interface.get_current_state()
        gpu_state = self.gpu_interface.get_current_state()
        
        # Convert to RL observation format (36-dimensional)
        observation = np.zeros(36, dtype=np.float32)
        
        # Task features (simplified - would need actual task queue)
        observation[0:9] = 0.5  # Neutral task features
        
        # System state features
        cpu_usage = linux_state['cpu']['usage_percent'] / 100.0
        cpu_temp = min(linux_state['cpu']['temperature_celsius'] / 100.0, 1.0)
        memory_usage = linux_state['memory']['usage_percent'] / 100.0
        
        observation[9] = cpu_usage
        observation[10] = cpu_temp
        observation[11] = memory_usage
        
        # GPU features
        if gpu_state['available'] and len(gpu_state['gpus']) > 0:
            gpu = gpu_state['gpus'][0]
            observation[12] = gpu['gpu_util_percent'] / 100.0
            observation[13] = gpu['memory_util_percent'] / 100.0
            observation[14] = min(gpu['temperature_celsius'] / 100.0, 1.0)
        
        # Fill remaining features with defaults or derived values
        observation[15:36] = 0.1  # Queue and history features
        
        return observation
    
    def _get_current_system_state(self) -> Dict[str, Any]:
        """Get comprehensive current system state"""
        return {
            'linux': self.linux_interface.get_current_state(),
            'gpu': self.gpu_interface.get_current_state(),
            'timestamp': time.time()
        }
    
    def _should_use_real_hardware(self) -> bool:
        """Decide whether to use real hardware for this step"""
        
        # For safety, start with low probability and gradually increase
        if len(self.real_executions) < 10:
            real_hardware_probability = 0.1  # 10% initially
        elif len(self.real_executions) < 100:
            real_hardware_probability = 0.3  # 30% after some experience
        else:
            real_hardware_probability = 0.8  # 80% after sufficient experience
        
        # Safety override
        if not self.safety_monitor.is_system_healthy():
            return False
        
        return np.random.random() < real_hardware_probability
    
    def _compute_reward_from_real_execution(self, execution: RealTaskExecution) -> float:
        """Compute RL reward from real execution results"""
        
        if not execution.was_successful:
            return -5.0  # Execution failure penalty
        
        # Simple reward based on execution time (lower is better)
        baseline_time = 1.0  # Expected baseline execution time
        time_ratio = execution.execution_time / baseline_time
        
        if time_ratio <= 1.0:
            reward = 1.0 - time_ratio * 0.5  # Reward for fast execution
        else:
            reward = -0.5 * (time_ratio - 1.0)  # Penalty for slow execution
        
        # Energy efficiency bonus/penalty
        energy_per_second = execution.energy_consumed / execution.execution_time
        if energy_per_second < 100:  # Efficient
            reward += 0.2
        elif energy_per_second > 300:  # Inefficient
            reward -= 0.3
        
        return float(np.clip(reward, -10.0, 2.0))
    
    def _check_episode_termination(self) -> bool:
        """Check if episode should terminate"""
        
        # Terminate on safety violations
        if len(self.safety_violations) > 3:
            return True
        
        # Terminate on system unhealthiness
        if not self.safety_monitor.is_system_healthy():
            return True
        
        # Normal episode length termination
        return self.episode_step >= self.config.get('max_episode_steps', 100)

class SafetyMonitor:
    """Safety monitoring for HIL operations"""
    
    def __init__(self):
        self.violation_history = []
        self.system_health_history = []
    
    def is_action_safe(self, action: Dict[str, Any], system_state: Dict[str, Any]) -> bool:
        """Check if action is safe to execute"""
        
        # Check system resource levels
        linux_state = system_state.get('linux', {})
        cpu_usage = linux_state.get('cpu', {}).get('usage_percent', 0)
        memory_usage = linux_state.get('memory', {}).get('usage_percent', 0)
        
        # Prevent actions when system is overloaded
        if cpu_usage > 90.0:
            return False
        
        if memory_usage > 90.0:
            return False
        
        # Check action parameters
        priority = action.get('priority', 0)
        if priority < -15:  # Prevent extremely high priority
            return False
        
        # Check recent violation history
        recent_violations = len([v for v in self.violation_history[-10:] if v])
        if recent_violations > 5:
            return False
        
        return True
    
    def is_system_healthy(self) -> bool:
        """Check overall system health"""
        try:
            # CPU check
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > 95.0:
                return False
            
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 95.0:
                return False
            
            # Disk space check
            disk = psutil.disk_usage('/')
            if disk.percent > 95.0:
                return False
            
            return True
            
        except Exception:
            return False

def main():
    """Demonstrate HIL framework"""
    
    print("=== Hardware-in-the-Loop Framework Demo ===\n")
    
    # Create HIL environment
    hil_config = {
        'use_simulation_fallback': True,
        'max_episode_steps': 20
    }
    
    hil_env = HILEnvironment(hil_config)
    
    print("Testing HIL Environment:")
    print("=" * 40)
    
    # Test episode
    obs = hil_env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    total_reward = 0
    for step in range(10):
        # Random action for testing
        action = np.array([
            np.random.randint(2),   # device
            np.random.randint(5),   # priority  
            np.random.randint(10)   # batch size
        ])
        
        obs, reward, done, info = hil_env.step(action)
        total_reward += reward
        
        print(f"Step {step}: action={action}, reward={reward:.3f}, "
              f"real_execution={info.get('real_execution', False)}")
        
        if 'safety_violation' in info:
            print(f"  ⚠️  Safety violation detected!")
        
        if done:
            break
    
    print(f"\nEpisode completed:")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Real executions: {len(hil_env.real_executions)}")
    print(f"Safety violations: {len(hil_env.safety_violations)}")
    
    # Test hardware interfaces directly
    print("\n" + "=" * 40)
    print("Testing Hardware Interfaces:")
    
    linux_interface = LinuxSchedulerInterface()
    system_state = linux_interface.get_current_state()
    
    print("Current System State:")
    print(f"  CPU Usage: {system_state['cpu']['usage_percent']:.1f}%")
    print(f"  Memory Usage: {system_state['memory']['usage_percent']:.1f}%")
    print(f"  CPU Temperature: {system_state['cpu']['temperature_celsius']:.1f}°C")
    print(f"  Process Count: {system_state['system']['process_count']}")
    
    # Execute a simple test task
    print("\nExecuting test task...")
    test_action = {
        'device': 'cpu',
        'priority': 0,
        'cpu_limit': 50,
        'memory_limit': 512,
        'batch_size': 1
    }
    
    execution_result = linux_interface.execute_action(test_action)
    print(f"Task executed successfully: {execution_result.was_successful}")
    print(f"Execution time: {execution_result.execution_time:.3f} seconds")
    print(f"Peak memory: {execution_result.memory_peak} MB")
    print(f"Energy consumed: {execution_result.energy_consumed:.2f} Joules")

if __name__ == '__main__':
    main()