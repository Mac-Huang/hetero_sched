#!/usr/bin/env python3
"""
Real-time System State Extraction for HeteroSched

Advanced system monitoring and state extraction from actual hardware schedulers,
providing comprehensive real-time information for RL agent training and execution.

This module goes beyond basic psutil monitoring to extract scheduler-specific
information from kernel interfaces, GPU runtime APIs, and system performance counters.

Research Innovation: First comprehensive real-time state extraction system
specifically designed for heterogeneous scheduling RL environments.

Authors: HeteroSched Research Team
"""

import os
import sys
import time
import threading
import json
import logging
import subprocess
import ctypes
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import psutil
import GPUtil

# Windows/Linux compatibility
try:
    import wmi  # Windows WMI interface
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

try:
    import pynvml  # NVIDIA Management Library
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SystemSnapshot:
    """Comprehensive system state snapshot"""
    timestamp: float
    cpu_state: Dict[str, Any]
    memory_state: Dict[str, Any] 
    gpu_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    thermal_state: Dict[str, Any]
    power_state: Dict[str, Any]
    network_state: Dict[str, Any]
    storage_state: Dict[str, Any]
    process_state: Dict[str, Any]

class CPUSchedulerMonitor:
    """Monitor CPU scheduler state and metrics"""
    
    def __init__(self):
        self.proc_sched_path = "/proc/sched_debug"  # Linux scheduler debug info
        self.proc_stat_path = "/proc/stat"
        self.proc_loadavg_path = "/proc/loadavg"
        self.is_linux = os.name == 'posix'
        
        # Windows performance counters
        if not self.is_linux and WMI_AVAILABLE:
            self.wmi_conn = wmi.WMI()
        
        logger.info(f"CPU scheduler monitor initialized (Linux: {self.is_linux})")
    
    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get detailed CPU scheduler state"""
        
        if self.is_linux:
            return self._get_linux_scheduler_state()
        else:
            return self._get_windows_scheduler_state()
    
    def _get_linux_scheduler_state(self) -> Dict[str, Any]:
        """Extract Linux CFS (Completely Fair Scheduler) state"""
        
        scheduler_info = {
            'scheduler_type': 'CFS',
            'policy_info': {},
            'runqueue_info': {},
            'load_balancing': {},
            'scheduling_domains': []
        }
        
        try:
            # Parse /proc/sched_debug for detailed scheduler info
            if os.path.exists(self.proc_sched_path):
                with open(self.proc_sched_path, 'r') as f:
                    sched_debug = f.read()
                    scheduler_info['runqueue_info'] = self._parse_sched_debug(sched_debug)
            
            # Parse /proc/stat for CPU statistics
            if os.path.exists(self.proc_stat_path):
                with open(self.proc_stat_path, 'r') as f:
                    stat_info = f.read()
                    scheduler_info['cpu_stats'] = self._parse_proc_stat(stat_info)
            
            # Parse /proc/loadavg for load information
            if os.path.exists(self.proc_loadavg_path):
                with open(self.proc_loadavg_path, 'r') as f:
                    loadavg = f.read().strip().split()
                    scheduler_info['load_avg'] = {
                        '1min': float(loadavg[0]),
                        '5min': float(loadavg[1]),
                        '15min': float(loadavg[2]),
                        'running_tasks': int(loadavg[3].split('/')[0]),
                        'total_tasks': int(loadavg[3].split('/')[1]),
                        'last_pid': int(loadavg[4])
                    }
            
        except Exception as e:
            logger.warning(f"Failed to parse Linux scheduler info: {e}")
        
        return scheduler_info
    
    def _get_windows_scheduler_state(self) -> Dict[str, Any]:
        """Extract Windows scheduler state via WMI"""
        
        scheduler_info = {
            'scheduler_type': 'Windows',
            'process_info': {},
            'thread_info': {},
            'performance_counters': {}
        }
        
        try:
            if WMI_AVAILABLE:
                # Get processor information
                processors = self.wmi_conn.Win32_Processor()
                for proc in processors:
                    scheduler_info['processor_info'] = {
                        'name': proc.Name,
                        'cores': proc.NumberOfCores,
                        'logical_processors': proc.NumberOfLogicalProcessors,
                        'load_percentage': proc.LoadPercentage,
                        'current_clock_speed': proc.CurrentClockSpeed,
                        'max_clock_speed': proc.MaxClockSpeed
                    }
                    break  # Usually just one processor entry
                
                # Get performance counter info
                perf_counters = self.wmi_conn.Win32_PerfRawData_PerfOS_Processor()
                for counter in perf_counters:
                    if counter.Name == "_Total":
                        scheduler_info['performance_counters'] = {
                            'processor_time': counter.PercentProcessorTime,
                            'user_time': counter.PercentUserTime,
                            'privileged_time': counter.PercentPrivilegedTime,
                            'interrupt_time': counter.PercentInterruptTime,
                            'idle_time': counter.PercentIdleTime
                        }
                        break
            
        except Exception as e:
            logger.warning(f"Failed to get Windows scheduler info: {e}")
        
        # Fallback to psutil for basic info
        scheduler_info['fallback_info'] = {
            'cpu_percent': psutil.cpu_percent(percpu=True),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
        }
        
        return scheduler_info
    
    def _parse_sched_debug(self, sched_debug: str) -> Dict[str, Any]:
        """Parse /proc/sched_debug output"""
        
        runqueue_info = {}
        lines = sched_debug.split('\n')
        
        current_cpu = None
        for line in lines:
            line = line.strip()
            
            # CPU runqueue information
            if line.startswith('cpu#'):
                current_cpu = line.split()[0]
                runqueue_info[current_cpu] = {}
            
            elif current_cpu and ':' in line:
                try:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse numeric values
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                    
                    runqueue_info[current_cpu][key] = value
                    
                except ValueError:
                    continue
        
        return runqueue_info
    
    def _parse_proc_stat(self, stat_content: str) -> Dict[str, Any]:
        """Parse /proc/stat for CPU statistics"""
        
        cpu_stats = {}
        lines = stat_content.split('\n')
        
        for line in lines:
            if line.startswith('cpu'):
                parts = line.split()
                cpu_name = parts[0]
                
                if len(parts) >= 8:
                    cpu_stats[cpu_name] = {
                        'user': int(parts[1]),
                        'nice': int(parts[2]), 
                        'system': int(parts[3]),
                        'idle': int(parts[4]),
                        'iowait': int(parts[5]),
                        'irq': int(parts[6]),
                        'softirq': int(parts[7]),
                        'steal': int(parts[8]) if len(parts) > 8 else 0,
                        'guest': int(parts[9]) if len(parts) > 9 else 0,
                        'guest_nice': int(parts[10]) if len(parts) > 10 else 0
                    }
        
        return cpu_stats

class GPUSchedulerMonitor:
    """Monitor GPU scheduler and runtime state"""
    
    def __init__(self):
        self.nvml_available = NVML_AVAILABLE
        self.gpus_available = False
        
        try:
            if self.nvml_available:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.gpus_available = self.device_count > 0
                logger.info(f"NVML initialized: {self.device_count} GPUs found")
            else:
                # Fallback to GPUtil
                self.gpus = GPUtil.getGPUs()
                self.gpus_available = len(self.gpus) > 0
                logger.info(f"GPUtil fallback: {len(self.gpus)} GPUs found")
                
        except Exception as e:
            logger.warning(f"GPU monitoring initialization failed: {e}")
            self.gpus_available = False
    
    def get_gpu_state(self) -> Dict[str, Any]:
        """Get comprehensive GPU state"""
        
        if not self.gpus_available:
            return {'available': False, 'message': 'No GPUs detected'}
        
        gpu_state = {
            'available': True,
            'devices': [],
            'driver_version': '',
            'cuda_version': '',
            'total_memory_mb': 0,
            'total_used_memory_mb': 0
        }
        
        try:
            if self.nvml_available:
                gpu_state = self._get_nvml_gpu_state()
            else:
                gpu_state = self._get_gputil_gpu_state()
                
        except Exception as e:
            logger.error(f"GPU state extraction failed: {e}")
            gpu_state['error'] = str(e)
        
        return gpu_state
    
    def _get_nvml_gpu_state(self) -> Dict[str, Any]:
        """Get GPU state using NVIDIA ML API"""
        
        gpu_state = {
            'available': True,
            'devices': [],
            'driver_version': pynvml.nvmlSystemGetDriverVersion().decode(),
            'cuda_version': '',
            'total_memory_mb': 0,
            'total_used_memory_mb': 0
        }
        
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Basic device info
            name = pynvml.nvmlDeviceGetName(handle).decode()
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
            
            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
            except:
                power = 0
            
            # Clock speeds
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except:
                graphics_clock = memory_clock = 0
            
            # Running processes
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_count = len(processes)
                process_memory = sum(proc.usedGpuMemory for proc in processes) // (1024*1024)
            except:
                process_count = 0
                process_memory = 0
            
            device_info = {
                'id': i,
                'name': name,
                'memory_total_mb': memory_info.total // (1024*1024),
                'memory_used_mb': memory_info.used // (1024*1024),
                'memory_free_mb': memory_info.free // (1024*1024),
                'utilization_gpu': utilization.gpu,
                'utilization_memory': utilization.memory,
                'temperature_celsius': temp,
                'power_draw_watts': power,
                'graphics_clock_mhz': graphics_clock,
                'memory_clock_mhz': memory_clock,
                'process_count': process_count,
                'process_memory_mb': process_memory
            }
            
            gpu_state['devices'].append(device_info)
            gpu_state['total_memory_mb'] += device_info['memory_total_mb']
            gpu_state['total_used_memory_mb'] += device_info['memory_used_mb']
        
        return gpu_state
    
    def _get_gputil_gpu_state(self) -> Dict[str, Any]:
        """Fallback GPU state using GPUtil"""
        
        gpu_state = {
            'available': True,
            'devices': [],
            'driver_version': 'unknown',
            'total_memory_mb': 0,
            'total_used_memory_mb': 0
        }
        
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            device_info = {
                'id': i,
                'name': gpu.name,
                'memory_total_mb': gpu.memoryTotal,
                'memory_used_mb': gpu.memoryUsed,
                'memory_free_mb': gpu.memoryFree,
                'utilization_gpu': gpu.load * 100,
                'utilization_memory': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'temperature_celsius': gpu.temperature,
                'power_draw_watts': 0,  # Not available in GPUtil
                'graphics_clock_mhz': 0,
                'memory_clock_mhz': 0,
                'process_count': 0,
                'process_memory_mb': 0
            }
            
            gpu_state['devices'].append(device_info)
            gpu_state['total_memory_mb'] += device_info['memory_total_mb']
            gpu_state['total_used_memory_mb'] += device_info['memory_used_mb']
        
        return gpu_state

class ThermalMonitor:
    """Monitor system thermal state and thermal management"""
    
    def __init__(self):
        self.thermal_zones = []
        self.is_linux = os.name == 'posix'
        
        if self.is_linux:
            self._discover_linux_thermal_zones()
        
        logger.info(f"Thermal monitor initialized ({len(self.thermal_zones)} zones)")
    
    def _discover_linux_thermal_zones(self):
        """Discover available thermal zones on Linux"""
        
        thermal_base = "/sys/class/thermal"
        if not os.path.exists(thermal_base):
            return
        
        for item in os.listdir(thermal_base):
            if item.startswith('thermal_zone'):
                zone_path = os.path.join(thermal_base, item)
                try:
                    # Read zone type
                    type_file = os.path.join(zone_path, 'type')
                    if os.path.exists(type_file):
                        with open(type_file, 'r') as f:
                            zone_type = f.read().strip()
                    else:
                        zone_type = 'unknown'
                    
                    self.thermal_zones.append({
                        'path': zone_path,
                        'name': item,
                        'type': zone_type
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read thermal zone {item}: {e}")
    
    def get_thermal_state(self) -> Dict[str, Any]:
        """Get comprehensive thermal state"""
        
        thermal_state = {
            'zones': [],
            'cpu_temperature': 50.0,  # Default fallback
            'gpu_temperatures': [],
            'thermal_throttling': False,
            'cooling_devices': []
        }
        
        try:
            # Linux thermal zones
            if self.is_linux:
                thermal_state = self._get_linux_thermal_state()
            
            # Fallback to psutil sensors
            try:
                sensors = psutil.sensors_temperatures()
                if sensors:
                    for sensor_name, sensor_list in sensors.items():
                        for sensor in sensor_list:
                            if 'cpu' in sensor_name.lower() or 'core' in sensor_name.lower():
                                thermal_state['cpu_temperature'] = sensor.current
                                break
            except Exception:
                pass
            
        except Exception as e:
            logger.warning(f"Thermal state extraction failed: {e}")
        
        return thermal_state
    
    def _get_linux_thermal_state(self) -> Dict[str, Any]:
        """Get thermal state from Linux thermal zones"""
        
        thermal_state = {
            'zones': [],
            'cpu_temperature': 50.0,
            'thermal_throttling': False,
            'cooling_devices': []
        }
        
        for zone in self.thermal_zones:
            try:
                # Read temperature
                temp_file = os.path.join(zone['path'], 'temp')
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        temp_millidegree = int(f.read().strip())
                        temp_celsius = temp_millidegree / 1000.0
                else:
                    temp_celsius = 0.0
                
                # Read trip points (thermal thresholds)
                trip_points = []
                for i in range(10):  # Check up to 10 trip points
                    trip_temp_file = os.path.join(zone['path'], f'trip_point_{i}_temp')
                    trip_type_file = os.path.join(zone['path'], f'trip_point_{i}_type')
                    
                    if os.path.exists(trip_temp_file) and os.path.exists(trip_type_file):
                        with open(trip_temp_file, 'r') as f:
                            trip_temp = int(f.read().strip()) / 1000.0
                        with open(trip_type_file, 'r') as f:
                            trip_type = f.read().strip()
                        
                        trip_points.append({
                            'temperature': trip_temp,
                            'type': trip_type
                        })
                    else:
                        break
                
                zone_info = {
                    'name': zone['name'],
                    'type': zone['type'],
                    'temperature': temp_celsius,
                    'trip_points': trip_points
                }
                
                thermal_state['zones'].append(zone_info)
                
                # Update CPU temperature if this is a CPU zone
                if 'cpu' in zone['type'].lower() or 'x86_pkg_temp' in zone['type']:
                    thermal_state['cpu_temperature'] = temp_celsius
                
                # Check for thermal throttling
                for trip in trip_points:
                    if trip['type'] in ['critical', 'hot'] and temp_celsius >= trip['temperature'] * 0.9:
                        thermal_state['thermal_throttling'] = True
                
            except Exception as e:
                logger.warning(f"Failed to read thermal zone {zone['name']}: {e}")
        
        return thermal_state

class SystemStateExtractor:
    """Main system state extraction coordinator"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.running = False
        self.state_history = deque(maxlen=100)  # Keep last 100 snapshots
        
        # Initialize component monitors
        self.cpu_monitor = CPUSchedulerMonitor()
        self.gpu_monitor = GPUSchedulerMonitor()
        self.thermal_monitor = ThermalMonitor()
        
        # Background monitoring thread
        self.monitor_thread = None
        
        logger.info("System state extractor initialized")
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Background monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Background monitoring stopped")
    
    def get_current_snapshot(self) -> SystemSnapshot:
        """Get current comprehensive system snapshot"""
        
        timestamp = time.time()
        
        # Collect state from all monitors
        cpu_state = self.cpu_monitor.get_scheduler_state()
        gpu_state = self.gpu_monitor.get_gpu_state()
        thermal_state = self.thermal_monitor.get_thermal_state()
        
        # Basic system information
        memory_state = self._get_memory_state()
        network_state = self._get_network_state()
        storage_state = self._get_storage_state()
        process_state = self._get_process_state()
        power_state = self._get_power_state()
        
        snapshot = SystemSnapshot(
            timestamp=timestamp,
            cpu_state=cpu_state,
            memory_state=memory_state,
            gpu_state=gpu_state,
            scheduler_state=cpu_state,  # CPU scheduler info
            thermal_state=thermal_state,
            power_state=power_state,
            network_state=network_state,
            storage_state=storage_state,
            process_state=process_state
        )
        
        return snapshot
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.running:
            try:
                snapshot = self.get_current_snapshot()
                self.state_history.append(snapshot)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.update_interval)
    
    def _get_memory_state(self) -> Dict[str, Any]:
        """Get memory subsystem state"""
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'virtual': {
                'total_mb': memory.total // (1024*1024),
                'available_mb': memory.available // (1024*1024),
                'used_mb': memory.used // (1024*1024),
                'free_mb': memory.free // (1024*1024),
                'percent': memory.percent,
                'cached_mb': getattr(memory, 'cached', 0) // (1024*1024),
                'buffers_mb': getattr(memory, 'buffers', 0) // (1024*1024)
            },
            'swap': {
                'total_mb': swap.total // (1024*1024),
                'used_mb': swap.used // (1024*1024),
                'free_mb': swap.free // (1024*1024),
                'percent': swap.percent
            }
        }
    
    def _get_network_state(self) -> Dict[str, Any]:
        """Get network subsystem state"""
        
        try:
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())
            
            return {
                'io_counters': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                },
                'connections': net_connections
            }
        except Exception as e:
            logger.warning(f"Network state extraction failed: {e}")
            return {}
    
    def _get_storage_state(self) -> Dict[str, Any]:
        """Get storage subsystem state"""
        
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            state = {
                'disk_usage': {
                    'total_mb': disk_usage.total // (1024*1024),
                    'used_mb': disk_usage.used // (1024*1024),
                    'free_mb': disk_usage.free // (1024*1024),
                    'percent': (disk_usage.used / disk_usage.total) * 100
                }
            }
            
            if disk_io:
                state['io_counters'] = {
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_time': disk_io.read_time,
                    'write_time': disk_io.write_time
                }
            
            return state
            
        except Exception as e:
            logger.warning(f"Storage state extraction failed: {e}")
            return {}
    
    def _get_process_state(self) -> Dict[str, Any]:
        """Get process subsystem state"""
        
        try:
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
            
            # Top CPU consumers
            top_cpu = sorted(processes, key=lambda p: p.info.get('cpu_percent', 0), reverse=True)[:5]
            
            # Top memory consumers  
            top_memory = sorted(processes, key=lambda p: p.info.get('memory_percent', 0), reverse=True)[:5]
            
            return {
                'total_processes': len(processes),
                'top_cpu_consumers': [
                    {
                        'pid': p.info['pid'],
                        'name': p.info['name'],
                        'cpu_percent': p.info.get('cpu_percent', 0)
                    } for p in top_cpu
                ],
                'top_memory_consumers': [
                    {
                        'pid': p.info['pid'],
                        'name': p.info['name'],
                        'memory_percent': p.info.get('memory_percent', 0)
                    } for p in top_memory
                ]
            }
            
        except Exception as e:
            logger.warning(f"Process state extraction failed: {e}")
            return {}
    
    def _get_power_state(self) -> Dict[str, Any]:
        """Get power management state (if available)"""
        
        power_state = {
            'battery_present': False,
            'power_plugged': True,
            'estimated_cpu_power': 0,
            'estimated_gpu_power': 0
        }
        
        try:
            # Battery information
            battery = psutil.sensors_battery()
            if battery:
                power_state.update({
                    'battery_present': True,
                    'battery_percent': battery.percent,
                    'power_plugged': battery.power_plugged,
                    'time_left_seconds': battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                })
            
            # Estimate power consumption (very rough approximation)
            cpu_percent = psutil.cpu_percent()
            power_state['estimated_cpu_power'] = 20 + (cpu_percent / 100.0) * 80  # 20-100W range
            
            # GPU power from GPU monitor if available
            gpu_state = self.gpu_monitor.get_gpu_state()
            if gpu_state.get('available') and gpu_state.get('devices'):
                total_gpu_power = sum(dev.get('power_draw_watts', 0) for dev in gpu_state['devices'])
                power_state['estimated_gpu_power'] = total_gpu_power
            
        except Exception as e:
            logger.warning(f"Power state extraction failed: {e}")
        
        return power_state
    
    def get_rl_compatible_state(self, snapshot: SystemSnapshot = None) -> np.ndarray:
        """Convert system snapshot to RL-compatible 36-dimensional state vector"""
        
        if snapshot is None:
            snapshot = self.get_current_snapshot()
        
        # Initialize 36-dimensional state vector
        state = np.zeros(36, dtype=np.float32)
        
        # Task features (0-8): Placeholder since we don't have actual task info
        state[0:9] = 0.5  # Neutral values
        
        # CPU state (9-12)
        cpu_usage = 0.0
        if 'fallback_info' in snapshot.cpu_state:
            cpu_usage = np.mean(snapshot.cpu_state['fallback_info'].get('cpu_percent', [0])) / 100.0
        
        state[9] = min(cpu_usage, 1.0)
        state[10] = min(snapshot.thermal_state.get('cpu_temperature', 50) / 100.0, 1.0)
        state[11] = snapshot.memory_state['virtual']['percent'] / 100.0
        
        # GPU state (12-15)
        if snapshot.gpu_state.get('available') and snapshot.gpu_state.get('devices'):
            gpu_dev = snapshot.gpu_state['devices'][0]  # First GPU
            state[12] = gpu_dev.get('utilization_gpu', 0) / 100.0
            state[13] = gpu_dev.get('utilization_memory', 0) / 100.0
            state[14] = min(gpu_dev.get('temperature_celsius', 50) / 100.0, 1.0)
        
        # System load (15-17)
        if 'load_avg' in snapshot.cpu_state:
            load_info = snapshot.cpu_state['load_avg']
            state[15] = min(load_info.get('1min', 0) / 8.0, 1.0)  # Normalize by typical core count
            state[16] = min(load_info.get('5min', 0) / 8.0, 1.0)
            state[17] = min(load_info.get('15min', 0) / 8.0, 1.0)
        
        # Storage and network (18-21)
        if 'disk_usage' in snapshot.storage_state:
            state[18] = snapshot.storage_state['disk_usage']['percent'] / 100.0
        
        if 'io_counters' in snapshot.network_state:
            # Normalize network rates (very rough)
            bytes_per_sec = snapshot.network_state['io_counters'].get('bytes_sent', 0) / 1e6  # MB/s
            state[19] = min(bytes_per_sec / 100.0, 1.0)  # Cap at 100 MB/s
        
        # Process and power state (20-23)
        state[20] = min(snapshot.process_state.get('total_processes', 0) / 1000.0, 1.0)
        state[21] = snapshot.power_state.get('estimated_cpu_power', 50) / 200.0  # Normalize to 200W max
        
        # Queue and performance history (22-35): Placeholders
        state[22:36] = 0.1  # Default values
        
        return np.clip(state, 0.0, 1.0)

def main():
    """Demonstrate system state extraction"""
    
    print("=== Real-time System State Extraction Demo ===\n")
    
    # Create system state extractor
    extractor = SystemStateExtractor(update_interval=2.0)
    
    print("Getting comprehensive system snapshot...")
    snapshot = extractor.get_current_snapshot()
    
    print("\nSystem State Summary:")
    print("=" * 50)
    print(f"Timestamp: {snapshot.timestamp}")
    
    # CPU State
    print(f"\nCPU Scheduler:")
    print(f"  Type: {snapshot.cpu_state.get('scheduler_type', 'Unknown')}")
    if 'fallback_info' in snapshot.cpu_state:
        cpu_info = snapshot.cpu_state['fallback_info']
        print(f"  Usage: {np.mean(cpu_info.get('cpu_percent', [0])):.1f}%")
        print(f"  Cores: {cpu_info.get('cpu_count', 0)}")
    
    # GPU State
    print(f"\nGPU:")
    if snapshot.gpu_state.get('available'):
        gpu_devices = snapshot.gpu_state.get('devices', [])
        print(f"  Devices: {len(gpu_devices)}")
        for i, gpu in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu.get('name', 'Unknown')}")
            print(f"    Utilization: {gpu.get('utilization_gpu', 0):.1f}%")
            print(f"    Memory: {gpu.get('memory_used_mb', 0)}/{gpu.get('memory_total_mb', 0)} MB")
            print(f"    Temperature: {gpu.get('temperature_celsius', 0):.1f}°C")
    else:
        print("  No GPUs available")
    
    # Memory State
    print(f"\nMemory:")
    memory = snapshot.memory_state['virtual']
    print(f"  Usage: {memory['percent']:.1f}% ({memory['used_mb']}/{memory['total_mb']} MB)")
    
    # Thermal State
    print(f"\nThermal:")
    print(f"  CPU Temperature: {snapshot.thermal_state.get('cpu_temperature', 0):.1f}°C")
    print(f"  Thermal Zones: {len(snapshot.thermal_state.get('zones', []))}")
    print(f"  Throttling: {snapshot.thermal_state.get('thermal_throttling', False)}")
    
    # RL-compatible state
    print(f"\nRL-Compatible State Vector:")
    rl_state = extractor.get_rl_compatible_state(snapshot)
    print(f"  Shape: {rl_state.shape}")
    print(f"  Range: [{rl_state.min():.3f}, {rl_state.max():.3f}]")
    print(f"  Sample values: {rl_state[:10]}")
    
    # Test background monitoring
    print(f"\nTesting background monitoring...")
    extractor.start_monitoring()
    
    print("Monitoring for 10 seconds...")
    time.sleep(10)
    
    extractor.stop_monitoring()
    
    print(f"Collected {len(extractor.state_history)} snapshots")
    if extractor.state_history:
        latest = extractor.state_history[-1]
        print(f"Latest snapshot timestamp: {latest.timestamp}")

if __name__ == '__main__':
    main()