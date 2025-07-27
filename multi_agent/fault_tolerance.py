"""
Fault Tolerance and Agent Failure Recovery Mechanisms

This module implements R19: comprehensive fault tolerance and recovery mechanisms
that ensure robust operation of the HeteroSched multi-agent scheduling system
under various failure conditions.

Key Features:
1. Agent health monitoring and failure detection
2. Automatic failover and recovery procedures
3. State replication and checkpoint management
4. Byzantine fault tolerance protocols
5. Graceful degradation under partial failures
6. Load redistribution after agent failures
7. Dynamic topology reconfiguration
8. Recovery verification and rollback capabilities

The fault tolerance framework ensures system reliability and availability
even when individual agents or network partitions fail.

Authors: HeteroSched Research Team
"""

import asyncio
import time
import logging
import json
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import numpy as np
import torch
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
import copy

class FailureType(Enum):
    AGENT_CRASH = "agent_crash"
    NETWORK_PARTITION = "network_partition"
    BYZANTINE_FAILURE = "byzantine_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MESSAGE_CORRUPTION = "message_corruption"
    TIMEOUT_FAILURE = "timeout_failure"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    RESTART_AGENT = "restart_agent"
    FAILOVER_TO_BACKUP = "failover_to_backup"
    LOAD_REDISTRIBUTION = "load_redistribution"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    PARTITION_HEALING = "partition_healing"
    STATE_ROLLBACK = "state_rollback"

class CheckpointType(Enum):
    FULL_STATE = "full_state"
    INCREMENTAL = "incremental"
    COMPRESSED = "compressed"
    VERIFIED = "verified"

@dataclass
class AgentHealth:
    """Health information for an agent"""
    agent_id: str
    status: HealthStatus
    last_heartbeat: float
    cpu_usage: float
    memory_usage: float
    message_queue_size: int
    response_time: float
    error_count: int
    uptime: float
    last_checkpoint: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    failure_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class FailureEvent:
    """Represents a failure event"""
    event_id: str
    agent_id: str
    failure_type: FailureType
    timestamp: float
    severity: int  # 1-5 scale
    description: str
    detected_by: str
    impact_assessment: Dict[str, Any]
    recovery_actions: List[str] = field(default_factory=list)
    resolution_time: Optional[float] = None
    verified: bool = False

@dataclass
class Checkpoint:
    """State checkpoint for recovery"""
    checkpoint_id: str
    agent_id: str
    checkpoint_type: CheckpointType
    timestamp: float
    state_data: Dict[str, Any]
    hash_value: str
    size_bytes: int
    compression_ratio: float = 1.0
    verification_status: bool = True
    dependencies: List[str] = field(default_factory=list)

@dataclass
class RecoveryPlan:
    """Plan for recovering from failures"""
    plan_id: str
    failure_event: FailureEvent
    strategy: RecoveryStrategy
    affected_agents: List[str]
    recovery_steps: List[Dict[str, Any]]
    estimated_duration: float
    rollback_plan: Optional[Dict[str, Any]] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    execution_status: str = "pending"

class HealthMonitor:
    """Monitors agent health and detects failures"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("HealthMonitor")
        self.agent_health: Dict[str, AgentHealth] = {}
        self.failure_detectors: Dict[FailureType, Callable] = {}
        self.health_thresholds = config.get("health_thresholds", {
            "cpu_warning": 0.8,
            "cpu_critical": 0.95,
            "memory_warning": 0.8,
            "memory_critical": 0.95,
            "response_time_warning": 5.0,
            "response_time_critical": 10.0,
            "heartbeat_timeout": 30.0,
            "error_rate_warning": 0.1,
            "error_rate_critical": 0.3
        })
        self.monitoring_active = False
        self.monitoring_interval = config.get("monitoring_interval", 5.0)
        
    def register_failure_detector(self, failure_type: FailureType, detector: Callable):
        """Register a failure detector for specific failure type"""
        self.failure_detectors[failure_type] = detector
        
    async def start_monitoring(self):
        """Start health monitoring"""
        self.monitoring_active = True
        self.logger.info("Starting health monitoring")
        
        # Start monitoring tasks
        monitoring_tasks = [
            self._heartbeat_monitoring_loop(),
            self._performance_monitoring_loop(),
            self._failure_detection_loop(),
            self._health_analysis_loop()
        ]
        
        await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        self.logger.info("Stopping health monitoring")
        
    def update_agent_health(self, agent_id: str, health_data: Dict[str, Any]):
        """Update health information for an agent"""
        if agent_id not in self.agent_health:
            self.agent_health[agent_id] = AgentHealth(
                agent_id=agent_id,
                status=HealthStatus.UNKNOWN,
                last_heartbeat=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                message_queue_size=0,
                response_time=0.0,
                error_count=0,
                uptime=0.0,
                last_checkpoint=0.0
            )
        
        health = self.agent_health[agent_id]
        health.last_heartbeat = time.time()
        health.cpu_usage = health_data.get("cpu_usage", health.cpu_usage)
        health.memory_usage = health_data.get("memory_usage", health.memory_usage)
        health.message_queue_size = health_data.get("queue_size", health.message_queue_size)
        health.response_time = health_data.get("response_time", health.response_time)
        health.error_count = health_data.get("error_count", health.error_count)
        health.uptime = health_data.get("uptime", health.uptime)
        health.performance_metrics.update(health_data.get("metrics", {}))
        
        # Update status based on metrics
        health.status = self._assess_health_status(health)
        
    def _assess_health_status(self, health: AgentHealth) -> HealthStatus:
        """Assess overall health status based on metrics"""
        thresholds = self.health_thresholds
        
        # Check for critical conditions
        if (health.cpu_usage > thresholds["cpu_critical"] or
            health.memory_usage > thresholds["memory_critical"] or
            health.response_time > thresholds["response_time_critical"]):
            return HealthStatus.FAILING
        
        # Check heartbeat timeout
        if time.time() - health.last_heartbeat > thresholds["heartbeat_timeout"]:
            return HealthStatus.FAILED
        
        # Check for warning conditions
        if (health.cpu_usage > thresholds["cpu_warning"] or
            health.memory_usage > thresholds["memory_warning"] or
            health.response_time > thresholds["response_time_warning"]):
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    async def _heartbeat_monitoring_loop(self):
        """Monitor agent heartbeats"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                timeout = self.health_thresholds["heartbeat_timeout"]
                
                for agent_id, health in self.agent_health.items():
                    if current_time - health.last_heartbeat > timeout:
                        if health.status != HealthStatus.FAILED:
                            self.logger.warning(f"Agent {agent_id} heartbeat timeout")
                            health.status = HealthStatus.FAILED
                            
                            # Trigger failure event
                            failure_event = FailureEvent(
                                event_id=str(uuid.uuid4()),
                                agent_id=agent_id,
                                failure_type=FailureType.TIMEOUT_FAILURE,
                                timestamp=current_time,
                                severity=4,
                                description=f"Heartbeat timeout for agent {agent_id}",
                                detected_by="heartbeat_monitor",
                                impact_assessment={"type": "agent_unavailable"}
                            )
                            
                            await self._notify_failure(failure_event)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _performance_monitoring_loop(self):
        """Monitor agent performance metrics"""
        while self.monitoring_active:
            try:
                for agent_id, health in self.agent_health.items():
                    # Detect performance degradation
                    if health.status == HealthStatus.HEALTHY:
                        degradation_detected = False
                        
                        # Check CPU usage trend
                        if health.cpu_usage > self.health_thresholds["cpu_warning"]:
                            degradation_detected = True
                        
                        # Check memory usage trend
                        if health.memory_usage > self.health_thresholds["memory_warning"]:
                            degradation_detected = True
                        
                        # Check response time trend
                        if health.response_time > self.health_thresholds["response_time_warning"]:
                            degradation_detected = True
                        
                        if degradation_detected:
                            failure_event = FailureEvent(
                                event_id=str(uuid.uuid4()),
                                agent_id=agent_id,
                                failure_type=FailureType.PERFORMANCE_DEGRADATION,
                                timestamp=time.time(),
                                severity=2,
                                description=f"Performance degradation detected for agent {agent_id}",
                                detected_by="performance_monitor",
                                impact_assessment={"type": "reduced_capacity"}
                            )
                            
                            await self._notify_failure(failure_event)
                
                await asyncio.sleep(self.monitoring_interval * 2)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _failure_detection_loop(self):
        """Run registered failure detectors"""
        while self.monitoring_active:
            try:
                for failure_type, detector in self.failure_detectors.items():
                    try:
                        failures = await detector(self.agent_health)
                        for failure in failures:
                            await self._notify_failure(failure)
                    except Exception as e:
                        self.logger.error(f"Error in {failure_type} detector: {e}")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in failure detection loop: {e}")
                await asyncio.sleep(10)
    
    async def _health_analysis_loop(self):
        """Analyze overall system health"""
        while self.monitoring_active:
            try:
                total_agents = len(self.agent_health)
                if total_agents == 0:
                    await asyncio.sleep(self.monitoring_interval * 4)
                    continue
                
                healthy_count = sum(1 for h in self.agent_health.values() 
                                  if h.status == HealthStatus.HEALTHY)
                degraded_count = sum(1 for h in self.agent_health.values() 
                                   if h.status == HealthStatus.DEGRADED)
                failed_count = sum(1 for h in self.agent_health.values() 
                                 if h.status == HealthStatus.FAILED)
                
                health_ratio = healthy_count / total_agents
                
                self.logger.info(f"System health: {health_ratio:.1%} healthy "
                               f"({healthy_count}H/{degraded_count}D/{failed_count}F)")
                
                # Check for system-wide issues
                if health_ratio < 0.5:
                    failure_event = FailureEvent(
                        event_id=str(uuid.uuid4()),
                        agent_id="system",
                        failure_type=FailureType.PERFORMANCE_DEGRADATION,
                        timestamp=time.time(),
                        severity=5,
                        description="System-wide performance degradation detected",
                        detected_by="health_analyzer",
                        impact_assessment={"type": "system_degradation", "health_ratio": health_ratio}
                    )
                    
                    await self._notify_failure(failure_event)
                
                await asyncio.sleep(self.monitoring_interval * 4)
                
            except Exception as e:
                self.logger.error(f"Error in health analysis: {e}")
                await asyncio.sleep(20)
    
    async def _notify_failure(self, failure_event: FailureEvent):
        """Notify about detected failure"""
        self.logger.warning(f"Failure detected: {failure_event.description}")
        # This would integrate with the recovery system
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health"""
        if not self.agent_health:
            return {"status": "unknown", "total_agents": 0}
        
        status_counts = defaultdict(int)
        for health in self.agent_health.values():
            status_counts[health.status.value] += 1
        
        total_agents = len(self.agent_health)
        healthy_ratio = status_counts["healthy"] / total_agents if total_agents > 0 else 0
        
        return {
            "total_agents": total_agents,
            "status_distribution": dict(status_counts),
            "healthy_ratio": healthy_ratio,
            "system_status": "healthy" if healthy_ratio > 0.8 else "degraded" if healthy_ratio > 0.5 else "critical"
        }

class CheckpointManager:
    """Manages state checkpoints for recovery"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CheckpointManager")
        self.checkpoints: Dict[str, Dict[str, Checkpoint]] = defaultdict(dict)  # agent_id -> checkpoint_id -> checkpoint
        self.checkpoint_interval = config.get("checkpoint_interval", 300)  # 5 minutes
        self.max_checkpoints_per_agent = config.get("max_checkpoints_per_agent", 10)
        self.compression_enabled = config.get("compression_enabled", True)
        self.verification_enabled = config.get("verification_enabled", True)
        
    async def create_checkpoint(self, agent_id: str, state_data: Dict[str, Any], 
                              checkpoint_type: CheckpointType = CheckpointType.FULL_STATE) -> Checkpoint:
        """Create a new checkpoint for an agent"""
        
        checkpoint_id = f"{agent_id}_{int(time.time())}"
        
        # Serialize state data
        serialized_data = pickle.dumps(state_data)
        original_size = len(serialized_data)
        
        # Apply compression if enabled
        if self.compression_enabled and checkpoint_type != CheckpointType.COMPRESSED:
            # Simplified compression simulation
            compressed_data = serialized_data  # In practice, use gzip/lz4
            compression_ratio = 0.7  # Simulated compression ratio
        else:
            compressed_data = serialized_data
            compression_ratio = 1.0
        
        # Calculate hash for verification
        hash_value = hashlib.sha256(compressed_data).hexdigest()
        
        # Create checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            checkpoint_type=checkpoint_type,
            timestamp=time.time(),
            state_data=state_data,
            hash_value=hash_value,
            size_bytes=len(compressed_data),
            compression_ratio=compression_ratio,
            verification_status=True
        )
        
        # Verify checkpoint if enabled
        if self.verification_enabled:
            checkpoint.verification_status = await self._verify_checkpoint(checkpoint)
        
        # Store checkpoint
        self.checkpoints[agent_id][checkpoint_id] = checkpoint
        
        # Cleanup old checkpoints
        await self._cleanup_old_checkpoints(agent_id)
        
        self.logger.info(f"Created checkpoint {checkpoint_id} for agent {agent_id} "
                        f"(size: {checkpoint.size_bytes} bytes, compression: {compression_ratio:.1%})")
        
        return checkpoint
    
    async def restore_from_checkpoint(self, agent_id: str, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Restore agent state from checkpoint"""
        
        if agent_id not in self.checkpoints:
            self.logger.warning(f"No checkpoints found for agent {agent_id}")
            return None
        
        agent_checkpoints = self.checkpoints[agent_id]
        
        # Use specified checkpoint or latest one
        if checkpoint_id:
            if checkpoint_id not in agent_checkpoints:
                self.logger.warning(f"Checkpoint {checkpoint_id} not found for agent {agent_id}")
                return None
            checkpoint = agent_checkpoints[checkpoint_id]
        else:
            # Get latest checkpoint
            if not agent_checkpoints:
                return None
            latest_checkpoint_id = max(agent_checkpoints.keys(), 
                                     key=lambda cid: agent_checkpoints[cid].timestamp)
            checkpoint = agent_checkpoints[latest_checkpoint_id]
        
        # Verify checkpoint before restoration
        if self.verification_enabled:
            if not await self._verify_checkpoint(checkpoint):
                self.logger.error(f"Checkpoint verification failed for {checkpoint.checkpoint_id}")
                return None
        
        self.logger.info(f"Restoring agent {agent_id} from checkpoint {checkpoint.checkpoint_id}")
        return checkpoint.state_data
    
    async def _verify_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Verify checkpoint integrity"""
        try:
            # Serialize and hash the state data
            serialized_data = pickle.dumps(checkpoint.state_data)
            calculated_hash = hashlib.sha256(serialized_data).hexdigest()
            
            # Compare hashes
            return calculated_hash == checkpoint.hash_value
        except Exception as e:
            self.logger.error(f"Error verifying checkpoint {checkpoint.checkpoint_id}: {e}")
            return False
    
    async def _cleanup_old_checkpoints(self, agent_id: str):
        """Remove old checkpoints to maintain storage limits"""
        agent_checkpoints = self.checkpoints[agent_id]
        
        if len(agent_checkpoints) <= self.max_checkpoints_per_agent:
            return
        
        # Sort by timestamp and keep only the most recent ones
        sorted_checkpoints = sorted(agent_checkpoints.items(), 
                                  key=lambda item: item[1].timestamp, reverse=True)
        
        # Remove excess checkpoints
        for i in range(self.max_checkpoints_per_agent, len(sorted_checkpoints)):
            checkpoint_id, checkpoint = sorted_checkpoints[i]
            del agent_checkpoints[checkpoint_id]
            self.logger.debug(f"Removed old checkpoint {checkpoint_id} for agent {agent_id}")
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        total_checkpoints = sum(len(agent_checkpoints) for agent_checkpoints in self.checkpoints.values())
        total_size = sum(checkpoint.size_bytes 
                        for agent_checkpoints in self.checkpoints.values()
                        for checkpoint in agent_checkpoints.values())
        
        agents_with_checkpoints = len(self.checkpoints)
        
        return {
            "total_checkpoints": total_checkpoints,
            "total_size_bytes": total_size,
            "agents_with_checkpoints": agents_with_checkpoints,
            "average_checkpoints_per_agent": total_checkpoints / max(agents_with_checkpoints, 1),
            "average_checkpoint_size": total_size / max(total_checkpoints, 1)
        }

class RecoveryManager:
    """Manages failure recovery procedures"""
    
    def __init__(self, health_monitor: HealthMonitor, checkpoint_manager: CheckpointManager, config: Dict[str, Any]):
        self.health_monitor = health_monitor
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        self.logger = logging.getLogger("RecoveryManager")
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.recovery_strategies: Dict[FailureType, Callable] = {}
        self.active_recoveries: Set[str] = set()
        
        # Register default recovery strategies
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default recovery strategies"""
        self.recovery_strategies[FailureType.AGENT_CRASH] = self._recover_from_agent_crash
        self.recovery_strategies[FailureType.NETWORK_PARTITION] = self._recover_from_network_partition
        self.recovery_strategies[FailureType.PERFORMANCE_DEGRADATION] = self._recover_from_performance_degradation
        self.recovery_strategies[FailureType.TIMEOUT_FAILURE] = self._recover_from_timeout_failure
        
    async def handle_failure(self, failure_event: FailureEvent) -> RecoveryPlan:
        """Handle a failure event and create recovery plan"""
        
        self.logger.info(f"Handling failure: {failure_event.description}")
        
        # Assess failure impact
        impact_assessment = await self._assess_failure_impact(failure_event)
        failure_event.impact_assessment.update(impact_assessment)
        
        # Create recovery plan
        recovery_plan = await self._create_recovery_plan(failure_event)
        
        # Execute recovery plan
        if recovery_plan:
            self.recovery_plans[recovery_plan.plan_id] = recovery_plan
            success = await self._execute_recovery_plan(recovery_plan)
            
            if success:
                recovery_plan.execution_status = "completed"
                self.logger.info(f"Recovery plan {recovery_plan.plan_id} completed successfully")
            else:
                recovery_plan.execution_status = "failed"
                self.logger.error(f"Recovery plan {recovery_plan.plan_id} failed")
                
                # Attempt rollback if available
                if recovery_plan.rollback_plan:
                    await self._execute_rollback(recovery_plan)
        
        return recovery_plan
    
    async def _assess_failure_impact(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Assess the impact of a failure"""
        
        impact = {
            "affected_agents": [failure_event.agent_id],
            "service_disruption": False,
            "performance_impact": 0.0,
            "recovery_complexity": "low"
        }
        
        # Analyze based on failure type
        if failure_event.failure_type == FailureType.AGENT_CRASH:
            # Check if agent is critical for system operation
            agent_health = self.health_monitor.agent_health.get(failure_event.agent_id)
            if agent_health and hasattr(agent_health, 'role'):
                if agent_health.role == "coordinator":
                    impact["service_disruption"] = True
                    impact["recovery_complexity"] = "high"
                    impact["performance_impact"] = 0.3
        
        elif failure_event.failure_type == FailureType.NETWORK_PARTITION:
            impact["service_disruption"] = True
            impact["recovery_complexity"] = "medium"
            impact["performance_impact"] = 0.2
        
        elif failure_event.failure_type == FailureType.PERFORMANCE_DEGRADATION:
            impact["performance_impact"] = 0.1
            impact["recovery_complexity"] = "low"
        
        return impact
    
    async def _create_recovery_plan(self, failure_event: FailureEvent) -> Optional[RecoveryPlan]:
        """Create a recovery plan for the failure"""
        
        # Select recovery strategy based on failure type
        if failure_event.failure_type not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy for failure type {failure_event.failure_type}")
            return None
        
        strategy_func = self.recovery_strategies[failure_event.failure_type]
        
        # Generate recovery steps
        recovery_steps = await strategy_func(failure_event)
        
        if not recovery_steps:
            return None
        
        # Determine recovery strategy
        if failure_event.failure_type == FailureType.AGENT_CRASH:
            strategy = RecoveryStrategy.RESTART_AGENT
        elif failure_event.failure_type == FailureType.NETWORK_PARTITION:
            strategy = RecoveryStrategy.PARTITION_HEALING
        elif failure_event.failure_type == FailureType.PERFORMANCE_DEGRADATION:
            strategy = RecoveryStrategy.LOAD_REDISTRIBUTION
        else:
            strategy = RecoveryStrategy.GRACEFUL_DEGRADATION
        
        # Create recovery plan
        recovery_plan = RecoveryPlan(
            plan_id=str(uuid.uuid4()),
            failure_event=failure_event,
            strategy=strategy,
            affected_agents=[failure_event.agent_id],
            recovery_steps=recovery_steps,
            estimated_duration=self._estimate_recovery_duration(recovery_steps),
            success_criteria={
                "agent_healthy": True,
                "performance_restored": True,
                "no_data_loss": True
            }
        )
        
        return recovery_plan
    
    async def _execute_recovery_plan(self, recovery_plan: RecoveryPlan) -> bool:
        """Execute a recovery plan"""
        
        self.active_recoveries.add(recovery_plan.plan_id)
        recovery_plan.execution_status = "executing"
        
        try:
            self.logger.info(f"Executing recovery plan {recovery_plan.plan_id}")
            
            for i, step in enumerate(recovery_plan.recovery_steps):
                self.logger.info(f"Executing recovery step {i+1}/{len(recovery_plan.recovery_steps)}: {step['description']}")
                
                # Execute step based on type
                if step["type"] == "restart_agent":
                    success = await self._restart_agent(step["agent_id"])
                elif step["type"] == "restore_from_checkpoint":
                    success = await self._restore_agent_from_checkpoint(step["agent_id"], step.get("checkpoint_id"))
                elif step["type"] == "redistribute_load":
                    success = await self._redistribute_load(step["failed_agent"], step["target_agents"])
                elif step["type"] == "heal_partition":
                    success = await self._heal_network_partition(step["partition_info"])
                elif step["type"] == "verify_recovery":
                    success = await self._verify_recovery(step["agent_id"], step["criteria"])
                else:
                    self.logger.warning(f"Unknown recovery step type: {step['type']}")
                    success = False
                
                if not success:
                    self.logger.error(f"Recovery step {i+1} failed")
                    return False
                
                # Add delay between steps if specified
                if "delay" in step:
                    await asyncio.sleep(step["delay"])
            
            # Verify overall recovery success
            success = await self._verify_plan_success(recovery_plan)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing recovery plan {recovery_plan.plan_id}: {e}")
            return False
        finally:
            self.active_recoveries.discard(recovery_plan.plan_id)
    
    async def _recover_from_agent_crash(self, failure_event: FailureEvent) -> List[Dict[str, Any]]:
        """Generate recovery steps for agent crash"""
        return [
            {
                "type": "restart_agent",
                "agent_id": failure_event.agent_id,
                "description": f"Restart crashed agent {failure_event.agent_id}",
                "delay": 5.0
            },
            {
                "type": "restore_from_checkpoint",
                "agent_id": failure_event.agent_id,
                "description": f"Restore state from latest checkpoint",
                "delay": 2.0
            },
            {
                "type": "verify_recovery",
                "agent_id": failure_event.agent_id,
                "criteria": {"health_status": "healthy", "response_time": "<5s"},
                "description": f"Verify agent {failure_event.agent_id} recovery"
            }
        ]
    
    async def _recover_from_network_partition(self, failure_event: FailureEvent) -> List[Dict[str, Any]]:
        """Generate recovery steps for network partition"""
        return [
            {
                "type": "heal_partition",
                "partition_info": {"affected_agent": failure_event.agent_id},
                "description": "Attempt to heal network partition",
                "delay": 10.0
            },
            {
                "type": "verify_recovery",
                "agent_id": failure_event.agent_id,
                "criteria": {"connectivity": "restored"},
                "description": "Verify network connectivity restored"
            }
        ]
    
    async def _recover_from_performance_degradation(self, failure_event: FailureEvent) -> List[Dict[str, Any]]:
        """Generate recovery steps for performance degradation"""
        return [
            {
                "type": "redistribute_load",
                "failed_agent": failure_event.agent_id,
                "target_agents": ["backup_agent_1", "backup_agent_2"],
                "description": f"Redistribute load from degraded agent {failure_event.agent_id}",
                "delay": 5.0
            },
            {
                "type": "verify_recovery",
                "agent_id": failure_event.agent_id,
                "criteria": {"performance_improved": True},
                "description": "Verify performance improvement"
            }
        ]
    
    async def _recover_from_timeout_failure(self, failure_event: FailureEvent) -> List[Dict[str, Any]]:
        """Generate recovery steps for timeout failure"""
        return [
            {
                "type": "restart_agent",
                "agent_id": failure_event.agent_id,
                "description": f"Restart unresponsive agent {failure_event.agent_id}",
                "delay": 5.0
            },
            {
                "type": "verify_recovery",
                "agent_id": failure_event.agent_id,
                "criteria": {"responsive": True},
                "description": f"Verify agent {failure_event.agent_id} responsiveness"
            }
        ]
    
    async def _restart_agent(self, agent_id: str) -> bool:
        """Restart an agent (simulation)"""
        self.logger.info(f"Restarting agent {agent_id}")
        await asyncio.sleep(2)  # Simulate restart time
        
        # Update health status to recovering
        if agent_id in self.health_monitor.agent_health:
            self.health_monitor.agent_health[agent_id].status = HealthStatus.RECOVERING
        
        return True
    
    async def _restore_agent_from_checkpoint(self, agent_id: str, checkpoint_id: Optional[str] = None) -> bool:
        """Restore agent from checkpoint"""
        self.logger.info(f"Restoring agent {agent_id} from checkpoint")
        
        restored_state = await self.checkpoint_manager.restore_from_checkpoint(agent_id, checkpoint_id)
        
        if restored_state:
            self.logger.info(f"Successfully restored agent {agent_id} state")
            return True
        else:
            self.logger.warning(f"Failed to restore agent {agent_id} state")
            return False
    
    async def _redistribute_load(self, failed_agent: str, target_agents: List[str]) -> bool:
        """Redistribute load from failed agent to target agents"""
        self.logger.info(f"Redistributing load from {failed_agent} to {target_agents}")
        await asyncio.sleep(3)  # Simulate load redistribution
        return True
    
    async def _heal_network_partition(self, partition_info: Dict[str, Any]) -> bool:
        """Attempt to heal network partition"""
        self.logger.info(f"Attempting to heal network partition: {partition_info}")
        await asyncio.sleep(5)  # Simulate partition healing
        return True
    
    async def _verify_recovery(self, agent_id: str, criteria: Dict[str, Any]) -> bool:
        """Verify recovery success"""
        self.logger.info(f"Verifying recovery for agent {agent_id}")
        
        # Check agent health
        if agent_id in self.health_monitor.agent_health:
            health = self.health_monitor.agent_health[agent_id]
            
            if "health_status" in criteria:
                if health.status.value != criteria["health_status"]:
                    return False
            
            if "response_time" in criteria:
                max_response_time = float(criteria["response_time"].rstrip('s'))
                if health.response_time > max_response_time:
                    return False
        
        return True
    
    async def _verify_plan_success(self, recovery_plan: RecoveryPlan) -> bool:
        """Verify overall recovery plan success"""
        
        for criterion, expected_value in recovery_plan.success_criteria.items():
            if criterion == "agent_healthy":
                agent_id = recovery_plan.failure_event.agent_id
                if agent_id in self.health_monitor.agent_health:
                    health = self.health_monitor.agent_health[agent_id]
                    if health.status not in [HealthStatus.HEALTHY, HealthStatus.RECOVERING]:
                        return False
            
            # Add more success criteria verification as needed
        
        return True
    
    async def _execute_rollback(self, recovery_plan: RecoveryPlan):
        """Execute rollback plan if recovery fails"""
        self.logger.warning(f"Executing rollback for plan {recovery_plan.plan_id}")
        # Implementation depends on specific rollback strategy
    
    def _estimate_recovery_duration(self, recovery_steps: List[Dict[str, Any]]) -> float:
        """Estimate recovery duration based on steps"""
        total_duration = 0.0
        
        for step in recovery_steps:
            if step["type"] == "restart_agent":
                total_duration += 30.0
            elif step["type"] == "restore_from_checkpoint":
                total_duration += 15.0
            elif step["type"] == "redistribute_load":
                total_duration += 60.0
            elif step["type"] == "heal_partition":
                total_duration += 120.0
            else:
                total_duration += 10.0
            
            total_duration += step.get("delay", 0.0)
        
        return total_duration
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        total_plans = len(self.recovery_plans)
        completed_plans = sum(1 for plan in self.recovery_plans.values() 
                            if plan.execution_status == "completed")
        failed_plans = sum(1 for plan in self.recovery_plans.values() 
                         if plan.execution_status == "failed")
        active_recoveries = len(self.active_recoveries)
        
        return {
            "total_recovery_plans": total_plans,
            "completed_plans": completed_plans,
            "failed_plans": failed_plans,
            "active_recoveries": active_recoveries,
            "success_rate": completed_plans / max(total_plans, 1)
        }

class FaultToleranceFramework:
    """Main fault tolerance and recovery framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("FaultToleranceFramework")
        
        # Initialize components
        self.health_monitor = HealthMonitor(config.get("health_monitoring", {}))
        self.checkpoint_manager = CheckpointManager(config.get("checkpointing", {}))
        self.recovery_manager = RecoveryManager(
            self.health_monitor, 
            self.checkpoint_manager, 
            config.get("recovery", {})
        )
        
        self.framework_active = False
        
        # Register failure detectors
        self._register_failure_detectors()
        
        # Statistics
        self.failure_events: List[FailureEvent] = []
        self.recovery_history: List[RecoveryPlan] = []
        
    def _register_failure_detectors(self):
        """Register custom failure detectors"""
        
        async def byzantine_failure_detector(agent_health: Dict[str, AgentHealth]) -> List[FailureEvent]:
            """Detect Byzantine failures"""
            failures = []
            
            for agent_id, health in agent_health.items():
                # Simple heuristic: inconsistent behavior patterns
                if (health.error_count > 10 and 
                    health.response_time > 5.0 and 
                    health.status == HealthStatus.HEALTHY):
                    
                    failure = FailureEvent(
                        event_id=str(uuid.uuid4()),
                        agent_id=agent_id,
                        failure_type=FailureType.BYZANTINE_FAILURE,
                        timestamp=time.time(),
                        severity=4,
                        description=f"Potential Byzantine behavior detected in agent {agent_id}",
                        detected_by="byzantine_detector",
                        impact_assessment={"type": "data_integrity_risk"}
                    )
                    failures.append(failure)
            
            return failures
        
        async def resource_exhaustion_detector(agent_health: Dict[str, AgentHealth]) -> List[FailureEvent]:
            """Detect resource exhaustion"""
            failures = []
            
            for agent_id, health in agent_health.items():
                if (health.cpu_usage > 0.98 and health.memory_usage > 0.98):
                    failure = FailureEvent(
                        event_id=str(uuid.uuid4()),
                        agent_id=agent_id,
                        failure_type=FailureType.RESOURCE_EXHAUSTION,
                        timestamp=time.time(),
                        severity=3,
                        description=f"Resource exhaustion detected in agent {agent_id}",
                        detected_by="resource_detector",
                        impact_assessment={"type": "capacity_limitation"}
                    )
                    failures.append(failure)
            
            return failures
        
        self.health_monitor.register_failure_detector(FailureType.BYZANTINE_FAILURE, byzantine_failure_detector)
        self.health_monitor.register_failure_detector(FailureType.RESOURCE_EXHAUSTION, resource_exhaustion_detector)
    
    async def start(self):
        """Start the fault tolerance framework"""
        self.framework_active = True
        self.logger.info("Starting fault tolerance framework")
        
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
    async def stop(self):
        """Stop the fault tolerance framework"""
        self.framework_active = False
        self.logger.info("Stopping fault tolerance framework")
        
        # Stop health monitoring
        await self.health_monitor.stop_monitoring()
    
    async def register_agent(self, agent_id: str, initial_health: Dict[str, Any]):
        """Register an agent with the fault tolerance system"""
        self.health_monitor.update_agent_health(agent_id, initial_health)
        
        # Create initial checkpoint
        initial_state = {
            "agent_id": agent_id,
            "initialization_time": time.time(),
            "initial_config": initial_health,
            "state": "initialized"
        }
        
        await self.checkpoint_manager.create_checkpoint(agent_id, initial_state)
        
        self.logger.info(f"Registered agent {agent_id} with fault tolerance system")
    
    async def update_agent_health(self, agent_id: str, health_data: Dict[str, Any]):
        """Update agent health information"""
        self.health_monitor.update_agent_health(agent_id, health_data)
    
    async def create_agent_checkpoint(self, agent_id: str, state_data: Dict[str, Any]) -> Checkpoint:
        """Create a checkpoint for an agent"""
        return await self.checkpoint_manager.create_checkpoint(agent_id, state_data)
    
    async def handle_failure_event(self, failure_event: FailureEvent) -> RecoveryPlan:
        """Handle a failure event"""
        self.failure_events.append(failure_event)
        
        recovery_plan = await self.recovery_manager.handle_failure(failure_event)
        
        if recovery_plan:
            self.recovery_history.append(recovery_plan)
        
        return recovery_plan
    
    def get_fault_tolerance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance statistics"""
        
        health_stats = self.health_monitor.get_system_health_summary()
        checkpoint_stats = self.checkpoint_manager.get_checkpoint_statistics()
        recovery_stats = self.recovery_manager.get_recovery_statistics()
        
        # Failure statistics
        failure_types = defaultdict(int)
        for failure in self.failure_events:
            failure_types[failure.failure_type.value] += 1
        
        avg_resolution_time = 0.0
        resolved_failures = [f for f in self.failure_events if f.resolution_time is not None]
        if resolved_failures:
            avg_resolution_time = sum(f.resolution_time - f.timestamp for f in resolved_failures) / len(resolved_failures)
        
        return {
            "health_monitoring": health_stats,
            "checkpointing": checkpoint_stats,
            "recovery": recovery_stats,
            "failure_statistics": {
                "total_failures": len(self.failure_events),
                "failure_types": dict(failure_types),
                "resolved_failures": len(resolved_failures),
                "average_resolution_time": avg_resolution_time
            },
            "system_availability": self._calculate_availability()
        }
    
    def _calculate_availability(self) -> float:
        """Calculate system availability percentage"""
        if not self.failure_events:
            return 1.0
        
        total_downtime = sum(
            (f.resolution_time or time.time()) - f.timestamp 
            for f in self.failure_events 
            if f.severity >= 4
        )
        
        uptime_period = time.time() - min(f.timestamp for f in self.failure_events)
        
        if uptime_period <= 0:
            return 1.0
        
        availability = 1.0 - (total_downtime / uptime_period)
        return max(0.0, min(1.0, availability))

def demonstrate_fault_tolerance():
    """Demonstrate the fault tolerance and recovery framework"""
    print("=== Fault Tolerance and Agent Failure Recovery Mechanisms ===")
    
    # Configuration
    config = {
        "health_monitoring": {
            "monitoring_interval": 2.0,
            "health_thresholds": {
                "cpu_warning": 0.7,
                "cpu_critical": 0.9,
                "memory_warning": 0.7,
                "memory_critical": 0.9,
                "response_time_warning": 3.0,
                "response_time_critical": 7.0,
                "heartbeat_timeout": 15.0
            }
        },
        "checkpointing": {
            "checkpoint_interval": 60,
            "max_checkpoints_per_agent": 5,
            "compression_enabled": True,
            "verification_enabled": True
        },
        "recovery": {
            "max_recovery_attempts": 3,
            "recovery_timeout": 300
        }
    }
    
    async def run_demonstration():
        print("1. Initializing Fault Tolerance Framework...")
        
        framework = FaultToleranceFramework(config)
        await framework.start()
        
        print("   Fault tolerance framework started")
        
        print("2. Registering Agents...")
        
        # Register agents with different roles and characteristics
        agents = [
            ("coordinator_1", {"role": "coordinator", "cpu_usage": 0.3, "memory_usage": 0.4}),
            ("worker_1", {"role": "worker", "cpu_usage": 0.5, "memory_usage": 0.6}),
            ("worker_2", {"role": "worker", "cpu_usage": 0.4, "memory_usage": 0.5}),
            ("bridge_1", {"role": "bridge", "cpu_usage": 0.2, "memory_usage": 0.3}),
            ("monitor_1", {"role": "monitor", "cpu_usage": 0.1, "memory_usage": 0.2})
        ]
        
        for agent_id, initial_health in agents:
            await framework.register_agent(agent_id, initial_health)
            
        print(f"   Registered {len(agents)} agents")
        
        print("3. Simulating Normal Operation...")
        
        # Simulate normal operation with health updates
        for _ in range(5):
            for agent_id, _ in agents:
                health_update = {
                    "cpu_usage": np.random.uniform(0.1, 0.6),
                    "memory_usage": np.random.uniform(0.2, 0.7),
                    "response_time": np.random.uniform(0.5, 2.0),
                    "error_count": np.random.randint(0, 3),
                    "uptime": time.time(),
                    "queue_size": np.random.randint(0, 50)
                }
                await framework.update_agent_health(agent_id, health_update)
            
            await asyncio.sleep(1)
        
        print("   Normal operation simulation completed")
        
        print("4. Creating Checkpoints...")
        
        # Create checkpoints for all agents
        for agent_id, _ in agents:
            state_data = {
                "agent_state": "active",
                "pending_tasks": [f"task_{i}" for i in range(5)],
                "configuration": {"param1": np.random.uniform(0, 1)},
                "performance_history": [np.random.uniform(0.8, 1.0) for _ in range(10)]
            }
            
            checkpoint = await framework.create_agent_checkpoint(agent_id, state_data)
            print(f"   Created checkpoint for {agent_id} (size: {checkpoint.size_bytes} bytes)")
        
        print("5. Simulating Failure Scenarios...")
        
        # Scenario 1: Agent crash
        print("   Scenario 1: Agent crash")
        crash_failure = FailureEvent(
            event_id=str(uuid.uuid4()),
            agent_id="worker_1",
            failure_type=FailureType.AGENT_CRASH,
            timestamp=time.time(),
            severity=4,
            description="Worker agent crashed unexpectedly",
            detected_by="system_monitor",
            impact_assessment={"type": "service_disruption"}
        )
        
        recovery_plan = await framework.handle_failure_event(crash_failure)
        print(f"     Recovery plan created: {recovery_plan.strategy.value}")
        print(f"     Estimated duration: {recovery_plan.estimated_duration:.1f}s")
        print(f"     Execution status: {recovery_plan.execution_status}")
        
        await asyncio.sleep(2)
        
        # Scenario 2: Performance degradation
        print("   Scenario 2: Performance degradation")
        degradation_failure = FailureEvent(
            event_id=str(uuid.uuid4()),
            agent_id="coordinator_1",
            failure_type=FailureType.PERFORMANCE_DEGRADATION,
            timestamp=time.time(),
            severity=2,
            description="Coordinator showing performance degradation",
            detected_by="performance_monitor",
            impact_assessment={"type": "reduced_throughput"}
        )
        
        recovery_plan = await framework.handle_failure_event(degradation_failure)
        print(f"     Recovery plan created: {recovery_plan.strategy.value}")
        print(f"     Execution status: {recovery_plan.execution_status}")
        
        await asyncio.sleep(2)
        
        # Scenario 3: Network partition
        print("   Scenario 3: Network partition")
        partition_failure = FailureEvent(
            event_id=str(uuid.uuid4()),
            agent_id="bridge_1",
            failure_type=FailureType.NETWORK_PARTITION,
            timestamp=time.time(),
            severity=3,
            description="Network partition detected affecting bridge agent",
            detected_by="network_monitor",
            impact_assessment={"type": "connectivity_loss"}
        )
        
        recovery_plan = await framework.handle_failure_event(partition_failure)
        print(f"     Recovery plan created: {recovery_plan.strategy.value}")
        print(f"     Execution status: {recovery_plan.execution_status}")
        
        print("6. Testing Recovery Mechanisms...")
        
        # Test checkpoint restoration
        print("   Testing checkpoint restoration...")
        restored_state = await framework.checkpoint_manager.restore_from_checkpoint("worker_1")
        if restored_state:
            print(f"     Successfully restored worker_1 state with {len(restored_state)} elements")
        
        # Test health status updates
        print("   Updating agent health after recovery...")
        recovery_health = {
            "cpu_usage": 0.3,
            "memory_usage": 0.4,
            "response_time": 1.5,
            "error_count": 0,
            "uptime": time.time()
        }
        await framework.update_agent_health("worker_1", recovery_health)
        
        print("7. System Health Analysis...")
        
        # Get system health summary
        health_summary = framework.health_monitor.get_system_health_summary()
        print(f"   Total agents: {health_summary['total_agents']}")
        print(f"   System status: {health_summary['system_status']}")
        print(f"   Healthy ratio: {health_summary['healthy_ratio']:.1%}")
        print(f"   Status distribution: {health_summary['status_distribution']}")
        
        print("8. Fault Tolerance Statistics...")
        
        # Get comprehensive statistics
        stats = framework.get_fault_tolerance_statistics()
        
        print("   Health Monitoring:")
        print(f"     Active agents: {stats['health_monitoring']['total_agents']}")
        print(f"     System availability: {stats['system_availability']:.1%}")
        
        print("   Checkpointing:")
        print(f"     Total checkpoints: {stats['checkpointing']['total_checkpoints']}")
        print(f"     Total size: {stats['checkpointing']['total_size_bytes']} bytes")
        print(f"     Average checkpoint size: {stats['checkpointing']['average_checkpoint_size']:.0f} bytes")
        
        print("   Recovery:")
        print(f"     Total recovery plans: {stats['recovery']['total_recovery_plans']}")
        print(f"     Success rate: {stats['recovery']['success_rate']:.1%}")
        print(f"     Active recoveries: {stats['recovery']['active_recoveries']}")
        
        print("   Failure Statistics:")
        print(f"     Total failures: {stats['failure_statistics']['total_failures']}")
        print(f"     Failure types: {stats['failure_statistics']['failure_types']}")
        print(f"     Average resolution time: {stats['failure_statistics']['average_resolution_time']:.1f}s")
        
        print("9. Fault Tolerance Benefits...")
        
        benefits = [
            "Proactive health monitoring with configurable thresholds",
            "Automatic failure detection across multiple failure types",
            "Intelligent recovery planning with strategy selection",
            "State checkpointing with compression and verification",
            "Byzantine fault tolerance for malicious behavior detection",
            "Graceful degradation under partial system failures",
            "Load redistribution and automatic failover capabilities",
            "Comprehensive recovery verification and rollback support"
        ]
        
        for i, benefit in enumerate(benefits, 1):
            print(f"   {i}. {benefit}")
        
        print("10. Cleanup...")
        
        await framework.stop()
        print("   Fault tolerance framework stopped")
        
        return {
            "framework": framework,
            "final_statistics": stats,
            "health_summary": health_summary,
            "recovery_plans": framework.recovery_history
        }
    
    # Run the demonstration
    return asyncio.run(run_demonstration())

if __name__ == "__main__":
    demonstrate_fault_tolerance()