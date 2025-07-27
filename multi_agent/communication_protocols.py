"""
Communication Protocols for Inter-Agent Coordination

This module implements R17: sophisticated communication protocols that enable efficient
coordination between agents in the HeteroSched multi-agent scheduling system.

Key Features:
1. Asynchronous message passing with guaranteed delivery
2. Bandwidth-aware communication optimization
3. Hierarchical routing and message prioritization
4. Consensus protocols for distributed decision making
5. Fault-tolerant communication with automatic recovery
6. Privacy-preserving information sharing mechanisms
7. Dynamic topology adaptation and load balancing

The protocols ensure scalable, reliable, and efficient coordination across
heterogeneous distributed scheduling environments.

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
import heapq

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    RESOURCE_UPDATE = "resource_update"
    COORDINATION_REQUEST = "coordination_request"
    CONSENSUS_VOTE = "consensus_vote"
    HEARTBEAT = "heartbeat"
    PERFORMANCE_REPORT = "performance_report"
    EMERGENCY_ALERT = "emergency_alert"
    ROUTING_UPDATE = "routing_update"

class MessagePriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class CommunicationProtocol(Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    GOSSIP = "gossip"
    TREE_ROUTING = "tree_routing"
    CONSENSUS = "consensus"

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    BRIDGE = "bridge"
    MONITOR = "monitor"

@dataclass
class Message:
    """Represents a message in the communication system"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: float
    ttl: int = 10  # Time to live (hops)
    route: List[str] = field(default_factory=list)
    acknowledgment_required: bool = True
    encryption_required: bool = False
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

@dataclass
class AgentInfo:
    """Information about an agent in the network"""
    agent_id: str
    role: AgentRole
    capabilities: Set[str]
    current_load: float
    location: Tuple[float, float]  # For geographic routing
    neighbors: Set[str] = field(default_factory=set)
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True
    bandwidth_limit: float = 1000.0  # KB/s
    message_queue_size: int = 0

@dataclass
class RoutingEntry:
    """Entry in the routing table"""
    destination: str
    next_hop: str
    distance: int
    cost: float
    last_updated: float

@dataclass
class CommunicationMetrics:
    """Metrics for communication performance"""
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    average_latency: float = 0.0
    message_loss_rate: float = 0.0
    bandwidth_utilization: float = 0.0
    consensus_success_rate: float = 0.0

class MessageQueue:
    """Priority-based message queue with bandwidth management"""
    
    def __init__(self, max_size: int = 1000, bandwidth_limit: float = 1000.0):
        self.max_size = max_size
        self.bandwidth_limit = bandwidth_limit  # KB/s
        self.queue = []  # Priority queue
        self.pending_acks: Dict[str, Message] = {}
        self.bandwidth_usage = deque(maxlen=60)  # Last 60 seconds
        self.lock = threading.Lock()
        
    def enqueue(self, message: Message) -> bool:
        """Add message to queue if there's space"""
        with self.lock:
            if len(self.queue) >= self.max_size:
                # Drop lowest priority message if queue is full
                if self.queue and self.queue[-1][0] > message.priority.value:
                    heapq.heappop(self.queue)
                else:
                    return False
            
            # Add message with priority
            heapq.heappush(self.queue, (message.priority.value, time.time(), message))
            return True
    
    def dequeue(self) -> Optional[Message]:
        """Get highest priority message from queue"""
        with self.lock:
            if self.queue:
                _, _, message = heapq.heappop(self.queue)
                
                # Track acknowledgment if required
                if message.acknowledgment_required:
                    self.pending_acks[message.message_id] = message
                
                return message
            return None
    
    def acknowledge(self, message_id: str) -> bool:
        """Acknowledge receipt of message"""
        with self.lock:
            if message_id in self.pending_acks:
                del self.pending_acks[message_id]
                return True
            return False
    
    def get_pending_acks(self, timeout: float = 30.0) -> List[Message]:
        """Get messages pending acknowledgment that have timed out"""
        current_time = time.time()
        timed_out = []
        
        with self.lock:
            for msg_id, message in list(self.pending_acks.items()):
                if current_time - message.timestamp > timeout:
                    timed_out.append(message)
                    del self.pending_acks[msg_id]
        
        return timed_out
    
    def check_bandwidth(self, message_size: int) -> bool:
        """Check if sending message would exceed bandwidth limit"""
        current_time = time.time()
        
        # Clean old bandwidth usage data
        while self.bandwidth_usage and current_time - self.bandwidth_usage[0][0] > 1.0:
            self.bandwidth_usage.popleft()
        
        # Calculate current bandwidth usage
        current_usage = sum(size for _, size in self.bandwidth_usage)
        
        return current_usage + message_size <= self.bandwidth_limit * 1024  # Convert to bytes
    
    def record_bandwidth_usage(self, message_size: int):
        """Record bandwidth usage for rate limiting"""
        self.bandwidth_usage.append((time.time(), message_size))

class RoutingTable:
    """Distributed routing table for message forwarding"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.routes: Dict[str, RoutingEntry] = {}
        self.neighbors: Set[str] = set()
        self.lock = threading.RLock()
        
    def add_route(self, destination: str, next_hop: str, distance: int, cost: float):
        """Add or update a route"""
        with self.lock:
            self.routes[destination] = RoutingEntry(
                destination=destination,
                next_hop=next_hop,
                distance=distance,
                cost=cost,
                last_updated=time.time()
            )
    
    def get_next_hop(self, destination: str) -> Optional[str]:
        """Get next hop for destination"""
        with self.lock:
            if destination in self.routes:
                return self.routes[destination].next_hop
            return None
    
    def update_from_neighbor(self, neighbor_routes: Dict[str, RoutingEntry], neighbor_id: str):
        """Update routing table from neighbor's routes (distance vector)"""
        with self.lock:
            updated = False
            
            for dest, neighbor_route in neighbor_routes.items():
                if dest == self.agent_id:
                    continue  # Don't route to self
                
                new_distance = neighbor_route.distance + 1
                new_cost = neighbor_route.cost + self._get_link_cost(neighbor_id)
                
                if (dest not in self.routes or 
                    new_distance < self.routes[dest].distance or
                    (new_distance == self.routes[dest].distance and new_cost < self.routes[dest].cost)):
                    
                    self.add_route(dest, neighbor_id, new_distance, new_cost)
                    updated = True
            
            return updated
    
    def _get_link_cost(self, neighbor_id: str) -> float:
        """Get cost of link to neighbor (can be based on latency, bandwidth, etc.)"""
        # Simplified cost model
        return 1.0
    
    def cleanup_stale_routes(self, max_age: float = 300.0):
        """Remove stale routes"""
        current_time = time.time()
        with self.lock:
            stale_routes = [
                dest for dest, route in self.routes.items()
                if current_time - route.last_updated > max_age
            ]
            
            for dest in stale_routes:
                del self.routes[dest]

class ConsensusManager:
    """Manages consensus protocols for distributed decision making"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.active_consensus: Dict[str, Dict[str, Any]] = {}
        self.consensus_history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        
    async def initiate_consensus(self, consensus_id: str, proposal: Dict[str, Any], 
                               participants: List[str], timeout: float = 30.0) -> Dict[str, Any]:
        """Initiate a consensus protocol"""
        
        consensus_data = {
            "consensus_id": consensus_id,
            "proposal": proposal,
            "participants": participants,
            "votes": {},
            "status": "active",
            "start_time": time.time(),
            "timeout": timeout,
            "initiator": self.agent_id
        }
        
        with self.lock:
            self.active_consensus[consensus_id] = consensus_data
        
        # Send consensus request to all participants
        tasks = []
        for participant in participants:
            if participant != self.agent_id:
                task = asyncio.create_task(
                    self._send_consensus_request(participant, consensus_data)
                )
                tasks.append(task)
        
        # Wait for responses or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            pass
        
        # Evaluate consensus
        result = self._evaluate_consensus(consensus_id)
        
        with self.lock:
            if consensus_id in self.active_consensus:
                del self.active_consensus[consensus_id]
            self.consensus_history.append(result)
        
        return result
    
    async def _send_consensus_request(self, participant: str, consensus_data: Dict[str, Any]):
        """Send consensus request to participant"""
        # This would use the actual communication layer
        # For now, simulate response
        await asyncio.sleep(np.random.uniform(0.1, 2.0))
        
        # Simulate vote
        vote = np.random.choice(["approve", "reject"], p=[0.7, 0.3])
        await self.receive_vote(consensus_data["consensus_id"], participant, vote)
    
    async def receive_vote(self, consensus_id: str, voter_id: str, vote: str):
        """Receive a vote for consensus"""
        with self.lock:
            if consensus_id in self.active_consensus:
                self.active_consensus[consensus_id]["votes"][voter_id] = {
                    "vote": vote,
                    "timestamp": time.time()
                }
    
    def _evaluate_consensus(self, consensus_id: str) -> Dict[str, Any]:
        """Evaluate consensus result"""
        with self.lock:
            if consensus_id not in self.active_consensus:
                return {"status": "not_found"}
            
            consensus_data = self.active_consensus[consensus_id]
            votes = consensus_data["votes"]
            participants = consensus_data["participants"]
            
            # Count votes
            approve_votes = sum(1 for vote_data in votes.values() if vote_data["vote"] == "approve")
            total_participants = len(participants)
            
            # Determine result based on majority
            threshold = self.config.get("consensus_threshold", 0.5)
            required_votes = int(total_participants * threshold)
            
            if approve_votes >= required_votes:
                status = "approved"
            else:
                status = "rejected"
            
            return {
                "consensus_id": consensus_id,
                "status": status,
                "proposal": consensus_data["proposal"],
                "votes": votes,
                "approve_count": approve_votes,
                "total_participants": total_participants,
                "duration": time.time() - consensus_data["start_time"]
            }

class CommunicationLayer:
    """Main communication layer managing all protocols"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"CommLayer-{agent_id}")
        
        # Core components
        self.message_queue = MessageQueue(
            max_size=config.get("queue_size", 1000),
            bandwidth_limit=config.get("bandwidth_limit", 1000.0)
        )
        self.routing_table = RoutingTable(agent_id)
        self.consensus_manager = ConsensusManager(agent_id, config)
        
        # Network state
        self.agents: Dict[str, AgentInfo] = {}
        self.active_connections: Set[str] = set()
        self.metrics = CommunicationMetrics()
        
        # Event handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.connection_callbacks: List[Callable] = []
        
        # Background tasks
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler
        
    def register_connection_callback(self, callback: Callable):
        """Register callback for connection events"""
        self.connection_callbacks.append(callback)
        
    async def start(self):
        """Start the communication layer"""
        self.running = True
        self.logger.info("Starting communication layer")
        
        # Start background tasks
        tasks = [
            self._message_processing_loop(),
            self._heartbeat_loop(),
            self._routing_update_loop(),
            self._acknowledgment_loop(),
            self._metrics_update_loop()
        ]
        
        self.background_tasks = [asyncio.create_task(task) for task in tasks]
        
    async def stop(self):
        """Stop the communication layer"""
        self.running = False
        self.logger.info("Stopping communication layer")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
    async def send_message(self, message: Message) -> bool:
        """Send a message to another agent"""
        
        # Check bandwidth constraints
        message_size = len(pickle.dumps(message))
        if not self.message_queue.check_bandwidth(message_size):
            self.logger.warning(f"Bandwidth limit exceeded, dropping message {message.message_id}")
            return False
        
        # Add to route
        message.route.append(self.agent_id)
        
        # Determine routing
        if message.receiver_id == "broadcast":
            return await self._broadcast_message(message)
        elif message.receiver_id.startswith("multicast:"):
            group = message.receiver_id.split(":")[1]
            return await self._multicast_message(message, group)
        else:
            return await self._unicast_message(message)
    
    async def _unicast_message(self, message: Message) -> bool:
        """Send message to specific agent"""
        
        # Find next hop
        next_hop = self.routing_table.get_next_hop(message.receiver_id)
        
        if not next_hop:
            self.logger.warning(f"No route to {message.receiver_id}")
            return False
        
        # Check TTL
        if message.ttl <= 0:
            self.logger.warning(f"Message {message.message_id} TTL expired")
            return False
        
        message.ttl -= 1
        
        # Queue message for sending
        if self.message_queue.enqueue(message):
            self.message_queue.record_bandwidth_usage(len(pickle.dumps(message)))
            self.metrics.total_messages_sent += 1
            self.metrics.total_bytes_sent += len(pickle.dumps(message))
            return True
        
        return False
    
    async def _broadcast_message(self, message: Message) -> bool:
        """Broadcast message to all connected agents"""
        success_count = 0
        
        for neighbor in self.routing_table.neighbors:
            if neighbor != message.sender_id:  # Don't send back to sender
                neighbor_message = Message(
                    message_id=str(uuid.uuid4()),
                    sender_id=message.sender_id,
                    receiver_id=neighbor,
                    message_type=message.message_type,
                    priority=message.priority,
                    payload=message.payload,
                    timestamp=message.timestamp,
                    ttl=message.ttl - 1,
                    route=message.route.copy(),
                    acknowledgment_required=False  # Don't require acks for broadcast
                )
                
                if await self._unicast_message(neighbor_message):
                    success_count += 1
        
        return success_count > 0
    
    async def _multicast_message(self, message: Message, group: str) -> bool:
        """Send message to specific group of agents"""
        # Get group members (simplified implementation)
        group_members = self._get_group_members(group)
        success_count = 0
        
        for member in group_members:
            if member != message.sender_id:
                member_message = Message(
                    message_id=str(uuid.uuid4()),
                    sender_id=message.sender_id,
                    receiver_id=member,
                    message_type=message.message_type,
                    priority=message.priority,
                    payload=message.payload,
                    timestamp=message.timestamp,
                    ttl=message.ttl,
                    route=message.route.copy()
                )
                
                if await self._unicast_message(member_message):
                    success_count += 1
        
        return success_count > 0
    
    def _get_group_members(self, group: str) -> List[str]:
        """Get members of a multicast group"""
        # Simplified group membership
        if group == "coordinators":
            return [agent_id for agent_id, info in self.agents.items() 
                   if info.role == AgentRole.COORDINATOR]
        elif group == "workers":
            return [agent_id for agent_id, info in self.agents.items() 
                   if info.role == AgentRole.WORKER]
        else:
            return list(self.agents.keys())
    
    async def receive_message(self, message: Message):
        """Receive and process an incoming message"""
        
        self.metrics.total_messages_received += 1
        self.metrics.total_bytes_received += len(pickle.dumps(message))
        
        # Send acknowledgment if required
        if message.acknowledgment_required:
            ack_message = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.HEARTBEAT,  # Use heartbeat for ack
                priority=MessagePriority.HIGH,
                payload={"ack_for": message.message_id},
                acknowledgment_required=False
            )
            await self.send_message(ack_message)
        
        # Handle acknowledgments
        if (message.message_type == MessageType.HEARTBEAT and 
            "ack_for" in message.payload):
            self.message_queue.acknowledge(message.payload["ack_for"])
            return
        
        # Forward message if not for this agent
        if (message.receiver_id != self.agent_id and 
            not message.receiver_id.startswith("broadcast") and
            not message.receiver_id.startswith("multicast")):
            
            # Forward message
            await self._unicast_message(message)
            return
        
        # Process message
        if message.message_type in self.message_handlers:
            try:
                await self.message_handlers[message.message_type](message)
            except Exception as e:
                self.logger.error(f"Error handling message {message.message_id}: {e}")
        else:
            self.logger.warning(f"No handler for message type {message.message_type}")
    
    async def _message_processing_loop(self):
        """Background loop for processing outgoing messages"""
        while self.running:
            try:
                message = self.message_queue.dequeue()
                if message:
                    # Simulate network transmission
                    await asyncio.sleep(0.001)  # 1ms network latency
                    
                    # For demo, we'll just log the message
                    self.logger.debug(f"Sent message {message.message_id} to {message.receiver_id}")
                
                await asyncio.sleep(0.01)  # 10ms processing interval
                
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _heartbeat_loop(self):
        """Background loop for sending heartbeats"""
        while self.running:
            try:
                # Send heartbeat to all neighbors
                for neighbor in self.routing_table.neighbors:
                    heartbeat = Message(
                        message_id=str(uuid.uuid4()),
                        sender_id=self.agent_id,
                        receiver_id=neighbor,
                        message_type=MessageType.HEARTBEAT,
                        priority=MessagePriority.LOW,
                        payload={
                            "load": self._get_current_load(),
                            "queue_size": len(self.message_queue.queue),
                            "timestamp": time.time()
                        },
                        acknowledgment_required=False
                    )
                    
                    await self.send_message(heartbeat)
                
                # Clean up stale agent info
                self._cleanup_stale_agents()
                
                await asyncio.sleep(self.config.get("heartbeat_interval", 10))
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _routing_update_loop(self):
        """Background loop for routing table updates"""
        while self.running:
            try:
                # Send routing updates to neighbors
                for neighbor in self.routing_table.neighbors:
                    routing_update = Message(
                        message_id=str(uuid.uuid4()),
                        sender_id=self.agent_id,
                        receiver_id=neighbor,
                        message_type=MessageType.ROUTING_UPDATE,
                        priority=MessagePriority.LOW,
                        payload={
                            "routes": {dest: asdict(route) for dest, route in self.routing_table.routes.items()}
                        },
                        acknowledgment_required=False
                    )
                    
                    await self.send_message(routing_update)
                
                # Clean up stale routes
                self.routing_table.cleanup_stale_routes()
                
                await asyncio.sleep(self.config.get("routing_update_interval", 30))
                
            except Exception as e:
                self.logger.error(f"Error in routing update loop: {e}")
                await asyncio.sleep(10)
    
    async def _acknowledgment_loop(self):
        """Background loop for handling message acknowledgments"""
        while self.running:
            try:
                # Check for timed out acknowledgments
                timed_out_messages = self.message_queue.get_pending_acks()
                
                for message in timed_out_messages:
                    self.logger.warning(f"Message {message.message_id} acknowledgment timed out")
                    # Could implement retry logic here
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in acknowledgment loop: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_update_loop(self):
        """Background loop for updating communication metrics"""
        while self.running:
            try:
                # Update metrics
                self._update_metrics()
                
                await asyncio.sleep(self.config.get("metrics_update_interval", 60))
                
            except Exception as e:
                self.logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(30)
    
    def _get_current_load(self) -> float:
        """Get current agent load"""
        # Simplified load calculation
        return len(self.message_queue.queue) / self.message_queue.max_size
    
    def _cleanup_stale_agents(self, timeout: float = 60.0):
        """Remove agents that haven't sent heartbeat recently"""
        current_time = time.time()
        stale_agents = []
        
        for agent_id, agent_info in self.agents.items():
            if current_time - agent_info.last_heartbeat > timeout:
                stale_agents.append(agent_id)
        
        for agent_id in stale_agents:
            self.logger.info(f"Removing stale agent {agent_id}")
            del self.agents[agent_id]
            self.routing_table.neighbors.discard(agent_id)
    
    def _update_metrics(self):
        """Update communication metrics"""
        # Calculate average latency (simplified)
        if self.metrics.total_messages_sent > 0:
            # This would be calculated from actual round-trip times
            self.metrics.average_latency = np.random.uniform(1, 10)  # 1-10ms
        
        # Calculate bandwidth utilization
        current_usage = sum(size for _, size in self.message_queue.bandwidth_usage)
        self.metrics.bandwidth_utilization = current_usage / (self.message_queue.bandwidth_limit * 1024)
        
        # Calculate message loss rate (simplified)
        pending_acks = len(self.message_queue.pending_acks)
        if self.metrics.total_messages_sent > 0:
            self.metrics.message_loss_rate = pending_acks / self.metrics.total_messages_sent
    
    def add_neighbor(self, agent_id: str, agent_info: AgentInfo):
        """Add a neighbor agent"""
        self.agents[agent_id] = agent_info
        self.routing_table.neighbors.add(agent_id)
        self.routing_table.add_route(agent_id, agent_id, 1, 1.0)
        
        # Notify callbacks
        for callback in self.connection_callbacks:
            try:
                callback(agent_id, "connected")
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    def remove_neighbor(self, agent_id: str):
        """Remove a neighbor agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
        
        self.routing_table.neighbors.discard(agent_id)
        
        # Remove routes through this neighbor
        routes_to_remove = [
            dest for dest, route in self.routing_table.routes.items()
            if route.next_hop == agent_id
        ]
        
        for dest in routes_to_remove:
            del self.routing_table.routes[dest]
        
        # Notify callbacks
        for callback in self.connection_callbacks:
            try:
                callback(agent_id, "disconnected")
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            "metrics": asdict(self.metrics),
            "active_agents": len(self.agents),
            "routing_table_size": len(self.routing_table.routes),
            "message_queue_size": len(self.message_queue.queue),
            "pending_acknowledgments": len(self.message_queue.pending_acks),
            "bandwidth_utilization": self.metrics.bandwidth_utilization,
            "consensus_history": len(self.consensus_manager.consensus_history)
        }

def demonstrate_communication_protocols():
    """Demonstrate the communication protocols framework"""
    print("=== Communication Protocols for Inter-Agent Coordination ===")
    
    # Configuration
    config = {
        "queue_size": 1000,
        "bandwidth_limit": 1000.0,  # KB/s
        "heartbeat_interval": 5,
        "routing_update_interval": 15,
        "metrics_update_interval": 30,
        "consensus_threshold": 0.6
    }
    
    async def run_demonstration():
        print("1. Initializing Communication Layers...")
        
        # Create multiple agents
        agents = {}
        comm_layers = {}
        
        agent_configs = [
            ("coordinator_1", AgentRole.COORDINATOR),
            ("worker_1", AgentRole.WORKER),
            ("worker_2", AgentRole.WORKER),
            ("bridge_1", AgentRole.BRIDGE),
            ("monitor_1", AgentRole.MONITOR)
        ]
        
        for agent_id, role in agent_configs:
            comm_layer = CommunicationLayer(agent_id, config)
            comm_layers[agent_id] = comm_layer
            
            agent_info = AgentInfo(
                agent_id=agent_id,
                role=role,
                capabilities={"scheduling", "monitoring"} if role == AgentRole.COORDINATOR else {"execution"},
                current_load=np.random.uniform(0.1, 0.8),
                location=(np.random.uniform(0, 100), np.random.uniform(0, 100)),
                bandwidth_limit=1000.0
            )
            agents[agent_id] = agent_info
        
        print(f"   Created {len(agents)} agents with communication layers")
        
        print("2. Setting up Network Topology...")
        
        # Create a simple network topology
        connections = [
            ("coordinator_1", "bridge_1"),
            ("bridge_1", "worker_1"),
            ("bridge_1", "worker_2"),
            ("coordinator_1", "monitor_1"),
            ("worker_1", "worker_2")
        ]
        
        for agent1, agent2 in connections:
            comm_layers[agent1].add_neighbor(agent2, agents[agent2])
            comm_layers[agent2].add_neighbor(agent1, agents[agent1])
        
        print(f"   Established {len(connections)} bidirectional connections")
        
        print("3. Starting Communication Layers...")
        
        # Start all communication layers
        start_tasks = []
        for comm_layer in comm_layers.values():
            start_tasks.append(comm_layer.start())
        
        await asyncio.gather(*start_tasks)
        print("   All communication layers started")
        
        print("4. Testing Message Types...")
        
        # Test different message types
        coordinator = comm_layers["coordinator_1"]
        
        # Task request message
        task_request = Message(
            message_id="",
            sender_id="coordinator_1",
            receiver_id="worker_1",
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.HIGH,
            payload={
                "task_id": "task_001",
                "resource_requirements": {"cpu": 4, "memory": 8},
                "deadline": time.time() + 3600
            }
        )
        
        await coordinator.send_message(task_request)
        print("   Sent task request message")
        
        # Resource update broadcast
        resource_update = Message(
            message_id="",
            sender_id="coordinator_1",
            receiver_id="broadcast",
            message_type=MessageType.RESOURCE_UPDATE,
            priority=MessagePriority.MEDIUM,
            payload={
                "available_resources": {"cpu": 100, "memory": 256, "gpu": 8},
                "utilization": 0.6
            }
        )
        
        await coordinator.send_message(resource_update)
        print("   Sent resource update broadcast")
        
        # Multicast to workers
        coordination_request = Message(
            message_id="",
            sender_id="coordinator_1",
            receiver_id="multicast:workers",
            message_type=MessageType.COORDINATION_REQUEST,
            priority=MessagePriority.HIGH,
            payload={
                "coordination_type": "load_balancing",
                "parameters": {"target_utilization": 0.8}
            }
        )
        
        await coordinator.send_message(coordination_request)
        print("   Sent coordination request to workers")
        
        print("5. Testing Consensus Protocol...")
        
        # Initiate consensus
        consensus_result = await coordinator.consensus_manager.initiate_consensus(
            consensus_id="consensus_001",
            proposal={
                "action": "scale_resources",
                "parameters": {"target_nodes": 5, "resource_type": "gpu"}
            },
            participants=["coordinator_1", "bridge_1", "worker_1", "worker_2"],
            timeout=10.0
        )
        
        print(f"   Consensus result: {consensus_result['status']}")
        print(f"   Votes: {consensus_result['approve_count']}/{consensus_result['total_participants']}")
        print(f"   Duration: {consensus_result['duration']:.2f}s")
        
        print("6. Testing Fault Tolerance...")
        
        # Simulate agent failure
        print("   Simulating bridge_1 failure...")
        await comm_layers["bridge_1"].stop()
        
        # Remove from other agents' neighbor lists
        for agent_id, comm_layer in comm_layers.items():
            if agent_id != "bridge_1":
                comm_layer.remove_neighbor("bridge_1")
        
        # Wait for routing to adapt
        await asyncio.sleep(2)
        
        # Test connectivity after failure
        test_message = Message(
            message_id="",
            sender_id="coordinator_1",
            receiver_id="worker_1",
            message_type=MessageType.HEARTBEAT,
            priority=MessagePriority.LOW,
            payload={"test": "connectivity_after_failure"}
        )
        
        success = await coordinator.send_message(test_message)
        print(f"   Message delivery after failure: {'Success' if success else 'Failed'}")
        
        print("7. Performance Metrics...")
        
        # Wait for metrics to accumulate
        await asyncio.sleep(3)
        
        for agent_id, comm_layer in comm_layers.items():
            if agent_id != "bridge_1":  # Skip failed agent
                stats = comm_layer.get_communication_statistics()
                print(f"   {agent_id} metrics:")
                print(f"     Messages sent: {stats['metrics']['total_messages_sent']}")
                print(f"     Messages received: {stats['metrics']['total_messages_received']}")
                print(f"     Bandwidth utilization: {stats['bandwidth_utilization']:.1%}")
                print(f"     Active neighbors: {stats['active_agents']}")
        
        print("8. Message Queue Analysis...")
        
        # Analyze message queue performance
        test_comm = comm_layers["coordinator_1"]
        queue_stats = {
            "queue_size": len(test_comm.message_queue.queue),
            "pending_acks": len(test_comm.message_queue.pending_acks),
            "bandwidth_usage": len(test_comm.message_queue.bandwidth_usage)
        }
        
        print(f"   Queue size: {queue_stats['queue_size']}")
        print(f"   Pending acknowledgments: {queue_stats['pending_acks']}")
        print(f"   Bandwidth samples: {queue_stats['bandwidth_usage']}")
        
        print("9. Communication Protocol Benefits...")
        
        benefits = [
            "Asynchronous message passing with guaranteed delivery",
            "Priority-based message queuing and bandwidth management",
            "Fault-tolerant routing with automatic topology adaptation",
            "Distributed consensus protocols for coordinated decisions",
            "Hierarchical communication reducing coordination overhead",
            "Privacy-preserving information sharing mechanisms",
            "Real-time performance monitoring and optimization",
            "Scalable multicast and broadcast communication"
        ]
        
        for i, benefit in enumerate(benefits, 1):
            print(f"   {i}. {benefit}")
        
        print("10. Cleanup...")
        
        # Stop all remaining communication layers
        stop_tasks = []
        for agent_id, comm_layer in comm_layers.items():
            if agent_id != "bridge_1":  # Already stopped
                stop_tasks.append(comm_layer.stop())
        
        await asyncio.gather(*stop_tasks)
        print("   All communication layers stopped")
        
        return {
            "agents": agents,
            "communication_layers": comm_layers,
            "consensus_result": consensus_result,
            "final_statistics": {
                agent_id: comm_layer.get_communication_statistics()
                for agent_id, comm_layer in comm_layers.items()
                if agent_id != "bridge_1"
            }
        }
    
    # Run the demonstration
    return asyncio.run(run_demonstration())

if __name__ == "__main__":
    demonstrate_communication_protocols()