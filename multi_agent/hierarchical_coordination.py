"""
Hierarchical Coordination Between Global and Local Agents for HeteroSched

This module implements R16: a sophisticated hierarchical multi-agent coordination framework
that enables efficient scheduling across multiple levels of the system hierarchy.

Key Features:
1. Global-Local Agent Hierarchy with distinct responsibilities
2. Dynamic delegation strategies based on system state and workload
3. Consensus mechanisms for conflict resolution
4. Adaptive coordination protocols that scale with system size
5. Load balancing and fault tolerance across agent levels

The framework supports heterogeneous scheduling at datacenter, cluster, and node levels
while maintaining coordination efficiency and decision quality.

Authors: HeteroSched Research Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging
from collections import defaultdict, deque
import threading
import queue

class AgentLevel(Enum):
    GLOBAL = "global"
    CLUSTER = "cluster" 
    NODE = "node"
    DEVICE = "device"

class CoordinationProtocol(Enum):
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    AUCTION = "auction"
    DELEGATION = "delegation"

class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Agent:
    """Represents an agent in the hierarchical coordination system"""
    agent_id: str
    level: AgentLevel
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    managed_resources: Set[str] = field(default_factory=set)
    current_load: float = 0.0
    capacity: float = 1.0
    is_active: bool = True
    last_heartbeat: float = field(default_factory=time.time)

@dataclass
class SchedulingTask:
    """Represents a task to be scheduled in the hierarchical system"""
    task_id: str
    resource_requirements: Dict[str, float]
    deadline: Optional[float] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    affinity_constraints: Set[str] = field(default_factory=set)
    anti_affinity_constraints: Set[str] = field(default_factory=set)
    estimated_duration: float = 1.0
    
@dataclass
class CoordinationMessage:
    """Message passed between agents in the coordination protocol"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: TaskPriority = TaskPriority.MEDIUM

@dataclass
class SchedulingDecision:
    """Represents a scheduling decision made by an agent"""
    decision_id: str
    agent_id: str
    task_id: str
    assigned_resources: Dict[str, str]
    confidence: float
    rationale: str
    timestamp: float = field(default_factory=time.time)

class GlobalAgent:
    """
    Global-level agent responsible for high-level coordination and resource allocation
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.cluster_agents: Dict[str, Agent] = {}
        self.global_state = {}
        self.pending_tasks: Dict[str, SchedulingTask] = {}
        self.coordination_history: List[CoordinationMessage] = []
        
        # Neural network for global decision making
        self.policy_network = GlobalPolicyNetwork(config)
        self.value_network = GlobalValueNetwork(config)
        
        # Coordination mechanisms
        self.delegation_strategy = DelegationStrategy(config)
        self.consensus_manager = ConsensusManager(config)
        self.load_balancer = LoadBalancer(config)
        
        # Communication queues
        self.message_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        
        self.logger = logging.getLogger(f"GlobalAgent-{agent_id}")
        
    async def coordinate_scheduling(self, tasks: List[SchedulingTask]) -> List[SchedulingDecision]:
        """
        Main coordination method for global scheduling decisions
        """
        self.logger.info(f"Coordinating scheduling for {len(tasks)} tasks")
        
        # Analyze global system state
        global_state = await self.analyze_global_state()
        
        # Categorize tasks by complexity and requirements
        task_categories = self.categorize_tasks(tasks)
        
        decisions = []
        
        # Handle different task categories with appropriate strategies
        for category, task_list in task_categories.items():
            if category == "critical_global":
                # Handle critical tasks requiring global coordination
                category_decisions = await self.handle_critical_global_tasks(task_list, global_state)
            elif category == "cluster_delegatable":
                # Delegate to cluster agents with coordination
                category_decisions = await self.delegate_to_clusters(task_list, global_state)
            elif category == "distributed":
                # Handle multi-cluster distributed tasks
                category_decisions = await self.handle_distributed_tasks(task_list, global_state)
            else:
                # Default delegation strategy
                category_decisions = await self.default_delegation(task_list, global_state)
            
            decisions.extend(category_decisions)
        
        # Apply global constraints and optimization
        optimized_decisions = await self.optimize_global_allocation(decisions, global_state)
        
        # Communicate decisions to cluster agents
        await self.communicate_decisions(optimized_decisions)
        
        return optimized_decisions
    
    def categorize_tasks(self, tasks: List[SchedulingTask]) -> Dict[str, List[SchedulingTask]]:
        """Categorize tasks based on global coordination requirements"""
        categories = {
            "critical_global": [],
            "cluster_delegatable": [],
            "distributed": [],
            "standard": []
        }
        
        for task in tasks:
            # Critical tasks that require global oversight
            if task.priority == TaskPriority.CRITICAL or self._requires_global_coordination(task):
                categories["critical_global"].append(task)
            # Tasks that span multiple clusters
            elif self._is_distributed_task(task):
                categories["distributed"].append(task)
            # Tasks that can be delegated to cluster agents
            elif self._can_delegate_to_cluster(task):
                categories["cluster_delegatable"].append(task)
            else:
                categories["standard"].append(task)
        
        return categories
    
    async def analyze_global_state(self) -> Dict[str, Any]:
        """Analyze the current global system state"""
        state = {
            "total_capacity": 0.0,
            "total_load": 0.0,
            "cluster_states": {},
            "resource_availability": {},
            "network_topology": {},
            "bottlenecks": [],
            "predicted_load": {}
        }
        
        # Aggregate cluster states
        for cluster_id, cluster_agent in self.cluster_agents.items():
            cluster_state = await self.request_cluster_state(cluster_id)
            state["cluster_states"][cluster_id] = cluster_state
            state["total_capacity"] += cluster_state.get("capacity", 0.0)
            state["total_load"] += cluster_state.get("current_load", 0.0)
        
        # Identify bottlenecks and optimization opportunities
        state["bottlenecks"] = self._identify_bottlenecks(state["cluster_states"])
        state["predicted_load"] = self._predict_future_load(state["cluster_states"])
        
        return state
    
    async def delegate_to_clusters(self, tasks: List[SchedulingTask], 
                                 global_state: Dict[str, Any]) -> List[SchedulingDecision]:
        """Delegate tasks to appropriate cluster agents"""
        decisions = []
        
        # Group tasks by target cluster using intelligent assignment
        cluster_assignments = self.delegation_strategy.assign_tasks_to_clusters(
            tasks, global_state["cluster_states"]
        )
        
        # Send delegation messages to cluster agents
        delegation_futures = []
        for cluster_id, cluster_tasks in cluster_assignments.items():
            future = self._delegate_to_cluster(cluster_id, cluster_tasks, global_state)
            delegation_futures.append(future)
        
        # Wait for all cluster responses with timeout
        cluster_responses = await asyncio.gather(*delegation_futures, return_exceptions=True)
        
        # Process cluster responses and handle failures
        for cluster_id, response in zip(cluster_assignments.keys(), cluster_responses):
            if isinstance(response, Exception):
                self.logger.error(f"Delegation to cluster {cluster_id} failed: {response}")
                # Implement fallback strategy
                fallback_decisions = await self._handle_delegation_failure(
                    cluster_id, cluster_assignments[cluster_id], global_state
                )
                decisions.extend(fallback_decisions)
            else:
                decisions.extend(response)
        
        return decisions
    
    def _requires_global_coordination(self, task: SchedulingTask) -> bool:
        """Check if task requires global-level coordination"""
        # Tasks with cross-cluster dependencies
        if len(task.affinity_constraints) > 1:
            return True
        
        # High resource requirements that might span clusters
        total_resources = sum(task.resource_requirements.values())
        if total_resources > self.config.get("global_coordination_threshold", 0.8):
            return True
        
        # Tasks with strict SLA requirements
        if task.deadline and task.deadline < self.config.get("critical_deadline_threshold", 60):
            return True
        
        return False

class ClusterAgent:
    """
    Cluster-level agent responsible for intra-cluster coordination
    """
    
    def __init__(self, agent_id: str, cluster_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.cluster_id = cluster_id
        self.config = config
        self.node_agents: Dict[str, Agent] = {}
        self.cluster_state = {}
        self.local_queue: Dict[str, SchedulingTask] = {}
        
        # Neural networks for cluster-level decisions
        self.policy_network = ClusterPolicyNetwork(config)
        self.coordination_network = CoordinationNetwork(config)
        
        # Local coordination mechanisms
        self.local_scheduler = LocalScheduler(config)
        self.resource_manager = ClusterResourceManager(config)
        self.communication_manager = CommunicationManager(config)
        
        self.logger = logging.getLogger(f"ClusterAgent-{agent_id}")
        
    async def handle_delegated_tasks(self, tasks: List[SchedulingTask], 
                                   global_context: Dict[str, Any]) -> List[SchedulingDecision]:
        """Handle tasks delegated from the global agent"""
        self.logger.info(f"Handling {len(tasks)} delegated tasks")
        
        # Update local state with global context
        await self.update_cluster_state(global_context)
        
        # Categorize tasks for local vs node delegation
        local_tasks, node_tasks = self.categorize_local_tasks(tasks)
        
        decisions = []
        
        # Handle tasks that can be resolved at cluster level
        if local_tasks:
            local_decisions = await self.schedule_local_tasks(local_tasks)
            decisions.extend(local_decisions)
        
        # Delegate appropriate tasks to node agents
        if node_tasks:
            node_decisions = await self.delegate_to_nodes(node_tasks)
            decisions.extend(node_decisions)
        
        # Apply cluster-level optimization
        optimized_decisions = await self.optimize_cluster_allocation(decisions)
        
        return optimized_decisions
    
    def categorize_local_tasks(self, tasks: List[SchedulingTask]) -> Tuple[List[SchedulingTask], List[SchedulingTask]]:
        """Categorize tasks for local handling vs node delegation"""
        local_tasks = []
        node_tasks = []
        
        for task in tasks:
            # Tasks requiring cluster-wide coordination
            if self._requires_cluster_coordination(task):
                local_tasks.append(task)
            # Tasks that can be handled by individual nodes
            else:
                node_tasks.append(task)
        
        return local_tasks, node_tasks
    
    async def coordinate_with_peers(self, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with peer cluster agents for cross-cluster tasks"""
        peer_clusters = coordination_request.get("peer_clusters", [])
        coordination_type = coordination_request.get("type", "resource_sharing")
        
        if coordination_type == "resource_sharing":
            return await self._coordinate_resource_sharing(peer_clusters, coordination_request)
        elif coordination_type == "load_balancing":
            return await self._coordinate_load_balancing(peer_clusters, coordination_request)
        elif coordination_type == "consensus":
            return await self._participate_in_consensus(peer_clusters, coordination_request)
        else:
            self.logger.warning(f"Unknown coordination type: {coordination_type}")
            return {"status": "error", "message": "Unknown coordination type"}

class NodeAgent:
    """
    Node-level agent responsible for local resource management
    """
    
    def __init__(self, agent_id: str, node_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.node_id = node_id
        self.config = config
        self.local_resources = {}
        self.active_tasks: Dict[str, SchedulingTask] = {}
        self.performance_history = deque(maxlen=1000)
        
        # Local decision making
        self.local_policy = NodePolicyNetwork(config)
        self.resource_monitor = ResourceMonitor(config)
        self.task_executor = TaskExecutor(config)
        
        self.logger = logging.getLogger(f"NodeAgent-{agent_id}")
    
    async def execute_local_scheduling(self, tasks: List[SchedulingTask]) -> List[SchedulingDecision]:
        """Execute scheduling decisions at the node level"""
        self.logger.info(f"Executing local scheduling for {len(tasks)} tasks")
        
        # Monitor current resource state
        resource_state = await self.resource_monitor.get_current_state()
        
        # Make local scheduling decisions
        decisions = []
        for task in tasks:
            decision = await self.make_local_decision(task, resource_state)
            if decision:
                decisions.append(decision)
                # Update resource state for next decision
                resource_state = self.update_resource_state(resource_state, decision)
        
        return decisions
    
    async def make_local_decision(self, task: SchedulingTask, 
                                resource_state: Dict[str, Any]) -> Optional[SchedulingDecision]:
        """Make a scheduling decision for a single task"""
        # Check resource availability
        if not self._can_accommodate_task(task, resource_state):
            return None
        
        # Use local policy to make decision
        state_tensor = self._encode_state(task, resource_state)
        action_probs = self.local_policy(state_tensor)
        
        # Select action and create decision
        action = self._select_action(action_probs, task, resource_state)
        
        if action is not None:
            decision = SchedulingDecision(
                decision_id=f"{self.node_id}_{task.task_id}_{int(time.time())}",
                agent_id=self.agent_id,
                task_id=task.task_id,
                assigned_resources=action["resources"],
                confidence=action["confidence"],
                rationale=f"Local node decision based on resource availability"
            )
            return decision
        
        return None

# Neural Network Architectures for Hierarchical Coordination

class GlobalPolicyNetwork(nn.Module):
    """Neural network for global-level policy decisions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Encoding layers for global state
        self.cluster_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.get("global_d_model", 256),
                nhead=config.get("global_nhead", 8),
                batch_first=True
            ),
            num_layers=config.get("global_encoder_layers", 6)
        )
        
        # Task encoding
        self.task_encoder = nn.Linear(config.get("task_features", 64), config.get("global_d_model", 256))
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(config.get("global_d_model", 256), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.get("num_global_actions", 128))
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.get("global_d_model", 256), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, cluster_states: torch.Tensor, task_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode cluster states
        cluster_encoded = self.cluster_encoder(cluster_states)
        cluster_global = cluster_encoded.mean(dim=1)  # Global cluster representation
        
        # Encode tasks
        task_encoded = self.task_encoder(task_features)
        
        # Combine representations
        combined = cluster_global + task_encoded.mean(dim=1)
        
        # Generate policy and value
        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return policy_logits, value

class ClusterPolicyNetwork(nn.Module):
    """Neural network for cluster-level coordination decisions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Multi-head attention for node coordination
        self.node_attention = nn.MultiheadAttention(
            embed_dim=config.get("cluster_d_model", 128),
            num_heads=config.get("cluster_nhead", 4),
            batch_first=True
        )
        
        # Task-node matching network
        self.matching_network = nn.Sequential(
            nn.Linear(config.get("cluster_d_model", 128) * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Coordination policy
        self.coordination_policy = nn.Sequential(
            nn.Linear(config.get("cluster_d_model", 128), 256),
            nn.ReLU(),
            nn.Linear(256, config.get("num_coordination_actions", 32))
        )
    
    def forward(self, node_states: torch.Tensor, task_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Apply attention over nodes
        attended_nodes, attention_weights = self.node_attention(
            node_states, node_states, node_states
        )
        
        # Compute task-node matching scores
        num_tasks, num_nodes = task_features.size(0), attended_nodes.size(1)
        matching_scores = torch.zeros(num_tasks, num_nodes)
        
        for i in range(num_tasks):
            for j in range(num_nodes):
                combined = torch.cat([task_features[i:i+1], attended_nodes[0, j:j+1]], dim=-1)
                matching_scores[i, j] = self.matching_network(combined).squeeze()
        
        # Generate coordination policy
        global_state = attended_nodes.mean(dim=1)
        coordination_logits = self.coordination_policy(global_state)
        
        return {
            "matching_scores": matching_scores,
            "coordination_logits": coordination_logits,
            "attention_weights": attention_weights
        }

class NodePolicyNetwork(nn.Module):
    """Neural network for node-level scheduling decisions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Resource state encoder
        self.resource_encoder = nn.Sequential(
            nn.Linear(config.get("resource_features", 32), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(config.get("task_features", 64), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Decision network
        self.decision_network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.get("num_node_actions", 16))
        )
    
    def forward(self, resource_state: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        resource_encoded = self.resource_encoder(resource_state)
        task_encoded = self.task_encoder(task_features)
        
        combined = torch.cat([resource_encoded, task_encoded], dim=-1)
        action_logits = self.decision_network(combined)
        
        return action_logits

# Coordination Mechanisms

class DelegationStrategy:
    """Implements intelligent delegation strategies for hierarchical coordination"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.delegation_history: List[Dict[str, Any]] = []
        
    def assign_tasks_to_clusters(self, tasks: List[SchedulingTask], 
                               cluster_states: Dict[str, Any]) -> Dict[str, List[SchedulingTask]]:
        """Assign tasks to clusters using intelligent delegation"""
        assignments = defaultdict(list)
        
        for task in tasks:
            best_cluster = self._select_best_cluster(task, cluster_states)
            assignments[best_cluster].append(task)
        
        return dict(assignments)
    
    def _select_best_cluster(self, task: SchedulingTask, 
                           cluster_states: Dict[str, Any]) -> str:
        """Select the best cluster for a given task"""
        scores = {}
        
        for cluster_id, state in cluster_states.items():
            score = self._compute_cluster_score(task, state)
            scores[cluster_id] = score
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _compute_cluster_score(self, task: SchedulingTask, cluster_state: Dict[str, Any]) -> float:
        """Compute a score for assigning a task to a cluster"""
        # Resource availability score
        resource_score = self._compute_resource_availability_score(task, cluster_state)
        
        # Load balancing score
        load_score = 1.0 - cluster_state.get("current_load", 0.0)
        
        # Affinity score
        affinity_score = self._compute_affinity_score(task, cluster_state)
        
        # Weighted combination
        total_score = (
            0.4 * resource_score +
            0.3 * load_score +
            0.3 * affinity_score
        )
        
        return total_score

class ConsensusManager:
    """Manages consensus protocols for multi-agent coordination"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_consensus: Dict[str, Dict[str, Any]] = {}
        
    async def initiate_consensus(self, consensus_id: str, participants: List[str], 
                               proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate a consensus protocol among specified agents"""
        self.active_consensus[consensus_id] = {
            "participants": participants,
            "proposal": proposal,
            "votes": {},
            "status": "active",
            "start_time": time.time()
        }
        
        # Send consensus request to all participants
        responses = await self._send_consensus_requests(consensus_id, participants, proposal)
        
        # Process responses and determine consensus
        result = await self._process_consensus_responses(consensus_id, responses)
        
        return result
    
    async def _send_consensus_requests(self, consensus_id: str, participants: List[str], 
                                     proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Send consensus requests to all participants"""
        # Implementation would send actual network requests
        # For demonstration, return mock responses
        responses = []
        for participant in participants:
            response = {
                "participant_id": participant,
                "vote": "approve" if np.random.random() > 0.2 else "reject",
                "rationale": f"Decision from {participant}",
                "confidence": np.random.uniform(0.7, 1.0)
            }
            responses.append(response)
        
        return responses

class LoadBalancer:
    """Implements load balancing strategies across the hierarchy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def compute_load_distribution(self, agents: Dict[str, Agent], 
                                tasks: List[SchedulingTask]) -> Dict[str, List[SchedulingTask]]:
        """Compute optimal load distribution across agents"""
        # Calculate current loads
        current_loads = {agent_id: agent.current_load for agent_id, agent in agents.items()}
        
        # Predict future loads
        predicted_loads = self._predict_loads(current_loads, tasks)
        
        # Optimize distribution
        distribution = self._optimize_distribution(agents, tasks, predicted_loads)
        
        return distribution
    
    def _predict_loads(self, current_loads: Dict[str, float], 
                      tasks: List[SchedulingTask]) -> Dict[str, float]:
        """Predict future loads based on current state and incoming tasks"""
        predicted = current_loads.copy()
        
        # Simple prediction model - can be enhanced with ML
        total_task_load = sum(task.estimated_duration for task in tasks)
        avg_increase = total_task_load / len(predicted) if predicted else 0
        
        for agent_id in predicted:
            predicted[agent_id] += avg_increase * (1 + np.random.normal(0, 0.1))
        
        return predicted

def demonstrate_hierarchical_coordination():
    """Demonstrate the hierarchical coordination system"""
    print("=== Hierarchical Multi-Agent Coordination Demo ===")
    
    # Configuration
    config = {
        "global_d_model": 256,
        "global_nhead": 8,
        "global_encoder_layers": 6,
        "cluster_d_model": 128,
        "cluster_nhead": 4,
        "task_features": 64,
        "resource_features": 32,
        "num_global_actions": 128,
        "num_coordination_actions": 32,
        "num_node_actions": 16,
        "global_coordination_threshold": 0.8,
        "critical_deadline_threshold": 60
    }
    
    print("1. Initializing Global Agent...")
    global_agent = GlobalAgent("global_001", config)
    
    print("2. Creating Cluster Agents...")
    cluster_agents = []
    for i in range(3):
        cluster_agent = ClusterAgent(f"cluster_{i:03d}", f"cluster_{i}", config)
        cluster_agents.append(cluster_agent)
        global_agent.cluster_agents[f"cluster_{i}"] = Agent(
            agent_id=f"cluster_{i:03d}",
            level=AgentLevel.CLUSTER,
            parent_id="global_001",
            capacity=100.0,
            current_load=np.random.uniform(0.3, 0.7)
        )
    
    print("3. Creating Node Agents...")
    node_agents = []
    for cluster_idx in range(3):
        for node_idx in range(5):
            node_agent = NodeAgent(f"node_{cluster_idx}_{node_idx:02d}", f"node_{cluster_idx}_{node_idx}", config)
            node_agents.append(node_agent)
            
            # Add to cluster agent
            cluster_agents[cluster_idx].node_agents[f"node_{cluster_idx}_{node_idx}"] = Agent(
                agent_id=f"node_{cluster_idx}_{node_idx:02d}",
                level=AgentLevel.NODE,
                parent_id=f"cluster_{cluster_idx:03d}",
                capacity=20.0,
                current_load=np.random.uniform(0.2, 0.8)
            )
    
    print("4. Generating Sample Tasks...")
    tasks = []
    for i in range(50):
        task = SchedulingTask(
            task_id=f"task_{i:04d}",
            resource_requirements={
                "cpu": np.random.uniform(1, 8),
                "memory": np.random.uniform(1, 16),
                "gpu": np.random.choice([0, 1, 2])
            },
            deadline=time.time() + np.random.uniform(60, 3600),
            priority=np.random.choice(list(TaskPriority)),
            estimated_duration=np.random.uniform(10, 300)
        )
        tasks.append(task)
    
    print("5. Demonstrating Task Categorization...")
    task_categories = global_agent.categorize_tasks(tasks)
    for category, task_list in task_categories.items():
        print(f"   {category}: {len(task_list)} tasks")
    
    print("6. Testing Neural Network Architectures...")
    
    # Test Global Policy Network
    global_policy = GlobalPolicyNetwork(config)
    cluster_states = torch.randn(1, 3, config["global_d_model"])
    task_features = torch.randn(1, 10, config["task_features"])
    
    policy_logits, values = global_policy(cluster_states, task_features)
    print(f"   Global Policy Output Shape: {policy_logits.shape}, Values: {values.shape}")
    
    # Test Cluster Policy Network
    cluster_policy = ClusterPolicyNetwork(config)
    node_states = torch.randn(1, 5, config["cluster_d_model"])
    cluster_task_features = torch.randn(10, config["task_features"])
    
    cluster_output = cluster_policy(node_states, cluster_task_features)
    print(f"   Cluster Policy - Matching Scores: {cluster_output['matching_scores'].shape}")
    print(f"   Cluster Policy - Coordination: {cluster_output['coordination_logits'].shape}")
    
    # Test Node Policy Network
    node_policy = NodePolicyNetwork(config)
    resource_state = torch.randn(config["resource_features"])
    node_task_features = torch.randn(config["task_features"])
    
    node_output = node_policy(resource_state, node_task_features)
    print(f"   Node Policy Output Shape: {node_output.shape}")
    
    print("7. Testing Delegation Strategy...")
    delegation_strategy = DelegationStrategy(config)
    
    # Mock cluster states
    cluster_states_dict = {}
    for i in range(3):
        cluster_states_dict[f"cluster_{i}"] = {
            "capacity": 100.0,
            "current_load": np.random.uniform(0.3, 0.7),
            "available_resources": {
                "cpu": np.random.uniform(50, 100),
                "memory": np.random.uniform(100, 200),
                "gpu": np.random.randint(0, 10)
            }
        }
    
    task_assignments = delegation_strategy.assign_tasks_to_clusters(tasks[:10], cluster_states_dict)
    print("   Task Assignment Results:")
    for cluster_id, assigned_tasks in task_assignments.items():
        print(f"     {cluster_id}: {len(assigned_tasks)} tasks")
    
    print("8. Testing Load Balancer...")
    load_balancer = LoadBalancer(config)
    
    agents_dict = {}
    for i in range(3):
        agents_dict[f"cluster_{i}"] = Agent(
            agent_id=f"cluster_{i}",
            level=AgentLevel.CLUSTER,
            capacity=100.0,
            current_load=np.random.uniform(0.3, 0.7)
        )
    
    load_distribution = load_balancer.compute_load_distribution(agents_dict, tasks[:15])
    print("   Load Distribution Results:")
    for agent_id, assigned_tasks in load_distribution.items():
        print(f"     {agent_id}: {len(assigned_tasks)} tasks")
    
    print("9. Performance Analysis...")
    
    # Analyze coordination efficiency
    total_agents = 1 + len(cluster_agents) + len(node_agents)
    coordination_overhead = len(task_categories["critical_global"]) / len(tasks)
    delegation_ratio = (len(task_categories["cluster_delegatable"]) + 
                       len(task_categories["standard"])) / len(tasks)
    
    print(f"   Total Agents: {total_agents}")
    print(f"   Coordination Overhead: {coordination_overhead:.2%}")
    print(f"   Delegation Ratio: {delegation_ratio:.2%}")
    print(f"   Hierarchical Levels: {len(set(agent.level for agents in [global_agent.cluster_agents.values()] for agent in agents))}")
    
    # Analyze decision latency (simulated)
    decision_latencies = {
        "global_decisions": np.random.uniform(10, 50, len(task_categories["critical_global"])),
        "cluster_decisions": np.random.uniform(2, 15, len(task_categories["cluster_delegatable"])),
        "node_decisions": np.random.uniform(0.5, 5, len(task_categories["standard"]))
    }
    
    print("   Decision Latency Analysis:")
    for decision_type, latencies in decision_latencies.items():
        if len(latencies) > 0:
            print(f"     {decision_type}: {np.mean(latencies):.1f}ms avg, {np.max(latencies):.1f}ms max")
    
    print("\n=== Hierarchical Coordination Benefits ===")
    benefits = [
        "Scalable coordination across multiple system levels",
        "Intelligent delegation reducing global coordination overhead", 
        "Fault tolerance through hierarchical redundancy",
        "Adaptive load balancing with predictive capabilities",
        "Context-aware task assignment using attention mechanisms"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")
    
    return {
        "global_agent": global_agent,
        "cluster_agents": cluster_agents,
        "node_agents": node_agents,
        "task_categories": task_categories,
        "neural_networks": {
            "global_policy": global_policy,
            "cluster_policy": cluster_policy, 
            "node_policy": node_policy
        }
    }

if __name__ == "__main__":
    demonstrate_hierarchical_coordination()