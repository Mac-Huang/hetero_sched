"""
Multi-Agent Module for HeteroSched

This module implements distributed multi-agent systems for heterogeneous scheduling,
including federated architectures, consensus protocols, and fault-tolerant coordination.

Key Components:
- Distributed agent architecture with hierarchical coordination
- Consensus-based decision making protocols
- Real-time resource monitoring and task assignment
- Fault-tolerant communication systems
- Game-theoretic multi-agent coordination

Authors: HeteroSched Research Team
"""

from .distributed_scheduler import (
    DistributedAgent,
    GlobalCoordinator, 
    ClusterManager,
    LocalScheduler,
    AgentRole,
    MessageType,
    ConsensusState,
    AgentMessage,
    ResourceInfo,
    TaskRequest,
    ConsensusProposal,
    CommunicationLayer,
    create_distributed_system
)

__all__ = [
    'DistributedAgent',
    'GlobalCoordinator',
    'ClusterManager', 
    'LocalScheduler',
    'AgentRole',
    'MessageType',
    'ConsensusState',
    'AgentMessage',
    'ResourceInfo',
    'TaskRequest',
    'ConsensusProposal',
    'CommunicationLayer',
    'create_distributed_system'
]