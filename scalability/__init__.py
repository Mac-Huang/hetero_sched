"""
Scalability Module for HeteroSched

This module implements comprehensive scalability solutions for large-scale
heterogeneous scheduling systems, including hierarchical state representation,
graph neural networks, federated learning, and adaptive optimization techniques.

Key Components:
- Hierarchical state representation for multi-level system abstraction
- Graph neural networks for variable-topology scheduling
- Federated learning for distributed training across clusters
- Adaptive action space pruning for computational efficiency
- Experience replay prioritization for handling rare critical events

Authors: HeteroSched Research Team
"""

from .hierarchical_state import (
    StateLevel,
    AggregationType,
    NodeState,
    HierarchicalState,
    StateEncoder,
    AttentionAggregator,
    GraphStateEncoder,
    GraphConvLayer,
    HierarchicalStateManager,
    create_sample_topology
)

from .graph_neural_scheduler import (
    NodeType,
    EdgeType,
    SchedulingAction,
    GraphNode,
    GraphEdge,
    SchedulingTask,
    GraphState,
    HeterogeneousGraphConv,
    TopologyEncoder,
    GraphSchedulingPolicy,
    GraphExperienceReplay,
    DynamicGraphScheduler,
    create_sample_graph_state
)

from .federated_learning import (
    ClientType,
    AggregationMethod,
    PrivacyMechanism,
    ClientProfile,
    FederatedModel,
    LocalUpdate,
    DifferentialPrivacy,
    PersonalizedModel,
    FederatedAggregator,
    FederatedScheduler,
    create_sample_clients
)

from .action_space_pruning import (
    PruningStrategy,
    ActionType,
    SchedulingAction,
    ActionPruningConfig,
    SystemState,
    FeasibilityAnalyzer,
    ValueBasedPruner,
    HierarchicalActionClusterer,
    LearnedActionPruner,
    AdaptiveActionSpacePruner,
    create_sample_actions,
    create_sample_system_state
)

from .prioritized_experience_replay import (
    EventType,
    PriorityType,
    SchedulingExperience,
    RareEventDetector,
    CuriosityDrivenExploration,
    HindsightExperienceReplay,
    MultiObjectivePriorityCalculator,
    PrioritizedExperienceReplay,
    create_sample_experiences
)

__all__ = [
    # Hierarchical State
    'StateLevel',
    'AggregationType',
    'NodeState', 
    'HierarchicalState',
    'StateEncoder',
    'AttentionAggregator',
    'GraphStateEncoder',
    'GraphConvLayer',
    'HierarchicalStateManager',
    'create_sample_topology',
    
    # Graph Neural Scheduler
    'NodeType',
    'EdgeType',
    'SchedulingAction',
    'GraphNode',
    'GraphEdge',
    'SchedulingTask',
    'GraphState',
    'HeterogeneousGraphConv',
    'TopologyEncoder',
    'GraphSchedulingPolicy',
    'GraphExperienceReplay',
    'DynamicGraphScheduler',
    'create_sample_graph_state',
    
    # Federated Learning
    'ClientType',
    'AggregationMethod',
    'PrivacyMechanism',
    'ClientProfile',
    'FederatedModel',
    'LocalUpdate',
    'DifferentialPrivacy',
    'PersonalizedModel',
    'FederatedAggregator',
    'FederatedScheduler',
    'create_sample_clients',
    
    # Action Space Pruning
    'PruningStrategy',
    'ActionType',
    'SchedulingAction',
    'ActionPruningConfig',
    'SystemState',
    'FeasibilityAnalyzer',
    'ValueBasedPruner',
    'HierarchicalActionClusterer',
    'LearnedActionPruner',
    'AdaptiveActionSpacePruner',
    'create_sample_actions',
    'create_sample_system_state',
    
    # Prioritized Experience Replay
    'EventType',
    'PriorityType',
    'SchedulingExperience',
    'RareEventDetector',
    'CuriosityDrivenExploration',
    'HindsightExperienceReplay',
    'MultiObjectivePriorityCalculator',
    'PrioritizedExperienceReplay',
    'create_sample_experiences'
]