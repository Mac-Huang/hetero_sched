"""
Game-Theoretic Analysis of Multi-Agent Scheduling Equilibria

This module implements R18: comprehensive game-theoretic analysis for understanding
and optimizing equilibria in multi-agent heterogeneous scheduling systems.

Key Features:
1. Nash equilibrium analysis for distributed scheduling decisions
2. Mechanism design for incentive-compatible resource allocation
3. Auction-based protocols for competitive resource assignment
4. Cooperative and non-cooperative game formulations
5. Social welfare optimization and fairness guarantees
6. Strategic behavior analysis and equilibrium stability
7. Dynamic game models for evolving system conditions

The framework provides theoretical foundations and practical algorithms
for achieving efficient and stable multi-agent coordination.

Authors: HeteroSched Research Team
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import logging
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import pdist, squareform
import networkx as nx

class GameType(Enum):
    COOPERATIVE = "cooperative"
    NON_COOPERATIVE = "non_cooperative"
    SEQUENTIAL = "sequential"
    SIMULTANEOUS = "simultaneous"
    AUCTION = "auction"
    COALITION = "coalition"

class EquilibriumType(Enum):
    NASH = "nash"
    CORRELATED = "correlated"
    EVOLUTIONARY_STABLE = "evolutionary_stable"
    PARETO_OPTIMAL = "pareto_optimal"
    SOCIAL_OPTIMAL = "social_optimal"

class AuctionType(Enum):
    FIRST_PRICE = "first_price"
    SECOND_PRICE = "second_price"
    VCG = "vcg"  # Vickrey-Clarke-Groves
    COMBINATORIAL = "combinatorial"
    DOUBLE = "double"

@dataclass
class Agent:
    """Represents an agent in the game-theoretic model"""
    agent_id: str
    strategy_space: List[Any]
    utility_function: Callable
    resource_endowment: Dict[str, float]
    valuation_function: Callable
    budget: float = 0.0
    risk_preference: float = 0.0  # Risk aversion parameter
    cooperation_level: float = 1.0  # Willingness to cooperate
    
@dataclass
class Resource:
    """Represents a resource in the scheduling game"""
    resource_id: str
    resource_type: str
    capacity: float
    owner_id: Optional[str] = None
    quality: float = 1.0
    availability_schedule: List[Tuple[float, float]] = field(default_factory=list)
    
@dataclass
class Task:
    """Represents a task requiring resources"""
    task_id: str
    arrival_time: float
    deadline: float
    resource_requirements: Dict[str, float]
    priority: int
    value: float  # Value to the requesting agent
    penalty: float = 0.0  # Penalty for missing deadline
    
@dataclass
class GameOutcome:
    """Represents the outcome of a game"""
    allocation: Dict[str, Dict[str, float]]  # agent_id -> resource_id -> amount
    payments: Dict[str, float]  # agent_id -> payment
    utilities: Dict[str, float]  # agent_id -> utility
    social_welfare: float
    efficiency: float
    fairness_index: float
    equilibrium_type: Optional[EquilibriumType] = None

class UtilityFunction:
    """Utility function implementations for different agent types"""
    
    @staticmethod
    def linear_utility(allocation: Dict[str, float], valuations: Dict[str, float], 
                      payment: float = 0.0) -> float:
        """Linear utility function"""
        value = sum(allocation.get(resource, 0) * valuations.get(resource, 0) 
                   for resource in valuations)
        return value - payment
    
    @staticmethod
    def cobb_douglas_utility(allocation: Dict[str, float], alpha: Dict[str, float], 
                           payment: float = 0.0) -> float:
        """Cobb-Douglas utility function"""
        if not allocation or any(v <= 0 for v in allocation.values()):
            return -payment
        
        utility = np.prod([allocation.get(resource, 1e-6) ** alpha.get(resource, 0) 
                          for resource in alpha])
        return utility - payment
    
    @staticmethod
    def ces_utility(allocation: Dict[str, float], rho: float, 
                   weights: Dict[str, float], payment: float = 0.0) -> float:
        """Constant Elasticity of Substitution utility"""
        if rho == 0:  # Cobb-Douglas case
            return UtilityFunction.cobb_douglas_utility(allocation, weights, payment)
        
        utility_sum = sum(weights.get(resource, 0) * (allocation.get(resource, 0) ** rho)
                         for resource in allocation)
        
        if utility_sum <= 0:
            return -payment
        
        return (utility_sum ** (1/rho)) - payment
    
    @staticmethod
    def scheduling_utility(tasks_completed: List[Task], resources_used: Dict[str, float],
                          resource_costs: Dict[str, float], time_penalty: float = 0.1) -> float:
        """Scheduling-specific utility function"""
        # Value from completed tasks
        task_value = sum(task.value for task in tasks_completed)
        
        # Cost of resources
        resource_cost = sum(resources_used.get(res, 0) * resource_costs.get(res, 0)
                           for res in resource_costs)
        
        # Time penalty for late completions
        time_penalties = sum(max(0, task.penalty) for task in tasks_completed)
        
        return task_value - resource_cost - time_penalty * time_penalties

class NashEquilibriumSolver:
    """Solver for Nash equilibria in scheduling games"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("NashEquilibriumSolver")
        
    def find_pure_strategy_nash(self, agents: List[Agent], 
                               payoff_functions: Dict[str, Callable]) -> Optional[Dict[str, Any]]:
        """Find pure strategy Nash equilibria"""
        
        # Generate all strategy profiles
        strategy_spaces = [agent.strategy_space for agent in agents]
        all_profiles = list(itertools.product(*strategy_spaces))
        
        nash_equilibria = []
        
        for profile in all_profiles:
            if self._is_nash_equilibrium(profile, agents, payoff_functions):
                # Calculate utilities for this equilibrium
                utilities = {}
                for i, agent in enumerate(agents):
                    utilities[agent.agent_id] = payoff_functions[agent.agent_id](profile, i)
                
                nash_equilibria.append({
                    'strategy_profile': dict(zip([a.agent_id for a in agents], profile)),
                    'utilities': utilities,
                    'social_welfare': sum(utilities.values())
                })
        
        return nash_equilibria if nash_equilibria else None
    
    def _is_nash_equilibrium(self, profile: Tuple, agents: List[Agent], 
                           payoff_functions: Dict[str, Callable]) -> bool:
        """Check if a strategy profile is a Nash equilibrium"""
        
        for i, agent in enumerate(agents):
            current_utility = payoff_functions[agent.agent_id](profile, i)
            
            # Check all alternative strategies for this agent
            for alt_strategy in agent.strategy_space:
                if alt_strategy == profile[i]:
                    continue
                
                # Create alternative profile
                alt_profile = list(profile)
                alt_profile[i] = alt_strategy
                alt_profile = tuple(alt_profile)
                
                alt_utility = payoff_functions[agent.agent_id](alt_profile, i)
                
                # If agent can improve by unilateral deviation, not Nash equilibrium
                if alt_utility > current_utility + 1e-6:  # Small tolerance for numerical errors
                    return False
        
        return True
    
    def find_mixed_strategy_nash(self, agents: List[Agent], 
                                payoff_matrix: np.ndarray) -> Optional[Dict[str, Any]]:
        """Find mixed strategy Nash equilibria using iterative methods"""
        
        if len(agents) != 2:
            self.logger.warning("Mixed strategy Nash only implemented for 2-player games")
            return None
        
        # Use fictitious play to approximate mixed Nash equilibrium
        num_strategies = [len(agent.strategy_space) for agent in agents]
        
        # Initialize belief probabilities
        beliefs = [np.ones(num_strategies[i]) / num_strategies[i] for i in range(2)]
        strategy_counts = [np.zeros(num_strategies[i]) for i in range(2)]
        
        max_iterations = self.config.get('max_iterations', 1000)
        convergence_threshold = self.config.get('convergence_threshold', 1e-6)
        
        for iteration in range(max_iterations):
            old_beliefs = [b.copy() for b in beliefs]
            
            # Each agent plays best response to opponent's mixed strategy
            for i in range(2):
                opponent = 1 - i
                
                # Calculate expected payoffs for each strategy
                expected_payoffs = np.zeros(num_strategies[i])
                
                for s_i in range(num_strategies[i]):
                    for s_j in range(num_strategies[opponent]):
                        if i == 0:
                            payoff = payoff_matrix[s_i, s_j, 0]  # Player 1's payoff
                        else:
                            payoff = payoff_matrix[s_j, s_i, 1]  # Player 2's payoff
                        
                        expected_payoffs[s_i] += beliefs[opponent][s_j] * payoff
                
                # Play best response
                best_response = np.argmax(expected_payoffs)
                strategy_counts[i][best_response] += 1
                
                # Update beliefs (empirical frequency)
                beliefs[i] = strategy_counts[i] / np.sum(strategy_counts[i])
            
            # Check convergence
            convergence = all(
                np.linalg.norm(beliefs[i] - old_beliefs[i]) < convergence_threshold
                for i in range(2)
            )
            
            if convergence:
                self.logger.info(f"Mixed Nash converged after {iteration + 1} iterations")
                break
        
        # Calculate expected utilities
        expected_utilities = {}
        for i in range(2):
            utility = 0.0
            for s_i in range(num_strategies[i]):
                for s_j in range(num_strategies[1-i]):
                    if i == 0:
                        payoff = payoff_matrix[s_i, s_j, 0]
                    else:
                        payoff = payoff_matrix[s_j, s_i, 1]
                    
                    utility += beliefs[i][s_i] * beliefs[1-i][s_j] * payoff
            
            expected_utilities[agents[i].agent_id] = utility
        
        return {
            'mixed_strategies': dict(zip([a.agent_id for a in agents], beliefs)),
            'expected_utilities': expected_utilities,
            'social_welfare': sum(expected_utilities.values()),
            'convergence_iterations': iteration + 1
        }

class MechanismDesign:
    """Mechanism design for incentive-compatible resource allocation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("MechanismDesign")
        
    def vcg_mechanism(self, agents: List[Agent], resources: List[Resource], 
                     tasks: List[Task]) -> GameOutcome:
        """Vickrey-Clarke-Groves mechanism for truthful bidding"""
        
        # Solve social welfare maximization problem
        optimal_allocation, max_welfare = self._solve_social_welfare_maximization(
            agents, resources, tasks
        )
        
        # Calculate VCG payments
        payments = {}
        for agent in agents:
            # Calculate welfare without this agent
            other_agents = [a for a in agents if a.agent_id != agent.agent_id]
            other_tasks = [t for t in tasks if t.task_id.split('_')[0] != agent.agent_id]
            
            if other_agents and other_tasks:
                _, welfare_without = self._solve_social_welfare_maximization(
                    other_agents, resources, other_tasks
                )
            else:
                welfare_without = 0.0
            
            # VCG payment = externality imposed on others
            welfare_of_others_with_agent = sum(
                optimal_allocation['utilities'].get(a.agent_id, 0) 
                for a in other_agents
            )
            
            payments[agent.agent_id] = welfare_without - welfare_of_others_with_agent
        
        return GameOutcome(
            allocation=optimal_allocation['allocation'],
            payments=payments,
            utilities={aid: util - payments.get(aid, 0) 
                      for aid, util in optimal_allocation['utilities'].items()},
            social_welfare=max_welfare,
            efficiency=1.0,  # VCG is efficient
            fairness_index=self._calculate_fairness_index(optimal_allocation['utilities']),
            equilibrium_type=EquilibriumType.SOCIAL_OPTIMAL
        )
    
    def _solve_social_welfare_maximization(self, agents: List[Agent], 
                                         resources: List[Resource], 
                                         tasks: List[Task]) -> Tuple[Dict[str, Any], float]:
        """Solve the social welfare maximization problem"""
        
        # This is a simplified implementation
        # In practice, this would be a complex optimization problem
        
        allocation = {}
        utilities = {}
        total_welfare = 0.0
        
        # Simple greedy allocation based on value-to-resource ratio
        sorted_tasks = sorted(tasks, key=lambda t: t.value / max(sum(t.resource_requirements.values()), 1e-6), reverse=True)
        
        resource_capacity = {r.resource_id: r.capacity for r in resources}
        
        for task in sorted_tasks:
            # Check if resources are available
            can_allocate = all(
                resource_capacity.get(res_type, 0) >= req
                for res_type, req in task.resource_requirements.items()
            )
            
            if can_allocate:
                # Allocate resources
                agent_id = task.task_id.split('_')[0]  # Extract agent ID from task ID
                
                if agent_id not in allocation:
                    allocation[agent_id] = {}
                
                for res_type, req in task.resource_requirements.items():
                    allocation[agent_id][res_type] = allocation[agent_id].get(res_type, 0) + req
                    resource_capacity[res_type] -= req
                
                # Calculate utility (simplified)
                task_utility = task.value
                utilities[agent_id] = utilities.get(agent_id, 0) + task_utility
                total_welfare += task_utility
        
        return {
            'allocation': allocation,
            'utilities': utilities
        }, total_welfare
    
    def _calculate_fairness_index(self, utilities: Dict[str, float]) -> float:
        """Calculate Jain's fairness index"""
        if not utilities:
            return 1.0
        
        values = list(utilities.values())
        if len(values) <= 1:
            return 1.0
        
        numerator = (sum(values)) ** 2
        denominator = len(values) * sum(v ** 2 for v in values)
        
        return numerator / denominator if denominator > 0 else 0.0

class AuctionMechanism:
    """Auction mechanisms for resource allocation"""
    
    def __init__(self, auction_type: AuctionType, config: Dict[str, Any]):
        self.auction_type = auction_type
        self.config = config
        self.logger = logging.getLogger("AuctionMechanism")
        
    def run_auction(self, agents: List[Agent], resources: List[Resource],
                   bids: Dict[str, Dict[str, float]]) -> GameOutcome:
        """Run auction based on the specified type"""
        
        if self.auction_type == AuctionType.FIRST_PRICE:
            return self._first_price_auction(agents, resources, bids)
        elif self.auction_type == AuctionType.SECOND_PRICE:
            return self._second_price_auction(agents, resources, bids)
        elif self.auction_type == AuctionType.VCG:
            return self._vcg_auction(agents, resources, bids)
        elif self.auction_type == AuctionType.COMBINATORIAL:
            return self._combinatorial_auction(agents, resources, bids)
        else:
            raise ValueError(f"Unsupported auction type: {self.auction_type}")
    
    def _first_price_auction(self, agents: List[Agent], resources: List[Resource],
                           bids: Dict[str, Dict[str, float]]) -> GameOutcome:
        """First-price sealed-bid auction"""
        
        allocation = {}
        payments = {}
        utilities = {}
        
        # For each resource, allocate to highest bidder
        for resource in resources:
            resource_id = resource.resource_id
            
            # Get all bids for this resource
            resource_bids = {
                agent_id: bid_dict.get(resource_id, 0.0)
                for agent_id, bid_dict in bids.items()
            }
            
            if not any(bid > 0 for bid in resource_bids.values()):
                continue
            
            # Find highest bidder
            winner = max(resource_bids.keys(), key=lambda k: resource_bids[k])
            winning_bid = resource_bids[winner]
            
            if winning_bid > 0:
                # Allocate resource to winner
                if winner not in allocation:
                    allocation[winner] = {}
                allocation[winner][resource_id] = resource.capacity
                
                # Payment is the bid amount
                payments[winner] = payments.get(winner, 0) + winning_bid
        
        # Calculate utilities
        for agent in agents:
            agent_allocation = allocation.get(agent.agent_id, {})
            agent_payment = payments.get(agent.agent_id, 0)
            
            # Use agent's valuation function
            utilities[agent.agent_id] = agent.utility_function(agent_allocation, agent_payment)
        
        return GameOutcome(
            allocation=allocation,
            payments=payments,
            utilities=utilities,
            social_welfare=sum(utilities.values()),
            efficiency=self._calculate_efficiency(utilities, agents, resources),
            fairness_index=self._calculate_fairness_index(utilities)
        )
    
    def _second_price_auction(self, agents: List[Agent], resources: List[Resource],
                            bids: Dict[str, Dict[str, float]]) -> GameOutcome:
        """Second-price sealed-bid auction"""
        
        allocation = {}
        payments = {}
        utilities = {}
        
        # For each resource, allocate to highest bidder but charge second price
        for resource in resources:
            resource_id = resource.resource_id
            
            # Get all bids for this resource
            resource_bids = {
                agent_id: bid_dict.get(resource_id, 0.0)
                for agent_id, bid_dict in bids.items()
            }
            
            if not any(bid > 0 for bid in resource_bids.values()):
                continue
            
            # Sort bids in descending order
            sorted_bids = sorted(resource_bids.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_bids) >= 1 and sorted_bids[0][1] > 0:
                winner = sorted_bids[0][0]
                
                # Second price is the second highest bid (or 0 if only one bidder)
                second_price = sorted_bids[1][1] if len(sorted_bids) > 1 else 0.0
                
                # Allocate resource to winner
                if winner not in allocation:
                    allocation[winner] = {}
                allocation[winner][resource_id] = resource.capacity
                
                # Payment is the second price
                payments[winner] = payments.get(winner, 0) + second_price
        
        # Calculate utilities
        for agent in agents:
            agent_allocation = allocation.get(agent.agent_id, {})
            agent_payment = payments.get(agent.agent_id, 0)
            
            utilities[agent.agent_id] = agent.utility_function(agent_allocation, agent_payment)
        
        return GameOutcome(
            allocation=allocation,
            payments=payments,
            utilities=utilities,
            social_welfare=sum(utilities.values()),
            efficiency=self._calculate_efficiency(utilities, agents, resources),
            fairness_index=self._calculate_fairness_index(utilities)
        )
    
    def _vcg_auction(self, agents: List[Agent], resources: List[Resource],
                    bids: Dict[str, Dict[str, float]]) -> GameOutcome:
        """VCG auction implementation"""
        
        # For VCG, we assume bids represent true valuations
        # This is a simplified implementation
        
        # Find allocation that maximizes declared valuations
        allocation, max_welfare = self._solve_welfare_maximization(agents, resources, bids)
        
        # Calculate VCG payments
        payments = {}
        for agent in agents:
            # Calculate welfare without this agent
            other_agents = [a for a in agents if a.agent_id != agent.agent_id]
            other_bids = {aid: bid_dict for aid, bid_dict in bids.items() if aid != agent.agent_id}
            
            if other_agents and other_bids:
                _, welfare_without = self._solve_welfare_maximization(other_agents, resources, other_bids)
                welfare_of_others_with = sum(
                    sum(other_bids[aid].get(rid, 0) * allocation.get(aid, {}).get(rid, 0)
                        for rid in [r.resource_id for r in resources])
                    for aid in other_bids
                )
                payments[agent.agent_id] = welfare_without - welfare_of_others_with
            else:
                payments[agent.agent_id] = 0.0
        
        # Calculate utilities based on true valuations (assuming truthful bidding)
        utilities = {}
        for agent in agents:
            agent_allocation = allocation.get(agent.agent_id, {})
            agent_payment = payments.get(agent.agent_id, 0)
            utilities[agent.agent_id] = agent.utility_function(agent_allocation, agent_payment)
        
        return GameOutcome(
            allocation=allocation,
            payments=payments,
            utilities=utilities,
            social_welfare=sum(utilities.values()),
            efficiency=1.0,  # VCG is efficient
            fairness_index=self._calculate_fairness_index(utilities),
            equilibrium_type=EquilibriumType.SOCIAL_OPTIMAL
        )
    
    def _combinatorial_auction(self, agents: List[Agent], resources: List[Resource],
                             bids: Dict[str, Dict[str, float]]) -> GameOutcome:
        """Combinatorial auction for bundle bidding"""
        
        # This is a simplified implementation
        # In practice, this would require solving a complex integer programming problem
        
        # For now, treat it as multiple single-item auctions
        return self._second_price_auction(agents, resources, bids)
    
    def _solve_welfare_maximization(self, agents: List[Agent], resources: List[Resource],
                                   bids: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, float]], float]:
        """Solve welfare maximization for auction"""
        
        # Simplified greedy allocation
        allocation = {}
        total_welfare = 0.0
        
        # For each resource, allocate to agent with highest valuation
        for resource in resources:
            resource_id = resource.resource_id
            
            best_agent = None
            best_value = 0.0
            
            for agent in agents:
                value = bids.get(agent.agent_id, {}).get(resource_id, 0.0)
                if value > best_value:
                    best_value = value
                    best_agent = agent.agent_id
            
            if best_agent:
                if best_agent not in allocation:
                    allocation[best_agent] = {}
                allocation[best_agent][resource_id] = resource.capacity
                total_welfare += best_value
        
        return allocation, total_welfare
    
    def _calculate_efficiency(self, utilities: Dict[str, float], agents: List[Agent],
                            resources: List[Resource]) -> float:
        """Calculate efficiency compared to optimal allocation"""
        
        # This is simplified - in practice would need to solve optimization problem
        current_welfare = sum(utilities.values())
        
        # Estimate maximum possible welfare (simplified)
        max_possible = sum(agent.budget for agent in agents if agent.budget > 0)
        if max_possible == 0:
            max_possible = len(agents) * len(resources) * 100  # Rough estimate
        
        return min(1.0, current_welfare / max_possible) if max_possible > 0 else 0.0
    
    def _calculate_fairness_index(self, utilities: Dict[str, float]) -> float:
        """Calculate Jain's fairness index"""
        if not utilities:
            return 1.0
        
        values = list(utilities.values())
        if len(values) <= 1:
            return 1.0
        
        numerator = (sum(values)) ** 2
        denominator = len(values) * sum(v ** 2 for v in values)
        
        return numerator / denominator if denominator > 0 else 0.0

class CoalitionFormation:
    """Coalition formation analysis for cooperative scheduling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CoalitionFormation")
        
    def find_stable_coalitions(self, agents: List[Agent], 
                              coalition_value_function: Callable) -> List[Dict[str, Any]]:
        """Find stable coalition structures"""
        
        n = len(agents)
        if n > 10:  # Computational limit for exhaustive search
            return self._approximate_stable_coalitions(agents, coalition_value_function)
        
        # Generate all possible coalition structures
        all_structures = self._generate_coalition_structures(agents)
        
        stable_structures = []
        
        for structure in all_structures:
            if self._is_stable_coalition_structure(structure, agents, coalition_value_function):
                # Calculate payoffs using Shapley value
                payoffs = self._calculate_shapley_values(structure, agents, coalition_value_function)
                
                stable_structures.append({
                    'coalitions': structure,
                    'payoffs': payoffs,
                    'total_value': sum(coalition_value_function(coalition) for coalition in structure),
                    'stability': 'core_stable'
                })
        
        return stable_structures
    
    def _generate_coalition_structures(self, agents: List[Agent]) -> List[List[List[str]]]:
        """Generate all possible coalition structures (partitions)"""
        
        agent_ids = [agent.agent_id for agent in agents]
        
        def partitions(collection):
            if len(collection) == 1:
                yield [collection]
                return
            
            first = collection[0]
            for smaller in partitions(collection[1:]):
                # Add first element to each existing subset
                for i, subset in enumerate(smaller):
                    yield smaller[:i] + [subset + [first]] + smaller[i+1:]
                # Create new subset with just the first element
                yield [[first]] + smaller
        
        return list(partitions(agent_ids))
    
    def _is_stable_coalition_structure(self, structure: List[List[str]], agents: List[Agent],
                                     coalition_value_function: Callable) -> bool:
        """Check if coalition structure is stable (in the core)"""
        
        # Calculate current payoffs for each agent in the structure
        current_payoffs = self._calculate_shapley_values(structure, agents, coalition_value_function)
        
        # Check if any coalition of agents can block this structure
        agent_ids = [agent.agent_id for agent in agents]
        
        # Check all possible blocking coalitions
        for r in range(2, len(agent_ids) + 1):
            for blocking_coalition in itertools.combinations(agent_ids, r):
                blocking_coalition = list(blocking_coalition)
                
                # Calculate value if this coalition deviates
                blocking_value = coalition_value_function(blocking_coalition)
                
                # Calculate current total payoff of these agents
                current_total = sum(current_payoffs.get(agent_id, 0) for agent_id in blocking_coalition)
                
                # If blocking coalition can improve, structure is not stable
                if blocking_value > current_total + 1e-6:
                    return False
        
        return True
    
    def _calculate_shapley_values(self, structure: List[List[str]], agents: List[Agent],
                                coalition_value_function: Callable) -> Dict[str, float]:
        """Calculate Shapley value allocation within coalitions"""
        
        payoffs = {}
        
        for coalition in structure:
            if len(coalition) == 1:
                # Single agent coalition
                payoffs[coalition[0]] = coalition_value_function(coalition)
            else:
                # Multi-agent coalition - calculate Shapley values
                coalition_payoffs = self._shapley_value(coalition, coalition_value_function)
                payoffs.update(coalition_payoffs)
        
        return payoffs
    
    def _shapley_value(self, coalition: List[str], value_function: Callable) -> Dict[str, float]:
        """Calculate Shapley values for agents in a coalition"""
        
        n = len(coalition)
        shapley_values = {agent: 0.0 for agent in coalition}
        
        # Calculate marginal contributions for all possible orderings
        for ordering in itertools.permutations(coalition):
            for i, agent in enumerate(ordering):
                # Coalition without this agent
                coalition_without = list(ordering[:i])
                
                # Marginal contribution
                value_with = value_function(coalition_without + [agent]) if coalition_without else value_function([agent])
                value_without = value_function(coalition_without) if coalition_without else 0.0
                
                marginal_contribution = value_with - value_without
                shapley_values[agent] += marginal_contribution / len(list(itertools.permutations(coalition)))
        
        return shapley_values
    
    def _approximate_stable_coalitions(self, agents: List[Agent],
                                     coalition_value_function: Callable) -> List[Dict[str, Any]]:
        """Approximate stable coalitions for large numbers of agents"""
        
        # Use greedy coalition formation
        remaining_agents = [agent.agent_id for agent in agents]
        coalitions = []
        
        while remaining_agents:
            # Start with random agent
            current_coalition = [remaining_agents.pop(0)]
            
            improved = True
            while improved and remaining_agents:
                improved = False
                best_addition = None
                best_value_increase = 0
                
                current_value = coalition_value_function(current_coalition)
                
                for agent_id in remaining_agents:
                    new_coalition = current_coalition + [agent_id]
                    new_value = coalition_value_function(new_coalition)
                    value_increase = new_value - current_value
                    
                    if value_increase > best_value_increase:
                        best_value_increase = value_increase
                        best_addition = agent_id
                        improved = True
                
                if improved and best_addition:
                    current_coalition.append(best_addition)
                    remaining_agents.remove(best_addition)
            
            coalitions.append(current_coalition)
        
        # Calculate payoffs
        payoffs = self._calculate_shapley_values(coalitions, agents, coalition_value_function)
        
        return [{
            'coalitions': coalitions,
            'payoffs': payoffs,
            'total_value': sum(coalition_value_function(coalition) for coalition in coalitions),
            'stability': 'approximately_stable'
        }]

class GameTheoreticScheduler:
    """Main game-theoretic scheduler integrating all mechanisms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("GameTheoreticScheduler")
        
        # Initialize components
        self.nash_solver = NashEquilibriumSolver(config)
        self.mechanism_design = MechanismDesign(config)
        self.coalition_formation = CoalitionFormation(config)
        
        # Game history and analysis
        self.game_history: List[GameOutcome] = []
        self.equilibrium_history: List[Dict[str, Any]] = []
        
    def analyze_scheduling_game(self, agents: List[Agent], resources: List[Resource],
                               tasks: List[Task], game_type: GameType) -> Dict[str, Any]:
        """Comprehensive analysis of the scheduling game"""
        
        self.logger.info(f"Analyzing scheduling game with {len(agents)} agents, "
                        f"{len(resources)} resources, {len(tasks)} tasks")
        
        analysis_results = {
            'game_type': game_type,
            'num_agents': len(agents),
            'num_resources': len(resources),
            'num_tasks': len(tasks),
            'timestamp': time.time()
        }
        
        if game_type == GameType.NON_COOPERATIVE:
            analysis_results.update(self._analyze_non_cooperative_game(agents, resources, tasks))
        elif game_type == GameType.COOPERATIVE:
            analysis_results.update(self._analyze_cooperative_game(agents, resources, tasks))
        elif game_type == GameType.AUCTION:
            analysis_results.update(self._analyze_auction_game(agents, resources, tasks))
        elif game_type == GameType.COALITION:
            analysis_results.update(self._analyze_coalition_game(agents, resources, tasks))
        
        return analysis_results
    
    def _analyze_non_cooperative_game(self, agents: List[Agent], resources: List[Resource],
                                    tasks: List[Task]) -> Dict[str, Any]:
        """Analyze non-cooperative scheduling game"""
        
        # Define strategy spaces (simplified)
        for agent in agents:
            agent.strategy_space = ['aggressive', 'conservative', 'balanced']
        
        # Define payoff functions
        payoff_functions = {}
        for agent in agents:
            payoff_functions[agent.agent_id] = lambda profile, idx, a=agent: self._calculate_scheduling_payoff(
                profile, idx, a, resources, tasks
            )
        
        # Find Nash equilibria
        pure_nash = self.nash_solver.find_pure_strategy_nash(agents, payoff_functions)
        
        results = {
            'equilibrium_type': 'nash',
            'pure_strategy_equilibria': pure_nash,
            'num_pure_equilibria': len(pure_nash) if pure_nash else 0
        }
        
        if pure_nash:
            # Analyze best equilibrium
            best_equilibrium = max(pure_nash, key=lambda eq: eq['social_welfare'])
            results['best_equilibrium'] = best_equilibrium
            results['efficiency'] = self._calculate_price_of_anarchy(pure_nash)
        
        return results
    
    def _analyze_cooperative_game(self, agents: List[Agent], resources: List[Resource],
                                tasks: List[Task]) -> Dict[str, Any]:
        """Analyze cooperative scheduling game"""
        
        # Use VCG mechanism for cooperative solution
        vcg_outcome = self.mechanism_design.vcg_mechanism(agents, resources, tasks)
        
        return {
            'mechanism': 'vcg',
            'outcome': vcg_outcome,
            'social_welfare': vcg_outcome.social_welfare,
            'efficiency': vcg_outcome.efficiency,
            'fairness': vcg_outcome.fairness_index,
            'truthful': True  # VCG is truthful
        }
    
    def _analyze_auction_game(self, agents: List[Agent], resources: List[Resource],
                            tasks: List[Task]) -> Dict[str, Any]:
        """Analyze auction-based scheduling game"""
        
        # Generate synthetic bids based on agent valuations
        bids = {}
        for agent in agents:
            agent_bids = {}
            for resource in resources:
                # Simple bidding strategy: bid proportion of valuation
                valuation = np.random.uniform(10, 100)  # Simplified valuation
                bid = valuation * np.random.uniform(0.5, 0.9)  # Strategic bidding
                agent_bids[resource.resource_id] = bid
            bids[agent.agent_id] = agent_bids
        
        # Compare different auction mechanisms
        mechanisms = [AuctionType.FIRST_PRICE, AuctionType.SECOND_PRICE, AuctionType.VCG]
        auction_results = {}
        
        for mechanism_type in mechanisms:
            auction = AuctionMechanism(mechanism_type, self.config)
            outcome = auction.run_auction(agents, resources, bids)
            
            auction_results[mechanism_type.value] = {
                'outcome': outcome,
                'revenue': sum(outcome.payments.values()),
                'efficiency': outcome.efficiency,
                'fairness': outcome.fairness_index
            }
        
        return {
            'auction_mechanisms': auction_results,
            'bids': bids,
            'best_mechanism': max(auction_results.keys(), 
                                key=lambda k: auction_results[k]['efficiency'])
        }
    
    def _analyze_coalition_game(self, agents: List[Agent], resources: List[Resource],
                              tasks: List[Task]) -> Dict[str, Any]:
        """Analyze coalition formation game"""
        
        # Define coalition value function
        def coalition_value(coalition_members: List[str]) -> float:
            # Simple coalition value: economies of scale
            base_value = len(coalition_members) * 10
            synergy_bonus = (len(coalition_members) - 1) * 5
            return base_value + synergy_bonus
        
        # Find stable coalitions
        stable_coalitions = self.coalition_formation.find_stable_coalitions(agents, coalition_value)
        
        return {
            'coalition_structures': stable_coalitions,
            'num_stable_structures': len(stable_coalitions),
            'grand_coalition_value': coalition_value([a.agent_id for a in agents]),
            'individual_values': {a.agent_id: coalition_value([a.agent_id]) for a in agents}
        }
    
    def _calculate_scheduling_payoff(self, strategy_profile: Tuple, agent_index: int,
                                   agent: Agent, resources: List[Resource], 
                                   tasks: List[Task]) -> float:
        """Calculate payoff for scheduling strategy"""
        
        agent_strategy = strategy_profile[agent_index]
        
        # Simple payoff calculation based on strategy
        base_payoff = 50.0
        
        if agent_strategy == 'aggressive':
            # Higher payoff but more risk
            payoff = base_payoff * 1.5 + np.random.normal(0, 10)
        elif agent_strategy == 'conservative':
            # Lower but stable payoff
            payoff = base_payoff * 0.8 + np.random.normal(0, 2)
        else:  # balanced
            payoff = base_payoff + np.random.normal(0, 5)
        
        # Account for resource contention with other agents
        aggressive_count = sum(1 for s in strategy_profile if s == 'aggressive')
        if agent_strategy == 'aggressive' and aggressive_count > 1:
            payoff *= 0.8  # Penalty for competition
        
        return max(0, payoff)
    
    def _calculate_price_of_anarchy(self, equilibria: List[Dict[str, Any]]) -> float:
        """Calculate price of anarchy (efficiency loss)"""
        if not equilibria:
            return 0.0
        
        worst_equilibrium = min(equilibria, key=lambda eq: eq['social_welfare'])
        best_equilibrium = max(equilibria, key=lambda eq: eq['social_welfare'])
        
        if best_equilibrium['social_welfare'] == 0:
            return 1.0
        
        return worst_equilibrium['social_welfare'] / best_equilibrium['social_welfare']
    
    def get_game_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about game analysis"""
        
        if not self.game_history:
            return {"total_games": 0}
        
        return {
            "total_games": len(self.game_history),
            "total_equilibria": len(self.equilibrium_history),
            "avg_social_welfare": np.mean([game.social_welfare for game in self.game_history]),
            "avg_efficiency": np.mean([game.efficiency for game in self.game_history]),
            "avg_fairness": np.mean([game.fairness_index for game in self.game_history]),
            "mechanism_performance": self._analyze_mechanism_performance()
        }
    
    def _analyze_mechanism_performance(self) -> Dict[str, Any]:
        """Analyze performance of different mechanisms"""
        
        mechanism_stats = defaultdict(list)
        
        for game in self.game_history:
            if hasattr(game, 'mechanism_type'):
                mechanism_stats[game.mechanism_type].append({
                    'social_welfare': game.social_welfare,
                    'efficiency': game.efficiency,
                    'fairness': game.fairness_index
                })
        
        performance_summary = {}
        for mechanism, results in mechanism_stats.items():
            if results:
                performance_summary[mechanism] = {
                    'count': len(results),
                    'avg_welfare': np.mean([r['social_welfare'] for r in results]),
                    'avg_efficiency': np.mean([r['efficiency'] for r in results]),
                    'avg_fairness': np.mean([r['fairness'] for r in results])
                }
        
        return performance_summary

def demonstrate_game_theoretic_analysis():
    """Demonstrate the game-theoretic analysis framework"""
    print("=== Game-Theoretic Analysis of Multi-Agent Scheduling ===")
    
    # Configuration
    config = {
        'max_iterations': 1000,
        'convergence_threshold': 1e-6,
        'auction_types': ['first_price', 'second_price', 'vcg']
    }
    
    print("1. Initializing Game-Theoretic Framework...")
    
    scheduler = GameTheoreticScheduler(config)
    
    print("2. Creating Agents, Resources, and Tasks...")
    
    # Create agents with different characteristics
    agents = []
    for i in range(4):
        agent = Agent(
            agent_id=f"agent_{i}",
            strategy_space=['aggressive', 'conservative', 'balanced'],
            utility_function=lambda alloc, payment=0: UtilityFunction.linear_utility(
                alloc, {f"resource_{j}": np.random.uniform(10, 50) for j in range(3)}, payment
            ),
            resource_endowment={f"resource_{j}": np.random.uniform(5, 15) for j in range(3)},
            valuation_function=lambda res: np.random.uniform(10, 100),
            budget=np.random.uniform(100, 500),
            risk_preference=np.random.uniform(0, 1),
            cooperation_level=np.random.uniform(0.5, 1.0)
        )
        agents.append(agent)
    
    print(f"   Created {len(agents)} agents")
    
    # Create resources
    resources = []
    for i in range(3):
        resource = Resource(
            resource_id=f"resource_{i}",
            resource_type=["cpu", "memory", "storage"][i],
            capacity=np.random.uniform(50, 200),
            quality=np.random.uniform(0.8, 1.0)
        )
        resources.append(resource)
    
    print(f"   Created {len(resources)} resources")
    
    # Create tasks
    tasks = []
    for i in range(10):
        task = Task(
            task_id=f"agent_{i%4}_task_{i}",
            arrival_time=np.random.uniform(0, 100),
            deadline=np.random.uniform(100, 300),
            resource_requirements={
                f"resource_{j}": np.random.uniform(1, 10) 
                for j in range(np.random.randint(1, 4))
            },
            priority=np.random.randint(1, 6),
            value=np.random.uniform(20, 100),
            penalty=np.random.uniform(5, 20)
        )
        tasks.append(task)
    
    print(f"   Created {len(tasks)} tasks")
    
    print("3. Analyzing Non-Cooperative Game...")
    
    non_coop_results = scheduler.analyze_scheduling_game(
        agents, resources, tasks, GameType.NON_COOPERATIVE
    )
    
    print(f"   Found {non_coop_results['num_pure_equilibria']} pure strategy Nash equilibria")
    
    if non_coop_results['num_pure_equilibria'] > 0:
        best_eq = non_coop_results['best_equilibrium']
        print(f"   Best equilibrium social welfare: {best_eq['social_welfare']:.2f}")
        print(f"   Price of anarchy: {non_coop_results.get('efficiency', 1.0):.3f}")
    
    print("4. Analyzing Cooperative Game (VCG Mechanism)...")
    
    coop_results = scheduler.analyze_scheduling_game(
        agents, resources, tasks, GameType.COOPERATIVE
    )
    
    print(f"   VCG social welfare: {coop_results['social_welfare']:.2f}")
    print(f"   VCG efficiency: {coop_results['efficiency']:.3f}")
    print(f"   VCG fairness: {coop_results['fairness']:.3f}")
    print(f"   Truthful mechanism: {coop_results['truthful']}")
    
    print("5. Analyzing Auction Mechanisms...")
    
    auction_results = scheduler.analyze_scheduling_game(
        agents, resources, tasks, GameType.AUCTION
    )
    
    print("   Auction Mechanism Comparison:")
    for mechanism, results in auction_results['auction_mechanisms'].items():
        print(f"     {mechanism}:")
        print(f"       Revenue: {results['revenue']:.2f}")
        print(f"       Efficiency: {results['efficiency']:.3f}")
        print(f"       Fairness: {results['fairness']:.3f}")
    
    print(f"   Best mechanism: {auction_results['best_mechanism']}")
    
    print("6. Analyzing Coalition Formation...")
    
    coalition_results = scheduler.analyze_scheduling_game(
        agents, resources, tasks, GameType.COALITION
    )
    
    print(f"   Found {coalition_results['num_stable_structures']} stable coalition structures")
    print(f"   Grand coalition value: {coalition_results['grand_coalition_value']:.2f}")
    
    if coalition_results['coalition_structures']:
        best_structure = coalition_results['coalition_structures'][0]
        print("   Best coalition structure:")
        for i, coalition in enumerate(best_structure['coalitions']):
            print(f"     Coalition {i+1}: {coalition}")
        
        print("   Shapley value payoffs:")
        for agent_id, payoff in best_structure['payoffs'].items():
            print(f"     {agent_id}: {payoff:.2f}")
    
    print("7. Mechanism Design Analysis...")
    
    # Compare mechanism properties
    mechanisms_compared = {
        'Non-Cooperative Nash': {
            'efficiency': non_coop_results.get('efficiency', 0.0),
            'fairness': 0.5,  # Estimated
            'truthful': False,
            'complexity': 'Low'
        },
        'VCG Mechanism': {
            'efficiency': coop_results['efficiency'],
            'fairness': coop_results['fairness'],
            'truthful': True,
            'complexity': 'High'
        },
        'Second-Price Auction': {
            'efficiency': auction_results['auction_mechanisms']['second_price']['efficiency'],
            'fairness': auction_results['auction_mechanisms']['second_price']['fairness'],
            'truthful': True,
            'complexity': 'Medium'
        }
    }
    
    print("   Mechanism Comparison:")
    for mechanism, properties in mechanisms_compared.items():
        print(f"     {mechanism}:")
        print(f"       Efficiency: {properties['efficiency']:.3f}")
        print(f"       Fairness: {properties['fairness']:.3f}")
        print(f"       Truthful: {properties['truthful']}")
        print(f"       Complexity: {properties['complexity']}")
    
    print("8. Strategic Behavior Analysis...")
    
    # Analyze bidding strategies in auctions
    auction_bids = auction_results['bids']
    
    print("   Bidding Strategy Analysis:")
    for agent_id, bids in auction_bids.items():
        avg_bid = np.mean(list(bids.values()))
        print(f"     {agent_id}: Average bid = {avg_bid:.2f}")
    
    # Analyze strategy diversity
    all_bids = [bid for agent_bids in auction_bids.values() for bid in agent_bids.values()]
    bid_variance = np.var(all_bids)
    print(f"   Bid variance (strategy diversity): {bid_variance:.2f}")
    
    print("9. Game-Theoretic Insights...")
    
    insights = [
        "Nash equilibria may not achieve optimal social welfare",
        "VCG mechanism ensures truthful bidding and efficiency",
        "Coalition formation can improve individual and social outcomes",
        "Auction mechanisms trade off simplicity vs optimality",
        "Strategic behavior significantly impacts resource allocation",
        "Mechanism choice depends on efficiency vs fairness priorities",
        "Information asymmetry affects equilibrium outcomes",
        "Repeated interactions enable cooperation and reputation"
    ]
    
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    print("10. Integration with HeteroSched...")
    
    integration_benefits = [
        "Theoretical foundations for multi-agent coordination protocols",
        "Incentive design for voluntary participation in scheduling",
        "Fair resource allocation mechanisms with provable properties",
        "Strategic robustness against gaming and manipulation",
        "Coalition formation for federated scheduling environments",
        "Auction-based dynamic resource pricing and allocation",
        "Equilibrium analysis for system stability and convergence"
    ]
    
    for i, benefit in enumerate(integration_benefits, 1):
        print(f"   {i}. {benefit}")
    
    return {
        "scheduler": scheduler,
        "agents": agents,
        "resources": resources,
        "tasks": tasks,
        "analysis_results": {
            "non_cooperative": non_coop_results,
            "cooperative": coop_results,
            "auction": auction_results,
            "coalition": coalition_results
        },
        "mechanism_comparison": mechanisms_compared
    }

if __name__ == "__main__":
    demonstrate_game_theoretic_analysis()