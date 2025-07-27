"""
R44: Tutorial and Educational Materials for RL Scheduling

This module creates comprehensive tutorial and educational materials for learning
reinforcement learning approaches to heterogeneous scheduling. It includes 
interactive tutorials, code examples, exercises, and assessment materials
suitable for students, researchers, and practitioners.
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import zipfile
import warnings
warnings.filterwarnings('ignore')


class TutorialLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearningObjective(Enum):
    CONCEPTS = "understanding_concepts"
    IMPLEMENTATION = "hands_on_implementation"
    ANALYSIS = "performance_analysis"
    RESEARCH = "research_methods"


@dataclass
class TutorialModule:
    """Represents a tutorial module"""
    title: str
    level: TutorialLevel
    objectives: List[LearningObjective]
    prerequisites: List[str]
    duration_minutes: int
    description: str
    content_sections: List[str]
    exercises: List[str]
    code_examples: List[str]


@dataclass
class LearningPath:
    """Represents a structured learning path"""
    name: str
    description: str
    target_audience: str
    modules: List[str]
    estimated_hours: int
    prerequisites: List[str]


class EducationalMaterialsGenerator:
    """Generates comprehensive educational materials for RL scheduling"""
    
    def __init__(self, output_dir: str = "tutorial_materials"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tutorial structure
        self.modules = self._create_tutorial_modules()
        self.learning_paths = self._create_learning_paths()
        
    def _create_tutorial_modules(self) -> Dict[str, TutorialModule]:
        """Create comprehensive tutorial modules"""
        modules = {}
        
        # Module 1: Introduction to RL and Scheduling
        modules["intro_rl_scheduling"] = TutorialModule(
            title="Introduction to RL and Scheduling",
            level=TutorialLevel.BEGINNER,
            objectives=[LearningObjective.CONCEPTS],
            prerequisites=["Basic programming knowledge", "Linear algebra"],
            duration_minutes=90,
            description="Introduction to reinforcement learning concepts and scheduling problems",
            content_sections=[
                "What is Reinforcement Learning?",
                "Scheduling Problems Overview",
                "Why RL for Scheduling?",
                "Key Challenges and Opportunities"
            ],
            exercises=[
                "Identify RL components in a simple scheduling scenario",
                "Compare heuristic vs. RL approaches",
                "Design a basic reward function"
            ],
            code_examples=[
                "simple_mdp_example.py",
                "basic_scheduler_comparison.py"
            ]
        )
        
        # Module 2: MDP Formulation for Scheduling
        modules["mdp_formulation"] = TutorialModule(
            title="MDP Formulation for Scheduling Problems",
            level=TutorialLevel.INTERMEDIATE,
            objectives=[LearningObjective.CONCEPTS, LearningObjective.IMPLEMENTATION],
            prerequisites=["intro_rl_scheduling", "Probability theory"],
            duration_minutes=120,
            description="Learn to formulate scheduling problems as Markov Decision Processes",
            content_sections=[
                "MDP Components: States, Actions, Rewards",
                "State Space Design for Scheduling",
                "Action Space Considerations",
                "Reward Function Engineering",
                "Transition Dynamics Modeling"
            ],
            exercises=[
                "Formulate job scheduling as MDP",
                "Design state representation for cloud scheduling",
                "Implement reward function for multi-objective scheduling"
            ],
            code_examples=[
                "job_scheduling_mdp.py",
                "cloud_scheduling_formulation.py",
                "multi_objective_rewards.py"
            ]
        )
        
        # Module 3: Deep RL Algorithms for Scheduling
        modules["deep_rl_algorithms"] = TutorialModule(
            title="Deep RL Algorithms for Scheduling",
            level=TutorialLevel.INTERMEDIATE,
            objectives=[LearningObjective.IMPLEMENTATION, LearningObjective.ANALYSIS],
            prerequisites=["mdp_formulation", "Neural networks", "PyTorch or TensorFlow"],
            duration_minutes=180,
            description="Implement and compare deep RL algorithms for scheduling tasks",
            content_sections=[
                "DQN for Discrete Scheduling Decisions",
                "Policy Gradient Methods",
                "Actor-Critic Architectures",
                "Continuous Control for Resource Allocation",
                "Algorithm Selection Guidelines"
            ],
            exercises=[
                "Implement DQN for job scheduling",
                "Train PPO agent for resource allocation",
                "Compare different RL algorithms on same problem"
            ],
            code_examples=[
                "dqn_job_scheduler.py",
                "ppo_resource_allocator.py",
                "algorithm_comparison.py"
            ]
        )
        
        # Module 4: Multi-Agent RL for Distributed Scheduling
        modules["multi_agent_rl"] = TutorialModule(
            title="Multi-Agent RL for Distributed Scheduling",
            level=TutorialLevel.ADVANCED,
            objectives=[LearningObjective.IMPLEMENTATION, LearningObjective.RESEARCH],
            prerequisites=["deep_rl_algorithms", "Distributed systems"],
            duration_minutes=150,
            description="Design and implement multi-agent systems for distributed scheduling",
            content_sections=[
                "Multi-Agent RL Fundamentals",
                "Coordination Mechanisms",
                "Communication Protocols",
                "Hierarchical Multi-Agent Systems",
                "Scalability Considerations"
            ],
            exercises=[
                "Implement independent learning agents",
                "Design communication protocol",
                "Compare centralized vs. decentralized approaches"
            ],
            code_examples=[
                "independent_learning_agents.py",
                "coordinated_multi_agent.py",
                "hierarchical_scheduling.py"
            ]
        )
        
        # Module 5: Advanced Topics and Research Frontiers
        modules["advanced_topics"] = TutorialModule(
            title="Advanced Topics and Research Frontiers",
            level=TutorialLevel.EXPERT,
            objectives=[LearningObjective.RESEARCH, LearningObjective.ANALYSIS],
            prerequisites=["multi_agent_rl", "Research experience"],
            duration_minutes=240,
            description="Explore cutting-edge research in RL for scheduling",
            content_sections=[
                "Meta-Learning for Adaptation",
                "Transfer Learning Across Domains",
                "Causal Inference and Interpretability",
                "Quantum-Inspired Optimization",
                "Future Research Directions"
            ],
            exercises=[
                "Implement meta-learning algorithm",
                "Design transfer learning experiment",
                "Analyze policy interpretability"
            ],
            code_examples=[
                "meta_learning_scheduler.py",
                "transfer_learning_demo.py",
                "interpretability_analysis.py"
            ]
        )
        
        return modules
        
    def _create_learning_paths(self) -> Dict[str, LearningPath]:
        """Create structured learning paths for different audiences"""
        paths = {}
        
        # Path for Computer Science Students
        paths["cs_student"] = LearningPath(
            name="Computer Science Student Path",
            description="Comprehensive introduction to RL scheduling for CS students",
            target_audience="Undergraduate/Graduate CS students",
            modules=[
                "intro_rl_scheduling",
                "mdp_formulation", 
                "deep_rl_algorithms",
                "multi_agent_rl"
            ],
            estimated_hours=12,
            prerequisites=["Data structures", "Algorithms", "Machine learning basics"]
        )
        
        # Path for Industry Practitioners
        paths["industry_practitioner"] = LearningPath(
            name="Industry Practitioner Path",
            description="Practical RL scheduling for industry applications",
            target_audience="Software engineers and system architects",
            modules=[
                "intro_rl_scheduling",
                "mdp_formulation",
                "deep_rl_algorithms"
            ],
            estimated_hours=8,
            prerequisites=["Programming experience", "System design knowledge"]
        )
        
        # Path for Researchers
        paths["researcher"] = LearningPath(
            name="Research Path",
            description="Advanced RL scheduling for researchers",
            target_audience="PhD students and research scientists",
            modules=[
                "intro_rl_scheduling",
                "mdp_formulation",
                "deep_rl_algorithms", 
                "multi_agent_rl",
                "advanced_topics"
            ],
            estimated_hours=16,
            prerequisites=["Advanced mathematics", "Research experience", "RL background"]
        )
        
        return paths
        
    def generate_module_content(self, module_name: str) -> Dict[str, str]:
        """Generate complete content for a tutorial module"""
        module = self.modules[module_name]
        content = {}
        
        # Generate main tutorial content
        content["tutorial"] = self._generate_tutorial_markdown(module)
        
        # Generate code examples
        for example in module.code_examples:
            content[example] = self._generate_code_example(example, module)
            
        # Generate exercises
        content["exercises.md"] = self._generate_exercises(module)
        
        # Generate solutions
        content["solutions.md"] = self._generate_solutions(module)
        
        return content
        
    def _generate_tutorial_markdown(self, module: TutorialModule) -> str:
        """Generate tutorial content in markdown format"""
        content = f"""# {module.title}

## Overview

**Level:** {module.level.value.title()}  
**Duration:** {module.duration_minutes} minutes  
**Prerequisites:** {', '.join(module.prerequisites)}

{module.description}

## Learning Objectives

By completing this module, you will be able to:
{chr(10).join(f"- {obj.value.replace('_', ' ').title()}" for obj in module.objectives)}

## Prerequisites Check

Before starting this module, ensure you have:
{chr(10).join(f"- [ ] {prereq}" for prereq in module.prerequisites)}

## Content Sections

"""
        
        # Generate content for each section
        for i, section in enumerate(module.content_sections, 1):
            content += f"\n### {i}. {section}\n\n"
            content += self._generate_section_content(section, module)
            
        # Add exercises section
        content += f"""
## Hands-On Exercises

Complete the following exercises to reinforce your learning:

{chr(10).join(f"{i}. {exercise}" for i, exercise in enumerate(module.exercises, 1))}

## Code Examples

This module includes the following code examples:
{chr(10).join(f"- `{example}`: {self._get_example_description(example)}" for example in module.code_examples)}

## Assessment

Test your understanding with the quiz and practical assignments provided in the exercises section.

## Next Steps

After completing this module, you should:
1. Review the key concepts covered
2. Complete all exercises and verify your solutions
3. Experiment with the provided code examples
4. Proceed to the next module in your learning path

## Additional Resources

- Research papers on the topic
- Online RL courses and tutorials  
- Community forums and discussion groups
- Implementation repositories and benchmarks

---

*This tutorial is part of the HeteroSched educational materials suite.*
"""
        return content
        
    def _generate_section_content(self, section: str, module: TutorialModule) -> str:
        """Generate content for a specific section"""
        if "Introduction" in section or "What is" in section:
            return """
This section introduces the fundamental concepts and provides the necessary background
for understanding the material. We'll cover key definitions, historical context,
and practical motivations.

Key points to remember:
- Definitions of core concepts
- Historical development and current state
- Practical applications and use cases
- Common challenges and limitations

**Interactive Example:** Try the accompanying code example to see these concepts in action.
"""
        elif "Formulation" in section or "Design" in section:
            return """
In this section, we focus on the practical aspects of problem formulation and system design.
You'll learn systematic approaches to modeling real-world problems.

Design considerations:
- Problem decomposition strategies
- State and action space modeling
- Performance metrics and objectives
- Implementation constraints

**Hands-on Activity:** Complete the associated exercise to practice these design skills.
"""
        elif "Algorithms" in section or "Methods" in section:
            return """
This section covers algorithmic approaches and implementation details. We'll examine
different methods, their strengths and weaknesses, and when to use each approach.

Algorithm topics:
- Core algorithmic principles
- Implementation considerations  
- Performance characteristics
- Comparative analysis

**Programming Exercise:** Implement the discussed algorithms using the provided templates.
"""
        else:
            return """
This section explores advanced concepts and current research directions in the field.
Understanding these topics will prepare you for cutting-edge applications.

Advanced topics:
- Recent research developments
- Open challenges and opportunities
- Future research directions
- Practical deployment considerations

**Research Activity:** Explore the latest papers and implementations in this area.
"""
            
    def _get_example_description(self, example: str) -> str:
        """Get description for code example"""
        descriptions = {
            "simple_mdp_example.py": "Basic MDP implementation for scheduling",
            "basic_scheduler_comparison.py": "Comparison of heuristic and RL schedulers",
            "job_scheduling_mdp.py": "Complete job scheduling MDP formulation",
            "cloud_scheduling_formulation.py": "Cloud resource scheduling as MDP",
            "multi_objective_rewards.py": "Multi-objective reward function design",
            "dqn_job_scheduler.py": "DQN implementation for job scheduling",
            "ppo_resource_allocator.py": "PPO agent for resource allocation",
            "algorithm_comparison.py": "Comparison framework for RL algorithms",
            "independent_learning_agents.py": "Independent multi-agent learning",
            "coordinated_multi_agent.py": "Coordinated multi-agent system",
            "hierarchical_scheduling.py": "Hierarchical multi-agent scheduling",
            "meta_learning_scheduler.py": "Meta-learning for scheduler adaptation",
            "transfer_learning_demo.py": "Transfer learning across scheduling domains",
            "interpretability_analysis.py": "Policy interpretability analysis tools"
        }
        return descriptions.get(example, "Code example and implementation")
        
    def _generate_code_example(self, example_name: str, module: TutorialModule) -> str:
        """Generate code example content"""
        if "simple_mdp" in example_name:
            return '''"""
Simple MDP Example for Scheduling
Educational tutorial code for understanding MDP basics
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

class SimpleMDP:
    """
    Simple MDP for job scheduling to demonstrate basic concepts
    
    State: Number of jobs in queue (0-5)
    Actions: Processing speed (slow=0, fast=1) 
    Reward: Negative queue length (prefer empty queue)
    """
    
    def __init__(self):
        self.max_queue_size = 5
        self.states = list(range(self.max_queue_size + 1))  # 0 to 5 jobs
        self.actions = [0, 1]  # 0=slow, 1=fast processing
        
        # Transition probabilities
        self.transitions = self._build_transition_matrix()
        
        # Rewards: negative queue length (want fewer jobs queued)
        self.rewards = {state: -state for state in self.states}
        
    def _build_transition_matrix(self) -> Dict[Tuple[int, int], Dict[int, float]]:
        """Build transition probability matrix P(s'|s,a)"""
        transitions = {}
        
        for state in self.states:
            for action in self.actions:
                transitions[(state, action)] = {}
                
                # Job arrival probability (Poisson-like)
                arrival_prob = 0.3
                
                # Processing probability depends on action
                process_prob = 0.5 if action == 0 else 0.8  # slow vs fast
                
                # Calculate next state probabilities
                for next_state in self.states:
                    prob = 0.0
                    
                    if state == 0:  # Empty queue
                        if next_state == 0:
                            prob = 1 - arrival_prob  # No arrival
                        elif next_state == 1:
                            prob = arrival_prob  # One arrival
                    else:  # Jobs in queue
                        # Job processed
                        if state > 0 and next_state == state - 1:
                            prob += process_prob * (1 - arrival_prob)
                        # Job processed, new arrival
                        if next_state == state:
                            prob += process_prob * arrival_prob
                        # No processing, new arrival
                        if next_state == min(state + 1, self.max_queue_size):
                            prob += (1 - process_prob) * arrival_prob
                        # No processing, no arrival
                        if next_state == state:
                            prob += (1 - process_prob) * (1 - arrival_prob)
                            
                    transitions[(state, action)][next_state] = prob
                    
        return transitions
        
    def value_iteration(self, gamma: float = 0.9, theta: float = 1e-6) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        Solve MDP using value iteration
        
        Returns:
            values: Optimal value function V*(s)
            policy: Optimal policy π*(s)
        """
        # Initialize value function
        V = {s: 0.0 for s in self.states}
        
        iteration = 0
        while True:
            delta = 0
            old_V = V.copy()
            
            # Update each state
            for s in self.states:
                # Compute action values Q(s,a)
                action_values = []
                for a in self.actions:
                    q_value = self.rewards[s]  # Immediate reward
                    
                    # Add discounted future value
                    for s_next in self.states:
                        prob = self.transitions[(s, a)].get(s_next, 0)
                        q_value += gamma * prob * old_V[s_next]
                        
                    action_values.append(q_value)
                
                # Bellman update
                V[s] = max(action_values)
                delta = max(delta, abs(V[s] - old_V[s]))
                
            iteration += 1
            if delta < theta:
                break
                
        # Extract optimal policy
        policy = {}
        for s in self.states:
            action_values = []
            for a in self.actions:
                q_value = self.rewards[s]
                for s_next in self.states:
                    prob = self.transitions[(s, a)].get(s_next, 0)
                    q_value += gamma * prob * V[s_next]
                action_values.append(q_value)
            policy[s] = self.actions[np.argmax(action_values)]
            
        print(f"Value iteration converged in {iteration} iterations")
        return V, policy
        
    def simulate_episode(self, policy: Dict[int, int], max_steps: int = 100) -> List[Tuple[int, int, float]]:
        """Simulate one episode using given policy"""
        trajectory = []
        state = np.random.choice(self.states)
        
        for step in range(max_steps):
            action = policy[state]
            reward = self.rewards[state]
            
            trajectory.append((state, action, reward))
            
            # Sample next state
            next_state_probs = self.transitions[(state, action)]
            next_states = list(next_state_probs.keys())
            probs = list(next_state_probs.values())
            state = np.random.choice(next_states, p=probs)
            
        return trajectory
        
    def visualize_policy(self, policy: Dict[int, int]):
        """Visualize the optimal policy"""
        states = list(policy.keys())
        actions = [policy[s] for s in states]
        action_labels = ['Slow', 'Fast']
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if a == 0 else 'green' for a in actions]
        bars = plt.bar(states, [1]*len(states), color=colors, alpha=0.7)
        
        # Add action labels
        for i, (state, action) in enumerate(policy.items()):
            plt.text(state, 0.5, action_labels[action], 
                    ha='center', va='center', fontweight='bold')
                    
        plt.xlabel('Queue Length (State)')
        plt.ylabel('Action')
        plt.title('Optimal Scheduling Policy')
        plt.xticks(states)
        plt.ylim(0, 1)
        
        # Add legend
        import matplotlib.patches as mpatches
        slow_patch = mpatches.Patch(color='red', alpha=0.7, label='Slow Processing')
        fast_patch = mpatches.Patch(color='green', alpha=0.7, label='Fast Processing')
        plt.legend(handles=[slow_patch, fast_patch])
        
        plt.tight_layout()
        plt.show()


def demonstrate_simple_mdp():
    """Demonstrate the simple MDP scheduling example"""
    print("=== Simple MDP Scheduling Example ===")
    
    # Create and solve MDP
    mdp = SimpleMDP()
    values, policy = mdp.value_iteration()
    
    # Display results
    print("\\nOptimal Values:")
    for state in mdp.states:
        print(f"V*({state}) = {values[state]:.3f}")
        
    print("\\nOptimal Policy:")
    action_names = {0: 'Slow', 1: 'Fast'}
    for state in mdp.states:
        print(f"π*({state}) = {action_names[policy[state]]}")
        
    # Simulate episodes
    print("\\nSimulating episodes with optimal policy...")
    total_reward = 0
    num_episodes = 10
    
    for episode in range(num_episodes):
        trajectory = mdp.simulate_episode(policy, max_steps=20)
        episode_reward = sum(reward for _, _, reward in trajectory)
        total_reward += episode_reward
        print(f"Episode {episode+1}: Total reward = {episode_reward}")
        
    avg_reward = total_reward / num_episodes
    print(f"\\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    # Visualize policy
    mdp.visualize_policy(policy)
    
    return mdp, values, policy


if __name__ == "__main__":
    demonstrate_simple_mdp()
'''
        
        elif "dqn_job_scheduler" in example_name:
            return '''"""
DQN Job Scheduler Implementation
Deep Q-Network for job scheduling in heterogeneous systems
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class JobSchedulingEnvironment:
    """
    Job scheduling environment for DQN training
    
    State: [queue_lengths, node_capacities, job_priorities]
    Actions: Assign job to node (0 to num_nodes-1) or wait (num_nodes)
    Reward: Based on completion time and resource utilization
    """
    
    def __init__(self, num_nodes=4, max_jobs=10):
        self.num_nodes = num_nodes
        self.max_jobs = max_jobs
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Initialize queues and capacities
        self.node_queues = [0] * self.num_nodes  # Jobs in each queue
        self.node_capacities = [4, 8, 6, 10][:self.num_nodes]  # Processing capacity
        self.current_job = self._generate_job()
        self.time_step = 0
        self.completed_jobs = 0
        
        return self._get_state()
        
    def _generate_job(self):
        """Generate a new job with random requirements"""
        return {
            'cpu_requirement': np.random.randint(1, 5),
            'priority': np.random.randint(1, 4),  # 1=low, 3=high
            'arrival_time': self.time_step
        }
        
    def _get_state(self):
        """Get current state representation"""
        state = []
        
        # Queue lengths (normalized)
        state.extend([q / self.max_jobs for q in self.node_queues])
        
        # Node capacities (normalized)
        state.extend([c / max(self.node_capacities) for c in self.node_capacities])
        
        # Current job features (normalized)
        if self.current_job:
            state.append(self.current_job['cpu_requirement'] / 5.0)
            state.append(self.current_job['priority'] / 3.0)
        else:
            state.extend([0, 0])
            
        return np.array(state, dtype=np.float32)
        
    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        reward = 0
        done = False
        info = {}
        
        if self.current_job is None:
            # No job to schedule
            reward = -0.1  # Small penalty for idle time
        elif action < self.num_nodes:
            # Assign job to node
            node_id = action
            job_requirement = self.current_job['cpu_requirement']
            
            if self.node_queues[node_id] + job_requirement <= self.node_capacities[node_id]:
                # Valid assignment
                self.node_queues[node_id] += job_requirement
                
                # Calculate reward based on multiple factors
                utilization = sum(self.node_queues) / sum(self.node_capacities)
                priority_bonus = self.current_job['priority'] * 0.1
                load_balance = 1.0 - np.std(self.node_queues) / (np.mean(self.node_queues) + 1e-6)
                
                reward = utilization + priority_bonus + load_balance
                self.completed_jobs += 1
                
                info['job_assigned'] = True
                info['node_id'] = node_id
            else:
                # Invalid assignment (overload)
                reward = -1.0
                info['job_assigned'] = False
                info['reason'] = 'capacity_exceeded'
        else:
            # Wait action
            reward = -0.05  # Small penalty for waiting
            info['job_assigned'] = False
            info['reason'] = 'wait_action'
            
        # Process queues (simulate job completion)
        self._process_queues()
        
        # Generate next job
        if self.current_job and info.get('job_assigned', False):
            self.current_job = self._generate_job() if np.random.random() < 0.8 else None
            
        self.time_step += 1
        
        # Episode termination conditions
        if self.time_step >= 200 or self.completed_jobs >= 50:
            done = True
            
        next_state = self._get_state()
        return next_state, reward, done, info
        
    def _process_queues(self):
        """Simulate job processing and queue updates"""
        for i in range(self.num_nodes):
            if self.node_queues[i] > 0:
                # Process jobs based on capacity
                processing_rate = 0.3  # Jobs processed per time step
                processed = min(self.node_queues[i], processing_rate * self.node_capacities[i])
                self.node_queues[i] = max(0, self.node_queues[i] - processed)


class DQN(nn.Module):
    """Deep Q-Network for job scheduling"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    """DQN Agent for job scheduling"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
        
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        # Compute Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn_scheduler(episodes=1000):
    """Train DQN agent for job scheduling"""
    print("=== Training DQN Job Scheduler ===")
    
    # Initialize environment and agent
    env = JobSchedulingEnvironment(num_nodes=4)
    state_size = len(env._get_state())
    action_size = env.num_nodes + 1  # +1 for wait action
    
    agent = DQNAgent(state_size, action_size)
    
    # Training loop
    scores = []
    avg_scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
                
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Jobs: {env.completed_jobs}")
                  
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, label='Episode Score')
    plt.plot(avg_scores, label='Average Score (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Progress')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    # Test trained agent
    test_scores = []
    for _ in range(10):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state, training=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        test_scores.append(total_reward)
        
    plt.bar(range(len(test_scores)), test_scores)
    plt.xlabel('Test Episode')
    plt.ylabel('Score')
    plt.title('Trained Agent Performance')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\\nTraining completed!")
    print(f"Average test score: {np.mean(test_scores):.2f}")
    
    return agent, env


if __name__ == "__main__":
    train_dqn_scheduler()
'''
        
        else:
            # Generic code template
            return f'''"""
{example_name.replace('_', ' ').title().replace('.py', '')}
Educational tutorial code for {module.title}
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

class TutorialExample:
    """Tutorial example implementation"""
    
    def __init__(self):
        self.name = "{example_name}"
        print(f"Initializing {{self.name}} tutorial example")
        
    def demonstrate(self):
        """Demonstrate the key concepts"""
        print("This is a template for the tutorial example.")
        print("Replace this with actual implementation based on the module content.")
        
        # Add relevant code here
        pass
        
    def visualize_results(self):
        """Create visualizations for the tutorial"""
        plt.figure(figsize=(10, 6))
        
        # Add plotting code here
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title(f"Visualization for {{self.name}}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.show()


def main():
    """Main function to run the tutorial example"""
    example = TutorialExample()
    example.demonstrate()
    example.visualize_results()


if __name__ == "__main__":
    main()
'''
            
    def _generate_exercises(self, module: TutorialModule) -> str:
        """Generate exercises for the module"""
        content = f"""# Exercises for {module.title}

## Instructions

Complete the following exercises to reinforce your understanding of the concepts covered in this module. Work through them in order, as later exercises may build on earlier ones.

**Estimated Time:** {module.duration_minutes // 2} minutes

## Exercise Set

"""
        
        for i, exercise in enumerate(module.exercises, 1):
            content += f"""
### Exercise {i}: {exercise}

**Objective:** Practice {exercise.lower()}

**Instructions:**
1. Review the relevant section in the tutorial
2. Implement the solution step by step  
3. Test your implementation with provided examples
4. Verify results against expected outputs

**Starter Code:**
```python
# TODO: Implement your solution here
def exercise_{i}_solution():
    '''Implement {exercise.lower()}'''
    pass

# Test your solution
if __name__ == "__main__":
    result = exercise_{i}_solution()
    print(f"Result: {{result}}")
```

**Expected Output:**
[Describe what the correct output should look like]

**Hints:**
- Consider the key concepts from section X
- Remember to handle edge cases
- Use the provided utility functions if helpful

---
"""
        
        content += """
## Submission Guidelines

1. Save your solutions in separate files named `exercise_1.py`, `exercise_2.py`, etc.
2. Include comments explaining your approach
3. Test your code with the provided test cases
4. Document any assumptions or design decisions

## Self-Assessment

After completing the exercises, ask yourself:
- [ ] Do I understand the core concepts?
- [ ] Can I explain my solution approach?
- [ ] Does my code handle edge cases correctly?
- [ ] Am I ready for the next module?

## Getting Help

If you're stuck:
1. Review the tutorial content again
2. Check the hints and expected outputs
3. Look at the provided code examples
4. Consult the solutions guide (after attempting on your own)
5. Ask questions in the community forum
"""
        return content
        
    def _generate_solutions(self, module: TutorialModule) -> str:
        """Generate solutions for exercises"""
        content = f"""# Solutions for {module.title}

**Important:** Only refer to these solutions after attempting the exercises yourself. Learning happens through struggle and problem-solving, not by copying solutions.

## Solution Guidelines

Each solution includes:
- Complete implementation
- Explanation of the approach
- Discussion of key concepts
- Alternative approaches where applicable

"""
        
        for i, exercise in enumerate(module.exercises, 1):
            content += f"""
## Solution {i}: {exercise}

### Approach
[Explain the problem-solving approach and key insights]

### Implementation
```python
def exercise_{i}_solution():
    '''
    Solution for: {exercise}
    
    Key concepts demonstrated:
    - [List key concepts]
    '''
    # Implementation here
    pass

# Alternative implementation
def exercise_{i}_alternative():
    '''Alternative approach using different method'''
    pass
```

### Explanation
[Detailed explanation of the solution, why it works, and how it relates to the tutorial concepts]

### Extension Ideas
- Try implementing with different parameters
- Optimize for performance or memory usage
- Add visualization or logging features

---
"""
        
        content += """
## Learning Notes

Key takeaways from these exercises:
1. [Main concept 1]
2. [Main concept 2]  
3. [Main concept 3]

Common mistakes to avoid:
- [Common mistake 1]
- [Common mistake 2]

## Next Steps

Now that you've completed these exercises:
1. Review any concepts that were challenging
2. Experiment with the code examples
3. Try the extension ideas for deeper learning
4. Proceed to the next module

Remember: The goal is understanding, not just getting the right answer!
"""
        return content
        
    def generate_learning_path_guide(self, path_name: str) -> str:
        """Generate a comprehensive guide for a learning path"""
        path = self.learning_paths[path_name]
        
        guide = f"""# {path.name}

## Overview

**Target Audience:** {path.target_audience}  
**Estimated Duration:** {path.estimated_hours} hours  
**Prerequisites:** {', '.join(path.prerequisites)}

{path.description}

## Learning Path Structure

This learning path consists of {len(path.modules)} modules designed to take you from beginner to advanced understanding of RL for heterogeneous scheduling.

### Module Sequence

"""
        
        for i, module_name in enumerate(path.modules, 1):
            module = self.modules[module_name]
            guide += f"""
#### Module {i}: {module.title}
- **Level:** {module.level.value.title()}
- **Duration:** {module.duration_minutes} minutes
- **Focus:** {', '.join(obj.value.replace('_', ' ').title() for obj in module.objectives)}
- **Description:** {module.description}

"""
        
        guide += f"""
## Study Schedule Recommendations

### Full-Time Study (1-2 weeks)
- Complete 1-2 modules per day
- Spend extra time on exercises and code examples
- Join study groups or find learning partners

### Part-Time Study (4-6 weeks) 
- Complete 1 module every 2-3 days
- Dedicate 2-3 hours per study session
- Review previous modules before starting new ones

### Self-Paced Study (2-3 months)
- Complete 1 module per week
- Take time to experiment and explore
- Focus on deep understanding over speed

## Assessment and Certification

### Module Assessments
Each module includes:
- Knowledge check quizzes
- Hands-on coding exercises  
- Practical projects

### Final Project
Complete a comprehensive project that demonstrates mastery of:
- RL algorithm implementation
- Scheduling problem formulation
- Performance evaluation and analysis
- Documentation and presentation

### Certification Requirements
To earn certification, you must:
- [ ] Complete all module assessments with 80%+ score
- [ ] Submit and pass the final project
- [ ] Participate in peer review process
- [ ] Complete the final comprehensive exam

## Resources and Support

### Required Materials
- Python programming environment
- PyTorch or TensorFlow
- Jupyter notebooks
- Git for version control

### Recommended Reading
- "Reinforcement Learning: An Introduction" by Sutton & Barto
- "Deep Reinforcement Learning Hands-On" by Maxim Lapan
- Recent research papers (provided in each module)

### Community Support
- Discussion forums for each module
- Weekly virtual office hours
- Peer study groups
- Mentor matching program

## Tips for Success

### Before Starting
1. Ensure you meet all prerequisites
2. Set up your development environment
3. Create a study schedule and stick to it
4. Join the community forums

### During Study
1. Take notes and summarize key concepts
2. Implement code examples from scratch
3. Experiment with parameters and modifications
4. Ask questions when stuck

### After Completion
1. Contribute to open-source projects
2. Share your learning experience
3. Mentor new learners
4. Stay updated with latest research

## Career Applications

Skills learned in this path prepare you for:
- **Research Positions:** ML research scientist, PhD studies
- **Industry Roles:** Systems engineer, cloud architect, HPC specialist
- **Entrepreneurship:** Starting companies using RL for system optimization
- **Consulting:** Helping organizations implement intelligent scheduling

## Next Steps After Completion

Consider pursuing:
- Advanced research projects
- Industry internships or positions
- PhD studies in related areas
- Contributing to open-source RL projects
- Attending conferences and workshops

---

**Ready to start your learning journey? Begin with Module 1: {self.modules[path.modules[0]].title}**
"""
        return guide
        
    def create_complete_tutorial_package(self) -> Dict[str, Any]:
        """Create complete tutorial package with all materials"""
        package = {
            "modules": {},
            "learning_paths": {},
            "assessment": {},
            "resources": {}
        }
        
        # Generate all module content
        for module_name, module in self.modules.items():
            module_dir = self.output_dir / "modules" / module_name
            module_dir.mkdir(parents=True, exist_ok=True)
            
            module_content = self.generate_module_content(module_name)
            module_files = {}
            
            for filename, content in module_content.items():
                file_path = module_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                module_files[filename] = str(file_path)
                
            package["modules"][module_name] = module_files
            
        # Generate learning path guides
        for path_name, path in self.learning_paths.items():
            guide_content = self.generate_learning_path_guide(path_name)
            guide_path = self.output_dir / "learning_paths" / f"{path_name}_guide.md"
            guide_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(guide_path, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            package["learning_paths"][path_name] = str(guide_path)
            
        # Generate assessment materials
        assessment_content = self._generate_assessment_materials()
        assessment_dir = self.output_dir / "assessment"
        assessment_dir.mkdir(exist_ok=True)
        
        for filename, content in assessment_content.items():
            file_path = assessment_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            package["assessment"][filename] = str(file_path)
            
        # Generate additional resources
        resources_content = self._generate_additional_resources()
        resources_dir = self.output_dir / "resources"
        resources_dir.mkdir(exist_ok=True)
        
        for filename, content in resources_content.items():
            file_path = resources_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            package["resources"][filename] = str(file_path)
            
        return package
        
    def _generate_assessment_materials(self) -> Dict[str, str]:
        """Generate assessment and evaluation materials"""
        materials = {}
        
        # Final project specification
        materials["final_project.md"] = """# Final Project: RL Scheduler Implementation

## Project Overview

Design and implement a complete reinforcement learning solution for a heterogeneous scheduling problem of your choice.

## Requirements

### 1. Problem Definition (20 points)
- Clearly define the scheduling problem
- Justify why RL is appropriate
- Identify key challenges and constraints

### 2. MDP Formulation (25 points)  
- Define state space, action space, and reward function
- Justify design choices
- Discuss alternative formulations

### 3. Implementation (35 points)
- Implement at least one RL algorithm
- Include proper documentation and comments
- Demonstrate working code with examples

### 4. Evaluation (15 points)
- Compare against baseline methods
- Analyze performance across multiple metrics
- Discuss results and limitations

### 5. Presentation (5 points)
- Clear written report (10-15 pages)
- Code repository with README
- Optional: video demonstration

## Grading Rubric

[Detailed rubric with specific criteria for each section]

## Timeline

- Week 1: Project proposal due
- Week 2-4: Implementation phase
- Week 5: Final submission and presentation

## Sample Projects

- Multi-objective HPC job scheduling
- Dynamic cloud resource allocation
- Edge computing task placement
- Real-time embedded system scheduling
"""
        
        # Comprehensive exam
        materials["comprehensive_exam.md"] = """# Comprehensive Exam

## Format
- 50 multiple choice questions (2 points each)
- 5 short answer questions (10 points each)
- 2 programming problems (25 points each)
- Total: 200 points, 3 hours

## Topics Covered

### Theory (30%)
- MDP formulation and solution methods
- RL algorithm principles and convergence
- Multi-agent coordination mechanisms

### Implementation (40%)
- Algorithm implementation and debugging
- Performance optimization techniques
- Code analysis and design patterns

### Analysis (30%)
- Experimental design and evaluation
- Result interpretation and reporting
- Comparison methodologies

## Sample Questions

[Include representative sample questions for each section]

## Preparation Guidelines

1. Review all module materials thoroughly
2. Practice implementing algorithms from scratch
3. Work through all tutorial exercises again
4. Study the provided sample questions
5. Form study groups with peers
"""
        
        return materials
        
    def _generate_additional_resources(self) -> Dict[str, str]:
        """Generate additional learning resources"""
        resources = {}
        
        # Research paper list
        resources["recommended_papers.md"] = """# Recommended Research Papers

## Foundational Papers

### Reinforcement Learning Basics
1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
3. Schulman, J., et al. (2017). Proximal policy optimization algorithms.

### Scheduling Theory
1. Pinedo, M. L. (2016). Scheduling: theory, algorithms, and systems.
2. Brucker, P. (2007). Scheduling algorithms.

### RL for Scheduling
1. Zhang, C., et al. (2020). Deep reinforcement learning for job scheduling in HPC clusters.
2. Mao, H., et al. (2019). Resource management with deep reinforcement learning.
3. Liu, C., et al. (2015). Multiobjective reinforcement learning: A comprehensive overview.

## Recent Advances

[List of recent papers organized by topic]

## Reading Strategy

1. Start with foundational papers
2. Focus on papers relevant to your interests
3. Take notes on key contributions
4. Implement interesting ideas
5. Discuss with peers and instructors
"""
        
        # FAQ document
        resources["faq.md"] = """# Frequently Asked Questions

## Technical Questions

### Q: What programming experience do I need?
A: You should be comfortable with Python and have basic understanding of machine learning concepts.

### Q: Which RL library should I use?
A: We recommend starting with stable-baselines3 for beginners, then moving to PyTorch for custom implementations.

### Q: How do I handle large state spaces?
A: Consider state representation learning, hierarchical approaches, or function approximation techniques.

## Course Structure

### Q: Can I skip modules if I already know the material?
A: We recommend completing the assessments to verify your knowledge, but you can move quickly through familiar content.

### Q: How much time should I spend on each module?
A: Follow the suggested durations, but adjust based on your background and learning style.

### Q: What if I get stuck on an exercise?
A: Try the hints first, then consult the community forum, and finally check the solutions.

## Career and Applications

### Q: What jobs can this prepare me for?
A: Research scientist, systems engineer, cloud architect, HPC specialist, and many others.

### Q: Should I pursue a PhD after this course?
A: This course provides good preparation for PhD studies in related areas.

### Q: How can I stay updated with latest research?
A: Follow key conferences (ICML, NeurIPS, MLSys), join mailing lists, and participate in community forums.
"""
        
        return resources


def demonstrate_tutorial_materials():
    """Demonstrate the educational materials generation"""
    print("=== R44: Educational Materials Generation ===")
    
    # Initialize generator
    generator = EducationalMaterialsGenerator()
    
    # Generate complete tutorial package
    print("\nGenerating comprehensive tutorial package...")
    package = generator.create_complete_tutorial_package()
    
    print(f"\nTutorial package generation completed!")
    print(f"Output directory: {generator.output_dir}")
    
    # Show package structure
    print(f"\nPackage Structure:")
    for category, items in package.items():
        print(f"\n{category.title()}:")
        if isinstance(items, dict):
            for name, path in items.items():
                if isinstance(path, dict):
                    print(f"  {name}/")
                    for file_name in path.keys():
                        print(f"    {file_name}")
                else:
                    print(f"  {name}")
        else:
            print(f"  {items}")
    
    # Show module statistics
    print(f"\nModule Statistics:")
    for name, module in generator.modules.items():
        print(f"- {module.title}")
        print(f"  Level: {module.level.value}")
        print(f"  Duration: {module.duration_minutes} minutes")
        print(f"  Exercises: {len(module.exercises)}")
        print(f"  Code examples: {len(module.code_examples)}")
    
    # Show learning path information
    print(f"\nLearning Paths:")
    for name, path in generator.learning_paths.items():
        print(f"- {path.name}")
        print(f"  Target: {path.target_audience}")
        print(f"  Duration: {path.estimated_hours} hours")
        print(f"  Modules: {len(path.modules)}")
    
    # Generate summary statistics
    total_modules = len(generator.modules)
    total_exercises = sum(len(m.exercises) for m in generator.modules.values())
    total_code_examples = sum(len(m.code_examples) for m in generator.modules.values())
    total_duration = sum(m.duration_minutes for m in generator.modules.values())
    
    print(f"\nOverall Statistics:")
    print(f"- Total modules: {total_modules}")
    print(f"- Total exercises: {total_exercises}")
    print(f"- Total code examples: {total_code_examples}")
    print(f"- Total duration: {total_duration // 60} hours {total_duration % 60} minutes")
    print(f"- Learning paths: {len(generator.learning_paths)}")
    
    return generator, package


if __name__ == "__main__":
    generator, package = demonstrate_tutorial_materials()