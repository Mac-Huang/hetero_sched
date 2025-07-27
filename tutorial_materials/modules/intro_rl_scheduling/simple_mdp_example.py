"""
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
    print("\nOptimal Values:")
    for state in mdp.states:
        print(f"V*({state}) = {values[state]:.3f}")
        
    print("\nOptimal Policy:")
    action_names = {0: 'Slow', 1: 'Fast'}
    for state in mdp.states:
        print(f"π*({state}) = {action_names[policy[state]]}")
        
    # Simulate episodes
    print("\nSimulating episodes with optimal policy...")
    total_reward = 0
    num_episodes = 10
    
    for episode in range(num_episodes):
        trajectory = mdp.simulate_episode(policy, max_steps=20)
        episode_reward = sum(reward for _, _, reward in trajectory)
        total_reward += episode_reward
        print(f"Episode {episode+1}: Total reward = {episode_reward}")
        
    avg_reward = total_reward / num_episodes
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    # Visualize policy
    mdp.visualize_policy(policy)
    
    return mdp, values, policy


if __name__ == "__main__":
    demonstrate_simple_mdp()
