"""
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
    
    print(f"\nTraining completed!")
    print(f"Average test score: {np.mean(test_scores):.2f}")
    
    return agent, env


if __name__ == "__main__":
    train_dqn_scheduler()
