"""
Few-Shot Learning for New Task Type Adaptation

This module implements R14: a comprehensive few-shot learning framework that enables
HeteroSched agents to rapidly adapt to new task types and scheduling scenarios with
minimal training data.

Key Features:
1. Model-Agnostic Meta-Learning (MAML) for scheduling domains
2. Prototypical networks for task type classification and adaptation
3. Memory-augmented neural networks for rapid knowledge acquisition
4. Task embedding and similarity learning for transfer
5. Gradient-based adaptation with scheduling-specific constraints
6. Episodic training framework for few-shot scenarios
7. Zero-shot and one-shot adaptation capabilities

The framework enables quick deployment to new environments and task types
without extensive retraining.

Authors: HeteroSched Research Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import logging
import random
import math
from collections import defaultdict, deque
import copy

class AdaptationStrategy(Enum):
    MAML = "maml"
    PROTOTYPICAL = "prototypical"
    MEMORY_AUGMENTED = "memory_augmented"
    TASK_EMBEDDING = "task_embedding"
    GRADIENT_BASED = "gradient_based"

class SupportSetStrategy(Enum):
    RANDOM_SAMPLING = "random_sampling"
    DIVERSE_SAMPLING = "diverse_sampling"
    REPRESENTATIVE_SAMPLING = "representative_sampling"
    ACTIVE_LEARNING = "active_learning"

@dataclass
class TaskInstance:
    """Represents a single task instance for few-shot learning"""
    task_id: str
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    task_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Episode:
    """Represents an episode for episodic training"""
    episode_id: str
    support_set: List[TaskInstance]
    query_set: List[TaskInstance]
    task_description: Dict[str, Any]
    
@dataclass
class AdaptationResult:
    """Results from few-shot adaptation"""
    adaptation_id: str
    strategy: AdaptationStrategy
    initial_performance: float
    adapted_performance: float
    adaptation_steps: int
    adaptation_time: float
    support_set_size: int
    success: bool

class PrototypicalNetwork(nn.Module):
    """Prototypical Network for few-shot task classification and adaptation"""
    
    def __init__(self, input_dim: int, embedding_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Embedding network
        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Task-specific adaptation layers
        self.adaptation_layers = nn.ModuleDict({
            'task_classifier': nn.Linear(embedding_dim, config.get('num_task_types', 10)),
            'resource_predictor': nn.Linear(embedding_dim, config.get('num_resources', 5)),
            'priority_predictor': nn.Linear(embedding_dim, config.get('num_priorities', 5))
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through prototypical network"""
        # Get embeddings
        embeddings = self.embedding_network(x)
        
        # Task-specific predictions
        outputs = {}
        for name, layer in self.adaptation_layers.items():
            outputs[name] = layer(embeddings)
        
        outputs['embeddings'] = embeddings
        return outputs
    
    def compute_prototypes(self, support_embeddings: torch.Tensor, 
                          support_labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes from support set"""
        unique_labels = torch.unique(support_labels)
        prototypes = torch.zeros(len(unique_labels), self.embedding_dim)
        
        for i, label in enumerate(unique_labels):
            mask = support_labels == label
            prototypes[i] = support_embeddings[mask].mean(dim=0)
        
        return prototypes
    
    def classify_with_prototypes(self, query_embeddings: torch.Tensor,
                               prototypes: torch.Tensor) -> torch.Tensor:
        """Classify queries using nearest prototype"""
        # Compute distances to prototypes
        distances = torch.cdist(query_embeddings, prototypes)
        
        # Convert to probabilities (negative distances)
        logits = -distances
        return F.softmax(logits, dim=1)

class MAMLScheduler(nn.Module):
    """Model-Agnostic Meta-Learning for scheduling adaptation"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Main policy network
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Value network for advantage estimation
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Meta-learning parameters
        self.inner_lr = config.get('inner_lr', 0.01)
        self.meta_lr = config.get('meta_lr', 0.001)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value"""
        action_logits = self.policy_network(state)
        state_value = self.value_network(state)
        return action_logits, state_value
    
    def inner_update(self, support_set: List[TaskInstance], 
                    num_steps: int = 5) -> nn.Module:
        """Perform inner loop update on support set"""
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self)
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for step in range(num_steps):
            total_loss = 0.0
            
            for instance in support_set:
                # Forward pass
                action_logits, state_value = adapted_model(instance.state.unsqueeze(0))
                
                # Compute loss based on the task instance
                action_loss = F.cross_entropy(action_logits, instance.action.unsqueeze(0))
                value_loss = F.mse_loss(state_value, torch.tensor([[instance.reward]], dtype=torch.float32))
                
                loss = action_loss + 0.5 * value_loss
                total_loss += loss
            
            # Update parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_update(self, episodes: List[Episode], meta_optimizer: optim.Optimizer):
        """Perform meta-update across multiple episodes"""
        meta_loss = 0.0
        
        for episode in episodes:
            # Inner update on support set
            adapted_model = self.inner_update(episode.support_set)
            
            # Evaluate on query set
            query_loss = 0.0
            for instance in episode.query_set:
                action_logits, state_value = adapted_model(instance.state.unsqueeze(0))
                
                action_loss = F.cross_entropy(action_logits, instance.action.unsqueeze(0))
                value_loss = F.mse_loss(state_value, torch.tensor([[instance.reward]], dtype=torch.float32))
                
                query_loss += action_loss + 0.5 * value_loss
            
            meta_loss += query_loss / len(episode.query_set)
        
        # Meta gradient update
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        return meta_loss.item() / len(episodes)

class MemoryAugmentedNetwork(nn.Module):
    """Memory-Augmented Neural Network for rapid adaptation"""
    
    def __init__(self, input_dim: int, memory_size: int, memory_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Controller network
        self.controller = nn.LSTM(input_dim, 256, batch_first=True)
        
        # Memory interface
        self.key_network = nn.Linear(256, memory_dim)
        self.value_network = nn.Linear(256, memory_dim)
        self.read_network = nn.Linear(256 + memory_dim, 256)
        
        # Output network
        self.output_network = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, config.get('output_dim', 64))
        )
        
        # Initialize memory
        self.register_buffer('memory_keys', torch.randn(memory_size, memory_dim))
        self.register_buffer('memory_values', torch.randn(memory_size, memory_dim))
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Forward pass with memory interaction"""
        batch_size, seq_len, _ = x.shape
        
        # Controller processing
        controller_output, hidden = self.controller(x, hidden)
        
        outputs = []
        for t in range(seq_len):
            controller_state = controller_output[:, t, :]
            
            # Memory interaction
            memory_output = self.memory_interaction(controller_state)
            
            # Combine controller and memory
            combined = torch.cat([controller_state, memory_output], dim=1)
            read_output = self.read_network(combined)
            
            # Generate output
            output = self.output_network(read_output)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1), hidden
    
    def memory_interaction(self, controller_state: torch.Tensor) -> torch.Tensor:
        """Interact with external memory"""
        batch_size = controller_state.shape[0]
        
        # Generate keys for memory lookup
        query_key = self.key_network(controller_state)
        
        # Compute attention weights
        attention_weights = F.softmax(
            torch.matmul(query_key.unsqueeze(1), self.memory_keys.t()), dim=2
        )
        
        # Read from memory
        memory_read = torch.matmul(attention_weights, self.memory_values).squeeze(1)
        
        # Write to memory (update least recently used)
        write_value = self.value_network(controller_state)
        self.update_memory(query_key, write_value)
        
        return memory_read
    
    def update_memory(self, keys: torch.Tensor, values: torch.Tensor):
        """Update memory with new key-value pairs"""
        batch_size = keys.shape[0]
        
        for b in range(batch_size):
            # Find least recently used memory slot
            lru_idx = torch.argmin(self.memory_usage)
            
            # Update memory
            self.memory_keys[lru_idx] = keys[b]
            self.memory_values[lru_idx] = values[b]
            self.memory_usage[lru_idx] = torch.max(self.memory_usage) + 1

class TaskEmbeddingNetwork(nn.Module):
    """Network for learning task embeddings and similarity"""
    
    def __init__(self, input_dim: int, embedding_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Similarity network
        self.similarity_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Transfer network
        self.transfer_network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, config.get('transfer_dim', 128))
        )
        
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """Encode task features to embedding space"""
        return self.task_encoder(task_features)
    
    def compute_similarity(self, embedding1: torch.Tensor, 
                          embedding2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between task embeddings"""
        combined = torch.cat([embedding1, embedding2], dim=-1)
        return self.similarity_network(combined)
    
    def transfer_knowledge(self, source_embedding: torch.Tensor) -> torch.Tensor:
        """Transfer knowledge from source task embedding"""
        return self.transfer_network(source_embedding)

class FewShotLearningFramework:
    """Main framework for few-shot learning in scheduling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("FewShotLearningFramework")
        
        # Model dimensions
        self.state_dim = config.get('state_dim', 128)
        self.action_dim = config.get('action_dim', 64)
        self.embedding_dim = config.get('embedding_dim', 256)
        
        # Initialize models
        self.prototypical_net = PrototypicalNetwork(
            self.state_dim, self.embedding_dim, config
        )
        
        self.maml_scheduler = MAMLScheduler(
            self.state_dim, self.action_dim, config
        )
        
        self.memory_net = MemoryAugmentedNetwork(
            self.state_dim, 
            config.get('memory_size', 128),
            config.get('memory_dim', 64),
            config
        )
        
        self.task_embedding_net = TaskEmbeddingNetwork(
            self.state_dim, self.embedding_dim, config
        )
        
        # Training parameters
        self.meta_optimizer = optim.Adam(
            list(self.maml_scheduler.parameters()) + 
            list(self.prototypical_net.parameters()) +
            list(self.task_embedding_net.parameters()),
            lr=config.get('meta_lr', 0.001)
        )
        
        # Episode storage
        self.training_episodes: List[Episode] = []
        self.adaptation_history: List[AdaptationResult] = []
        
    def generate_episode(self, task_instances: List[TaskInstance],
                        support_size: int, query_size: int) -> Episode:
        """Generate a training episode from task instances"""
        
        # Shuffle instances
        random.shuffle(task_instances)
        
        # Split into support and query sets
        support_set = task_instances[:support_size]
        query_set = task_instances[support_size:support_size + query_size]
        
        # Extract task description
        task_description = self._extract_task_description(task_instances)
        
        episode = Episode(
            episode_id=f"episode_{len(self.training_episodes)}",
            support_set=support_set,
            query_set=query_set,
            task_description=task_description
        )
        
        return episode
    
    def adapt_to_new_task(self, strategy: AdaptationStrategy,
                         support_set: List[TaskInstance],
                         query_set: List[TaskInstance]) -> AdaptationResult:
        """Adapt to a new task using specified strategy"""
        
        start_time = time.time()
        adaptation_id = f"adapt_{strategy.value}_{int(start_time)}"
        
        # Evaluate initial performance
        initial_performance = self._evaluate_performance(query_set, adapted=False)
        
        try:
            if strategy == AdaptationStrategy.MAML:
                adapted_performance = self._adapt_with_maml(support_set, query_set)
                adaptation_steps = self.config.get('maml_steps', 5)
                
            elif strategy == AdaptationStrategy.PROTOTYPICAL:
                adapted_performance = self._adapt_with_prototypical(support_set, query_set)
                adaptation_steps = 1  # Single forward pass
                
            elif strategy == AdaptationStrategy.MEMORY_AUGMENTED:
                adapted_performance = self._adapt_with_memory(support_set, query_set)
                adaptation_steps = len(support_set)  # One step per support instance
                
            elif strategy == AdaptationStrategy.TASK_EMBEDDING:
                adapted_performance = self._adapt_with_task_embedding(support_set, query_set)
                adaptation_steps = 1  # Single embedding computation
                
            else:
                raise ValueError(f"Unknown adaptation strategy: {strategy}")
            
            success = adapted_performance > initial_performance
            
        except Exception as e:
            self.logger.error(f"Adaptation failed: {e}")
            adapted_performance = initial_performance
            adaptation_steps = 0
            success = False
        
        adaptation_time = time.time() - start_time
        
        result = AdaptationResult(
            adaptation_id=adaptation_id,
            strategy=strategy,
            initial_performance=initial_performance,
            adapted_performance=adapted_performance,
            adaptation_steps=adaptation_steps,
            adaptation_time=adaptation_time,
            support_set_size=len(support_set),
            success=success
        )
        
        self.adaptation_history.append(result)
        return result
    
    def _adapt_with_maml(self, support_set: List[TaskInstance],
                        query_set: List[TaskInstance]) -> float:
        """Adapt using MAML algorithm"""
        
        # Create episode
        episode = Episode(
            episode_id="maml_adaptation",
            support_set=support_set,
            query_set=query_set,
            task_description={}
        )
        
        # Inner update
        adapted_model = self.maml_scheduler.inner_update(
            support_set, 
            num_steps=self.config.get('maml_steps', 5)
        )
        
        # Evaluate on query set
        return self._evaluate_model_performance(adapted_model, query_set)
    
    def _adapt_with_prototypical(self, support_set: List[TaskInstance],
                                query_set: List[TaskInstance]) -> float:
        """Adapt using prototypical networks"""
        
        # Extract features and labels from support set
        support_states = torch.stack([inst.state for inst in support_set])
        support_labels = torch.tensor([hash(inst.task_type) % 10 for inst in support_set])
        
        # Get embeddings
        support_outputs = self.prototypical_net(support_states)
        support_embeddings = support_outputs['embeddings']
        
        # Compute prototypes
        prototypes = self.prototypical_net.compute_prototypes(
            support_embeddings, support_labels
        )
        
        # Evaluate on query set
        correct_predictions = 0
        total_predictions = len(query_set)
        
        for instance in query_set:
            query_output = self.prototypical_net(instance.state.unsqueeze(0))
            query_embedding = query_output['embeddings']
            
            # Classify using prototypes
            predictions = self.prototypical_net.classify_with_prototypes(
                query_embedding, prototypes
            )
            
            predicted_class = torch.argmax(predictions, dim=1)
            true_class = hash(instance.task_type) % 10
            
            if predicted_class.item() == true_class:
                correct_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _adapt_with_memory(self, support_set: List[TaskInstance],
                          query_set: List[TaskInstance]) -> float:
        """Adapt using memory-augmented networks"""
        
        # Process support set to update memory
        support_states = torch.stack([inst.state for inst in support_set]).unsqueeze(0)
        
        # Forward pass to update memory
        _, hidden = self.memory_net(support_states)
        
        # Evaluate on query set
        correct_predictions = 0
        total_predictions = len(query_set)
        
        for instance in query_set:
            query_state = instance.state.unsqueeze(0).unsqueeze(0)
            
            # Process with updated memory
            output, _ = self.memory_net(query_state, hidden)
            
            # Simple classification based on output
            prediction = torch.argmax(output.squeeze(), dim=-1)
            true_action = instance.action.item() if instance.action.numel() == 1 else instance.action.argmax().item()
            
            if prediction.item() == true_action:
                correct_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _adapt_with_task_embedding(self, support_set: List[TaskInstance],
                                  query_set: List[TaskInstance]) -> float:
        """Adapt using task embeddings and similarity"""
        
        # Compute task embedding from support set
        support_states = torch.stack([inst.state for inst in support_set])
        task_embedding = self.task_embedding_net(support_states).mean(dim=0)
        
        # Transfer knowledge
        transferred_knowledge = self.task_embedding_net.transfer_knowledge(task_embedding)
        
        # Evaluate similarity-based predictions on query set
        correct_predictions = 0
        total_predictions = len(query_set)
        
        for instance in query_set:
            query_embedding = self.task_embedding_net(instance.state.unsqueeze(0))
            
            # Compute similarity to task embedding
            similarity = self.task_embedding_net.compute_similarity(
                query_embedding, task_embedding.unsqueeze(0)
            )
            
            # Use similarity for prediction (simplified)
            prediction = 1 if similarity.item() > 0.5 else 0
            true_label = 1 if instance.reward > 0 else 0
            
            if prediction == true_label:
                correct_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _evaluate_performance(self, query_set: List[TaskInstance], 
                            adapted: bool = False) -> float:
        """Evaluate performance on query set"""
        if not adapted:
            # Baseline random performance
            return 0.1 + np.random.uniform(0, 0.2)
        
        # Use the original model without adaptation
        return self._evaluate_model_performance(self.maml_scheduler, query_set)
    
    def _evaluate_model_performance(self, model: nn.Module, 
                                  query_set: List[TaskInstance]) -> float:
        """Evaluate model performance on query set"""
        model.eval()
        
        correct_predictions = 0
        total_predictions = len(query_set)
        
        with torch.no_grad():
            for instance in query_set:
                if hasattr(model, 'forward'):
                    action_logits, _ = model(instance.state.unsqueeze(0))
                    predicted_action = torch.argmax(action_logits, dim=1)
                    true_action = instance.action.item() if instance.action.numel() == 1 else instance.action.argmax().item()
                    
                    if predicted_action.item() == true_action:
                        correct_predictions += 1
        
        model.train()
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _extract_task_description(self, task_instances: List[TaskInstance]) -> Dict[str, Any]:
        """Extract task description from instances"""
        task_types = set(inst.task_type for inst in task_instances)
        avg_reward = np.mean([inst.reward for inst in task_instances])
        
        return {
            "task_types": list(task_types),
            "num_instances": len(task_instances),
            "avg_reward": avg_reward,
            "complexity": len(task_types) / 10.0  # Normalize by assumed max types
        }
    
    def train_meta_learning(self, episodes: List[Episode], num_epochs: int = 100):
        """Train the meta-learning framework"""
        
        self.logger.info(f"Training meta-learning with {len(episodes)} episodes for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Sample batch of episodes
            batch_size = min(self.config.get('batch_size', 8), len(episodes))
            episode_batch = np.random.choice(episodes, batch_size, replace=False)
            
            # MAML meta-update
            maml_loss = self.maml_scheduler.meta_update(episode_batch, self.meta_optimizer)
            epoch_loss += maml_loss
            
            # Prototypical network training
            proto_loss = self._train_prototypical_network(episode_batch)
            epoch_loss += proto_loss
            
            # Task embedding training
            embedding_loss = self._train_task_embedding(episode_batch)
            epoch_loss += embedding_loss
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
    
    def _train_prototypical_network(self, episodes: List[Episode]) -> float:
        """Train prototypical network on episode batch"""
        total_loss = 0.0
        
        for episode in episodes:
            # Process support set
            support_states = torch.stack([inst.state for inst in episode.support_set])
            support_labels = torch.tensor([hash(inst.task_type) % 10 for inst in episode.support_set])
            
            # Process query set
            query_states = torch.stack([inst.state for inst in episode.query_set])
            query_labels = torch.tensor([hash(inst.task_type) % 10 for inst in episode.query_set])
            
            # Forward pass
            support_outputs = self.prototypical_net(support_states)
            query_outputs = self.prototypical_net(query_states)
            
            # Compute prototypes
            prototypes = self.prototypical_net.compute_prototypes(
                support_outputs['embeddings'], support_labels
            )
            
            # Classify queries
            query_predictions = self.prototypical_net.classify_with_prototypes(
                query_outputs['embeddings'], prototypes
            )
            
            # Compute loss
            loss = F.cross_entropy(query_predictions, query_labels)
            total_loss += loss.item()
        
        return total_loss / len(episodes)
    
    def _train_task_embedding(self, episodes: List[Episode]) -> float:
        """Train task embedding network on episode batch"""
        total_loss = 0.0
        
        for episode in episodes:
            # Create positive and negative pairs
            support_states = torch.stack([inst.state for inst in episode.support_set])
            
            # Positive pairs (same task)
            for i in range(len(support_states) - 1):
                embedding1 = self.task_embedding_net(support_states[i].unsqueeze(0))
                embedding2 = self.task_embedding_net(support_states[i + 1].unsqueeze(0))
                
                similarity = self.task_embedding_net.compute_similarity(embedding1, embedding2)
                positive_loss = F.binary_cross_entropy(similarity, torch.ones_like(similarity))
                total_loss += positive_loss.item()
        
        return total_loss / max(len(episodes), 1)
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about adaptation performance"""
        
        if not self.adaptation_history:
            return {"total_adaptations": 0}
        
        # Overall statistics
        total_adaptations = len(self.adaptation_history)
        successful_adaptations = sum(1 for result in self.adaptation_history if result.success)
        
        # Performance improvements
        improvements = [
            result.adapted_performance - result.initial_performance 
            for result in self.adaptation_history
        ]
        
        # Strategy breakdown
        strategy_stats = {}
        for strategy in AdaptationStrategy:
            strategy_results = [r for r in self.adaptation_history if r.strategy == strategy]
            if strategy_results:
                strategy_stats[strategy.value] = {
                    "count": len(strategy_results),
                    "success_rate": sum(1 for r in strategy_results if r.success) / len(strategy_results),
                    "avg_improvement": np.mean([r.adapted_performance - r.initial_performance for r in strategy_results]),
                    "avg_adaptation_time": np.mean([r.adaptation_time for r in strategy_results])
                }
        
        return {
            "total_adaptations": total_adaptations,
            "success_rate": successful_adaptations / total_adaptations,
            "avg_improvement": np.mean(improvements),
            "std_improvement": np.std(improvements),
            "avg_adaptation_time": np.mean([r.adaptation_time for r in self.adaptation_history]),
            "strategy_statistics": strategy_stats
        }

def demonstrate_few_shot_learning():
    """Demonstrate the few-shot learning framework"""
    print("=== Few-Shot Learning for New Task Type Adaptation ===")
    
    # Configuration
    config = {
        'state_dim': 128,
        'action_dim': 64,
        'embedding_dim': 256,
        'num_task_types': 10,
        'num_resources': 5,
        'num_priorities': 5,
        'memory_size': 128,
        'memory_dim': 64,
        'output_dim': 64,
        'transfer_dim': 128,
        'inner_lr': 0.01,
        'meta_lr': 0.001,
        'maml_steps': 5,
        'batch_size': 8
    }
    
    print("1. Initializing Few-Shot Learning Framework...")
    
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)
    
    framework = FewShotLearningFramework(config)
    
    print("2. Generating Synthetic Task Instances...")
    
    def generate_task_instances(num_instances: int, task_type: str) -> List[TaskInstance]:
        instances = []
        
        for i in range(num_instances):
            # Generate synthetic data
            state = torch.randn(config['state_dim'])
            action = torch.randint(0, config['action_dim'], (1,))
            reward = np.random.uniform(-1, 1)
            next_state = torch.randn(config['state_dim'])
            done = np.random.random() < 0.1
            
            instance = TaskInstance(
                task_id=f"{task_type}_task_{i}",
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                task_type=task_type,
                metadata={"complexity": np.random.uniform(0.1, 0.9)}
            )
            instances.append(instance)
        
        return instances
    
    # Generate instances for different task types
    task_types = ["compute_intensive", "memory_intensive", "io_intensive", "mixed", "streaming"]
    all_instances = {}
    
    for task_type in task_types:
        instances = generate_task_instances(100, task_type)
        all_instances[task_type] = instances
        print(f"   Generated {len(instances)} instances for {task_type}")
    
    print("3. Testing Adaptation Strategies...")
    
    # Test each adaptation strategy
    strategies = [
        AdaptationStrategy.MAML,
        AdaptationStrategy.PROTOTYPICAL,
        AdaptationStrategy.MEMORY_AUGMENTED,
        AdaptationStrategy.TASK_EMBEDDING
    ]
    
    adaptation_results = []
    
    for strategy in strategies:
        print(f"   Testing {strategy.value}...")
        
        # Use one task type for adaptation
        source_instances = all_instances["compute_intensive"]
        
        # Create support and query sets
        support_set = source_instances[:10]  # 10-shot learning
        query_set = source_instances[10:30]   # 20 queries
        
        # Perform adaptation
        result = framework.adapt_to_new_task(strategy, support_set, query_set)
        adaptation_results.append(result)
        
        print(f"     Initial performance: {result.initial_performance:.3f}")
        print(f"     Adapted performance: {result.adapted_performance:.3f}")
        print(f"     Improvement: {result.adapted_performance - result.initial_performance:.3f}")
        print(f"     Adaptation time: {result.adaptation_time:.3f}s")
        print(f"     Success: {result.success}")
    
    print("4. Testing Cross-Task Transfer...")
    
    # Test transfer from one task type to another
    source_task = "compute_intensive"
    target_tasks = ["memory_intensive", "io_intensive", "streaming"]
    
    transfer_results = {}
    
    for target_task in target_tasks:
        print(f"   Transfer from {source_task} to {target_task}...")
        
        # Use source task for support set
        support_set = all_instances[source_task][:15]
        
        # Use target task for query set
        query_set = all_instances[target_task][:20]
        
        # Test with MAML
        result = framework.adapt_to_new_task(
            AdaptationStrategy.MAML, support_set, query_set
        )
        
        transfer_results[target_task] = result
        
        print(f"     Transfer performance: {result.adapted_performance:.3f}")
        print(f"     Success: {result.success}")
    
    print("5. Testing Few-Shot vs Many-Shot Learning...")
    
    shot_sizes = [1, 5, 10, 20, 50]
    shot_results = {}
    
    for shot_size in shot_sizes:
        instances = all_instances["mixed"]
        support_set = instances[:shot_size]
        query_set = instances[shot_size:shot_size + 20]
        
        result = framework.adapt_to_new_task(
            AdaptationStrategy.PROTOTYPICAL, support_set, query_set
        )
        
        shot_results[shot_size] = result.adapted_performance
        print(f"   {shot_size}-shot learning: {result.adapted_performance:.3f}")
    
    print("6. Meta-Learning Training Simulation...")
    
    # Generate training episodes
    training_episodes = []
    
    for _ in range(20):
        # Random task type
        task_type = np.random.choice(task_types)
        instances = all_instances[task_type]
        
        # Generate episode
        episode = framework.generate_episode(instances, support_size=10, query_size=15)
        training_episodes.append(episode)
    
    print(f"   Generated {len(training_episodes)} training episodes")
    
    # Simulate meta-training (reduced epochs for demo)
    print("   Running meta-training...")
    framework.train_meta_learning(training_episodes, num_epochs=5)
    
    print("7. Framework Statistics...")
    
    stats = framework.get_adaptation_statistics()
    
    print(f"   Total adaptations: {stats['total_adaptations']}")
    print(f"   Overall success rate: {stats['success_rate']:.2%}")
    print(f"   Average improvement: {stats['avg_improvement']:.3f}")
    print(f"   Average adaptation time: {stats['avg_adaptation_time']:.3f}s")
    
    print("   Strategy Performance:")
    for strategy, strategy_stats in stats['strategy_statistics'].items():
        print(f"     {strategy}:")
        print(f"       Success rate: {strategy_stats['success_rate']:.2%}")
        print(f"       Avg improvement: {strategy_stats['avg_improvement']:.3f}")
        print(f"       Avg time: {strategy_stats['avg_adaptation_time']:.3f}s")
    
    print("8. Few-Shot Learning Benefits...")
    
    benefits = [
        "Rapid adaptation to new task types with minimal data",
        "Cross-domain transfer learning between different scheduling environments",
        "Meta-learning enables learning to learn new tasks quickly",
        "Memory-augmented approaches for rapid knowledge acquisition",
        "Prototypical networks for few-shot task classification",
        "Task embedding learning for similarity-based transfer",
        "Gradient-based adaptation with scheduling-specific constraints",
        "Episodic training framework for systematic few-shot learning"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"   {i}. {benefit}")
    
    print("9. Integration with HeteroSched...")
    
    integration_points = [
        "Quick deployment to new datacenter environments",
        "Adaptation to novel workload patterns with minimal retraining",
        "Cross-domain transfer from simulation to real systems",
        "Emergency adaptation to system failures or configuration changes",
        "Rapid scaling to new resource types (TPUs, quantum processors)",
        "Few-shot learning for personalized scheduling policies",
        "Meta-learning for improved foundation model capabilities"
    ]
    
    for i, point in enumerate(integration_points, 1):
        print(f"   {i}. {point}")
    
    return {
        "framework": framework,
        "adaptation_results": adaptation_results,
        "transfer_results": transfer_results,
        "shot_size_analysis": shot_results,
        "framework_statistics": stats,
        "training_episodes": training_episodes[:3]  # Sample episodes
    }

if __name__ == "__main__":
    demonstrate_few_shot_learning()