#!/usr/bin/env python3
"""
Foundation Model Architecture for Heterogeneous Scheduling

This module implements a foundational deep learning architecture specifically designed
for heterogeneous task scheduling that can be pre-trained on diverse workloads and
fine-tuned for specific deployment scenarios.

Research Innovation: First foundation model architecture specifically designed for
heterogeneous scheduling with self-supervised pre-training and multi-task learning.

Key Components:
- Transformer-based architecture with scheduling-specific inductive biases
- Multi-scale temporal modeling for different scheduling horizons
- Cross-modal attention for CPU/GPU/memory resource modeling
- Self-supervised pre-training on synthetic and real workloads
- Few-shot adaptation to new scheduling environments
- Hierarchical representation learning for scalability

Authors: HeteroSched Research Team
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import math
from collections import deque

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

@dataclass
class FoundationModelConfig:
    """Configuration for the foundation model architecture"""
    
    # Architecture parameters
    d_model: int = 512                    # Model dimension
    n_heads: int = 8                      # Number of attention heads
    n_layers: int = 12                    # Number of transformer layers
    d_ff: int = 2048                      # Feed-forward dimension
    dropout: float = 0.1                  # Dropout rate
    
    # Input/Output dimensions
    state_dim: int = 36                   # System state dimension
    action_dim: int = 100                 # Action space size
    resource_types: int = 4               # CPU, GPU, Memory, Network
    task_embedding_dim: int = 128         # Task embedding dimension
    
    # Temporal modeling
    max_sequence_length: int = 1000       # Maximum sequence length
    temporal_scales: List[int] = field(default_factory=lambda: [1, 4, 16, 64])  # Multi-scale
    
    # Multi-task learning
    n_pretraining_tasks: int = 8          # Number of pre-training tasks
    task_specific_heads: bool = True      # Task-specific output heads
    
    # Foundation model specific
    enable_self_supervised: bool = True   # Self-supervised pre-training
    masked_prediction_ratio: float = 0.15 # Masking ratio for pre-training
    enable_contrastive_learning: bool = True  # Contrastive learning
    
    # Fine-tuning
    enable_few_shot: bool = True          # Few-shot learning capability
    adapter_dim: int = 64                 # Adapter layers dimension
    enable_prompt_tuning: bool = True     # Prompt-based fine-tuning

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class ResourceAwareAttention(nn.Module):
    """Resource-aware multi-head attention for heterogeneous systems"""
    
    def __init__(self, d_model: int, n_heads: int, resource_types: int):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.resource_types = resource_types
        self.d_k = d_model // n_heads
        
        # Standard attention components
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Resource-specific attention biases
        self.resource_bias = nn.Parameter(torch.randn(n_heads, resource_types, resource_types))
        
        # Resource type embeddings
        self.resource_embeddings = nn.Embedding(resource_types, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, resource_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply resource-aware bias if resource mask is provided
        if resource_mask is not None:
            # resource_mask: [batch_size, seq_len] with resource type indices
            resource_bias_expanded = self.resource_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            # Apply resource bias (simplified - full implementation would be more complex)
            for b in range(batch_size):
                for i in range(seq_len):
                    for j in range(seq_len):
                        res_i = resource_mask[b, i].long()
                        res_j = resource_mask[b, j].long()
                        if res_i < self.resource_types and res_j < self.resource_types:
                            scores[b, :, i, j] += resource_bias_expanded[b, :, res_i, res_j]
        
        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) -> expand to (batch_size, n_heads, seq_len, seq_len)
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.n_heads, seq_len, seq_len)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Softmax and apply to values
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)

class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoding for different scheduling horizons"""
    
    def __init__(self, d_model: int, temporal_scales: List[int]):
        super().__init__()
        
        self.d_model = d_model
        self.temporal_scales = temporal_scales
        self.n_scales = len(temporal_scales)
        
        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList([
            nn.Conv1d(d_model, d_model // self.n_scales, kernel_size=scale, 
                     stride=scale, padding=0)
            for scale in temporal_scales
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # Transpose for conv1d: [batch_size, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        
        # Multi-scale encoding
        scale_outputs = []
        for i, encoder in enumerate(self.scale_encoders):
            # Skip scales that are larger than sequence length
            if self.temporal_scales[i] > seq_len:
                # Use average pooling as fallback
                fallback_out = F.avg_pool1d(x_conv, kernel_size=min(self.temporal_scales[i], seq_len), 
                                          stride=1, padding=min(self.temporal_scales[i]//2, seq_len//2))
                fallback_out = fallback_out[:, :x_conv.shape[1]//self.n_scales, :]
                # Ensure correct size
                if fallback_out.shape[2] != seq_len:
                    fallback_out = F.interpolate(fallback_out, size=seq_len, mode='linear', align_corners=False)
                scale_outputs.append(fallback_out)
            else:
                scale_out = encoder(x_conv)  # [batch_size, d_model//n_scales, new_seq_len]
                
                # Upsample back to original sequence length
                if scale_out.shape[2] != seq_len:
                    scale_out = F.interpolate(scale_out, size=seq_len, mode='linear', align_corners=False)
                scale_outputs.append(scale_out)
        
        # Concatenate scale outputs
        multi_scale = torch.cat(scale_outputs, dim=1)  # [batch_size, d_model, seq_len]
        multi_scale = multi_scale.transpose(1, 2)  # [batch_size, seq_len, d_model]
        
        # Fusion and residual connection
        fused = self.fusion(multi_scale)
        return self.layer_norm(x + fused)

class TaskEmbedding(nn.Module):
    """Hierarchical task embedding for heterogeneous workloads"""
    
    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        
        self.config = config
        
        # Task type embedding
        self.task_type_embedding = nn.Embedding(20, config.task_embedding_dim)  # 20 task types
        
        # Resource requirement embedding
        self.resource_embedding = nn.Linear(config.resource_types, config.task_embedding_dim)
        
        # Priority embedding
        self.priority_embedding = nn.Embedding(10, config.task_embedding_dim)  # 10 priority levels
        
        # Size/complexity embedding
        self.size_embedding = nn.Linear(1, config.task_embedding_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(config.task_embedding_dim * 4, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
    def forward(self, task_type: torch.Tensor, resource_req: torch.Tensor,
                priority: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_type: [batch_size, seq_len] - task type indices
            resource_req: [batch_size, seq_len, resource_types] - resource requirements
            priority: [batch_size, seq_len] - priority levels
            size: [batch_size, seq_len, 1] - task sizes
        """
        
        # Individual embeddings - ensure correct dtypes
        type_emb = self.task_type_embedding(task_type.long())
        resource_emb = self.resource_embedding(resource_req.float())
        priority_emb = self.priority_embedding(priority.long())
        size_emb = self.size_embedding(size.float())
        
        # Concatenate and fuse
        combined = torch.cat([type_emb, resource_emb, priority_emb, size_emb], dim=-1)
        return self.fusion(combined)

class FoundationTransformerLayer(nn.Module):
    """Transformer layer with scheduling-specific modifications"""
    
    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        
        self.config = config
        
        # Resource-aware attention
        self.attention = ResourceAwareAttention(
            config.d_model, config.n_heads, config.resource_types
        )
        
        # Multi-scale temporal encoding
        self.temporal_encoder = MultiScaleTemporalEncoder(
            config.d_model, config.temporal_scales
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, resource_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Resource-aware attention with residual connection
        attn_out = self.attention(x, resource_mask, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Multi-scale temporal encoding with residual connection
        temporal_out = self.temporal_encoder(x)
        x = self.norm2(x + self.dropout(temporal_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

class AdapterLayer(nn.Module):
    """Adapter layer for efficient fine-tuning"""
    
    def __init__(self, d_model: int, adapter_dim: int):
        super().__init__()
        
        self.down_project = nn.Linear(d_model, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return residual + x

class HeteroSchedFoundationModel(nn.Module):
    """Foundation model for heterogeneous scheduling"""
    
    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        
        self.config = config
        
        # Input embeddings
        self.state_embedding = nn.Linear(config.state_dim, config.d_model)
        self.task_embedding = TaskEmbedding(config)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_sequence_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            FoundationTransformerLayer(config) for _ in range(config.n_layers)
        ])
        
        # Adapter layers (for fine-tuning)
        if config.adapter_dim > 0:
            self.adapters = nn.ModuleList([
                AdapterLayer(config.d_model, config.adapter_dim) 
                for _ in range(config.n_layers)
            ])
        else:
            self.adapters = None
        
        # Pre-training heads
        if config.enable_self_supervised:
            self.masked_prediction_head = nn.Linear(config.d_model, config.state_dim)
            self.contrastive_projection = nn.Linear(config.d_model, 256)
        
        # Multi-task output heads
        self.output_heads = nn.ModuleDict({
            'action_prediction': nn.Linear(config.d_model, config.action_dim),
            'value_estimation': nn.Linear(config.d_model, 1),
            'resource_utilization': nn.Linear(config.d_model, config.resource_types),
            'latency_prediction': nn.Linear(config.d_model, 1),
            'energy_prediction': nn.Linear(config.d_model, 1),
            'throughput_prediction': nn.Linear(config.d_model, 1),
            'queue_length_prediction': nn.Linear(config.d_model, 1),
            'system_stability': nn.Linear(config.d_model, 2)  # stable/unstable
        })
        
        # Prompt embeddings for prompt tuning
        if config.enable_prompt_tuning:
            self.prompt_embeddings = nn.Parameter(
                torch.randn(config.n_pretraining_tasks, 10, config.d_model)  # 10 prompt tokens per task
            )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, state_sequence: torch.Tensor, task_sequence: Dict[str, torch.Tensor],
                resource_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                task_id: Optional[int] = None,
                output_head: str = 'action_prediction') -> torch.Tensor:
        """
        Forward pass of the foundation model
        
        Args:
            state_sequence: [batch_size, seq_len, state_dim] - system state sequence
            task_sequence: Dict with task information
            resource_mask: [batch_size, seq_len] - resource type mask
            attention_mask: [batch_size, seq_len, seq_len] - attention mask
            task_id: Task ID for prompt tuning
            output_head: Which output head to use
        """
        
        batch_size, seq_len, _ = state_sequence.shape
        
        # Embed state sequence
        state_emb = self.state_embedding(state_sequence)
        
        # Embed task sequence
        task_emb = self.task_embedding(
            task_sequence['task_type'],
            task_sequence['resource_req'],
            task_sequence['priority'],
            task_sequence['size']
        )
        
        # Combine embeddings
        x = state_emb + task_emb
        
        # Add prompt tokens if using prompt tuning
        if self.config.enable_prompt_tuning and task_id is not None:
            prompt_tokens = self.prompt_embeddings[task_id].unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([prompt_tokens, x], dim=1)
            
            # Adjust masks for prompt tokens
            if attention_mask is not None:
                prompt_mask = torch.ones(batch_size, 10, x.size(1), device=x.device)
                attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            
            if resource_mask is not None:
                prompt_resource_mask = torch.zeros(batch_size, 10, device=x.device, dtype=resource_mask.dtype)
                resource_mask = torch.cat([prompt_resource_mask, resource_mask], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, resource_mask, attention_mask)
            
            # Apply adapter if available
            if self.adapters is not None:
                x = self.adapters[i](x)
        
        # Remove prompt tokens if they were added
        if self.config.enable_prompt_tuning and task_id is not None:
            x = x[:, 10:, :]  # Remove first 10 tokens (prompts)
        
        # Apply output head
        if output_head in self.output_heads:
            output = self.output_heads[output_head](x)
        else:
            raise ValueError(f"Unknown output head: {output_head}")
        
        return output
    
    def get_representations(self, state_sequence: torch.Tensor, 
                          task_sequence: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get learned representations without output head"""
        
        batch_size, seq_len, _ = state_sequence.shape
        
        # Embed sequences
        state_emb = self.state_embedding(state_sequence)
        task_emb = self.task_embedding(
            task_sequence['task_type'],
            task_sequence['resource_req'],
            task_sequence['priority'],
            task_sequence['size']
        )
        
        # Combine and encode
        x = state_emb + task_emb
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            if self.adapters is not None:
                x = self.adapters[i](x)
        
        return x
    
    def masked_language_modeling_loss(self, state_sequence: torch.Tensor,
                                    task_sequence: Dict[str, torch.Tensor],
                                    mask_ratio: float = 0.15) -> torch.Tensor:
        """Compute masked language modeling loss for pre-training"""
        
        batch_size, seq_len, state_dim = state_sequence.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len) < mask_ratio
        masked_state = state_sequence.clone()
        masked_state[mask] = 0  # Mask with zeros
        
        # Forward pass with masked input
        representations = self.get_representations(masked_state, task_sequence)
        predictions = self.masked_prediction_head(representations)
        
        # Compute loss only on masked positions
        loss = F.mse_loss(predictions[mask], state_sequence[mask])
        
        return loss
    
    def contrastive_loss(self, state_sequence: torch.Tensor,
                        task_sequence: Dict[str, torch.Tensor],
                        temperature: float = 0.1) -> torch.Tensor:
        """Compute contrastive learning loss"""
        
        batch_size = state_sequence.shape[0]
        
        # Get representations
        representations = self.get_representations(state_sequence, task_sequence)
        
        # Pool representations (mean pooling)
        pooled_repr = torch.mean(representations, dim=1)  # [batch_size, d_model]
        
        # Project to contrastive space
        projections = self.contrastive_projection(pooled_repr)  # [batch_size, 256]
        
        # Normalize
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / temperature
        
        # Labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=projections.device)
        
        # Contrastive loss (InfoNCE)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def freeze_backbone(self):
        """Freeze backbone for fine-tuning"""
        for param in self.transformer_layers.parameters():
            param.requires_grad = False
        for param in self.state_embedding.parameters():
            param.requires_grad = False
        for param in self.task_embedding.parameters():
            param.requires_grad = False
    
    def unfreeze_adapters(self):
        """Unfreeze adapter layers for fine-tuning"""
        if self.adapters is not None:
            for param in self.adapters.parameters():
                param.requires_grad = True

class FoundationModelTrainer:
    """Trainer for the foundation model with pre-training and fine-tuning"""
    
    def __init__(self, model: HeteroSchedFoundationModel, config: FoundationModelConfig):
        self.model = model
        self.config = config
        
        # Optimizers
        self.pretraining_optimizer = optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
        
        self.finetuning_optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=1e-5, weight_decay=0.01
        )
        
        # Learning rate schedulers
        self.pretraining_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.pretraining_optimizer, T_max=1000
        )
        
        # Training statistics
        self.training_stats = {
            'pretraining_losses': [],
            'finetuning_losses': [],
            'validation_scores': []
        }
        
    def pretrain_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single pre-training step"""
        
        self.model.train()
        self.pretraining_optimizer.zero_grad()
        
        state_sequence = batch['state_sequence']
        task_sequence = {
            'task_type': batch['task_type'],
            'resource_req': batch['resource_req'],
            'priority': batch['priority'],
            'size': batch['size']
        }
        
        total_loss = 0.0
        losses = {}
        
        # Masked language modeling loss
        if self.config.enable_self_supervised:
            mlm_loss = self.model.masked_language_modeling_loss(
                state_sequence, task_sequence, self.config.masked_prediction_ratio
            )
            total_loss += mlm_loss
            losses['mlm_loss'] = float(mlm_loss)
        
        # Contrastive learning loss
        if self.config.enable_contrastive_learning:
            contrastive_loss = self.model.contrastive_loss(state_sequence, task_sequence)
            total_loss += 0.5 * contrastive_loss
            losses['contrastive_loss'] = float(contrastive_loss)
        
        # Multi-task prediction losses
        for head_name, head in self.model.output_heads.items():
            if head_name in batch:  # Only compute loss if we have labels
                predictions = self.model(
                    state_sequence, task_sequence, output_head=head_name
                )
                
                if head_name == 'system_stability':
                    # Classification loss
                    task_loss = F.cross_entropy(
                        predictions.view(-1, 2), batch[head_name].view(-1)
                    )
                elif head_name == 'action_prediction':
                    # Classification loss for actions
                    task_loss = F.cross_entropy(
                        predictions.view(-1, predictions.size(-1)), batch[head_name].view(-1)
                    )
                else:
                    # Regression loss - handle different output dimensions
                    if predictions.dim() == 3 and predictions.size(-1) > 1:
                        # Multi-dimensional output (e.g., resource_utilization)
                        task_loss = F.mse_loss(predictions, batch[head_name])
                    else:
                        # Single-dimensional output
                        task_loss = F.mse_loss(
                            predictions.squeeze(-1), batch[head_name]
                        )
                
                total_loss += 0.1 * task_loss  # Weighted multi-task loss
                losses[f'{head_name}_loss'] = float(task_loss)
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.pretraining_optimizer.step()
        self.pretraining_scheduler.step()
        
        losses['total_loss'] = float(total_loss)
        self.training_stats['pretraining_losses'].append(losses)
        
        return losses
    
    def finetune_step(self, batch: Dict[str, torch.Tensor], 
                     task_id: int, target_head: str) -> Dict[str, float]:
        """Single fine-tuning step"""
        
        self.model.train()
        self.finetuning_optimizer.zero_grad()
        
        state_sequence = batch['state_sequence']
        task_sequence = {
            'task_type': batch['task_type'],
            'resource_req': batch['resource_req'],
            'priority': batch['priority'],
            'size': batch['size']
        }
        
        # Forward pass with task-specific prompt
        predictions = self.model(
            state_sequence, task_sequence, 
            task_id=task_id, output_head=target_head
        )
        
        # Compute task-specific loss
        if target_head == 'system_stability':
            loss = F.cross_entropy(
                predictions.view(-1, 2), batch[target_head].view(-1)
            )
        elif target_head == 'action_prediction':
            # Classification loss for actions
            loss = F.cross_entropy(
                predictions.view(-1, predictions.size(-1)), batch[target_head].view(-1)
            )
        else:
            # Regression loss - handle different output dimensions
            if predictions.dim() == 3 and predictions.size(-1) > 1:
                # Multi-dimensional output (e.g., resource_utilization)
                loss = F.mse_loss(predictions, batch[target_head])
            else:
                # Single-dimensional output
                loss = F.mse_loss(
                    predictions.squeeze(-1), batch[target_head]
                )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.finetuning_optimizer.step()
        
        losses = {'finetuning_loss': float(loss)}
        self.training_stats['finetuning_losses'].append(losses)
        
        return losses
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'pretraining_optimizer_state_dict': self.pretraining_optimizer.state_dict(),
            'finetuning_optimizer_state_dict': self.finetuning_optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.pretraining_optimizer.load_state_dict(checkpoint['pretraining_optimizer_state_dict'])
        self.finetuning_optimizer.load_state_dict(checkpoint['finetuning_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        return checkpoint['epoch']

def create_synthetic_batch(batch_size: int = 8, seq_len: int = 100, 
                          config: FoundationModelConfig = None) -> Dict[str, torch.Tensor]:
    """Create synthetic batch for testing"""
    
    if config is None:
        config = FoundationModelConfig()
    
    batch = {
        'state_sequence': torch.randn(batch_size, seq_len, config.state_dim),
        'task_type': torch.randint(0, 20, (batch_size, seq_len)),
        'resource_req': torch.rand(batch_size, seq_len, config.resource_types),
        'priority': torch.randint(0, 10, (batch_size, seq_len)),
        'size': torch.rand(batch_size, seq_len, 1),
        
        # Labels for multi-task learning
        'action_prediction': torch.randint(0, config.action_dim, (batch_size, seq_len)),
        'value_estimation': torch.randn(batch_size, seq_len),
        'resource_utilization': torch.rand(batch_size, seq_len, config.resource_types),
        'latency_prediction': torch.rand(batch_size, seq_len),
        'energy_prediction': torch.rand(batch_size, seq_len),
        'throughput_prediction': torch.rand(batch_size, seq_len),
        'queue_length_prediction': torch.randint(0, 100, (batch_size, seq_len)).float(),
        'system_stability': torch.randint(0, 2, (batch_size, seq_len))
    }
    
    return batch

def main():
    """Demonstrate foundation model architecture"""
    
    print("=== Foundation Model Architecture for Heterogeneous Scheduling ===\n")
    
    # Configuration
    config = FoundationModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,  # Smaller for demo
        state_dim=36,
        action_dim=100,
        max_sequence_length=200
    )
    
    print("1. Model Configuration:")
    print(f"   Model Dimension: {config.d_model}")
    print(f"   Attention Heads: {config.n_heads}")
    print(f"   Transformer Layers: {config.n_layers}")
    print(f"   State Dimension: {config.state_dim}")
    print(f"   Action Dimension: {config.action_dim}")
    print(f"   Max Sequence Length: {config.max_sequence_length}")
    print(f"   Multi-task Heads: {config.n_pretraining_tasks}")
    
    print("\n2. Initializing Foundation Model...")
    
    # Initialize model
    model = HeteroSchedFoundationModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    print("\n3. Testing Forward Pass...")
    
    # Create synthetic batch
    batch = create_synthetic_batch(batch_size=4, seq_len=50, config=config)
    
    # Test different output heads
    output_heads = ['action_prediction', 'value_estimation', 'resource_utilization']
    
    for head in output_heads:
        with torch.no_grad():
            output = model(
                batch['state_sequence'],
                {
                    'task_type': batch['task_type'],
                    'resource_req': batch['resource_req'],
                    'priority': batch['priority'],
                    'size': batch['size']
                },
                output_head=head
            )
            print(f"   {head}: {output.shape}")
    
    print("\n4. Testing Pre-training Components...")
    
    # Test masked language modeling
    with torch.no_grad():
        mlm_loss = model.masked_language_modeling_loss(
            batch['state_sequence'],
            {
                'task_type': batch['task_type'],
                'resource_req': batch['resource_req'],
                'priority': batch['priority'],
                'size': batch['size']
            }
        )
        print(f"   Masked LM Loss: {mlm_loss:.4f}")
    
    # Test contrastive learning
    with torch.no_grad():
        contrastive_loss = model.contrastive_loss(
            batch['state_sequence'],
            {
                'task_type': batch['task_type'],
                'resource_req': batch['resource_req'],
                'priority': batch['priority'],
                'size': batch['size']
            }
        )
        print(f"   Contrastive Loss: {contrastive_loss:.4f}")
    
    print("\n5. Testing Trainer...")
    
    # Initialize trainer
    trainer = FoundationModelTrainer(model, config)
    
    # Pre-training step
    pretrain_losses = trainer.pretrain_step(batch)
    print(f"   Pre-training losses:")
    for loss_name, loss_value in pretrain_losses.items():
        print(f"     {loss_name}: {loss_value:.4f}")
    
    # Fine-tuning setup
    print("\n6. Testing Fine-tuning...")
    
    # Freeze backbone and unfreeze adapters
    model.freeze_backbone()
    model.unfreeze_adapters()
    
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Frozen Parameters: {frozen_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Trainable Ratio: {trainable_params / total_params:.1%}")
    
    # Fine-tuning step
    finetune_losses = trainer.finetune_step(batch, task_id=0, target_head='action_prediction')
    print(f"   Fine-tuning loss: {finetune_losses['finetuning_loss']:.4f}")
    
    print("\n7. Testing Prompt Tuning...")
    
    if config.enable_prompt_tuning:
        with torch.no_grad():
            # Test with different task prompts
            for task_id in range(3):
                output = model(
                    batch['state_sequence'],
                    {
                        'task_type': batch['task_type'],
                        'resource_req': batch['resource_req'],
                        'priority': batch['priority'],
                        'size': batch['size']
                    },
                    task_id=task_id,
                    output_head='action_prediction'
                )
                print(f"   Task {task_id} output shape: {output.shape}")
    
    # Test representation extraction
    print("\n8. Testing Representation Learning...")
    
    with torch.no_grad():
        representations = model.get_representations(
            batch['state_sequence'],
            {
                'task_type': batch['task_type'],
                'resource_req': batch['resource_req'],
                'priority': batch['priority'],
                'size': batch['size']
            }
        )
        print(f"   Representation shape: {representations.shape}")
        print(f"   Representation mean: {representations.mean():.4f}")
        print(f"   Representation std: {representations.std():.4f}")
    
    print("\n[SUCCESS] Foundation Model Architecture Test Completed!")
    print("\nKey Foundation Model Features Demonstrated:")
    print("+ Transformer-based architecture with scheduling-specific modifications")
    print("+ Multi-scale temporal modeling for different scheduling horizons")
    print("+ Resource-aware attention mechanisms")
    print("+ Self-supervised pre-training with masked prediction and contrastive learning")
    print("+ Multi-task learning with 8 different prediction heads")
    print("+ Efficient fine-tuning with adapter layers and prompt tuning")
    print("+ Hierarchical task embeddings for heterogeneous workloads")
    print("+ Scalable architecture suitable for foundation model deployment")

if __name__ == '__main__':
    main()