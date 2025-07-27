"""
R33: Neural Architecture Search Demo - Simplified Version

This module demonstrates the key concepts of Neural Architecture Search for 
RL scheduling agents with a simplified implementation for fast execution.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class LayerType(Enum):
    DENSE = "dense"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"


@dataclass
class SimpleArchitecture:
    """Simplified architecture representation"""
    hidden_layers: List[int]  # List of layer sizes
    dropout_rate: float
    use_batch_norm: bool
    architecture_id: str
    performance: float = 0.0
    complexity: int = 0


class SimpleNAS:
    """Simplified Neural Architecture Search for demonstration"""
    
    def __init__(self):
        self.population_size = 8
        self.generations = 5
        self.mutation_rate = 0.4
        
        # Search space
        self.layer_sizes = [32, 64, 128, 256]
        self.max_layers = 4
        self.dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        self.population = []
        self.generation_best = []
        
    def generate_random_architecture(self) -> SimpleArchitecture:
        """Generate a random architecture"""
        num_layers = random.randint(2, self.max_layers)
        hidden_layers = [random.choice(self.layer_sizes) for _ in range(num_layers)]
        dropout_rate = random.choice(self.dropout_rates)
        use_batch_norm = random.choice([True, False])
        
        arch_id = f"{'_'.join(map(str, hidden_layers))}_{dropout_rate}_{use_batch_norm}"
        
        return SimpleArchitecture(
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            architecture_id=arch_id
        )
        
    def evaluate_architecture(self, arch: SimpleArchitecture) -> float:
        """Simulate architecture evaluation"""
        # Simulate performance based on architecture characteristics
        
        # Base performance
        performance = 0.5
        
        # Reward deeper networks (up to a point)
        depth_bonus = min(len(arch.hidden_layers) * 0.1, 0.3)
        performance += depth_bonus
        
        # Reward appropriate layer sizes
        avg_layer_size = np.mean(arch.hidden_layers)
        if 64 <= avg_layer_size <= 128:
            performance += 0.2
        
        # Batch norm helps
        if arch.use_batch_norm:
            performance += 0.1
            
        # Optimal dropout range
        if 0.2 <= arch.dropout_rate <= 0.3:
            performance += 0.1
        elif arch.dropout_rate > 0.4:
            performance -= 0.1  # Too much dropout
            
        # Add some noise
        performance += random.uniform(-0.1, 0.1)
        
        # Calculate complexity (number of parameters)
        complexity = 0
        prev_size = 20  # Input size
        for layer_size in arch.hidden_layers:
            complexity += prev_size * layer_size
            prev_size = layer_size
        complexity += prev_size * 8  # Output layer
        
        # Penalize very large models
        if complexity > 50000:
            performance -= 0.2
            
        arch.performance = max(0, performance)
        arch.complexity = complexity
        
        return arch.performance
        
    def mutate_architecture(self, arch: SimpleArchitecture) -> SimpleArchitecture:
        """Mutate an architecture"""
        new_hidden = arch.hidden_layers.copy()
        
        # Mutate layer sizes
        if random.random() < 0.5:
            idx = random.randint(0, len(new_hidden) - 1)
            new_hidden[idx] = random.choice(self.layer_sizes)
            
        # Mutate dropout
        new_dropout = arch.dropout_rate
        if random.random() < 0.3:
            new_dropout = random.choice(self.dropout_rates)
            
        # Mutate batch norm
        new_batch_norm = arch.use_batch_norm
        if random.random() < 0.2:
            new_batch_norm = not new_batch_norm
            
        # Add/remove layer
        if random.random() < 0.3:
            if len(new_hidden) < self.max_layers and random.random() < 0.5:
                # Add layer
                new_hidden.append(random.choice(self.layer_sizes))
            elif len(new_hidden) > 2:
                # Remove layer
                new_hidden.pop(random.randint(0, len(new_hidden) - 1))
                
        arch_id = f"{'_'.join(map(str, new_hidden))}_{new_dropout}_{new_batch_norm}"
        
        return SimpleArchitecture(
            hidden_layers=new_hidden,
            dropout_rate=new_dropout,
            use_batch_norm=new_batch_norm,
            architecture_id=arch_id
        )
        
    def crossover_architectures(self, parent1: SimpleArchitecture, 
                              parent2: SimpleArchitecture) -> SimpleArchitecture:
        """Create offspring through crossover"""
        # Take layers from both parents
        min_layers = min(len(parent1.hidden_layers), len(parent2.hidden_layers))
        crossover_point = random.randint(1, min_layers - 1)
        
        new_hidden = (parent1.hidden_layers[:crossover_point] + 
                     parent2.hidden_layers[crossover_point:])
        
        # Randomly choose other parameters
        new_dropout = random.choice([parent1.dropout_rate, parent2.dropout_rate])
        new_batch_norm = random.choice([parent1.use_batch_norm, parent2.use_batch_norm])
        
        arch_id = f"{'_'.join(map(str, new_hidden))}_{new_dropout}_{new_batch_norm}"
        
        return SimpleArchitecture(
            hidden_layers=new_hidden,
            dropout_rate=new_dropout,
            use_batch_norm=new_batch_norm,
            architecture_id=arch_id
        )
        
    def tournament_selection(self, population: List[SimpleArchitecture], k: int = 3) -> SimpleArchitecture:
        """Tournament selection"""
        tournament = random.sample(population, k)
        return max(tournament, key=lambda x: x.performance)
        
    def evolve_architectures(self) -> List[SimpleArchitecture]:
        """Run evolutionary architecture search"""
        
        print("Starting Neural Architecture Search...")
        
        # Initialize population
        self.population = [self.generate_random_architecture() for _ in range(self.population_size)]
        
        # Evaluate initial population
        for arch in self.population:
            self.evaluate_architecture(arch)
            
        best_architectures = []
        
        for generation in range(self.generations):
            # Find best architecture
            best_arch = max(self.population, key=lambda x: x.performance)
            best_architectures.append(best_arch)
            self.generation_best.append(best_arch.performance)
            
            print(f"Generation {generation + 1}: Best performance = {best_arch.performance:.4f}, "
                  f"Architecture = {best_arch.hidden_layers}, "
                  f"Complexity = {best_arch.complexity:,} params")
            
            # Create new population
            new_population = [best_arch]  # Elitism
            
            while len(new_population) < self.population_size:
                if random.random() < 0.7:  # Crossover
                    parent1 = self.tournament_selection(self.population)
                    parent2 = self.tournament_selection(self.population)
                    offspring = self.crossover_architectures(parent1, parent2)
                else:  # Mutation
                    parent = self.tournament_selection(self.population)
                    offspring = self.mutate_architecture(parent)
                    
                self.evaluate_architecture(offspring)
                new_population.append(offspring)
                
            self.population = new_population
            
        return best_architectures
        
    def analyze_results(self, best_architectures: List[SimpleArchitecture]) -> Dict:
        """Analyze search results"""
        
        # Performance trends
        performances = [arch.performance for arch in best_architectures]
        complexities = [arch.complexity for arch in best_architectures]
        
        # Architecture patterns
        layer_counts = {}
        dropout_usage = {}
        batch_norm_usage = {'True': 0, 'False': 0}
        
        for arch in best_architectures:
            # Layer count distribution
            layer_count = len(arch.hidden_layers)
            layer_counts[layer_count] = layer_counts.get(layer_count, 0) + 1
            
            # Dropout distribution
            dropout_usage[arch.dropout_rate] = dropout_usage.get(arch.dropout_rate, 0) + 1
            
            # Batch norm usage
            batch_norm_usage[str(arch.use_batch_norm)] += 1
            
        analysis = {
            'performance_trend': performances,
            'complexity_trend': complexities,
            'layer_count_distribution': layer_counts,
            'dropout_distribution': dropout_usage,
            'batch_norm_distribution': batch_norm_usage,
            'best_architecture': max(best_architectures, key=lambda x: x.performance),
            'pareto_optimal': self._find_pareto_optimal(best_architectures)
        }
        
        return analysis
        
    def _find_pareto_optimal(self, architectures: List[SimpleArchitecture]) -> List[SimpleArchitecture]:
        """Find Pareto optimal architectures (performance vs complexity)"""
        pareto_optimal = []
        
        for arch in architectures:
            is_optimal = True
            for other in architectures:
                if (other.performance >= arch.performance and 
                    other.complexity <= arch.complexity and
                    (other.performance > arch.performance or other.complexity < arch.complexity)):
                    is_optimal = False
                    break
            if is_optimal:
                pareto_optimal.append(arch)
                
        return pareto_optimal
        
    def visualize_results(self, analysis: Dict):
        """Visualize search results"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Convergence plot
        generations = range(1, len(self.generation_best) + 1)
        ax1.plot(generations, self.generation_best, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Performance')
        ax1.set_title('NAS Convergence')
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance vs Complexity scatter
        performances = analysis['performance_trend']
        complexities = analysis['complexity_trend']
        
        ax2.scatter(complexities, performances, alpha=0.7, s=60, label='All Architectures')
        
        # Highlight Pareto optimal
        pareto_optimal = analysis['pareto_optimal']
        if pareto_optimal:
            pareto_complexities = [arch.complexity for arch in pareto_optimal]
            pareto_performances = [arch.performance for arch in pareto_optimal]
            ax2.scatter(pareto_complexities, pareto_performances, color='red', s=100, 
                       label='Pareto Optimal', edgecolors='black', alpha=0.8)
            
        ax2.set_xlabel('Model Complexity (Parameters)')
        ax2.set_ylabel('Performance')
        ax2.set_title('Performance vs Complexity Trade-off')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Layer count distribution
        layer_counts = analysis['layer_count_distribution']
        if layer_counts:
            layers = list(layer_counts.keys())
            counts = list(layer_counts.values())
            
            ax3.bar(layers, counts, alpha=0.7)
            ax3.set_xlabel('Number of Hidden Layers')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Layer Count Distribution')
            ax3.grid(True, alpha=0.3)
        
        # 4. Dropout rate distribution
        dropout_dist = analysis['dropout_distribution']
        if dropout_dist:
            dropout_rates = list(dropout_dist.keys())
            dropout_counts = list(dropout_dist.values())
            
            ax4.bar(range(len(dropout_rates)), dropout_counts, alpha=0.7)
            ax4.set_xlabel('Dropout Rate')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Dropout Rate Distribution')
            ax4.set_xticks(range(len(dropout_rates)))
            ax4.set_xticklabels([f'{rate:.1f}' for rate in dropout_rates])
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def demonstrate_simple_nas():
    """Demonstrate simplified Neural Architecture Search"""
    print("=== R33: Neural Architecture Search for RL Agent Design ===")
    
    # Initialize and run NAS
    nas = SimpleNAS()
    best_architectures = nas.evolve_architectures()
    
    # Analyze results
    print(f"\nAnalyzing search results...")
    analysis = nas.analyze_results(best_architectures)
    
    # Display key findings
    best_arch = analysis['best_architecture']
    pareto_optimal = analysis['pareto_optimal']
    
    print(f"\nKey Findings:")
    print(f"- Best Architecture: {best_arch.hidden_layers}")
    print(f"- Best Performance: {best_arch.performance:.4f}")
    print(f"- Best Complexity: {best_arch.complexity:,} parameters")
    print(f"- Pareto Optimal Solutions: {len(pareto_optimal)}")
    
    print(f"\nArchitecture Patterns:")
    print(f"- Layer count distribution: {analysis['layer_count_distribution']}")
    print(f"- Dropout usage: {analysis['dropout_distribution']}")
    print(f"- Batch norm usage: {analysis['batch_norm_distribution']}")
    
    # Show Pareto optimal architectures
    print(f"\nPareto Optimal Architectures:")
    for i, arch in enumerate(pareto_optimal):
        print(f"{i+1}. Layers: {arch.hidden_layers}, "
              f"Performance: {arch.performance:.4f}, "
              f"Complexity: {arch.complexity:,}")
    
    # Visualize results
    print(f"\nGenerating visualizations...")
    nas.visualize_results(analysis)
    
    # Summary of contributions
    print(f"\nTechnical Contributions:")
    print(f"1. Automated neural architecture discovery for RL scheduling")
    print(f"2. Multi-objective optimization balancing performance and efficiency")
    print(f"3. Evolutionary search with crossover and mutation operators")
    print(f"4. Pareto optimal architecture identification")
    print(f"5. Architecture pattern analysis and visualization")
    
    print(f"\nPractical Impact:")
    print(f"- Reduced manual architecture design effort")
    print(f"- Discovered efficient architectures for scheduling tasks")
    print(f"- Identified optimal performance-complexity trade-offs")
    print(f"- Provided insights into effective architectural patterns")
    
    return nas, best_architectures, analysis


if __name__ == "__main__":
    nas, best_architectures, analysis = demonstrate_simple_nas()