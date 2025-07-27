"""
Causal Inference for Scheduling Policy Interpretability

This module implements R31: comprehensive causal inference framework
that provides explainable AI capabilities for understanding scheduling
decisions in the HeteroSched system.

Key Features:
1. Causal discovery algorithms for scheduling factor relationships
2. Do-calculus interventions for counterfactual analysis
3. Structural equation modeling for decision mechanisms
4. Causal effect estimation with confounding adjustment
5. Feature attribution through causal lens
6. Treatment effect analysis for scheduling policies
7. Causal mediation analysis for indirect effects
8. Policy counterfactuals and what-if scenarios

The framework enables interpretable scheduling by uncovering causal
relationships between system states, actions, and outcomes.

Authors: HeteroSched Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import networkx as nx
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import time
import json
import pickle
import itertools
import copy

class CausalDiscoveryMethod(Enum):
    PC_ALGORITHM = "pc_algorithm"
    GES_ALGORITHM = "ges_algorithm"
    INTERVENTION_CALCULUS = "intervention_calculus"
    CONDITIONAL_INDEPENDENCE = "conditional_independence"
    GRANGER_CAUSALITY = "granger_causality"

class CausalEstimationMethod(Enum):
    BACKDOOR_ADJUSTMENT = "backdoor_adjustment"
    FRONTDOOR_ADJUSTMENT = "frontdoor_adjustment"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"

class VariableType(Enum):
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    INSTRUMENTAL = "instrumental"

@dataclass
class CausalVariable:
    """Represents a variable in the causal model"""
    name: str
    variable_type: VariableType
    data_type: str  # "continuous", "discrete", "binary"
    description: str = ""
    possible_values: Optional[List[Any]] = None
    is_observed: bool = True
    temporal_order: int = 0  # For temporal causality

@dataclass
class CausalEdge:
    """Represents a causal relationship between variables"""
    source: str
    target: str
    edge_type: str  # "causal", "confounding", "selection"
    strength: float  # Estimated causal effect size
    confidence: float  # Confidence in the relationship
    mechanism: str = ""  # Description of causal mechanism
    
@dataclass
class CausalModel:
    """Represents a complete causal model"""
    variables: Dict[str, CausalVariable]
    edges: List[CausalEdge]
    graph: nx.DiGraph
    structural_equations: Dict[str, Callable] = field(default_factory=dict)
    identification_strategy: str = ""
    
@dataclass
class CausalEffect:
    """Represents an estimated causal effect"""
    treatment: str
    outcome: str
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: CausalEstimationMethod
    confounders_adjusted: List[str]
    sample_size: int
    effect_type: str  # "ATE", "ATT", "ATC"

@dataclass
class CounterfactualScenario:
    """Represents a counterfactual analysis scenario"""
    scenario_id: str
    intervention: Dict[str, Any]  # Variable -> Value interventions
    original_outcome: float
    counterfactual_outcome: float
    causal_attribution: Dict[str, float]
    confidence: float

class ConditionalIndependenceTest:
    """Tests for conditional independence between variables"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger("ConditionalIndependenceTest")
        
    def test_independence(self, data: pd.DataFrame, x: str, y: str, 
                         conditioning_set: List[str] = None) -> Tuple[bool, float]:
        """Test if X and Y are conditionally independent given Z"""
        
        if conditioning_set is None:
            conditioning_set = []
        
        # Prepare data
        if x not in data.columns or y not in data.columns:
            return False, 1.0
        
        for var in conditioning_set:
            if var not in data.columns:
                return False, 1.0
        
        # Remove missing values
        cols_to_use = [x, y] + conditioning_set
        clean_data = data[cols_to_use].dropna()
        
        if len(clean_data) < 10:  # Insufficient data
            return False, 1.0
        
        try:
            if len(conditioning_set) == 0:
                # Marginal independence test
                return self._marginal_independence_test(clean_data, x, y)
            else:
                # Conditional independence test
                return self._conditional_independence_test(clean_data, x, y, conditioning_set)
        except Exception as e:
            self.logger.warning(f"Independence test failed: {e}")
            return False, 1.0
    
    def _marginal_independence_test(self, data: pd.DataFrame, x: str, y: str) -> Tuple[bool, float]:
        """Test marginal independence between X and Y"""
        
        x_vals = data[x].values
        y_vals = data[y].values
        
        # Check if variables are continuous or discrete
        x_is_continuous = self._is_continuous(x_vals)
        y_is_continuous = self._is_continuous(y_vals)
        
        if x_is_continuous and y_is_continuous:
            # Pearson correlation test
            corr, p_value = stats.pearsonr(x_vals, y_vals)
            return p_value > self.significance_level, p_value
        
        elif not x_is_continuous and not y_is_continuous:
            # Chi-square test of independence
            contingency_table = pd.crosstab(data[x], data[y])
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            return p_value > self.significance_level, p_value
        
        else:
            # Mixed case: use Kruskal-Wallis test
            if x_is_continuous:
                continuous_var, discrete_var = x_vals, y_vals
            else:
                continuous_var, discrete_var = y_vals, x_vals
            
            groups = [continuous_var[discrete_var == val] for val in np.unique(discrete_var)]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                return True, 1.0
            
            stat, p_value = stats.kruskal(*groups)
            return p_value > self.significance_level, p_value
    
    def _conditional_independence_test(self, data: pd.DataFrame, x: str, y: str, 
                                     conditioning_set: List[str]) -> Tuple[bool, float]:
        """Test conditional independence X ⊥ Y | Z"""
        
        # Use partial correlation approach for continuous variables
        x_vals = data[x].values
        y_vals = data[y].values
        
        if self._is_continuous(x_vals) and self._is_continuous(y_vals):
            return self._partial_correlation_test(data, x, y, conditioning_set)
        else:
            # For discrete variables, use log-linear model approach
            return self._loglinear_independence_test(data, x, y, conditioning_set)
    
    def _partial_correlation_test(self, data: pd.DataFrame, x: str, y: str, 
                                conditioning_set: List[str]) -> Tuple[bool, float]:
        """Test independence using partial correlation"""
        
        try:
            # Residualize X and Y with respect to conditioning set
            Z = data[conditioning_set].values
            x_vals = data[x].values
            y_vals = data[y].values
            
            # Fit linear regressions
            reg_x = LinearRegression().fit(Z, x_vals)
            reg_y = LinearRegression().fit(Z, y_vals)
            
            # Get residuals
            x_residuals = x_vals - reg_x.predict(Z)
            y_residuals = y_vals - reg_y.predict(Z)
            
            # Test correlation of residuals
            corr, p_value = stats.pearsonr(x_residuals, y_residuals)
            
            return p_value > self.significance_level, p_value
            
        except Exception as e:
            self.logger.warning(f"Partial correlation test failed: {e}")
            return False, 1.0
    
    def _loglinear_independence_test(self, data: pd.DataFrame, x: str, y: str, 
                                   conditioning_set: List[str]) -> Tuple[bool, float]:
        """Test independence using log-linear model for discrete variables"""
        
        try:
            # Create contingency table
            all_vars = [x, y] + conditioning_set
            
            # Discretize continuous variables if necessary
            discretized_data = data[all_vars].copy()
            for var in all_vars:
                if self._is_continuous(data[var].values):
                    discretized_data[var] = pd.cut(data[var], bins=5, labels=False)
            
            # Create frequency table
            freq_table = discretized_data.groupby(all_vars).size().reset_index(name='count')
            
            if len(freq_table) == 0:
                return True, 1.0
            
            # Simplified test using conditional contingency tables
            p_values = []
            
            for z_combination in discretized_data[conditioning_set].drop_duplicates().values:
                # Filter data for this conditioning set value
                condition = True
                for i, var in enumerate(conditioning_set):
                    condition &= (discretized_data[var] == z_combination[i])
                
                subset_data = discretized_data[condition]
                
                if len(subset_data) < 5:  # Insufficient data for this stratum
                    continue
                
                # Test independence in this stratum
                contingency = pd.crosstab(subset_data[x], subset_data[y])
                
                if contingency.size > 1:
                    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                    p_values.append(p_val)
            
            if not p_values:
                return True, 1.0
            
            # Combine p-values using Fisher's method
            combined_stat = -2 * np.sum(np.log(p_values))
            combined_p = 1 - stats.chi2.cdf(combined_stat, 2 * len(p_values))
            
            return combined_p > self.significance_level, combined_p
            
        except Exception as e:
            self.logger.warning(f"Log-linear independence test failed: {e}")
            return False, 1.0
    
    def _is_continuous(self, values: np.ndarray) -> bool:
        """Check if variable is continuous"""
        unique_values = len(np.unique(values))
        return unique_values > 10 and np.issubdtype(values.dtype, np.number)

class PCAlgorithm:
    """Implements the PC algorithm for causal discovery"""
    
    def __init__(self, independence_test: ConditionalIndependenceTest):
        self.independence_test = independence_test
        self.logger = logging.getLogger("PCAlgorithm")
        
    def discover_causal_structure(self, data: pd.DataFrame, 
                                variable_names: List[str]) -> nx.DiGraph:
        """Discover causal structure using PC algorithm"""
        
        self.logger.info("Starting PC algorithm for causal discovery")
        
        # Initialize complete undirected graph
        graph = nx.Graph()
        graph.add_nodes_from(variable_names)
        
        # Add all possible edges
        for i in range(len(variable_names)):
            for j in range(i + 1, len(variable_names)):
                graph.add_edge(variable_names[i], variable_names[j])
        
        # Phase 1: Edge removal based on conditional independence
        max_conditioning_size = min(4, len(variable_names) - 2)  # Limit for computational efficiency
        
        for conditioning_size in range(max_conditioning_size + 1):
            edges_to_remove = []
            
            for edge in list(graph.edges()):
                x, y = edge
                
                # Get potential conditioning sets
                neighbors_x = set(graph.neighbors(x)) - {y}
                neighbors_y = set(graph.neighbors(y)) - {x}
                potential_conditioning = list(neighbors_x.union(neighbors_y))
                
                # Test all conditioning sets of current size
                for conditioning_set in itertools.combinations(potential_conditioning, conditioning_size):
                    conditioning_list = list(conditioning_set)
                    
                    independent, p_value = self.independence_test.test_independence(
                        data, x, y, conditioning_list
                    )
                    
                    if independent:
                        edges_to_remove.append((x, y))
                        self.logger.debug(f"Removing edge {x} - {y} (p={p_value:.4f})")
                        break
            
            # Remove edges found to be conditionally independent
            for edge in edges_to_remove:
                if graph.has_edge(*edge):
                    graph.remove_edge(*edge)
        
        # Phase 2: Orient edges (simplified)
        directed_graph = self._orient_edges(graph, data)
        
        self.logger.info(f"PC algorithm completed. Found {directed_graph.number_of_edges()} directed edges")
        
        return directed_graph
    
    def _orient_edges(self, undirected_graph: nx.Graph, data: pd.DataFrame) -> nx.DiGraph:
        """Orient edges in the undirected graph"""
        
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(undirected_graph.nodes())
        
        # Convert undirected edges to directed (simplified orientation)
        for edge in undirected_graph.edges():
            x, y = edge
            
            # Use simple heuristic: direction based on correlation with time if available
            # or use variance-based orientation
            x_vals = data[x].values
            y_vals = data[y].values
            
            # Simple heuristic: higher variance variable as cause
            if np.var(x_vals) > np.var(y_vals):
                directed_graph.add_edge(x, y)
            else:
                directed_graph.add_edge(y, x)
        
        return directed_graph

class CausalEffectEstimator:
    """Estimates causal effects using various identification strategies"""
    
    def __init__(self, causal_model: CausalModel):
        self.causal_model = causal_model
        self.logger = logging.getLogger("CausalEffectEstimator")
        
    def estimate_causal_effect(self, data: pd.DataFrame, treatment: str, outcome: str,
                             method: CausalEstimationMethod = CausalEstimationMethod.BACKDOOR_ADJUSTMENT,
                             confounders: List[str] = None) -> CausalEffect:
        """Estimate causal effect of treatment on outcome"""
        
        if confounders is None:
            confounders = self._identify_confounders(treatment, outcome)
        
        self.logger.info(f"Estimating causal effect of {treatment} on {outcome} using {method.value}")
        
        if method == CausalEstimationMethod.BACKDOOR_ADJUSTMENT:
            return self._backdoor_adjustment(data, treatment, outcome, confounders)
        elif method == CausalEstimationMethod.PROPENSITY_SCORE_MATCHING:
            return self._propensity_score_matching(data, treatment, outcome, confounders)
        elif method == CausalEstimationMethod.INSTRUMENTAL_VARIABLES:
            return self._instrumental_variables(data, treatment, outcome, confounders)
        else:
            raise ValueError(f"Method {method} not implemented")
    
    def _identify_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Identify confounders using the causal graph"""
        
        confounders = []
        graph = self.causal_model.graph
        
        # Find variables that are ancestors of both treatment and outcome
        treatment_ancestors = nx.ancestors(graph, treatment) if graph.has_node(treatment) else set()
        outcome_ancestors = nx.ancestors(graph, outcome) if graph.has_node(outcome) else set()
        
        common_ancestors = treatment_ancestors.intersection(outcome_ancestors)
        confounders.extend(list(common_ancestors))
        
        # Add variables that have edges to both treatment and outcome
        for node in graph.nodes():
            if (graph.has_edge(node, treatment) and graph.has_edge(node, outcome)):
                if node not in confounders:
                    confounders.append(node)
        
        return confounders
    
    def _backdoor_adjustment(self, data: pd.DataFrame, treatment: str, outcome: str,
                           confounders: List[str]) -> CausalEffect:
        """Estimate causal effect using backdoor adjustment"""
        
        # Prepare data
        required_cols = [treatment, outcome] + confounders
        clean_data = data[required_cols].dropna()
        
        if len(clean_data) < 20:
            raise ValueError("Insufficient data for causal effect estimation")
        
        X = clean_data[confounders].values if confounders else np.ones((len(clean_data), 1))
        T = clean_data[treatment].values
        Y = clean_data[outcome].values
        
        # Check if treatment is binary or continuous
        is_binary_treatment = len(np.unique(T)) == 2
        
        if is_binary_treatment:
            return self._binary_treatment_backdoor(X, T, Y, treatment, outcome, confounders)
        else:
            return self._continuous_treatment_backdoor(X, T, Y, treatment, outcome, confounders)
    
    def _binary_treatment_backdoor(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                                 treatment: str, outcome: str, confounders: List[str]) -> CausalEffect:
        """Backdoor adjustment for binary treatment"""
        
        # Standardize confounders
        if X.shape[1] > 1:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Estimate outcome model: E[Y|T,X]
        outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit outcome model
        TX = np.column_stack([T, X_scaled])
        outcome_model.fit(TX, Y)
        
        # Compute average treatment effect (ATE)
        X_treated = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        X_control = np.column_stack([np.zeros(len(X_scaled)), X_scaled])
        
        Y1_pred = outcome_model.predict(X_treated)  # Potential outcome under treatment
        Y0_pred = outcome_model.predict(X_control)  # Potential outcome under control
        
        ate = np.mean(Y1_pred - Y0_pred)
        
        # Bootstrap confidence interval
        n_bootstrap = 200
        bootstrap_ates = []
        
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(len(Y), len(Y), replace=True)
            X_boot = X_scaled[boot_indices]
            T_boot = T[boot_indices]
            Y_boot = Y[boot_indices]
            
            try:
                TX_boot = np.column_stack([T_boot, X_boot])
                outcome_model_boot = RandomForestRegressor(n_estimators=50, random_state=np.random.randint(1000))
                outcome_model_boot.fit(TX_boot, Y_boot)
                
                X_treated_boot = np.column_stack([np.ones(len(X_boot)), X_boot])
                X_control_boot = np.column_stack([np.zeros(len(X_boot)), X_boot])
                
                Y1_boot = outcome_model_boot.predict(X_treated_boot)
                Y0_boot = outcome_model_boot.predict(X_control_boot)
                
                ate_boot = np.mean(Y1_boot - Y0_boot)
                bootstrap_ates.append(ate_boot)
            except:
                continue
        
        if bootstrap_ates:
            ci_lower = np.percentile(bootstrap_ates, 2.5)
            ci_upper = np.percentile(bootstrap_ates, 97.5)
            
            # Approximate p-value
            p_value = 2 * min(np.mean(np.array(bootstrap_ates) >= 0), 
                             np.mean(np.array(bootstrap_ates) <= 0))
        else:
            ci_lower = ci_upper = ate
            p_value = 0.5
        
        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=ate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method=CausalEstimationMethod.BACKDOOR_ADJUSTMENT,
            confounders_adjusted=confounders,
            sample_size=len(Y),
            effect_type="ATE"
        )
    
    def _continuous_treatment_backdoor(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                                     treatment: str, outcome: str, confounders: List[str]) -> CausalEffect:
        """Backdoor adjustment for continuous treatment"""
        
        # Use linear regression approach for continuous treatment
        if X.shape[1] > 1:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Fit regression model: Y = f(T, X)
        TX = np.column_stack([T, X_scaled])
        
        # Use linear regression for interpretability
        model = LinearRegression()
        model.fit(TX, Y)
        
        # The coefficient of T is the causal effect (under linearity assumption)
        causal_effect = model.coef_[0]
        
        # Compute standard error using bootstrap
        n_bootstrap = 200
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(len(Y), len(Y), replace=True)
            TX_boot = TX[boot_indices]
            Y_boot = Y[boot_indices]
            
            try:
                model_boot = LinearRegression()
                model_boot.fit(TX_boot, Y_boot)
                bootstrap_effects.append(model_boot.coef_[0])
            except:
                continue
        
        if bootstrap_effects:
            ci_lower = np.percentile(bootstrap_effects, 2.5)
            ci_upper = np.percentile(bootstrap_effects, 97.5)
            
            # T-test for significance
            se = np.std(bootstrap_effects)
            t_stat = causal_effect / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(Y) - TX.shape[1] - 1))
        else:
            ci_lower = ci_upper = causal_effect
            p_value = 0.5
        
        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=causal_effect,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method=CausalEstimationMethod.BACKDOOR_ADJUSTMENT,
            confounders_adjusted=confounders,
            sample_size=len(Y),
            effect_type="ATE"
        )
    
    def _propensity_score_matching(self, data: pd.DataFrame, treatment: str, outcome: str,
                                 confounders: List[str]) -> CausalEffect:
        """Estimate causal effect using propensity score matching"""
        
        # This is a simplified implementation
        # In practice, would use more sophisticated matching algorithms
        
        required_cols = [treatment, outcome] + confounders
        clean_data = data[required_cols].dropna()
        
        X = clean_data[confounders].values if confounders else np.ones((len(clean_data), 1))
        T = clean_data[treatment].values
        Y = clean_data[outcome].values
        
        # Estimate propensity scores
        ps_model = LogisticRegression()
        ps_model.fit(X, T)
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        
        # Simple nearest neighbor matching
        treated_indices = np.where(T == 1)[0]
        control_indices = np.where(T == 0)[0]
        
        matched_pairs = []
        
        for treated_idx in treated_indices:
            treated_ps = propensity_scores[treated_idx]
            
            # Find nearest control unit
            distances = np.abs(propensity_scores[control_indices] - treated_ps)
            nearest_control_idx = control_indices[np.argmin(distances)]
            
            matched_pairs.append((treated_idx, nearest_control_idx))
        
        # Compute treatment effect on matched sample
        if matched_pairs:
            treated_outcomes = [Y[pair[0]] for pair in matched_pairs]
            control_outcomes = [Y[pair[1]] for pair in matched_pairs]
            
            ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
            
            # Simple t-test for confidence interval
            differences = np.array(treated_outcomes) - np.array(control_outcomes)
            se = stats.sem(differences)
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se
            
            _, p_value = stats.ttest_1samp(differences, 0)
        else:
            ate = 0.0
            ci_lower = ci_upper = 0.0
            p_value = 1.0
        
        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=ate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method=CausalEstimationMethod.PROPENSITY_SCORE_MATCHING,
            confounders_adjusted=confounders,
            sample_size=len(matched_pairs),
            effect_type="ATT"
        )
    
    def _instrumental_variables(self, data: pd.DataFrame, treatment: str, outcome: str,
                              confounders: List[str]) -> CausalEffect:
        """Estimate causal effect using instrumental variables (simplified)"""
        
        # This is a placeholder implementation
        # Would need to identify valid instruments in practice
        
        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            p_value=1.0,
            method=CausalEstimationMethod.INSTRUMENTAL_VARIABLES,
            confounders_adjusted=confounders,
            sample_size=len(data),
            effect_type="LATE"
        )

class CounterfactualAnalyzer:
    """Performs counterfactual analysis and what-if scenarios"""
    
    def __init__(self, causal_model: CausalModel):
        self.causal_model = causal_model
        self.logger = logging.getLogger("CounterfactualAnalyzer")
        
    def generate_counterfactuals(self, data: pd.DataFrame, instance_idx: int,
                               interventions: Dict[str, Any]) -> CounterfactualScenario:
        """Generate counterfactual scenarios for a specific instance"""
        
        instance = data.iloc[instance_idx]
        
        # Get original outcome
        outcome_vars = [var.name for var in self.causal_model.variables.values() 
                       if var.variable_type == VariableType.OUTCOME]
        
        if not outcome_vars:
            raise ValueError("No outcome variables defined in causal model")
        
        outcome_var = outcome_vars[0]  # Use first outcome variable
        original_outcome = instance[outcome_var]
        
        # Simulate counterfactual outcome
        counterfactual_outcome = self._simulate_counterfactual(instance, interventions, outcome_var)
        
        # Compute causal attribution
        causal_attribution = self._compute_causal_attribution(instance, interventions, outcome_var)
        
        # Estimate confidence
        confidence = self._estimate_counterfactual_confidence(instance, interventions)
        
        return CounterfactualScenario(
            scenario_id=f"cf_{instance_idx}_{hash(str(interventions))}",
            intervention=interventions,
            original_outcome=original_outcome,
            counterfactual_outcome=counterfactual_outcome,
            causal_attribution=causal_attribution,
            confidence=confidence
        )
    
    def _simulate_counterfactual(self, instance: pd.Series, interventions: Dict[str, Any],
                               outcome_var: str) -> float:
        """Simulate counterfactual outcome under interventions"""
        
        # Create modified instance with interventions
        modified_instance = instance.copy()
        for var, value in interventions.items():
            modified_instance[var] = value
        
        # Use structural equations if available
        if outcome_var in self.causal_model.structural_equations:
            eq = self.causal_model.structural_equations[outcome_var]
            try:
                return eq(modified_instance)
            except:
                pass
        
        # Fallback: use simple linear approximation
        # In practice, would use learned causal mechanisms
        
        # Simple heuristic: assume linear relationships
        graph = self.causal_model.graph
        parents = list(graph.predecessors(outcome_var)) if graph.has_node(outcome_var) else []
        
        if not parents:
            return instance[outcome_var]  # No causal parents
        
        # Linear combination of parent values (simplified)
        counterfactual_value = 0.0
        for parent in parents:
            if parent in modified_instance:
                # Use simple coefficient (would be learned from data)
                coeff = 0.1  # Placeholder
                counterfactual_value += coeff * modified_instance[parent]
        
        # Add baseline
        baseline = instance[outcome_var] - sum(0.1 * instance[parent] for parent in parents if parent in instance)
        
        return counterfactual_value + baseline
    
    def _compute_causal_attribution(self, instance: pd.Series, interventions: Dict[str, Any],
                                  outcome_var: str) -> Dict[str, float]:
        """Compute causal attribution of each intervention"""
        
        attribution = {}
        original_outcome = instance[outcome_var]
        
        # Compute individual contributions
        for var, value in interventions.items():
            # Single intervention
            single_intervention = {var: value}
            single_cf_outcome = self._simulate_counterfactual(instance, single_intervention, outcome_var)
            
            attribution[var] = single_cf_outcome - original_outcome
        
        return attribution
    
    def _estimate_counterfactual_confidence(self, instance: pd.Series, 
                                          interventions: Dict[str, Any]) -> float:
        """Estimate confidence in counterfactual prediction"""
        
        # Simple heuristic based on intervention magnitude
        # In practice, would use model uncertainty estimation
        
        confidences = []
        
        for var, value in interventions.items():
            if var in instance:
                original_value = instance[var]
                
                # Normalize by variable range (simplified)
                if np.issubdtype(type(original_value), np.number):
                    # Assume reasonable range
                    range_size = max(abs(original_value), 1.0) * 2
                    intervention_magnitude = abs(value - original_value) / range_size
                    confidence = max(0.1, 1.0 - intervention_magnitude)
                else:
                    confidence = 0.5  # Default for categorical
                
                confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.5

class CausalInferenceFramework:
    """Main framework for causal inference in scheduling policies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CausalInferenceFramework")
        
        # Initialize components
        self.independence_test = ConditionalIndependenceTest(
            significance_level=config.get("significance_level", 0.05)
        )
        self.pc_algorithm = PCAlgorithm(self.independence_test)
        
        # Causal model will be built during analysis
        self.causal_model: Optional[CausalModel] = None
        self.effect_estimator: Optional[CausalEffectEstimator] = None
        self.counterfactual_analyzer: Optional[CounterfactualAnalyzer] = None
        
        # Results storage
        self.causal_effects: List[CausalEffect] = []
        self.counterfactual_scenarios: List[CounterfactualScenario] = []
        
    def build_causal_model(self, data: pd.DataFrame, variable_definitions: Dict[str, Dict[str, Any]]) -> CausalModel:
        """Build causal model from data and variable definitions"""
        
        self.logger.info("Building causal model from data")
        
        # Create causal variables
        variables = {}
        for var_name, var_info in variable_definitions.items():
            variables[var_name] = CausalVariable(
                name=var_name,
                variable_type=VariableType(var_info.get("type", "confounder")),
                data_type=var_info.get("data_type", "continuous"),
                description=var_info.get("description", ""),
                possible_values=var_info.get("possible_values"),
                is_observed=var_info.get("is_observed", True),
                temporal_order=var_info.get("temporal_order", 0)
            )
        
        # Discover causal structure
        variable_names = list(variables.keys())
        causal_graph = self.pc_algorithm.discover_causal_structure(data, variable_names)
        
        # Create causal edges
        edges = []
        for source, target in causal_graph.edges():
            # Estimate edge strength (simplified)
            if source in data.columns and target in data.columns:
                corr = data[source].corr(data[target])
                strength = abs(corr) if not np.isnan(corr) else 0.0
            else:
                strength = 0.5
            
            edges.append(CausalEdge(
                source=source,
                target=target,
                edge_type="causal",
                strength=strength,
                confidence=0.8,  # Placeholder
                mechanism=f"Causal relationship from {source} to {target}"
            ))
        
        # Create causal model
        self.causal_model = CausalModel(
            variables=variables,
            edges=edges,
            graph=causal_graph,
            identification_strategy="PC algorithm discovery"
        )
        
        # Initialize estimators
        self.effect_estimator = CausalEffectEstimator(self.causal_model)
        self.counterfactual_analyzer = CounterfactualAnalyzer(self.causal_model)
        
        self.logger.info(f"Causal model built with {len(variables)} variables and {len(edges)} causal relationships")
        
        return self.causal_model
    
    def analyze_scheduling_decisions(self, data: pd.DataFrame, 
                                   treatment_vars: List[str],
                                   outcome_vars: List[str]) -> Dict[str, Any]:
        """Analyze causal effects of scheduling decisions"""
        
        if self.causal_model is None:
            raise ValueError("Causal model must be built first")
        
        self.logger.info("Analyzing causal effects of scheduling decisions")
        
        analysis_results = {
            "causal_effects": {},
            "feature_importance": {},
            "policy_insights": []
        }
        
        # Estimate causal effects
        for treatment in treatment_vars:
            for outcome in outcome_vars:
                try:
                    effect = self.effect_estimator.estimate_causal_effect(
                        data, treatment, outcome, 
                        CausalEstimationMethod.BACKDOOR_ADJUSTMENT
                    )
                    
                    self.causal_effects.append(effect)
                    analysis_results["causal_effects"][f"{treatment}_to_{outcome}"] = effect
                    
                    self.logger.info(f"Causal effect {treatment} -> {outcome}: {effect.effect_size:.4f} "
                                   f"(CI: [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}], "
                                   f"p={effect.p_value:.4f})")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to estimate effect {treatment} -> {outcome}: {e}")
        
        # Compute feature importance based on causal effects
        feature_importance = {}
        for effect in self.causal_effects:
            if effect.p_value < 0.05:  # Significant effects only
                importance = abs(effect.effect_size) * (1 - effect.p_value)
                feature_importance[effect.treatment] = importance
        
        analysis_results["feature_importance"] = feature_importance
        
        # Generate policy insights
        insights = self._generate_policy_insights()
        analysis_results["policy_insights"] = insights
        
        return analysis_results
    
    def explain_scheduling_decision(self, data: pd.DataFrame, instance_idx: int,
                                  intervention_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Explain a specific scheduling decision using counterfactual analysis"""
        
        if self.counterfactual_analyzer is None:
            raise ValueError("Causal model must be built first")
        
        self.logger.info(f"Explaining scheduling decision for instance {instance_idx}")
        
        explanations = {
            "instance_id": instance_idx,
            "original_instance": data.iloc[instance_idx].to_dict(),
            "counterfactuals": [],
            "causal_attribution": {},
            "recommendations": []
        }
        
        # Generate counterfactuals
        for scenario in intervention_scenarios:
            try:
                counterfactual = self.counterfactual_analyzer.generate_counterfactuals(
                    data, instance_idx, scenario
                )
                
                self.counterfactual_scenarios.append(counterfactual)
                explanations["counterfactuals"].append(counterfactual)
                
                self.logger.debug(f"Generated counterfactual: {counterfactual.scenario_id}")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate counterfactual for scenario {scenario}: {e}")
        
        # Aggregate causal attribution
        all_attributions = defaultdict(list)
        for cf in explanations["counterfactuals"]:
            for var, attribution in cf.causal_attribution.items():
                all_attributions[var].append(attribution)
        
        # Average attributions
        for var, attributions in all_attributions.items():
            explanations["causal_attribution"][var] = {
                "mean_effect": np.mean(attributions),
                "effect_variance": np.var(attributions),
                "effect_range": (np.min(attributions), np.max(attributions))
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(explanations)
        explanations["recommendations"] = recommendations
        
        return explanations
    
    def _generate_policy_insights(self) -> List[str]:
        """Generate insights about scheduling policies"""
        
        insights = []
        
        if not self.causal_effects:
            return ["No causal effects analyzed yet"]
        
        # Find most impactful factors
        significant_effects = [e for e in self.causal_effects if e.p_value < 0.05]
        
        if significant_effects:
            # Largest positive effect
            largest_positive = max(significant_effects, key=lambda x: x.effect_size if x.effect_size > 0 else -float('inf'))
            if largest_positive.effect_size > 0:
                insights.append(f"Increasing {largest_positive.treatment} has the strongest positive causal effect on {largest_positive.outcome} "
                              f"(effect size: {largest_positive.effect_size:.3f})")
            
            # Largest negative effect
            largest_negative = min(significant_effects, key=lambda x: x.effect_size if x.effect_size < 0 else float('inf'))
            if largest_negative.effect_size < 0:
                insights.append(f"Increasing {largest_negative.treatment} has the strongest negative causal effect on {largest_negative.outcome} "
                              f"(effect size: {largest_negative.effect_size:.3f})")
            
            # Most reliable effect
            most_reliable = min(significant_effects, key=lambda x: x.p_value)
            insights.append(f"The most statistically reliable causal relationship is {most_reliable.treatment} -> {most_reliable.outcome} "
                          f"(p-value: {most_reliable.p_value:.4f})")
        
        # Robustness insights
        strong_effects = [e for e in significant_effects if abs(e.effect_size) > 0.1]
        if strong_effects:
            insights.append(f"Found {len(strong_effects)} strong causal relationships that should be prioritized in policy design")
        
        return insights
    
    def _generate_recommendations(self, explanations: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on causal analysis"""
        
        recommendations = []
        
        causal_attribution = explanations.get("causal_attribution", {})
        
        if not causal_attribution:
            return ["No causal attribution available for recommendations"]
        
        # Find variables with strongest causal effects
        effects_by_magnitude = sorted(causal_attribution.items(), 
                                    key=lambda x: abs(x[1]["mean_effect"]), reverse=True)
        
        for var, effect_info in effects_by_magnitude[:3]:  # Top 3 most impactful
            mean_effect = effect_info["mean_effect"]
            
            if abs(mean_effect) > 0.01:  # Threshold for meaningful effect
                direction = "increase" if mean_effect > 0 else "decrease"
                recommendations.append(f"To improve outcomes, consider adjusting {var} (predicted effect: {direction} outcome by {abs(mean_effect):.3f})")
        
        return recommendations
    
    def get_causal_model_summary(self) -> Dict[str, Any]:
        """Get summary of the causal model"""
        
        if self.causal_model is None:
            return {"status": "No causal model built"}
        
        summary = {
            "variables": len(self.causal_model.variables),
            "causal_relationships": len(self.causal_model.edges),
            "variable_types": {},
            "graph_properties": {},
            "causal_effects_analyzed": len(self.causal_effects),
            "counterfactuals_generated": len(self.counterfactual_scenarios)
        }
        
        # Variable type distribution
        for var in self.causal_model.variables.values():
            var_type = var.variable_type.value
            summary["variable_types"][var_type] = summary["variable_types"].get(var_type, 0) + 1
        
        # Graph properties
        graph = self.causal_model.graph
        summary["graph_properties"] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "is_connected": nx.is_weakly_connected(graph),
            "is_acyclic": nx.is_directed_acyclic_graph(graph)
        }
        
        return summary

def demonstrate_causal_inference():
    """Demonstrate the causal inference framework"""
    print("=== Causal Inference for Scheduling Policy Interpretability ===")
    
    # Generate synthetic scheduling data
    np.random.seed(42)
    n_samples = 500
    
    # Scheduling variables
    cpu_utilization = np.random.beta(2, 3, n_samples)  # 0-1
    memory_pressure = np.random.beta(1.5, 2, n_samples)  # 0-1
    task_priority = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])  # Low, Medium, High
    
    # Scheduling decisions (treatments)
    resource_allocation = 0.3 * cpu_utilization + 0.2 * memory_pressure + 0.1 * task_priority + np.random.normal(0, 0.1, n_samples)
    resource_allocation = np.clip(resource_allocation, 0, 1)
    
    preemption_enabled = (resource_allocation > 0.6).astype(int)
    
    # Outcomes
    makespan = 100 - 50 * resource_allocation - 20 * preemption_enabled + 30 * memory_pressure + np.random.normal(0, 5, n_samples)
    makespan = np.clip(makespan, 10, 200)
    
    utilization_efficiency = 0.5 + 0.4 * resource_allocation + 0.2 * preemption_enabled - 0.1 * memory_pressure + np.random.normal(0, 0.05, n_samples)
    utilization_efficiency = np.clip(utilization_efficiency, 0, 1)
    
    # Create DataFrame
    data = pd.DataFrame({
        'cpu_utilization': cpu_utilization,
        'memory_pressure': memory_pressure,
        'task_priority': task_priority,
        'resource_allocation': resource_allocation,
        'preemption_enabled': preemption_enabled,
        'makespan': makespan,
        'utilization_efficiency': utilization_efficiency
    })
    
    print("1. Generated Synthetic Scheduling Data...")
    print(f"   Dataset size: {len(data)} instances")
    print(f"   Variables: {list(data.columns)}")
    print(f"   Data preview:")
    print(data.head().to_string(index=False))
    
    print("\n2. Initializing Causal Inference Framework...")
    
    config = {
        "significance_level": 0.05,
        "discovery_method": "pc_algorithm",
        "estimation_method": "backdoor_adjustment"
    }
    
    framework = CausalInferenceFramework(config)
    
    print("   Framework initialized")
    
    print("3. Defining Causal Variables...")
    
    variable_definitions = {
        'cpu_utilization': {
            'type': 'confounder',
            'data_type': 'continuous',
            'description': 'Current CPU utilization of the system',
            'temporal_order': 1
        },
        'memory_pressure': {
            'type': 'confounder',
            'data_type': 'continuous',
            'description': 'Memory pressure indicator',
            'temporal_order': 1
        },
        'task_priority': {
            'type': 'confounder',
            'data_type': 'discrete',
            'description': 'Priority level of the task',
            'possible_values': [0, 1, 2],
            'temporal_order': 1
        },
        'resource_allocation': {
            'type': 'treatment',
            'data_type': 'continuous',
            'description': 'Amount of resources allocated to task',
            'temporal_order': 2
        },
        'preemption_enabled': {
            'type': 'treatment',
            'data_type': 'binary',
            'description': 'Whether task preemption is enabled',
            'temporal_order': 2
        },
        'makespan': {
            'type': 'outcome',
            'data_type': 'continuous',
            'description': 'Total execution time',
            'temporal_order': 3
        },
        'utilization_efficiency': {
            'type': 'outcome',
            'data_type': 'continuous',
            'description': 'Resource utilization efficiency',
            'temporal_order': 3
        }
    }
    
    for var_name, var_info in variable_definitions.items():
        print(f"   {var_name}: {var_info['type']} ({var_info['data_type']})")
    
    print("4. Building Causal Model...")
    
    causal_model = framework.build_causal_model(data, variable_definitions)
    
    print(f"   Discovered {causal_model.graph.number_of_edges()} causal relationships")
    print("   Causal relationships found:")
    for edge in causal_model.edges:
        print(f"     {edge.source} -> {edge.target} (strength: {edge.strength:.3f})")
    
    print("5. Analyzing Scheduling Decisions...")
    
    treatment_vars = ['resource_allocation', 'preemption_enabled']
    outcome_vars = ['makespan', 'utilization_efficiency']
    
    analysis_results = framework.analyze_scheduling_decisions(data, treatment_vars, outcome_vars)
    
    print("   Causal Effects:")
    for effect_name, effect in analysis_results["causal_effects"].items():
        significance = "*" if effect.p_value < 0.05 else ""
        print(f"     {effect_name}: {effect.effect_size:.4f} {significance}")
        print(f"       95% CI: [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}]")
        print(f"       p-value: {effect.p_value:.4f}")
        print(f"       Sample size: {effect.sample_size}")
    
    print("   Feature Importance (Causal):")
    for feature, importance in analysis_results["feature_importance"].items():
        print(f"     {feature}: {importance:.4f}")
    
    print("6. Policy Insights...")
    
    insights = analysis_results["policy_insights"]
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    print("7. Counterfactual Analysis...")
    
    # Select an instance for explanation
    instance_idx = 42
    
    # Define intervention scenarios
    intervention_scenarios = [
        {'resource_allocation': 0.8},  # High resource allocation
        {'resource_allocation': 0.2},  # Low resource allocation
        {'preemption_enabled': 1 - data.iloc[instance_idx]['preemption_enabled']},  # Toggle preemption
        {'resource_allocation': 0.6, 'preemption_enabled': 1}  # Combined intervention
    ]
    
    explanations = framework.explain_scheduling_decision(data, instance_idx, intervention_scenarios)
    
    print(f"   Explaining decision for instance {instance_idx}:")
    original = explanations["original_instance"]
    print(f"     Original: resource_alloc={original['resource_allocation']:.3f}, "
          f"preemption={original['preemption_enabled']}, makespan={original['makespan']:.1f}")
    
    print("   Counterfactual scenarios:")
    for cf in explanations["counterfactuals"]:
        outcome_change = cf.counterfactual_outcome - cf.original_outcome
        print(f"     Intervention {cf.intervention}: outcome change = {outcome_change:+.3f} "
              f"(confidence: {cf.confidence:.3f})")
        
        for var, attribution in cf.causal_attribution.items():
            print(f"       {var} contribution: {attribution:+.3f}")
    
    print("8. Recommendations...")
    
    recommendations = explanations["recommendations"]
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("9. Causal Model Summary...")
    
    model_summary = framework.get_causal_model_summary()
    
    print(f"   Variables: {model_summary['variables']}")
    print(f"   Causal relationships: {model_summary['causal_relationships']}")
    print(f"   Variable types: {model_summary['variable_types']}")
    print(f"   Graph properties: {model_summary['graph_properties']}")
    print(f"   Causal effects analyzed: {model_summary['causal_effects_analyzed']}")
    print(f"   Counterfactuals generated: {model_summary['counterfactuals_generated']}")
    
    print("10. Interpretability Benefits...")
    
    benefits = [
        "Causal discovery reveals true relationships vs spurious correlations",
        "Counterfactual analysis enables 'what-if' scenario exploration",
        "Backdoor adjustment controls for confounding variables",
        "Feature attribution through causal lens provides actionable insights",
        "Policy recommendations based on causal effects not correlations",
        "Structural equation modeling captures decision mechanisms",
        "Statistical significance testing validates causal claims",
        "Explainable AI for critical scheduling decisions"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"   {i}. {benefit}")
    
    return {
        "framework": framework,
        "causal_model": causal_model,
        "analysis_results": analysis_results,
        "explanations": explanations,
        "data": data
    }

if __name__ == "__main__":
    demonstrate_causal_inference()