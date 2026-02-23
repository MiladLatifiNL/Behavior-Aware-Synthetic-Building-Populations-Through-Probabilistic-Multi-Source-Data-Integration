"""
Household coordination using graph algorithms for consistent matching.

This module ensures household-level consistency in matching by:
- Modeling households as graphs with family relationships
- Ensuring coordinated activity patterns within households
- Optimizing household-level match quality
- Handling multi-generational households
- Managing childcare and elder care dependencies
- Coordinating work schedules and commute patterns
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import pickle
import time

logger = logging.getLogger(__name__)


@dataclass
class HouseholdConfig:
    """Configuration for household coordination."""
    use_graph_matching: bool = True
    use_constraint_propagation: bool = True
    use_joint_optimization: bool = True
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    childcare_coordination: bool = True
    eldercare_coordination: bool = True
    work_schedule_coordination: bool = True
    activity_compatibility_weight: float = 0.3
    relationship_weight: float = 0.2
    schedule_alignment_weight: float = 0.5
    min_childcare_age: int = 12
    min_eldercare_age: int = 75
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class HouseholdGraph:
    """
    Graph representation of a household with family relationships.
    """
    
    def __init__(self, household_data: pd.DataFrame):
        """
        Initialize household graph.
        
        Args:
            household_data: DataFrame with household members
        """
        self.graph = nx.Graph()
        self.household_id = None
        self.members = {}
        self.relationships = {}
        self.care_dependencies = {}
        
        self._build_graph(household_data)
    
    def _build_graph(self, household_data: pd.DataFrame):
        """
        Build graph from household data.
        
        Args:
            household_data: DataFrame with household members
        """
        if 'household_id' in household_data.columns:
            self.household_id = household_data['household_id'].iloc[0]
        
        # Add nodes for each household member
        for idx, member in household_data.iterrows():
            person_id = member.get('person_id', idx)
            
            # Node attributes
            attributes = {
                'age': member.get('age', 30),
                'sex': member.get('sex', 'U'),
                'employed': member.get('employed', False),
                'work_hours': member.get('work_hours', 0),
                'is_parent': member.get('is_parent', False),
                'is_child': member.get('age', 30) < 18,
                'is_elderly': member.get('age', 30) >= 75,
                'needs_care': member.get('age', 30) < 12 or member.get('age', 30) >= 85,
                'can_provide_care': 18 <= member.get('age', 30) <= 70
            }
            
            self.graph.add_node(person_id, **attributes)
            self.members[person_id] = attributes
        
        # Infer relationships and add edges
        self._infer_relationships(household_data)
        
        # Identify care dependencies
        self._identify_care_dependencies()
    
    def _infer_relationships(self, household_data: pd.DataFrame):
        """
        Infer family relationships from demographics.
        
        Args:
            household_data: DataFrame with household members
        """
        members_list = list(self.members.keys())
        
        for i, person1 in enumerate(members_list):
            for person2 in members_list[i+1:]:
                attr1 = self.members[person1]
                attr2 = self.members[person2]
                
                age_diff = abs(attr1['age'] - attr2['age'])
                
                # Parent-child relationship (age difference > 15)
                if age_diff > 15:
                    if attr1['age'] > attr2['age'] and attr2['is_child']:
                        relationship = 'parent-child'
                        self.graph.add_edge(person1, person2, relationship=relationship, weight=1.0)
                    elif attr2['age'] > attr1['age'] and attr1['is_child']:
                        relationship = 'parent-child'
                        self.graph.add_edge(person1, person2, relationship=relationship, weight=1.0)
                
                # Spouse/partner relationship (similar age adults)
                elif age_diff < 10 and attr1['age'] >= 18 and attr2['age'] >= 18:
                    relationship = 'spouse'
                    self.graph.add_edge(person1, person2, relationship=relationship, weight=0.9)
                
                # Sibling relationship (similar age children)
                elif age_diff < 5 and (attr1['is_child'] or attr2['is_child']):
                    relationship = 'sibling'
                    self.graph.add_edge(person1, person2, relationship=relationship, weight=0.7)
                
                # Grandparent-grandchild
                elif age_diff > 40:
                    relationship = 'grandparent-grandchild'
                    self.graph.add_edge(person1, person2, relationship=relationship, weight=0.6)
                
                # Default household member relationship
                else:
                    relationship = 'household_member'
                    self.graph.add_edge(person1, person2, relationship=relationship, weight=0.3)
                
                self.relationships[(person1, person2)] = relationship
    
    def _identify_care_dependencies(self):
        """
        Identify care dependencies within household.
        """
        for person_id, attributes in self.members.items():
            if attributes['needs_care']:
                # Find potential caregivers
                caregivers = []
                
                for neighbor in self.graph.neighbors(person_id):
                    neighbor_attr = self.members[neighbor]
                    edge_data = self.graph.get_edge_data(person_id, neighbor)
                    
                    # Prioritize parents for childcare
                    if attributes['is_child'] and edge_data.get('relationship') == 'parent-child':
                        if neighbor_attr['can_provide_care']:
                            caregivers.append((neighbor, 1.0))  # High priority
                    
                    # Any adult for eldercare
                    elif attributes['is_elderly'] and neighbor_attr['can_provide_care']:
                        priority = 0.8 if edge_data.get('relationship') in ['spouse', 'parent-child'] else 0.5
                        caregivers.append((neighbor, priority))
                
                if caregivers:
                    self.care_dependencies[person_id] = caregivers
    
    def get_subgraphs(self) -> List[nx.Graph]:
        """
        Get connected components as subgraphs.
        
        Returns:
            List of subgraphs representing family units
        """
        return [self.graph.subgraph(c).copy() for c in nx.connected_components(self.graph)]
    
    def get_care_constraints(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get care dependency constraints.
        
        Returns:
            Dictionary of person_id to list of (caregiver_id, priority) tuples
        """
        return self.care_dependencies
    
    def get_relationship_matrix(self) -> np.ndarray:
        """
        Get relationship strength matrix.
        
        Returns:
            Adjacency matrix with relationship weights
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        matrix = np.zeros((n, n))
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if self.graph.has_edge(node1, node2):
                    matrix[i, j] = self.graph[node1][node2].get('weight', 0.5)
        
        return matrix


class ActivityCompatibilityNetwork(nn.Module):
    """
    Neural network for learning activity compatibility between household members.
    """
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128):
        """
        Initialize compatibility network.
        
        Args:
            input_dim: Dimension of person features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Person encoders
        self.person_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Activity encoders
        self.activity_encoder = nn.Sequential(
            nn.Linear(24, hidden_dim),  # 24 hours of activities
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Compatibility scorer
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Attention for person pairs
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, person1_features, person2_features, activity1, activity2):
        """
        Calculate activity compatibility between two household members.
        
        Args:
            person1_features: Features of person 1
            person2_features: Features of person 2
            activity1: Activity pattern of person 1
            activity2: Activity pattern of person 2
            
        Returns:
            Compatibility score between 0 and 1
        """
        # Encode persons
        p1_encoded = self.person_encoder(person1_features)
        p2_encoded = self.person_encoder(person2_features)
        
        # Apply attention between persons
        p1_attended, _ = self.attention(
            p1_encoded.unsqueeze(1),
            p2_encoded.unsqueeze(1),
            p2_encoded.unsqueeze(1)
        )
        p1_attended = p1_attended.squeeze(1)
        
        # Encode activities
        a1_encoded = self.activity_encoder(activity1)
        a2_encoded = self.activity_encoder(activity2)
        
        # Combine all features
        combined = torch.cat([
            p1_attended, p2_encoded,
            a1_encoded, a2_encoded
        ], dim=-1)
        
        # Calculate compatibility
        compatibility = self.compatibility_scorer(combined)
        
        return compatibility.squeeze(-1)


class ConstraintPropagation:
    """
    Constraint propagation for household matching consistency.
    """
    
    def __init__(self, config: HouseholdConfig):
        """
        Initialize constraint propagation.
        
        Args:
            config: Household coordination configuration
        """
        self.config = config
        self.constraints = defaultdict(list)
        self.domains = {}
        self.arc_consistency_queue = deque()
    
    def add_constraint(self, person1: int, person2: int, constraint_type: str, 
                      constraint_func: callable):
        """
        Add constraint between two persons.
        
        Args:
            person1: First person ID
            person2: Second person ID
            constraint_type: Type of constraint
            constraint_func: Function that checks if constraint is satisfied
        """
        self.constraints[(person1, person2)].append({
            'type': constraint_type,
            'func': constraint_func
        })
        
        # Add to arc consistency queue
        self.arc_consistency_queue.append((person1, person2))
    
    def set_domain(self, person: int, candidates: List[Any]):
        """
        Set domain of possible matches for a person.
        
        Args:
            person: Person ID
            candidates: List of candidate matches
        """
        self.domains[person] = candidates
    
    def propagate(self) -> bool:
        """
        Propagate constraints using AC-3 algorithm.
        
        Returns:
            True if consistent, False if inconsistent
        """
        iteration = 0
        
        while self.arc_consistency_queue and iteration < self.config.max_iterations:
            person1, person2 = self.arc_consistency_queue.popleft()
            
            if self._revise(person1, person2):
                if len(self.domains[person1]) == 0:
                    return False  # Inconsistent
                
                # Add neighbors to queue
                for person3 in self.domains:
                    if person3 != person1 and person3 != person2:
                        if (person3, person1) in self.constraints:
                            self.arc_consistency_queue.append((person3, person1))
            
            iteration += 1
        
        return True  # Consistent
    
    def _revise(self, person1: int, person2: int) -> bool:
        """
        Revise domain of person1 based on constraints with person2.
        
        Args:
            person1: First person
            person2: Second person
            
        Returns:
            True if domain was revised
        """
        revised = False
        
        if person1 not in self.domains or person2 not in self.domains:
            return False
        
        to_remove = []
        
        for match1 in self.domains[person1]:
            satisfied = False
            
            for match2 in self.domains[person2]:
                # Check all constraints
                all_satisfied = True
                
                for constraint in self.constraints.get((person1, person2), []):
                    if not constraint['func'](match1, match2):
                        all_satisfied = False
                        break
                
                if all_satisfied:
                    satisfied = True
                    break
            
            if not satisfied:
                to_remove.append(match1)
                revised = True
        
        # Remove inconsistent values
        for match in to_remove:
            self.domains[person1].remove(match)
        
        return revised
    
    def get_consistent_assignment(self) -> Dict[int, Any]:
        """
        Get consistent assignment after propagation.
        
        Returns:
            Dictionary of person to match assignment
        """
        assignment = {}
        
        for person, domain in self.domains.items():
            if len(domain) == 1:
                assignment[person] = domain[0]
            elif len(domain) > 1:
                # Choose best from remaining
                assignment[person] = domain[0]  # Simple heuristic
        
        return assignment


class HouseholdOptimizer:
    """
    Joint optimization for household-level matching.
    """
    
    def __init__(self, config: HouseholdConfig):
        """
        Initialize household optimizer.
        
        Args:
            config: Configuration
        """
        self.config = config
        self.compatibility_network = None
        
        if config.use_joint_optimization:
            self.compatibility_network = ActivityCompatibilityNetwork()
            self.compatibility_network.to(config.device)
    
    def optimize_household_matching(self, household_graph: HouseholdGraph,
                                   person_candidates: Dict[int, pd.DataFrame],
                                   activity_patterns: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Optimize matching for entire household.
        
        Args:
            household_graph: Graph representation of household
            person_candidates: Candidate matches for each person
            activity_patterns: Activity patterns for candidates
            
        Returns:
            Optimal assignment of persons to matches
        """
        logger.info(f"Optimizing household with {len(household_graph.members)} members")
        
        # Build cost matrix
        persons = list(household_graph.members.keys())
        n_persons = len(persons)
        
        # Get maximum candidates across all persons
        max_candidates = max(len(candidates) for candidates in person_candidates.values())
        
        # Initialize cost matrix (minimize negative quality)
        cost_matrix = np.full((n_persons, max_candidates), np.inf)
        
        for i, person in enumerate(persons):
            candidates = person_candidates.get(person, pd.DataFrame())
            
            for j, (_, candidate) in enumerate(candidates.iterrows()):
                if j >= max_candidates:
                    break
                
                # Individual match quality
                individual_quality = candidate.get('match_score', 0.5)
                
                # Household coordination penalty
                coordination_cost = self._calculate_coordination_cost(
                    person, candidate, household_graph, person_candidates
                )
                
                # Combined cost (negative for minimization)
                cost_matrix[i, j] = -(individual_quality - coordination_cost)
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Build assignment
        assignment = {}
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] != np.inf:
                person = persons[i]
                candidates = person_candidates[person]
                if j < len(candidates):
                    assignment[person] = candidates.iloc[j].name
        
        # Apply constraint propagation if enabled
        if self.config.use_constraint_propagation:
            assignment = self._apply_constraints(assignment, household_graph, 
                                                person_candidates, activity_patterns)
        
        logger.info(f"Household optimization completed with {len(assignment)} assignments")
        return assignment
    
    def _calculate_coordination_cost(self, person: int, candidate: pd.Series,
                                    household_graph: HouseholdGraph,
                                    person_candidates: Dict[int, pd.DataFrame]) -> float:
        """
        Calculate coordination cost for a person-candidate pair.
        
        Args:
            person: Person ID
            candidate: Candidate match
            household_graph: Household graph
            person_candidates: All person candidates
            
        Returns:
            Coordination cost (penalty)
        """
        cost = 0.0
        person_attr = household_graph.members[person]
        
        # Check care dependencies
        if person in household_graph.care_dependencies:
            caregivers = household_graph.care_dependencies[person]
            
            # Check if caregivers have compatible schedules
            for caregiver, priority in caregivers:
                if caregiver in person_candidates:
                    # Simple heuristic: penalize if caregiver works full-time
                    caregiver_candidates = person_candidates[caregiver]
                    if not caregiver_candidates.empty:
                        avg_work_hours = caregiver_candidates['work_hours'].mean() \
                                        if 'work_hours' in caregiver_candidates.columns else 40
                        
                        if avg_work_hours > 35:  # Full-time work
                            cost += 0.2 * priority
        
        # Check if person is a caregiver
        for dependent, caregivers in household_graph.care_dependencies.items():
            caregiver_ids = [c[0] for c in caregivers]
            if person in caregiver_ids:
                # Penalize if candidate has inflexible schedule
                if candidate.get('work_hours', 0) > 40:
                    cost += 0.15
                if candidate.get('has_fixed_schedule', False):
                    cost += 0.1
        
        # Work schedule coordination for spouses
        for neighbor in household_graph.graph.neighbors(person):
            edge_data = household_graph.graph.get_edge_data(person, neighbor)
            
            if edge_data.get('relationship') == 'spouse':
                # Prefer complementary schedules
                if neighbor in person_candidates:
                    neighbor_candidates = person_candidates[neighbor]
                    if not neighbor_candidates.empty and 'work_shift' in candidate:
                        # Check for same shift (might want to avoid for childcare)
                        same_shift_ratio = (neighbor_candidates['work_shift'] == 
                                          candidate['work_shift']).mean() \
                                          if 'work_shift' in neighbor_candidates.columns else 0
                        
                        # If they have children, prefer different shifts
                        has_children = any(household_graph.members[p]['is_child'] 
                                         for p in household_graph.members)
                        if has_children and same_shift_ratio > 0.7:
                            cost += 0.2
        
        return cost
    
    def _apply_constraints(self, initial_assignment: Dict[int, int],
                          household_graph: HouseholdGraph,
                          person_candidates: Dict[int, pd.DataFrame],
                          activity_patterns: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Apply constraint propagation to refine assignment.
        
        Args:
            initial_assignment: Initial assignment
            household_graph: Household graph
            person_candidates: Candidate matches
            activity_patterns: Activity patterns
            
        Returns:
            Refined assignment
        """
        propagator = ConstraintPropagation(self.config)
        
        # Set domains
        for person, candidates in person_candidates.items():
            propagator.set_domain(person, list(candidates.index))
        
        # Add childcare constraints
        if self.config.childcare_coordination:
            self._add_childcare_constraints(propagator, household_graph, activity_patterns)
        
        # Add eldercare constraints
        if self.config.eldercare_coordination:
            self._add_eldercare_constraints(propagator, household_graph, activity_patterns)
        
        # Add work schedule constraints
        if self.config.work_schedule_coordination:
            self._add_work_schedule_constraints(propagator, household_graph, person_candidates)
        
        # Propagate constraints
        if propagator.propagate():
            refined_assignment = propagator.get_consistent_assignment()
            
            # Merge with initial assignment
            for person, match in refined_assignment.items():
                if person not in initial_assignment or initial_assignment[person] is None:
                    initial_assignment[person] = match
        
        return initial_assignment
    
    def _add_childcare_constraints(self, propagator: ConstraintPropagation,
                                  household_graph: HouseholdGraph,
                                  activity_patterns: Dict[int, np.ndarray]):
        """
        Add childcare coordination constraints.
        """
        for person, caregivers in household_graph.care_dependencies.items():
            person_attr = household_graph.members[person]
            
            if person_attr['is_child']:
                for caregiver, priority in caregivers:
                    # Constraint: at least one caregiver must be home when child is home
                    def childcare_constraint(child_match, caregiver_match):
                        if child_match in activity_patterns and caregiver_match in activity_patterns:
                            child_activities = activity_patterns[child_match]
                            caregiver_activities = activity_patterns[caregiver_match]
                            
                            # Check overlap in home activities
                            child_home = child_activities == 'home'  # Simplified
                            caregiver_home = caregiver_activities == 'home'
                            
                            overlap = np.sum(child_home & caregiver_home)
                            return overlap > len(child_home) * 0.3  # At least 30% overlap
                        return True
                    
                    propagator.add_constraint(person, caregiver, 'childcare', childcare_constraint)
    
    def _add_eldercare_constraints(self, propagator: ConstraintPropagation,
                                  household_graph: HouseholdGraph,
                                  activity_patterns: Dict[int, np.ndarray]):
        """
        Add eldercare coordination constraints.
        """
        for person, caregivers in household_graph.care_dependencies.items():
            person_attr = household_graph.members[person]
            
            if person_attr['is_elderly'] and person_attr['needs_care']:
                for caregiver, priority in caregivers:
                    # Constraint: caregiver must have flexible schedule
                    def eldercare_constraint(elder_match, caregiver_match):
                        # Simple check - would be more sophisticated in practice
                        return True  # Placeholder
                    
                    propagator.add_constraint(person, caregiver, 'eldercare', eldercare_constraint)
    
    def _add_work_schedule_constraints(self, propagator: ConstraintPropagation,
                                      household_graph: HouseholdGraph,
                                      person_candidates: Dict[int, pd.DataFrame]):
        """
        Add work schedule coordination constraints.
        """
        # Find working couples
        for person1, person2 in household_graph.relationships:
            if household_graph.relationships[(person1, person2)] == 'spouse':
                attr1 = household_graph.members[person1]
                attr2 = household_graph.members[person2]
                
                if attr1['employed'] and attr2['employed']:
                    # Constraint: avoid both having inflexible schedules
                    def schedule_constraint(match1, match2):
                        if person1 in person_candidates and person2 in person_candidates:
                            cand1 = person_candidates[person1]
                            cand2 = person_candidates[person2]
                            
                            if match1 in cand1.index and match2 in cand2.index:
                                flex1 = cand1.loc[match1].get('schedule_flexibility', 0.5)
                                flex2 = cand2.loc[match2].get('schedule_flexibility', 0.5)
                                
                                # At least one should have flexible schedule
                                return flex1 > 0.3 or flex2 > 0.3
                        return True
                    
                    propagator.add_constraint(person1, person2, 'work_schedule', schedule_constraint)


class HouseholdCoordinationSystem:
    """
    Complete system for household-level coordination in matching.
    """
    
    def __init__(self, config: Optional[HouseholdConfig] = None):
        """
        Initialize household coordination system.
        
        Args:
            config: Configuration
        """
        self.config = config or HouseholdConfig()
        self.optimizer = HouseholdOptimizer(config)
        self.household_graphs = {}
        self.coordination_metrics = {}
    
    def process_household(self, household_df: pd.DataFrame,
                         person_matches: Dict[int, pd.DataFrame],
                         activity_data: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, int]:
        """
        Process a single household for coordinated matching.
        
        Args:
            household_df: DataFrame with household members
            person_matches: Candidate matches for each person
            activity_data: Optional activity pattern data
            
        Returns:
            Coordinated assignment for household
        """
        # Build household graph
        household_graph = HouseholdGraph(household_df)
        
        # Store for later analysis
        if 'household_id' in household_df.columns:
            household_id = household_df['household_id'].iloc[0]
            self.household_graphs[household_id] = household_graph
        
        # Optimize matching
        assignment = self.optimizer.optimize_household_matching(
            household_graph, person_matches, activity_data or {}
        )
        
        # Calculate coordination metrics
        metrics = self._calculate_coordination_metrics(household_graph, assignment, person_matches)
        self.coordination_metrics[household_id if 'household_id' in household_df.columns else 0] = metrics
        
        return assignment
    
    def process_multiple_households(self, households: List[pd.DataFrame],
                                  all_person_matches: Dict[int, pd.DataFrame],
                                  activity_data: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, int]:
        """
        Process multiple households with potential inter-household constraints.
        
        Args:
            households: List of household DataFrames
            all_person_matches: All candidate matches
            activity_data: Optional activity data
            
        Returns:
            Complete assignment for all households
        """
        logger.info(f"Processing {len(households)} households for coordinated matching")
        
        all_assignments = {}
        
        for household_df in households:
            # Get person matches for this household
            household_person_ids = household_df['person_id'].tolist() \
                                 if 'person_id' in household_df.columns else household_df.index.tolist()
            
            household_person_matches = {
                pid: all_person_matches[pid] 
                for pid in household_person_ids 
                if pid in all_person_matches
            }
            
            # Process household
            assignment = self.process_household(
                household_df, household_person_matches, activity_data
            )
            
            all_assignments.update(assignment)
        
        logger.info(f"Completed coordination for {len(all_assignments)} persons")
        return all_assignments
    
    def _calculate_coordination_metrics(self, household_graph: HouseholdGraph,
                                       assignment: Dict[int, int],
                                       person_matches: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate metrics for coordination quality.
        
        Args:
            household_graph: Household graph
            assignment: Final assignment
            person_matches: Candidate matches
            
        Returns:
            Dictionary of coordination metrics
        """
        metrics = {
            'household_size': len(household_graph.members),
            'n_assigned': len(assignment),
            'assignment_rate': len(assignment) / len(household_graph.members) if household_graph.members else 0,
            'n_care_dependencies': len(household_graph.care_dependencies),
            'n_relationships': len(household_graph.relationships)
        }
        
        # Check care dependency satisfaction
        care_satisfied = 0
        for dependent, caregivers in household_graph.care_dependencies.items():
            if dependent in assignment:
                # Check if at least one caregiver is assigned
                caregiver_assigned = any(cg[0] in assignment for cg in caregivers)
                if caregiver_assigned:
                    care_satisfied += 1
        
        metrics['care_satisfaction_rate'] = care_satisfied / len(household_graph.care_dependencies) \
                                           if household_graph.care_dependencies else 1.0
        
        # Calculate average match quality
        total_quality = 0
        for person, match in assignment.items():
            if person in person_matches:
                candidates = person_matches[person]
                if match in candidates.index:
                    quality = candidates.loc[match].get('match_score', 0.5)
                    total_quality += quality
        
        metrics['avg_match_quality'] = total_quality / len(assignment) if assignment else 0
        
        return metrics
    
    def get_coordination_report(self) -> Dict[str, Any]:
        """
        Get comprehensive coordination report.
        
        Returns:
            Dictionary with coordination statistics and metrics
        """
        if not self.coordination_metrics:
            return {'error': 'No households processed yet'}
        
        # Aggregate metrics
        total_households = len(self.coordination_metrics)
        total_persons = sum(m['household_size'] for m in self.coordination_metrics.values())
        total_assigned = sum(m['n_assigned'] for m in self.coordination_metrics.values())
        
        avg_metrics = {}
        for key in ['assignment_rate', 'care_satisfaction_rate', 'avg_match_quality']:
            values = [m[key] for m in self.coordination_metrics.values() if key in m]
            avg_metrics[f'avg_{key}'] = np.mean(values) if values else 0
        
        return {
            'total_households': total_households,
            'total_persons': total_persons,
            'total_assigned': total_assigned,
            'overall_assignment_rate': total_assigned / total_persons if total_persons > 0 else 0,
            **avg_metrics,
            'config': {
                'use_graph_matching': self.config.use_graph_matching,
                'use_constraint_propagation': self.config.use_constraint_propagation,
                'use_joint_optimization': self.config.use_joint_optimization
            }
        }
    
    def visualize_household(self, household_id: int) -> None:
        """
        Visualize household graph structure.
        
        Args:
            household_id: Household ID to visualize
        """
        if household_id not in self.household_graphs:
            logger.warning(f"Household {household_id} not found")
            return
        
        import matplotlib.pyplot as plt
        
        household_graph = self.household_graphs[household_id]
        
        # Create layout
        pos = nx.spring_layout(household_graph.graph, seed=42)
        
        # Draw nodes
        node_colors = []
        for node in household_graph.graph.nodes():
            attr = household_graph.members[node]
            if attr['is_child']:
                node_colors.append('lightblue')
            elif attr['is_elderly']:
                node_colors.append('lightgray')
            elif attr['employed']:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightyellow')
        
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(household_graph.graph, pos, node_color=node_colors, 
                              node_size=500, alpha=0.9)
        
        # Draw edges with relationship labels
        edge_labels = {}
        for (n1, n2), rel in household_graph.relationships.items():
            edge_labels[(n1, n2)] = rel.replace('_', ' ').title()
        
        nx.draw_networkx_edges(household_graph.graph, pos, alpha=0.5)
        nx.draw_networkx_edge_labels(household_graph.graph, pos, edge_labels, font_size=8)
        
        # Draw node labels
        node_labels = {node: f"P{node}\nAge:{attr['age']}" 
                      for node, attr in household_graph.members.items()}
        nx.draw_networkx_labels(household_graph.graph, pos, node_labels, font_size=10)
        
        plt.title(f"Household {household_id} Structure")
        plt.axis('off')
        plt.tight_layout()
        
        output_path = f"data/validation/household_{household_id}_graph.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Household graph saved to {output_path}")
        plt.close()
    
    def save(self, path: str):
        """
        Save coordination system state.
        
        Args:
            path: Path to save file
        """
        save_dict = {
            'config': self.config,
            'coordination_metrics': self.coordination_metrics,
            'household_graphs': {hid: {
                'members': hg.members,
                'relationships': hg.relationships,
                'care_dependencies': hg.care_dependencies
            } for hid, hg in self.household_graphs.items()}
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Household coordination system saved to {path}")