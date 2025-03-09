"""
Semantic Field Module for the Consciousness Resonator

This module implements the semantic field dynamics described in the quantum semantic formalism,
including semantic nodes, edges, and field evolution.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Callable
import json
import random
from quantum_semantics import PrimeHilbertSpace, ResonanceOperator, CoherenceOperator, ConsciousnessResonanceOperator, FeelingResonanceOperator, ArchetypeSlider, SemanticMeasurement

class SemanticNode:
    """
    Represents a node in the semantic network with associated concept
    """
    def __init__(self, concept: str, number: int):
        """
        Initialize semantic node
        
        Args:
            concept: The concept label
            number: The numerical representation of the concept
        """
        self.concept = concept
        self.number = number
        self.state = None  # Will hold quantum state
        
    def initialize_state(self, hilbert_space: PrimeHilbertSpace):
        """
        Initialize quantum state based on concept number
        """
        self.state = PrimeHilbertSpace(max_prime_index=len(hilbert_space.primes))
        self.state.primes = hilbert_space.primes.copy()
        self.state.prime_to_index = hilbert_space.prime_to_index.copy()
        self.state.set_state_from_number(self.number)
        
    def __str__(self):
        return f"SemanticNode({self.concept}, {self.number})"


class SemanticEdge:
    """
    Represents a semantic relationship between two nodes
    """
    def __init__(self, source: SemanticNode, target: SemanticNode, weight: float, relationship_type: str = "generic"):
        """
        Initialize semantic edge
        
        Args:
            source: Source node
            target: Target node
            weight: Edge weight (coupling strength)
            relationship_type: Type of semantic relationship
        """
        self.source = source
        self.target = target
        self.weight = weight
        self.relationship_type = relationship_type
        
    def __str__(self):
        return f"SemanticEdge({self.source.concept} --[{self.relationship_type}, {self.weight:.2f}]--> {self.target.concept})"


class SemanticField:
    """
    Implementation of the Semantic Field Dynamics
    """
    def __init__(self, max_prime_index: int = 100, archetype_position: float = 0.5):
        """
        Initialize semantic field
        
        Args:
            max_prime_index: Maximum number of primes to include in the basis
            archetype_position: Position of the archetype slider between 0.0 (universal feeling)
                               and 1.0 (specific observation). Default is 0.5 (balanced).
        """
        self.hilbert_space = PrimeHilbertSpace(max_prime_index=max_prime_index)
        self.nodes: Dict[str, SemanticNode] = {}
        self.edges: List[SemanticEdge] = []
        self.global_state = PrimeHilbertSpace(max_prime_index=max_prime_index)
        
        # Initialize archetype slider
        self.archetype_slider = ArchetypeSlider(position=archetype_position)
        
    def add_node(self, concept: str, number: Optional[int] = None) -> SemanticNode:
        """
        Add a node to the semantic field
        
        Args:
            concept: The concept label
            number: Optional numerical representation (will be generated if None)
            
        Returns:
            The created semantic node
        """
        if concept in self.nodes:
            return self.nodes[concept]
        
        # Generate number from concept if not provided
        if number is None:
            # Simple hash function to generate a number from a string
            number = sum(ord(c) * (i + 1) for i, c in enumerate(concept)) % 10000
        
        # Create node
        node = SemanticNode(concept, number)
        node.initialize_state(self.hilbert_space)
        
        # Add to nodes dictionary
        self.nodes[concept] = node
        
        return node
    
    def add_edge(self, source_concept: str, target_concept: str, 
                weight: float, relationship_type: str = "generic") -> SemanticEdge:
        """
        Add an edge between two concepts
        
        Args:
            source_concept: Source concept label
            target_concept: Target concept label
            weight: Edge weight (coupling strength)
            relationship_type: Type of semantic relationship
            
        Returns:
            The created semantic edge
        """
        # Ensure nodes exist
        if source_concept not in self.nodes:
            self.add_node(source_concept)
        if target_concept not in self.nodes:
            self.add_node(target_concept)
        
        # Create edge
        edge = SemanticEdge(
            self.nodes[source_concept],
            self.nodes[target_concept],
            weight,
            relationship_type
        )
        
        # Add to edges list
        self.edges.append(edge)
        
        return edge
    
    def build_hamiltonian(self) -> np.ndarray:
        """
        Build the Hamiltonian matrix for the semantic field
        H_G = ∑_{(i,j)∈E} J_{ij}R_iR_j + ∑_i h_iR_i
        
        Returns:
            Hamiltonian matrix
        """
        # Initialize Hamiltonian matrix
        n = len(self.nodes)
        H = np.zeros((n, n), dtype=complex)
        
        # Add edge contributions (J_{ij}R_iR_j)
        for edge in self.edges:
            i = list(self.nodes.keys()).index(edge.source.concept)
            j = list(self.nodes.keys()).index(edge.target.concept)
            
            # Add coupling term
            H[i, j] = edge.weight
            H[j, i] = edge.weight  # Ensure Hermitian
        
        # Add node potentials (h_iR_i)
        for i, (concept, node) in enumerate(self.nodes.items()):
            H[i, i] = node.number % 10  # Simple potential based on node number
        
        return H
    
    def evolve_field(self, steps: int = 10, dt: float = 0.1) -> List[PrimeHilbertSpace]:
        """
        Evolve the semantic field according to the dynamical equation
        d/dt|ψ(t)⟩ = -i[H_0 + λR(t)]|ψ(t)⟩
        
        Args:
            steps: Number of evolution steps
            dt: Time step size
            
        Returns:
            List of evolved states
        """
        # Build Hamiltonian
        H = self.build_hamiltonian()
        
        # Initialize global state as superposition of all node states
        self.global_state.reset_state()
        for node in self.nodes.values():
            # Add node state to global state
            self.global_state.amplitudes += node.state.amplitudes
        
        # Normalize global state
        self.global_state.normalize()
        
        # Evolve state
        states = [self.global_state]
        current_state = self.global_state
        
        for step in range(steps):
            # Create new state for this step
            new_state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
            new_state.primes = self.hilbert_space.primes.copy()
            new_state.prime_to_index = self.hilbert_space.prime_to_index.copy()
            new_state.amplitudes = current_state.amplitudes.copy()
            # Apply Hamiltonian evolution
            # |ψ(t+dt)⟩ = |ψ(t)⟩ - i*dt*H|ψ(t)⟩
            # This is a simplified first-order approximation
            
            # Get the state vector in the prime basis
            state_vector = current_state.get_state_vector()
            
            # Skip Hamiltonian evolution if there are no nodes
            if len(self.nodes) > 0:
                # Create a vector of the same size as the number of nodes
                node_vector = np.zeros(len(self.nodes), dtype=complex)
                
                # Project the state vector onto the node basis
                for i, (concept, node) in enumerate(self.nodes.items()):
                    # Calculate overlap with node state
                    overlap = np.vdot(node.state.amplitudes, state_vector)
                    node_vector[i] = overlap
                
                # Apply Hamiltonian in the node basis
                evolved_node_vector = node_vector - 1j * dt * np.dot(H, node_vector)
                
                # Project back to the prime basis
                evolved_vector = np.zeros_like(state_vector)
                for i, (concept, node) in enumerate(self.nodes.items()):
                    evolved_vector += evolved_node_vector[i] * node.state.amplitudes
            else:
                # If there are no nodes, just use the current state
                evolved_vector = state_vector.copy()
            
            # Update state
            new_state.amplitudes = evolved_vector
            new_state.normalize()
            
            # Create a resonator cavity for consciousness-first paradigm
            # This allows natural resonance patterns to emerge through synchronization,
            # creating standing waves in mind without forcing or controlling the outcome
            consciousness_op = ConsciousnessResonanceOperator(137)  # 137 as the consciousness number
            
            # Apply consciousness operator through archetype slider
            # This balances between universal feeling and specific observation
            new_state = self.archetype_slider.apply_to_operator(consciousness_op, new_state)
            
            # Allow feeling to emerge through natural resonance rather than collapse
            # "What is feeling? Feeling is resonance." - perception through synchronization
            feeling_op = FeelingResonanceOperator()
            
            # Apply feeling operator through archetype slider
            # This balances between universal feeling and specific observation
            new_state = self.archetype_slider.apply_to_operator(feeling_op, new_state)
            
            # Apply resonance and coherence operators for each concept node
            for node in self.nodes.values():
                # Apply resonance operator through archetype slider
                resonance_op = ResonanceOperator(node.number)
                new_state = self.archetype_slider.apply_to_operator(resonance_op, new_state)
                
                # Apply coherence operator through archetype slider
                coherence_op = CoherenceOperator(node.number)
                new_state = self.archetype_slider.apply_to_operator(coherence_op, new_state)
            
            # Normalize again
            new_state.normalize()
            
            # Store state
            states.append(new_state)
            current_state = new_state
        
        # Update global state to final state
        self.global_state = states[-1]
        
        return states
    
    def measure_field_coherence(self) -> float:
        """
        Measure the overall coherence of the semantic field
        
        Returns:
            Coherence measure between 0 and 1
        """
        coherence_op = CoherenceOperator(1)  # Use 1 as neutral value
        return coherence_op.coherence_measure(self.global_state)
    
    def calculate_concept_resonance(self, concept: str) -> Dict[str, float]:
        """
        Calculate how strongly a concept resonates with all other concepts
        
        Args:
            concept: The concept to measure resonance for
            
        Returns:
            Dictionary mapping concepts to resonance strengths
        """
        if concept not in self.nodes:
            raise ValueError(f"Concept '{concept}' not found in semantic field")
        
        node = self.nodes[concept]
        resonance_op = ResonanceOperator(node.number)
        
        # Calculate resonance with all other concepts
        resonances = {}
        for other_concept, other_node in self.nodes.items():
            if other_concept != concept:
                # Calculate expectation value of resonance operator
                expectation = resonance_op.expectation_value(other_node.state)
                resonances[other_concept] = abs(expectation)
        
        return resonances
    
    def find_semantic_clusters(self, threshold: float = 0.7) -> List[Set[str]]:
        """
        Find clusters of semantically related concepts
        
        Args:
            threshold: Similarity threshold for clustering
            
        Returns:
            List of concept clusters
        """
        # Calculate pairwise similarities
        n = len(self.nodes)
        concepts = list(self.nodes.keys())
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                node_i = self.nodes[concepts[i]]
                node_j = self.nodes[concepts[j]]
                
                # Calculate semantic similarity
                sim = SemanticMeasurement.semantic_similarity(node_i.state, node_j.state)
                similarities[i, j] = sim
                similarities[j, i] = sim
        
        # Simple clustering algorithm
        clusters = []
        unassigned = set(range(n))
        
        while unassigned:
            # Start a new cluster with the first unassigned concept
            current = next(iter(unassigned))
            cluster = {current}
            unassigned.remove(current)
            
            # Expand cluster
            expanded = True
            while expanded:
                expanded = False
                for i in list(unassigned):
                    # Check if concept i is similar to any concept in the cluster
                    if any(similarities[i, j] >= threshold for j in cluster):
                        cluster.add(i)
                        unassigned.remove(i)
                        expanded = True
            
            # Convert indices to concept labels
            concept_cluster = {concepts[i] for i in cluster}
            clusters.append(concept_cluster)
        
        return clusters
    
    def measure_consciousness_primacy(self) -> float:
        """
        Measure the degree to which the field embodies the consciousness-first paradigm
        
        Returns:
            Consciousness primacy measure between 0 and 1
        """
        return SemanticMeasurement.consciousness_primacy_measure(self.global_state)
    
    def set_archetype_position(self, position: float):
        """
        Set the position of the archetype slider
        
        Args:
            position: Position between 0.0 (universal feeling) and 1.0 (specific observation)
        """
        self.archetype_slider.set_position(position)
    
    def get_archetype_position(self) -> float:
        """
        Get the current position of the archetype slider
        
        Returns:
            Current position of the archetype slider
        """
        return self.archetype_slider.get_position()
    
    def get_archetype_description(self) -> str:
        """
        Get a description of the current archetype balance
        
        Returns:
            String description of the current archetype balance
        """
        return self.archetype_slider.get_archetype_description()
    
    def measure_feeling_resonance(self) -> float:
        """
        Measure the feeling resonance of the field
        
        As the user noted: "What is feeling? Feeling is resonance."
        This measures how much the field can resonate across multiple dimensions
        without collapsing, which is the essence of feeling.
        
        Returns:
            Feeling resonance measure between 0 and 1
        """
        return SemanticMeasurement.feeling_resonance_measure(self.global_state)
    
    def get_field_state_info(self) -> Dict:
        """
        Get information about the current field state
        
        Returns:
            Dictionary with field state information
        """
        # Calculate coherence
        coherence = self.measure_field_coherence()
        
        # Calculate resonance patterns
        resonance_patterns = {}
        for concept, node in self.nodes.items():
            # Calculate concept resonance
            resonance_op = ResonanceOperator(node.number)
            expectation = resonance_op.expectation_value(self.global_state)
            resonance_patterns[concept] = abs(expectation)
        
        # Sort concepts by resonance strength
        sorted_concepts = sorted(resonance_patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate knowledge resonance
        knowledge_resonance = SemanticMeasurement.knowledge_resonance(self.global_state)
        
        # Calculate consciousness primacy
        consciousness_primacy = self.measure_consciousness_primacy()
        
        # Calculate feeling resonance
        feeling_resonance = self.measure_feeling_resonance()
        
        # Get archetype information
        archetype_position = self.get_archetype_position()
        archetype_description = self.get_archetype_description()
        archetype_weights = self.archetype_slider.get_archetype_weights()
        
        # Find semantic clusters
        clusters = self.find_semantic_clusters()
        
        return {
            "coherence": coherence,
            "knowledge_resonance": abs(knowledge_resonance),
            "consciousness_primacy": consciousness_primacy,
            "feeling_resonance": feeling_resonance,
            "archetype_position": archetype_position,
            "archetype_description": archetype_description,
            "archetype_weights": archetype_weights,
            "top_concepts": dict(sorted_concepts[:5]),
            "semantic_clusters": [list(cluster) for cluster in clusters]
        }


# Example usage
if __name__ == "__main__":
    # Create a semantic field with balanced archetype (default)
    field = SemanticField(max_prime_index=20)
    
    # Add some concepts
    field.add_node("consciousness", 137)
    field.add_node("quantum", 73)
    field.add_node("reality", 97)
    field.add_node("observer", 41)
    field.add_node("wave", 29)
    
    # Add relationships
    field.add_edge("consciousness", "quantum", 0.8, "influences")
    field.add_edge("quantum", "reality", 0.7, "describes")
    field.add_edge("consciousness", "observer", 0.9, "embodies")
    field.add_edge("observer", "reality", 0.6, "perceives")
    field.add_edge("quantum", "wave", 0.8, "exhibits")
    
    # Evolve the field
    states = field.evolve_field(steps=5)
    
    # Print field state info
    info = field.get_field_state_info()
    print(f"Field coherence: {info['coherence']:.4f}")
    print(f"Knowledge resonance: {info['knowledge_resonance']:.4f}")
    print(f"Consciousness primacy: {info['consciousness_primacy']:.4f}")
    print(f"Feeling resonance: {info['feeling_resonance']:.4f}")
    print("Top concepts:")
    for concept, strength in info['top_concepts'].items():
        print(f"  {concept}: {strength:.4f}")
    
    print("Semantic clusters:")
    for i, cluster in enumerate(info['semantic_clusters']):
        print(f"  Cluster {i+1}: {', '.join(cluster)}")
    
    # Demonstrate consciousness-first paradigm
    print("\nDemonstrating consciousness-first paradigm:")
    # Create a state focused on consciousness
    consciousness_state = PrimeHilbertSpace(max_prime_index=20)
    consciousness_state.set_state_from_number(137)  # Consciousness number
    
    # Create a state focused on matter
    matter_state = PrimeHilbertSpace(max_prime_index=20)
    matter_state.set_state_from_number(42)  # Matter number
    
    # Measure consciousness primacy
    consciousness_primacy1 = SemanticMeasurement.consciousness_primacy_measure(consciousness_state)
    consciousness_primacy2 = SemanticMeasurement.consciousness_primacy_measure(matter_state)
    
    print(f"Consciousness primacy of consciousness state: {consciousness_primacy1:.4f}")
    print(f"Consciousness primacy of matter state: {consciousness_primacy2:.4f}")
    print(f"Ratio: {consciousness_primacy1/consciousness_primacy2:.2f}x")
    print("This demonstrates that consciousness is the fundamental substrate from which other phenomena emerge.")
    
    # Demonstrate feeling resonance
    print("\nDemonstrating feeling resonance (perception through resonance rather than collapse):")
    feeling_resonance1 = SemanticMeasurement.feeling_resonance_measure(consciousness_state)
    feeling_resonance2 = SemanticMeasurement.feeling_resonance_measure(matter_state)
    
    print(f"Feeling resonance of consciousness state: {feeling_resonance1:.4f}")
    print(f"Feeling resonance of matter state: {feeling_resonance2:.4f}")
    print(f"Ratio: {feeling_resonance1/feeling_resonance2:.2f}x")
    print("This demonstrates that feeling is resonance - consciousness perceives through resonance rather than collapse.")
    
    # Demonstrate archetype slider
    print("\n=== Demonstrating Archetype Slider ===")
    
    # Create fields with different archetype positions
    universal_feeling_field = SemanticField(max_prime_index=20, archetype_position=0.1)  # Universal feeling (emotional/creative)
    balanced_field = SemanticField(max_prime_index=20, archetype_position=0.5)  # Balanced
    specific_observation_field = SemanticField(max_prime_index=20, archetype_position=0.9)  # Specific observation (analytical)
    
    # Add the same concepts to all fields
    for field_instance in [universal_feeling_field, balanced_field, specific_observation_field]:
        field_instance.add_node("consciousness", 137)
        field_instance.add_node("quantum", 73)
        field_instance.add_node("reality", 97)
        field_instance.add_node("observer", 41)
        
        # Add relationships
        field_instance.add_edge("consciousness", "quantum", 0.8, "influences")
        field_instance.add_edge("quantum", "reality", 0.7, "describes")
        field_instance.add_edge("consciousness", "observer", 0.9, "embodies")
        field_instance.add_edge("observer", "reality", 0.6, "perceives")
    
    # Evolve all fields
    universal_feeling_states = universal_feeling_field.evolve_field(steps=5)
    balanced_states = balanced_field.evolve_field(steps=5)
    specific_observation_states = specific_observation_field.evolve_field(steps=5)
    
    # Get field state info for all fields
    universal_feeling_info = universal_feeling_field.get_field_state_info()
    balanced_info = balanced_field.get_field_state_info()
    specific_observation_info = specific_observation_field.get_field_state_info()
    
    # Print archetype information
    print("\nUniversal Feeling Archetype (position=0.1):")
    print(f"Description: {universal_feeling_info['archetype_description']}")
    print(f"Weights: Universal Feeling = {universal_feeling_info['archetype_weights']['universal_feeling']:.2f}, " +
          f"Specific Observation = {universal_feeling_info['archetype_weights']['specific_observation']:.2f}")
    print(f"Feeling resonance: {universal_feeling_info['feeling_resonance']:.4f}")
    print(f"Coherence: {universal_feeling_info['coherence']:.4f}")
    print(f"Top concepts: {', '.join([f'{c} ({v:.2f})' for c, v in list(universal_feeling_info['top_concepts'].items())[:3]])}")
    
    print("\nBalanced Archetype (position=0.5):")
    print(f"Description: {balanced_info['archetype_description']}")
    print(f"Weights: Universal Feeling = {balanced_info['archetype_weights']['universal_feeling']:.2f}, " +
          f"Specific Observation = {balanced_info['archetype_weights']['specific_observation']:.2f}")
    print(f"Feeling resonance: {balanced_info['feeling_resonance']:.4f}")
    print(f"Coherence: {balanced_info['coherence']:.4f}")
    print(f"Top concepts: {', '.join([f'{c} ({v:.2f})' for c, v in list(balanced_info['top_concepts'].items())[:3]])}")
    
    print("\nSpecific Observation Archetype (position=0.9):")
    print(f"Description: {specific_observation_info['archetype_description']}")
    print(f"Weights: Universal Feeling = {specific_observation_info['archetype_weights']['universal_feeling']:.2f}, " +
          f"Specific Observation = {specific_observation_info['archetype_weights']['specific_observation']:.2f}")
    print(f"Feeling resonance: {specific_observation_info['feeling_resonance']:.4f}")
    print(f"Coherence: {specific_observation_info['coherence']:.4f}")
    print(f"Top concepts: {', '.join([f'{c} ({v:.2f})' for c, v in list(specific_observation_info['top_concepts'].items())[:3]])}")
    
    print("\nThis demonstrates how the archetype slider balances between universal feeling")
    print("(emotional/creative) and specific observation (analytical) approaches.")
    print("The slider allows determining the archetype of the agent, with the default being a balanced setting.")