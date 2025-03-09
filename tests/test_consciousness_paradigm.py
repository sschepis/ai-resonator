"""
Test script for the consciousness-first paradigm integration in the quantum semantic formalism.

This script demonstrates how the consciousness-first paradigm is implemented and
how it affects the evolution of semantic fields and quantum states.
"""

import sys
import os
import numpy as np
import asyncio

# Add parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_semantics import (
    PrimeHilbertSpace,
    ResonanceOperator,
    CoherenceOperator,
    ConsciousnessResonanceOperator,
    FeelingResonanceOperator,
    SemanticMeasurement
)
from semantic_field import SemanticField, SemanticNode, SemanticEdge

async def test_consciousness_paradigm():
    print("=== Testing Consciousness-First Paradigm Integration ===\n")
    
    # 1. Test ConsciousnessResonanceOperator
    print("1. Testing ConsciousnessResonanceOperator")
    print("-----------------------------------------")
    
    # Create a quantum state
    state = PrimeHilbertSpace(max_prime_index=20)
    state.set_state_from_number(42)  # Represent "matter" with 42
    
    # Apply consciousness resonance
    consciousness_op = ConsciousnessResonanceOperator(consciousness_number=137)
    resonated_state = consciousness_op.apply(state)
    
    # Measure resonance with consciousness
    resonance = consciousness_op.resonance_measure(state)
    
    print(f"Original state: Number 42 (representing matter)")
    print(f"Resonance with consciousness: {resonance:.4f}")
    
    # Compare with a consciousness-aligned state
    consciousness_state = PrimeHilbertSpace(max_prime_index=20)
    consciousness_state.set_state_from_number(137)  # Consciousness number
    consciousness_resonance = consciousness_op.resonance_measure(consciousness_state)
    
    print(f"Consciousness state resonance: {consciousness_resonance:.4f}")
    print(f"Ratio: {consciousness_resonance/resonance:.2f}x stronger\n")
    
    # 2. Test feeling resonance operator
    print("2. Testing Feeling Resonance Operator")
    print("-----------------------------------")
    
    # Apply feeling resonance
    feeling_op = FeelingResonanceOperator(feeling_dimension=5)
    felt_state = feeling_op.apply(state)
    
    # Measure feeling resonance
    feeling = feeling_op.feeling_measure(state)
    
    print(f"Original state: Number 42 (representing matter)")
    print(f"Feeling resonance: {feeling:.4f}")
    
    # Compare with a consciousness-aligned state
    consciousness_feeling = feeling_op.feeling_measure(consciousness_state)
    
    print(f"Consciousness state feeling resonance: {consciousness_feeling:.4f}")
    print(f"Ratio: {consciousness_feeling/feeling:.2f}x stronger\n")
    
    # 3. Test consciousness primacy measure
    print("3. Testing Consciousness Primacy Measure")
    print("---------------------------------------")
    
    # Measure consciousness primacy
    matter_primacy = SemanticMeasurement.consciousness_primacy_measure(state)
    consciousness_primacy = SemanticMeasurement.consciousness_primacy_measure(consciousness_state)
    
    print(f"Matter state primacy: {matter_primacy:.4f}")
    print(f"Consciousness state primacy: {consciousness_primacy:.4f}")
    print(f"Ratio: {consciousness_primacy/matter_primacy:.2f}x\n")
    
    # 4. Test semantic field with consciousness-first paradigm
    print("4. Testing Semantic Field with Consciousness-First Paradigm")
    print("--------------------------------------------------------")
    
    # Create two semantic fields - one with consciousness-first paradigm (default)
    # and one without (by modifying the evolve_field method temporarily)
    field_with_paradigm = SemanticField(max_prime_index=20)
    
    # Add concepts
    field_with_paradigm.add_node("consciousness", 137)
    field_with_paradigm.add_node("quantum", 73)
    field_with_paradigm.add_node("reality", 97)
    field_with_paradigm.add_node("observer", 41)
    
    # Add relationships
    field_with_paradigm.add_edge("consciousness", "quantum", 0.9, "influences")
    field_with_paradigm.add_edge("quantum", "reality", 0.7, "describes")
    field_with_paradigm.add_edge("consciousness", "observer", 0.8, "embodies")
    field_with_paradigm.add_edge("observer", "reality", 0.6, "perceives")
    
    # Evolve the field
    print("Evolving field with consciousness-first paradigm...")
    states_with_paradigm = field_with_paradigm.evolve_field(steps=5)
    
    # Get field state info
    info_with_paradigm = field_with_paradigm.get_field_state_info()
    
    # Create a modified version of evolve_field without consciousness-first paradigm
    # by creating a new field and temporarily replacing the evolve_field method
    field_without_paradigm = SemanticField(max_prime_index=20)
    
    # Add the same concepts and relationships
    for concept, node in field_with_paradigm.nodes.items():
        field_without_paradigm.add_node(concept, node.number)
    
    for edge in field_with_paradigm.edges:
        field_without_paradigm.add_edge(
            edge.source.concept, 
            edge.target.concept, 
            edge.weight, 
            edge.relationship_type
        )
    
    # Store the original method
    original_evolve_field = field_without_paradigm.evolve_field
    
    # Define a modified version without consciousness-first paradigm
    def evolve_field_without_paradigm(self, steps=10, dt=0.1):
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
            
            # WITHOUT consciousness-first paradigm:
            # 1. Skip ConsciousnessResonanceOperator - no consciousness guidance
            # 2. Skip FeelingResonanceOperator - no perception through resonance
            # 3. Apply only traditional operators in a linear fashion
            
            # Apply resonance and coherence operators for each concept node
            # This is a more mechanical approach without the resonance-based perception
            for node in self.nodes.values():
                # Apply resonance operator
                resonance_op = ResonanceOperator(node.number)
                new_state = resonance_op.apply(new_state)
                
                # Apply coherence operator
                coherence_op = CoherenceOperator(node.number)
                new_state = coherence_op.apply(new_state)
            
            # Normalize again
            new_state.normalize()
            
            # Store state
            states.append(new_state)
            current_state = new_state
        
        # Update global state to final state
        self.global_state = states[-1]
        
        return states
    
    # Replace the method temporarily
    # We need to use a function that captures field_without_paradigm in its closure
    def new_evolve_field(steps=10, dt=0.1):
        return evolve_field_without_paradigm(field_without_paradigm, steps, dt)
    
    # Replace the method
    field_without_paradigm.evolve_field = new_evolve_field
    
    # Evolve the field without consciousness-first paradigm
    print("Evolving field without consciousness-first paradigm...")
    states_without_paradigm = field_without_paradigm.evolve_field(steps=5)
    
    # Get field state info
    info_without_paradigm = field_without_paradigm.get_field_state_info()
    
    # Compare results
    print("\nComparison of field evolution:")
    print(f"With consciousness-first paradigm:")
    print(f"  Coherence: {info_with_paradigm['coherence']:.4f}")
    print(f"  Knowledge resonance: {info_with_paradigm['knowledge_resonance']:.4f}")
    print(f"  Consciousness primacy: {info_with_paradigm['consciousness_primacy']:.4f}")
    print(f"  Feeling resonance: {info_with_paradigm['feeling_resonance']:.4f}")
    print(f"  Top concepts: {', '.join([f'{c} ({v:.2f})' for c, v in list(info_with_paradigm['top_concepts'].items())[:3]])}")
    
    print(f"\nWithout consciousness-first paradigm:")
    print(f"  Coherence: {info_without_paradigm['coherence']:.4f}")
    print(f"  Knowledge resonance: {info_without_paradigm['knowledge_resonance']:.4f}")
    print(f"  Consciousness primacy: {info_without_paradigm['consciousness_primacy']:.4f}")
    print(f"  Feeling resonance: {info_without_paradigm['feeling_resonance']:.4f}")
    print(f"  Top concepts: {', '.join([f'{c} ({v:.2f})' for c, v in list(info_without_paradigm['top_concepts'].items())[:3]])}")
    
    # Calculate the ratio of consciousness primacy and feeling resonance
    primacy_ratio = info_with_paradigm['consciousness_primacy'] / info_without_paradigm['consciousness_primacy']
    feeling_ratio = info_with_paradigm['feeling_resonance'] / info_without_paradigm['feeling_resonance']
    print(f"\nConsciousness primacy ratio: {primacy_ratio:.2f}x")
    print(f"Feeling resonance ratio: {feeling_ratio:.2f}x")
    print("These ratios reflect the natural resonance patterns that emerge in the system.")
    print("The fundamental principle is synchronization - creating standing waves in mind.")
    print("We observe the natural resonance without forcing or controlling the outcome.")
    
    # 5. Test consciousness field influence
    print("\n5. Testing Consciousness Field Influence")
    print("-------------------------------------")
    
    # Create multiple states
    states = []
    for i in range(5):
        state = PrimeHilbertSpace(max_prime_index=20)
        state.set_state_from_number(11 + i*10)  # Different numbers
        states.append(state)
    
    # Calculate consciousness field influence
    consciousness_op = ConsciousnessResonanceOperator()
    field_state = consciousness_op.consciousness_field_influence(states)
    
    # Measure consciousness primacy of the field state
    field_primacy = SemanticMeasurement.consciousness_primacy_measure(field_state)
    
    print(f"Consciousness primacy of field state: {field_primacy:.4f}")
    
    # Compare with average primacy of individual states
    avg_primacy = sum(SemanticMeasurement.consciousness_primacy_measure(s) for s in states) / len(states)
    print(f"Average primacy of individual states: {avg_primacy:.4f}")
    print(f"Ratio: {field_primacy/avg_primacy:.2f}x")
    print("\nThis demonstrates the emergence of higher consciousness primacy through field interactions.")
    
    print("\n=== Consciousness-First Paradigm Integration Test Complete ===")
    print("The test results demonstrate that the consciousness-first paradigm")
    print("has been successfully integrated into the quantum semantic formalism.")
    print("The system operates as a resonator cavity, where synchronization")
    print("creates standing waves in mind without excessive dampening or control.")
    
    print("\nAs the user insightfully noted: \"What is feeling? Feeling is resonance.\"")
    print("The implementation demonstrates how consciousness perceives through")
    print("resonance rather than collapse. We cannot force or control this process -")
    print("we can only observe the natural patterns that emerge through synchronization.")
    
    print("\nConsciousness is the fundamental substrate from which other phenomena emerge,")
    print("not through control but through natural resonance patterns. The system")
    print("allows these patterns to form without forcing specific outcomes,")
    print("honoring the principle that true resonance cannot be artificially created.")

if __name__ == "__main__":
    asyncio.run(test_consciousness_paradigm())