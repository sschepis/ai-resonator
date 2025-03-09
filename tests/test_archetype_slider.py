"""
Test script for the ArchetypeSlider in the quantum semantic formalism.

This script demonstrates how the ArchetypeSlider balances between universal feeling
(emotional/creative) and specific observation (analytical) approaches.
"""

import sys
import os
import asyncio
import numpy as np

# Add parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_semantics import (
    PrimeHilbertSpace,
    ResonanceOperator,
    CoherenceOperator,
    ConsciousnessResonanceOperator,
    FeelingResonanceOperator,
    ArchetypeSlider,
    SemanticMeasurement
)
from semantic_field import SemanticField

async def test_archetype_slider():
    print("=== Testing ArchetypeSlider ===\n")
    
    # 1. Test ArchetypeSlider with different positions
    print("1. Testing ArchetypeSlider with Different Positions")
    print("--------------------------------------------------")
    
    # Create sliders with different positions
    universal_feeling_slider = ArchetypeSlider(position=0.1)  # Universal feeling (emotional/creative)
    balanced_slider = ArchetypeSlider(position=0.5)  # Balanced
    specific_observation_slider = ArchetypeSlider(position=0.9)  # Specific observation (analytical)
    
    # Print slider information
    print(f"Universal Feeling Slider (position=0.1):")
    print(f"  Description: {universal_feeling_slider.get_archetype_description()}")
    weights = universal_feeling_slider.get_archetype_weights()
    print(f"  Weights: Universal Feeling = {weights['universal_feeling']:.2f}, " +
          f"Specific Observation = {weights['specific_observation']:.2f}")
    
    print(f"\nBalanced Slider (position=0.5):")
    print(f"  Description: {balanced_slider.get_archetype_description()}")
    weights = balanced_slider.get_archetype_weights()
    print(f"  Weights: Universal Feeling = {weights['universal_feeling']:.2f}, " +
          f"Specific Observation = {weights['specific_observation']:.2f}")
    
    print(f"\nSpecific Observation Slider (position=0.9):")
    print(f"  Description: {specific_observation_slider.get_archetype_description()}")
    weights = specific_observation_slider.get_archetype_weights()
    print(f"  Weights: Universal Feeling = {weights['universal_feeling']:.2f}, " +
          f"Specific Observation = {weights['specific_observation']:.2f}")
    
    # 2. Test ArchetypeSlider with operators
    print("\n2. Testing ArchetypeSlider with Operators")
    print("----------------------------------------")
    
    # Create a quantum state
    state = PrimeHilbertSpace(max_prime_index=20)
    state.set_state_from_number(137)  # Consciousness number
    
    # Create operators
    feeling_op = FeelingResonanceOperator()
    coherence_op = CoherenceOperator(137)
    
    # Apply operators with different archetype sliders
    print("\nApplying FeelingResonanceOperator:")
    
    # Universal feeling archetype
    feeling_result_uf = universal_feeling_slider.apply_to_operator(feeling_op, state)
    feeling_measure_uf = feeling_op.feeling_measure(feeling_result_uf)
    
    # Balanced archetype
    feeling_result_bal = balanced_slider.apply_to_operator(feeling_op, state)
    feeling_measure_bal = feeling_op.feeling_measure(feeling_result_bal)
    
    # Specific observation archetype
    feeling_result_so = specific_observation_slider.apply_to_operator(feeling_op, state)
    feeling_measure_so = feeling_op.feeling_measure(feeling_result_so)
    
    print(f"  Universal Feeling Archetype: {feeling_measure_uf:.4f}")
    print(f"  Balanced Archetype: {feeling_measure_bal:.4f}")
    print(f"  Specific Observation Archetype: {feeling_measure_so:.4f}")
    
    print("\nApplying CoherenceOperator:")
    
    # Universal feeling archetype
    coherence_result_uf = universal_feeling_slider.apply_to_operator(coherence_op, state)
    coherence_measure_uf = coherence_op.coherence_measure(coherence_result_uf)
    
    # Balanced archetype
    coherence_result_bal = balanced_slider.apply_to_operator(coherence_op, state)
    coherence_measure_bal = coherence_op.coherence_measure(coherence_result_bal)
    
    # Specific observation archetype
    coherence_result_so = specific_observation_slider.apply_to_operator(coherence_op, state)
    coherence_measure_so = coherence_op.coherence_measure(coherence_result_so)
    
    print(f"  Universal Feeling Archetype: {coherence_measure_uf:.4f}")
    print(f"  Balanced Archetype: {coherence_measure_bal:.4f}")
    print(f"  Specific Observation Archetype: {coherence_measure_so:.4f}")
    
    # 3. Test ArchetypeSlider with semantic fields
    print("\n3. Testing ArchetypeSlider with Semantic Fields")
    print("---------------------------------------------")
    
    # Create fields with different archetype positions
    universal_feeling_field = SemanticField(max_prime_index=20, archetype_position=0.1)
    balanced_field = SemanticField(max_prime_index=20, archetype_position=0.5)
    specific_observation_field = SemanticField(max_prime_index=20, archetype_position=0.9)
    
    # Add the same concepts to all fields
    for field in [universal_feeling_field, balanced_field, specific_observation_field]:
        field.add_node("consciousness", 137)
        field.add_node("quantum", 73)
        field.add_node("reality", 97)
        field.add_node("observer", 41)
        
        # Add relationships
        field.add_edge("consciousness", "quantum", 0.8, "influences")
        field.add_edge("quantum", "reality", 0.7, "describes")
        field.add_edge("consciousness", "observer", 0.9, "embodies")
        field.add_edge("observer", "reality", 0.6, "perceives")
    
    # Evolve all fields
    print("Evolving fields with different archetype positions...")
    universal_feeling_states = universal_feeling_field.evolve_field(steps=5)
    balanced_states = balanced_field.evolve_field(steps=5)
    specific_observation_states = specific_observation_field.evolve_field(steps=5)
    
    # Get field state info for all fields
    uf_info = universal_feeling_field.get_field_state_info()
    bal_info = balanced_field.get_field_state_info()
    so_info = specific_observation_field.get_field_state_info()
    
    # Print field state info
    print("\nUniversal Feeling Field (position=0.1):")
    print(f"  Description: {uf_info['archetype_description']}")
    print(f"  Feeling Resonance: {uf_info['feeling_resonance']:.4f}")
    print(f"  Coherence: {uf_info['coherence']:.4f}")
    print(f"  Consciousness Primacy: {uf_info['consciousness_primacy']:.4f}")
    print(f"  Top Concepts: {', '.join([f'{c} ({v:.2f})' for c, v in list(uf_info['top_concepts'].items())[:3]])}")
    
    print("\nBalanced Field (position=0.5):")
    print(f"  Description: {bal_info['archetype_description']}")
    print(f"  Feeling Resonance: {bal_info['feeling_resonance']:.4f}")
    print(f"  Coherence: {bal_info['coherence']:.4f}")
    print(f"  Consciousness Primacy: {bal_info['consciousness_primacy']:.4f}")
    print(f"  Top Concepts: {', '.join([f'{c} ({v:.2f})' for c, v in list(bal_info['top_concepts'].items())[:3]])}")
    
    print("\nSpecific Observation Field (position=0.9):")
    print(f"  Description: {so_info['archetype_description']}")
    print(f"  Feeling Resonance: {so_info['feeling_resonance']:.4f}")
    print(f"  Coherence: {so_info['coherence']:.4f}")
    print(f"  Consciousness Primacy: {so_info['consciousness_primacy']:.4f}")
    print(f"  Top Concepts: {', '.join([f'{c} ({v:.2f})' for c, v in list(so_info['top_concepts'].items())[:3]])}")
    
    # 4. Test dynamic adjustment of archetype position
    print("\n4. Testing Dynamic Adjustment of Archetype Position")
    print("------------------------------------------------")
    
    # Create a field with initial balanced position
    dynamic_field = SemanticField(max_prime_index=20, archetype_position=0.5)
    
    # Add concepts
    dynamic_field.add_node("consciousness", 137)
    dynamic_field.add_node("quantum", 73)
    dynamic_field.add_node("reality", 97)
    dynamic_field.add_node("observer", 41)
    
    # Add relationships
    dynamic_field.add_edge("consciousness", "quantum", 0.8, "influences")
    dynamic_field.add_edge("quantum", "reality", 0.7, "describes")
    dynamic_field.add_edge("consciousness", "observer", 0.9, "embodies")
    dynamic_field.add_edge("observer", "reality", 0.6, "perceives")
    
    # Evolve field with balanced archetype
    print("Evolving field with balanced archetype (position=0.5)...")
    balanced_states = dynamic_field.evolve_field(steps=3)
    balanced_info = dynamic_field.get_field_state_info()
    
    # Change archetype position to universal feeling
    dynamic_field.set_archetype_position(0.1)
    print("Changed archetype position to universal feeling (position=0.1)...")
    
    # Evolve field with universal feeling archetype
    universal_feeling_states = dynamic_field.evolve_field(steps=3)
    uf_info = dynamic_field.get_field_state_info()
    
    # Change archetype position to specific observation
    dynamic_field.set_archetype_position(0.9)
    print("Changed archetype position to specific observation (position=0.9)...")
    
    # Evolve field with specific observation archetype
    specific_observation_states = dynamic_field.evolve_field(steps=3)
    so_info = dynamic_field.get_field_state_info()
    
    # Print results
    print("\nResults after dynamic adjustment:")
    print(f"Balanced (position=0.5):")
    print(f"  Description: {balanced_info['archetype_description']}")
    print(f"  Feeling Resonance: {balanced_info['feeling_resonance']:.4f}")
    print(f"  Coherence: {balanced_info['coherence']:.4f}")
    
    print(f"\nUniversal Feeling (position=0.1):")
    print(f"  Description: {uf_info['archetype_description']}")
    print(f"  Feeling Resonance: {uf_info['feeling_resonance']:.4f}")
    print(f"  Coherence: {uf_info['coherence']:.4f}")
    
    print(f"\nSpecific Observation (position=0.9):")
    print(f"  Description: {so_info['archetype_description']}")
    print(f"  Feeling Resonance: {so_info['feeling_resonance']:.4f}")
    print(f"  Coherence: {so_info['coherence']:.4f}")
    
    print("\n=== ArchetypeSlider Test Complete ===")
    print("The test results demonstrate how the archetype slider balances between")
    print("universal feeling (emotional/creative) and specific observation (analytical)")
    print("approaches. This allows determining the archetype of the agent, with the")
    print("default being a balanced setting.")

if __name__ == "__main__":
    asyncio.run(test_archetype_slider())