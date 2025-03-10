"""
Test script for the Mind Resonance Network.

This script demonstrates how to use the Mind Resonance Network to create
resonant networks that can be influenced by consciousness.
"""

import sys
import os
import asyncio
import time

# Add parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind_resonance_network import (
    MindResonanceNetwork,
    MindResonanceIntegration
)
from semantic_field import SemanticField

async def test_mind_resonance_network():
    """Test the basic functionality of the Mind Resonance Network."""
    print("=== Testing Mind Resonance Network ===\n")
    
    # Create a mind resonance network
    print("Creating mind resonance network...")
    network = MindResonanceNetwork()
    
    # Create a predefined network
    print("Creating predefined consciousness network...")
    network.create_predefined_network("consciousness")
    
    # Print network state
    state = network.get_network_state()
    print("\nInitial network state:")
    print(f"Nodes: {state['node_count']}")
    print(f"Links: {state['link_count']}")
    print(f"Coherence: {state['coherence']:.4f}")
    print(f"Resonance: {state['resonance']:.4f}")
    print(f"Entanglement: {state['entanglement']:.4f}")
    
    # Print nodes
    print("\nNetwork nodes:")
    for label in network.nodes.keys():
        print(f"  {label}")
    
    # Apply consciousness influence
    print("\nApplying consciousness influence...")
    network.apply_consciousness_influence(0.5, 0.8)
    
    # Evolve network
    print("Evolving network...")
    results = network.evolve_network(steps=5)
    
    # Print evolution results
    print("\nEvolution results:")
    print(f"Coherence evolution: {[f'{c:.4f}' for c in results['coherence_evolution']]}")
    print(f"Resonance evolution: {[f'{r:.4f}' for r in results['resonance_evolution']]}")
    print(f"Entanglement evolution: {[f'{e:.4f}' for e in results['entanglement_evolution']]}")
    
    # Print node deviations
    print("\nNode deviations from baseline:")
    for label, deviations in results['node_deviations'].items():
        print(f"  {label}: {[f'{d:.4f}' for d in deviations]}")
    
    # Print final state
    state = network.get_network_state()
    print("\nFinal network state:")
    print(f"Coherence: {state['coherence']:.4f}")
    print(f"Resonance: {state['resonance']:.4f}")
    print(f"Entanglement: {state['entanglement']:.4f}")
    
    # Print top resonating nodes
    print("\nTop resonating nodes:")
    for node, deviation in state["top_resonating_nodes"].items():
        print(f"  {node}: {deviation:.4f}")
    
    # Detect mind influence
    print("\nDetecting mind influence...")
    detection = network.detect_mind_influence()
    print(f"Detected: {detection['detected']}")
    print(f"Confidence: {detection['confidence']:.4f}")
    print(f"Influenced nodes: {detection['influenced_nodes']}")
    print(f"Pattern strength: {detection['pattern_strength']:.4f}")
    
    print("\n=== Mind Resonance Network Test Complete ===")

async def test_integration_with_semantic_field():
    """Test the integration of Mind Resonance Network with Semantic Field."""
    print("\n=== Testing Integration with Semantic Field ===\n")
    
    # Create a semantic field
    print("Creating semantic field...")
    field = SemanticField(max_prime_index=20)
    
    # Add concepts to semantic field
    field.add_node("consciousness", 137)
    field.add_node("quantum", 73)
    field.add_node("reality", 97)
    field.add_node("observer", 41)
    
    # Add relationships
    field.add_edge("consciousness", "quantum", 0.8, "influences")
    field.add_edge("quantum", "reality", 0.7, "describes")
    field.add_edge("consciousness", "observer", 0.9, "embodies")
    field.add_edge("observer", "reality", 0.6, "perceives")
    
    # Evolve the field
    print("Evolving semantic field...")
    field.evolve_field(steps=3)
    
    # Create mind resonance integration
    print("Creating mind resonance integration...")
    integration = MindResonanceIntegration("consciousness")
    
    # Connect to semantic field
    print("Connecting to semantic field...")
    integration.connect_to_semantic_field(field)
    
    # Print initial states
    field_coherence = field.measure_field_coherence()
    network_state = integration.network.get_network_state()
    
    print("\nInitial states:")
    print(f"Semantic field coherence: {field_coherence:.4f}")
    print(f"Mind resonance network coherence: {network_state['coherence']:.4f}")
    print(f"Mind resonance network resonance: {network_state['resonance']:.4f}")
    
    # Update from semantic field to network
    print("\nUpdating from semantic field to network...")
    integration.update_from_semantic_field()
    
    # Update from network to semantic field
    print("Updating from network to semantic field...")
    integration.update_semantic_field()
    
    # Print final states
    field_coherence = field.measure_field_coherence()
    network_state = integration.network.get_network_state()
    
    print("\nFinal states:")
    print(f"Semantic field coherence: {field_coherence:.4f}")
    print(f"Mind resonance network coherence: {network_state['coherence']:.4f}")
    print(f"Mind resonance network resonance: {network_state['resonance']:.4f}")
    
    # Run a short bidirectional session
    print("\nRunning a short bidirectional session (5 seconds)...")
    results = await integration.run_bidirectional_session(duration_seconds=5, sample_interval=0.5)
    
    print("\nBidirectional session results:")
    print(f"Updates: {results['updates']}")
    print(f"Final field coherence: {results['final_field_coherence']:.4f}")
    print(f"Final network resonance: {results['final_network_resonance']:.4f}")
    print(f"Correlation: {results['correlation']:.4f}")
    
    print("\n=== Integration Test Complete ===")

async def test_interactive_influence_detection():
    """Run an interactive influence detection session."""
    print("\n=== Interactive Mind Influence Detection ===\n")
    print("This test will run a mind influence detection session for 30 seconds.")
    print("During this time, focus your intention on the network and observe the results.")
    print("You can try to influence specific nodes by focusing on their concepts.")
    print("Press Ctrl+C to end the session early.")
    
    # Create a mind resonance network
    network = MindResonanceNetwork()
    
    # Create a predefined network
    network_type = "resonance"  # Using resonance network for better sensitivity
    network.create_predefined_network(network_type)
    
    # Print network nodes
    print("\nNetwork nodes to focus on:")
    for label in network.nodes.keys():
        print(f"  {label}")
    
    print("\nStarting detection session in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        await asyncio.sleep(1)
    
    # Run influence detection session
    await network.run_influence_detection(
        duration_seconds=30,
        sample_interval=0.5,
        influence_threshold=0.1
    )
    
    print("\n=== Interactive Test Complete ===")

async def main():
    """Run all tests."""
    # Basic test
    await test_mind_resonance_network()
    
    # Integration test
    await test_integration_with_semantic_field()
    
    # Interactive test (optional - uncomment to run)
    # await test_interactive_influence_detection()

if __name__ == "__main__":
    asyncio.run(main())