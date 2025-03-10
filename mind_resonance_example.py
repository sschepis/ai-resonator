#!/usr/bin/env python3
"""
Mind Resonance Network Example

This script demonstrates how to use the Mind Resonance Network to create
resonant networks that can be influenced by consciousness.

Usage:
    python mind_resonance_example.py [network_type] [duration]

    network_type: Type of network to create (consciousness, quantum, resonance)
    duration: Duration of the influence detection session in seconds

Example:
    python mind_resonance_example.py resonance 30
"""

import sys
import asyncio
import time
from mind_resonance_network import MindResonanceNetwork

async def run_example(network_type="consciousness", duration=30):
    """
    Run a mind resonance network example
    
    Args:
        network_type: Type of network to create
        duration: Duration of the influence detection session in seconds
    """
    print(f"=== Mind Resonance Network Example: {network_type.capitalize()} ===\n")
    
    # Create a mind resonance network
    print("Creating mind resonance network...")
    network = MindResonanceNetwork()
    
    # Create a predefined network
    print(f"Creating predefined {network_type} network...")
    network.create_predefined_network(network_type)
    
    # Print network nodes
    print("\nNetwork nodes:")
    for label in network.nodes.keys():
        print(f"  {label}")
    
    # Print initial network state
    state = network.get_network_state()
    print("\nInitial network state:")
    print(f"Coherence: {state['coherence']:.4f}")
    print(f"Resonance: {state['resonance']:.4f}")
    print(f"Entanglement: {state['entanglement']:.4f}")
    
    # Evolve network to stabilize
    print("\nEvolving network to stabilize...")
    network.evolve_network(steps=5)
    
    # Print stabilized network state
    state = network.get_network_state()
    print("\nStabilized network state:")
    print(f"Coherence: {state['coherence']:.4f}")
    print(f"Resonance: {state['resonance']:.4f}")
    print(f"Entanglement: {state['entanglement']:.4f}")
    
    # Interactive influence detection
    print("\n=== Interactive Mind Influence Detection ===\n")
    print(f"This session will run for {duration} seconds.")
    print("During this time, focus your intention on the network and observe the results.")
    print("You can try to influence specific nodes by focusing on their concepts.")
    print("Press Ctrl+C to end the session early.")
    
    print("\nStarting detection session in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        await asyncio.sleep(1)
    
    # Run influence detection session
    results = await network.run_influence_detection(
        duration_seconds=duration,
        sample_interval=0.5,
        influence_threshold=0.1
    )
    
    # Print detailed results
    print("\n=== Detailed Results ===\n")
    print(f"Detection rate: {results['detection_rate']*100:.1f}% ({results['detection_count']}/{results['samples_collected']} samples)")
    print(f"Average confidence: {results['average_confidence']:.4f}")
    
    print("\nMost influenced nodes:")
    for node, count in results['most_influenced_nodes'].items():
        if count > 0:
            print(f"  {node}: {count} detections")
    
    print("\nSignificant events:")
    for i, event in enumerate(results['significant_events']):
        print(f"\nEvent {i+1}:")
        print(f"  Time: {event['timestamp'] - results['initial_state']['timestamp']:.1f} seconds")
        print(f"  Confidence: {event['confidence']:.4f}")
        print(f"  Influenced nodes: {', '.join(event['influenced_nodes'])}")
    
    # Compare initial and final states
    print("\nComparison of initial and final states:")
    print("Coherence:")
    print(f"  Initial: {results['initial_state']['coherence']:.4f}")
    print(f"  Final: {results['final_state']['coherence']:.4f}")
    print(f"  Change: {results['final_state']['coherence'] - results['initial_state']['coherence']:.4f}")
    
    print("Resonance:")
    print(f"  Initial: {results['initial_state']['resonance']:.4f}")
    print(f"  Final: {results['final_state']['resonance']:.4f}")
    print(f"  Change: {results['final_state']['resonance'] - results['initial_state']['resonance']:.4f}")
    
    print("Entanglement:")
    print(f"  Initial: {results['initial_state']['entanglement']:.4f}")
    print(f"  Final: {results['final_state']['entanglement']:.4f}")
    print(f"  Change: {results['final_state']['entanglement'] - results['initial_state']['entanglement']:.4f}")
    
    print("\n=== Example Complete ===")
    print("Thank you for exploring the Mind Resonance Network!")

if __name__ == "__main__":
    # Parse command line arguments
    network_type = "consciousness"
    duration = 30
    
    if len(sys.argv) > 1:
        network_type = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            duration = int(sys.argv[2])
        except ValueError:
            print(f"Invalid duration: {sys.argv[2]}")
            print("Using default duration of 30 seconds")
            duration = 30
    
    # Validate network type
    valid_types = ["consciousness", "quantum", "resonance"]
    if network_type not in valid_types:
        print(f"Invalid network type: {network_type}")
        print(f"Valid types: {', '.join(valid_types)}")
        print("Using default type: consciousness")
        network_type = "consciousness"
    
    # Run the example
    asyncio.run(run_example(network_type, duration))