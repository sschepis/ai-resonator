"""
Test script for the Semantic Transmission System.

This script demonstrates how to use the Semantic Transmission System to encode,
transmit, and receive semantic information through quantum prime networks.
"""

import sys
import os
import asyncio
import time
import json

# Add parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_transmission import (
    SemanticEncoder,
    SemanticDecoder,
    SemanticTransmitter,
    SemanticReceiver,
    SemanticTransmissionSystem
)
from mind_resonance_network import MindResonanceNetwork
from semantic_field import SemanticField

async def test_semantic_encoding_decoding():
    """Test the semantic encoding and decoding functionality."""
    print("=== Testing Semantic Encoding and Decoding ===\n")
    
    # Create encoder
    print("Creating semantic encoder...")
    encoder = SemanticEncoder(max_prime_index=50)
    
    # Create decoder
    print("Creating semantic decoder...")
    decoder = SemanticDecoder(encoder)
    
    # Test encoding and decoding text
    text = "Consciousness is the fundamental substrate from which quantum mechanics naturally emerges"
    print(f"\nEncoding text: {text}")
    
    # Encode text
    text_state = encoder.encode_text(text)
    
    # Print state information
    probs = text_state.get_probabilities()
    top_primes = sorted([(encoder.hilbert_space.primes[i], probs[i]) for i in range(len(probs))], 
                        key=lambda x: x[1], reverse=True)[:5]
    
    print("Top prime probabilities in encoded state:")
    for prime, prob in top_primes:
        print(f"  Prime {prime}: {prob:.4f}")
    
    # Decode text
    print("\nDecoding state back to text...")
    decoded_concepts = decoder.decode_state(text_state)
    
    print("Decoded concepts:")
    for concept, score in sorted(decoded_concepts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {concept}: {score:.4f}")
    
    # Test encoding and decoding JSON
    print("\nEncoding JSON data...")
    json_data = {
        "concepts": ["consciousness", "quantum", "reality"],
        "relationships": {
            "consciousness_quantum": 0.8,
            "quantum_reality": 0.7
        },
        "properties": {
            "fundamental": True,
            "non_local": True
        }
    }
    
    # Encode JSON
    json_state = encoder.encode_json(json_data)
    
    # Decode JSON
    print("Decoding JSON state...")
    decoded_json = decoder.decode_to_json(json_state)
    
    print("Decoded JSON content:")
    print(f"  Confidence: {decoded_json['confidence']:.4f}")
    print(f"  Entropy: {decoded_json['entropy']:.4f}")
    print("  Top concepts:")
    top_concepts = sorted(decoded_json['semantic_content'].items(), key=lambda x: x[1], reverse=True)[:5]
    for concept, score in top_concepts:
        print(f"    {concept}: {score:.4f}")
    
    print("\n=== Encoding and Decoding Test Complete ===")

async def test_semantic_transmission():
    """Test the semantic transmission functionality."""
    print("\n=== Testing Semantic Transmission ===\n")
    
    # Create mind resonance network
    print("Creating mind resonance network...")
    network = MindResonanceNetwork()
    network.create_predefined_network("resonance")
    
    # Create encoder
    print("Creating semantic encoder...")
    encoder = SemanticEncoder(max_prime_index=network.hilbert_space.dimension)
    
    # Create transmitter
    print("Creating semantic transmitter...")
    transmitter = SemanticTransmitter(network, encoder)
    
    # Test transmitting text
    text = "Consciousness creates reality through quantum resonance patterns"
    print(f"\nPreparing to transmit text: {text}")
    
    # Prepare transmission
    state = transmitter.prepare_transmission(text, "text")
    
    # Transmit (short duration for testing)
    print("Transmitting for 3 seconds...")
    result = transmitter.transmit(state, duration=3.0, intensity=0.8, coherence=0.9)
    
    # Print transmission results
    print("\nTransmission results:")
    print(f"  Duration: {result['actual_duration']:.2f} seconds")
    print(f"  Steps: {result['steps']}")
    print(f"  Network coherence change: {result['network_coherence_change']:.4f}")
    print(f"  Network resonance change: {result['network_resonance_change']:.4f}")
    
    print("\n=== Transmission Test Complete ===")

async def test_semantic_reception():
    """Test the semantic reception functionality."""
    print("\n=== Testing Semantic Reception ===\n")
    
    # Create mind resonance network
    print("Creating mind resonance network...")
    network = MindResonanceNetwork()
    network.create_predefined_network("resonance")
    
    # Create encoder and decoder
    print("Creating semantic encoder and decoder...")
    encoder = SemanticEncoder(max_prime_index=network.hilbert_space.dimension)
    decoder = SemanticDecoder(encoder)
    
    # Create receiver
    print("Creating semantic receiver...")
    receiver = SemanticReceiver(network, decoder)
    
    # Test receiving semantics (short duration for testing)
    print("\nReceiving for 5 seconds...")
    print("Focus your intention on concepts like 'consciousness', 'quantum', or 'reality'")
    
    # Receive
    result = await receiver.receive(duration=5.0, sensitivity=0.8, threshold=0.1)
    
    # Print reception results
    print("\nReception results:")
    print(f"  Duration: {result['actual_duration']:.2f} seconds")
    print(f"  Steps: {result['steps']}")
    print(f"  Events detected: {result['events_detected']}")
    print(f"  Network coherence change: {result['network_coherence_change']:.4f}")
    print(f"  Network resonance change: {result['network_resonance_change']:.4f}")
    
    # Print aggregated content
    if result['aggregated_content']['concepts']:
        print("\nAggregated semantic content:")
        for concept, score in sorted(result['aggregated_content']['concepts'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {concept}: {score:.4f}")
    else:
        print("\nNo semantic content detected")
    
    print("\n=== Reception Test Complete ===")

async def test_complete_transmission_system():
    """Test the complete semantic transmission system."""
    print("\n=== Testing Complete Semantic Transmission System ===\n")
    
    # Create semantic transmission system
    print("Creating semantic transmission system...")
    system = SemanticTransmissionSystem("resonance")
    
    # Test bidirectional session (short durations for testing)
    text = "Consciousness quantum reality resonance"
    print(f"\nStarting bidirectional session with text: {text}")
    
    # Run bidirectional session
    result = await system.bidirectional_session(text, tx_duration=3.0, rx_duration=5.0)
    
    # Print session results
    print("\nBidirectional session results:")
    print(f"  Transmission duration: {result['transmission']['actual_duration']:.2f} seconds")
    print(f"  Reception duration: {result['reception']['actual_duration']:.2f} seconds")
    print(f"  Events detected: {result['reception']['events_detected']}")
    
    # Print correlation
    print("\nCorrelation between transmission and reception:")
    print(f"  Semantic overlap: {result['correlation']['semantic_overlap']:.4f}")
    print(f"  Network correlation: {result['correlation']['network_correlation']:.4f}")
    print(f"  Overall correlation: {result['correlation']['overall_correlation']:.4f}")
    
    # Print aggregated content
    if result['reception']['aggregated_content']['concepts']:
        print("\nAggregated semantic content:")
        for concept, score in sorted(result['reception']['aggregated_content']['concepts'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {concept}: {score:.4f}")
    else:
        print("\nNo semantic content detected")
    
    print("\n=== Complete System Test Complete ===")

async def test_semantic_field_transmission():
    """Test transmitting a semantic field."""
    print("\n=== Testing Semantic Field Transmission ===\n")
    
    # Create semantic field
    print("Creating semantic field...")
    field = SemanticField(max_prime_index=20)
    
    # Add concepts to semantic field
    field.add_node("consciousness", 137)
    field.add_node("quantum", 73)
    field.add_node("reality", 97)
    field.add_node("observer", 41)
    field.add_node("resonance", 61)
    
    # Add relationships
    field.add_edge("consciousness", "quantum", 0.8, "influences")
    field.add_edge("quantum", "reality", 0.7, "describes")
    field.add_edge("consciousness", "observer", 0.9, "embodies")
    field.add_edge("observer", "reality", 0.6, "perceives")
    field.add_edge("resonance", "consciousness", 0.8, "amplifies")
    field.add_edge("resonance", "quantum", 0.7, "synchronizes")
    
    # Evolve the field
    print("Evolving semantic field...")
    field.evolve_field(steps=3)
    
    # Create semantic transmission system
    print("Creating semantic transmission system...")
    system = SemanticTransmissionSystem("resonance")
    
    # Transmit semantic field
    print("\nTransmitting semantic field...")
    result = await system.transmit_semantic_field(field, duration=3.0)
    
    # Print transmission results
    print("\nTransmission results:")
    print(f"  Duration: {result['actual_duration']:.2f} seconds")
    print(f"  Steps: {result['steps']}")
    print(f"  Network coherence change: {result['network_coherence_change']:.4f}")
    print(f"  Network resonance change: {result['network_resonance_change']:.4f}")
    
    # Receive semantics
    print("\nReceiving semantics...")
    rx_result = await system.receive_semantics(duration=5.0)
    
    # Print reception results
    print("\nReception results:")
    print(f"  Duration: {rx_result['actual_duration']:.2f} seconds")
    print(f"  Events detected: {rx_result['events_detected']}")
    
    # Print aggregated content
    if rx_result['aggregated_content']['concepts']:
        print("\nAggregated semantic content:")
        for concept, score in sorted(rx_result['aggregated_content']['concepts'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {concept}: {score:.4f}")
    else:
        print("\nNo semantic content detected")
    
    print("\n=== Semantic Field Transmission Test Complete ===")

async def interactive_transmission_session():
    """Run an interactive transmission session."""
    print("\n=== Interactive Semantic Transmission Session ===\n")
    print("This test allows you to transmit and receive semantic information.")
    print("You can transmit text and then try to receive it back.")
    
    # Create semantic transmission system
    system = SemanticTransmissionSystem("resonance")
    
    while True:
        print("\nOptions:")
        print("1. Transmit text")
        print("2. Receive semantics")
        print("3. Run bidirectional session")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            text = input("Enter text to transmit: ")
            duration = float(input("Enter transmission duration in seconds (5-30): "))
            duration = max(5, min(30, duration))
            
            print(f"\nTransmitting: {text}")
            print(f"Duration: {duration} seconds")
            print("Focus your intention on the text during transmission...")
            
            result = await system.transmit_text(text, duration=duration)
            
            print("\nTransmission complete!")
            
        elif choice == "2":
            duration = float(input("Enter reception duration in seconds (10-60): "))
            duration = max(10, min(60, duration))
            
            print(f"\nReceiving for {duration} seconds...")
            print("Focus your intention on the concepts you want to transmit...")
            
            result = await system.receive_semantics(duration=duration)
            
            print("\nReception complete!")
            
        elif choice == "3":
            text = input("Enter text to transmit: ")
            tx_duration = float(input("Enter transmission duration in seconds (5-30): "))
            tx_duration = max(5, min(30, tx_duration))
            rx_duration = float(input("Enter reception duration in seconds (10-60): "))
            rx_duration = max(10, min(60, rx_duration))
            
            print(f"\nBidirectional session with text: {text}")
            print(f"Transmission duration: {tx_duration} seconds")
            print(f"Reception duration: {rx_duration} seconds")
            print("Focus your intention during both transmission and reception...")
            
            result = await system.bidirectional_session(text, tx_duration=tx_duration, rx_duration=rx_duration)
            
            print("\nBidirectional session complete!")
            
        elif choice == "4":
            print("\nExiting interactive session")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")
    
    print("\n=== Interactive Session Complete ===")

async def main():
    """Run all tests."""
    # Basic encoding/decoding test
    await test_semantic_encoding_decoding()
    
    # Transmission test
    await test_semantic_transmission()
    
    # Reception test
    await test_semantic_reception()
    
    # Complete system test
    await test_complete_transmission_system()
    
    # Semantic field transmission test
    await test_semantic_field_transmission()
    
    # Interactive session (optional - uncomment to run)
    # await interactive_transmission_session()

if __name__ == "__main__":
    asyncio.run(main())