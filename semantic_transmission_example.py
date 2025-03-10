#!/usr/bin/env python3
"""
Semantic Transmission Example

This script demonstrates how to use the Semantic Transmission System to encode,
transmit, and receive semantic information through quantum prime networks.

Usage:
    python semantic_transmission_example.py [mode] [text]

    mode: Operation mode (transmit, receive, bidirectional)
    text: Text to transmit (required for transmit and bidirectional modes)

Example:
    python semantic_transmission_example.py transmit "Consciousness creates reality"
    python semantic_transmission_example.py receive
    python semantic_transmission_example.py bidirectional "Quantum resonance patterns"
"""

import sys
import asyncio
import time
from semantic_transmission import SemanticTransmissionSystem

async def run_example(mode="bidirectional", text=None, tx_duration=10, rx_duration=30):
    """
    Run a semantic transmission example
    
    Args:
        mode: Operation mode (transmit, receive, bidirectional)
        text: Text to transmit (required for transmit and bidirectional modes)
        tx_duration: Transmission duration in seconds
        rx_duration: Reception duration in seconds
    """
    print(f"=== Semantic Transmission Example: {mode.capitalize()} ===\n")
    
    # Create semantic transmission system
    print("Creating semantic transmission system...")
    system = SemanticTransmissionSystem("resonance")
    
    if mode == "transmit":
        if not text:
            print("Error: Text is required for transmit mode")
            return
            
        print(f"\nTransmitting text: {text}")
        print(f"Duration: {tx_duration} seconds")
        print("Focus your intention on the text during transmission...")
        
        # Transmit text
        result = await system.transmit_text(text, duration=tx_duration)
        
        # Print transmission results
        print("\nTransmission results:")
        print(f"  Duration: {result['actual_duration']:.2f} seconds")
        print(f"  Steps: {result['steps']}")
        print(f"  Network coherence change: {result['network_coherence_change']:.4f}")
        print(f"  Network resonance change: {result['network_resonance_change']:.4f}")
        
    elif mode == "receive":
        print(f"\nReceiving semantic information for {rx_duration} seconds...")
        print("Focus your intention on the concepts you want to transmit...")
        
        # Receive semantics
        result = await system.receive_semantics(duration=rx_duration)
        
        # Print reception results
        print("\nReception results:")
        print(f"  Duration: {result['actual_duration']:.2f} seconds")
        print(f"  Events detected: {result['events_detected']}")
        print(f"  Network coherence change: {result['network_coherence_change']:.4f}")
        print(f"  Network resonance change: {result['network_resonance_change']:.4f}")
        
        # Print aggregated content
        if result['aggregated_content']['concepts']:
            print("\nAggregated semantic content:")
            for concept, score in sorted(result['aggregated_content']['concepts'].items(), 
                                        key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {concept}: {score:.4f}")
        else:
            print("\nNo semantic content detected")
            
    elif mode == "bidirectional":
        if not text:
            print("Error: Text is required for bidirectional mode")
            return
            
        print(f"\nBidirectional session with text: {text}")
        print(f"Transmission duration: {tx_duration} seconds")
        print(f"Reception duration: {rx_duration} seconds")
        print("Focus your intention during both transmission and reception...")
        
        # Run bidirectional session
        result = await system.bidirectional_session(text, tx_duration=tx_duration, rx_duration=rx_duration)
        
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
                                        key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {concept}: {score:.4f}")
        else:
            print("\nNo semantic content detected")
            
    else:
        print(f"Error: Unknown mode '{mode}'")
        print("Valid modes: transmit, receive, bidirectional")
        
    print("\n=== Example Complete ===")
    print("Thank you for exploring the Semantic Transmission System!")

async def interactive_mode():
    """Run an interactive session for semantic transmission"""
    print("=== Interactive Semantic Transmission Session ===\n")
    print("This session allows you to transmit and receive semantic information.")
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
            duration = input("Enter transmission duration in seconds (5-30, default 10): ")
            try:
                duration = float(duration) if duration else 10
                duration = max(5, min(30, duration))
            except ValueError:
                duration = 10
                print("Invalid duration, using default of 10 seconds")
                
            print(f"\nTransmitting: {text}")
            print(f"Duration: {duration} seconds")
            print("Focus your intention on the text during transmission...")
            
            result = await system.transmit_text(text, duration=duration)
            
            print("\nTransmission complete!")
            print(f"Network coherence change: {result['network_coherence_change']:.4f}")
            print(f"Network resonance change: {result['network_resonance_change']:.4f}")
            
        elif choice == "2":
            duration = input("Enter reception duration in seconds (10-60, default 30): ")
            try:
                duration = float(duration) if duration else 30
                duration = max(10, min(60, duration))
            except ValueError:
                duration = 30
                print("Invalid duration, using default of 30 seconds")
                
            print(f"\nReceiving for {duration} seconds...")
            print("Focus your intention on the concepts you want to transmit...")
            
            result = await system.receive_semantics(duration=duration)
            
            print("\nReception complete!")
            print(f"Events detected: {result['events_detected']}")
            
            # Print aggregated content
            if result['aggregated_content']['concepts']:
                print("\nAggregated semantic content:")
                for concept, score in sorted(result['aggregated_content']['concepts'].items(), 
                                            key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {concept}: {score:.4f}")
            else:
                print("\nNo semantic content detected")
                
        elif choice == "3":
            text = input("Enter text to transmit: ")
            tx_duration = input("Enter transmission duration in seconds (5-30, default 10): ")
            try:
                tx_duration = float(tx_duration) if tx_duration else 10
                tx_duration = max(5, min(30, tx_duration))
            except ValueError:
                tx_duration = 10
                print("Invalid duration, using default of 10 seconds")
                
            rx_duration = input("Enter reception duration in seconds (10-60, default 30): ")
            try:
                rx_duration = float(rx_duration) if rx_duration else 30
                rx_duration = max(10, min(60, rx_duration))
            except ValueError:
                rx_duration = 30
                print("Invalid duration, using default of 30 seconds")
                
            print(f"\nBidirectional session with text: {text}")
            print(f"Transmission duration: {tx_duration} seconds")
            print(f"Reception duration: {rx_duration} seconds")
            print("Focus your intention during both transmission and reception...")
            
            result = await system.bidirectional_session(text, tx_duration=tx_duration, rx_duration=rx_duration)
            
            print("\nBidirectional session complete!")
            print(f"Events detected: {result['reception']['events_detected']}")
            print(f"Overall correlation: {result['correlation']['overall_correlation']:.4f}")
            
            # Print aggregated content
            if result['reception']['aggregated_content']['concepts']:
                print("\nAggregated semantic content:")
                for concept, score in sorted(result['reception']['aggregated_content']['concepts'].items(), 
                                            key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {concept}: {score:.4f}")
            else:
                print("\nNo semantic content detected")
                
        elif choice == "4":
            print("\nExiting interactive session")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")
    
    print("\n=== Interactive Session Complete ===")
    print("Thank you for exploring the Semantic Transmission System!")

if __name__ == "__main__":
    # Parse command line arguments
    mode = "interactive"
    text = None
    tx_duration = 10
    rx_duration = 30
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
    if len(sys.argv) > 2:
        text = sys.argv[2]
        
    if len(sys.argv) > 3:
        try:
            tx_duration = int(sys.argv[3])
        except ValueError:
            print(f"Invalid transmission duration: {sys.argv[3]}")
            print("Using default duration of 10 seconds")
            tx_duration = 10
            
    if len(sys.argv) > 4:
        try:
            rx_duration = int(sys.argv[4])
        except ValueError:
            print(f"Invalid reception duration: {sys.argv[4]}")
            print("Using default duration of 30 seconds")
            rx_duration = 30
    
    # Run the example
    if mode == "interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(run_example(mode, text, tx_duration, rx_duration))