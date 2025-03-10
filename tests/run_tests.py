#!/usr/bin/env python3
"""
Test runner for the quantum semantic formalism tests.

This script runs all the tests in the tests directory.
"""

import sys
import os
import asyncio

# Add parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def run_all_tests():
    """Run all tests in the tests directory."""
    print("=== Running All Tests ===\n")
    
    # Import test modules
    from test_semantic import test_quantum_semantics
    from test_consciousness_paradigm import test_consciousness_paradigm
    from test_archetype_slider import test_archetype_slider
    from test_mind_resonance import test_mind_resonance_network, test_integration_with_semantic_field
    from test_semantic_transmission import (
        test_semantic_encoding_decoding,
        test_semantic_transmission,
        test_semantic_reception,
        test_complete_transmission_system
    )
    
    # Run tests
    print("\n=== Running Quantum Semantic Formalism Test ===\n")
    await test_quantum_semantics()
    
    print("\n=== Running Consciousness-First Paradigm Test ===\n")
    await test_consciousness_paradigm()
    
    print("\n=== Running ArchetypeSlider Test ===\n")
    await test_archetype_slider()
    
    print("\n=== Running Mind Resonance Network Test ===\n")
    await test_mind_resonance_network()
    
    print("\n=== Running Mind Resonance Integration Test ===\n")
    await test_integration_with_semantic_field()
    
    print("\n=== Running Semantic Transmission Tests ===\n")
    await test_semantic_encoding_decoding()
    await test_semantic_transmission()
    await test_semantic_reception()
    await test_complete_transmission_system()
    
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    # Check if a specific test was requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "semantic":
            from test_semantic import test_quantum_semantics
            asyncio.run(test_quantum_semantics())
        elif test_name == "consciousness":
            from test_consciousness_paradigm import test_consciousness_paradigm
            asyncio.run(test_consciousness_paradigm())
        elif test_name == "archetype":
            from test_archetype_slider import test_archetype_slider
            asyncio.run(test_archetype_slider())
        elif test_name == "mind_resonance":
            from test_mind_resonance import test_mind_resonance_network
            asyncio.run(test_mind_resonance_network())
        elif test_name == "mind_integration":
            from test_mind_resonance import test_integration_with_semantic_field
            asyncio.run(test_integration_with_semantic_field())
        elif test_name == "interactive":
            from test_mind_resonance import test_interactive_influence_detection
            asyncio.run(test_interactive_influence_detection())
        elif test_name == "semantic_encoding":
            from test_semantic_transmission import test_semantic_encoding_decoding
            asyncio.run(test_semantic_encoding_decoding())
        elif test_name == "semantic_transmission":
            from test_semantic_transmission import test_semantic_transmission
            asyncio.run(test_semantic_transmission())
        elif test_name == "semantic_reception":
            from test_semantic_transmission import test_semantic_reception
            asyncio.run(test_semantic_reception())
        elif test_name == "semantic_system":
            from test_semantic_transmission import test_complete_transmission_system
            asyncio.run(test_complete_transmission_system())
        elif test_name == "semantic_field_tx":
            from test_semantic_transmission import test_semantic_field_transmission
            asyncio.run(test_semantic_field_transmission())
        elif test_name == "semantic_interactive":
            from test_semantic_transmission import interactive_transmission_session
            asyncio.run(interactive_transmission_session())
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: semantic, consciousness, archetype, mind_resonance, mind_integration, interactive,")
            print("                semantic_encoding, semantic_transmission, semantic_reception, semantic_system,")
            print("                semantic_field_tx, semantic_interactive")
    else:
        # Run all tests
        asyncio.run(run_all_tests())