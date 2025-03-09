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
    
    # Run tests
    print("\n=== Running Quantum Semantic Formalism Test ===\n")
    await test_quantum_semantics()
    
    print("\n=== Running Consciousness-First Paradigm Test ===\n")
    await test_consciousness_paradigm()
    
    print("\n=== Running ArchetypeSlider Test ===\n")
    await test_archetype_slider()
    
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
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: semantic, consciousness, archetype")
    else:
        # Run all tests
        asyncio.run(run_all_tests())