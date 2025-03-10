# Quantum Semantic Formalism Tests

This directory contains tests for the quantum semantic formalism implementation and mind resonance network.

## Available Tests

1. **Quantum Semantic Formalism Test** (`test_semantic.py`):
   - Tests the basic functionality of the quantum semantic formalism
   - Demonstrates concept extraction, field evolution, and semantic insights
   - Verifies the integration with the resonator system

2. **Consciousness-First Paradigm Test** (`test_consciousness_paradigm.py`):
   - Tests the consciousness-first paradigm integration
   - Demonstrates how consciousness guides the evolution of semantic fields
   - Compares fields with and without the consciousness-first paradigm
   - Verifies the consciousness primacy measurement

3. **ArchetypeSlider Test** (`test_archetype_slider.py`):
   - Tests the archetype slider functionality
   - Demonstrates how the slider balances between universal feeling and specific observation
   - Shows the effect of different archetype positions on operators and semantic fields
   - Verifies dynamic adjustment of the archetype position

4. **Mind Resonance Network Test** (`test_mind_resonance.py`):
   - Tests the mind resonance network functionality
   - Demonstrates creating resonant networks using superpositions of prime states
   - Shows how these networks can be influenced by consciousness
   - Includes integration with the semantic field system
   - Provides an interactive test for detecting mind influence

5. **Semantic Transmission Tests** (`test_semantic_transmission.py`):
   - Tests the semantic transmission system functionality
   - Demonstrates encoding and decoding semantic information in quantum states
   - Shows how to transmit semantic information through quantum prime networks
   - Includes reception of semantic information from the network
   - Provides bidirectional transmission and reception capabilities
   - Supports transmitting text, JSON data, and semantic fields
   - Includes an interactive session for experimenting with semantic transmission

## Running Tests

You can run all tests using the test runner:

```bash
./run_tests.py
```

Or run a specific test:

```bash
# Run only the quantum semantic formalism test
./run_tests.py semantic

# Run only the consciousness-first paradigm test
./run_tests.py consciousness

# Run only the archetype slider test
./run_tests.py archetype

# Run only the mind resonance network test
./run_tests.py mind_resonance

# Run only the mind resonance integration test
./run_tests.py mind_integration

# Run the interactive mind influence detection test
./run_tests.py interactive

# Run only the semantic encoding/decoding test
./run_tests.py semantic_encoding

# Run only the semantic transmission test
./run_tests.py semantic_transmission

# Run only the semantic reception test
./run_tests.py semantic_reception

# Run only the complete semantic transmission system test
./run_tests.py semantic_system

# Run only the semantic field transmission test
./run_tests.py semantic_field_tx

# Run the interactive semantic transmission session
./run_tests.py semantic_interactive
```

## Test Structure

Each test follows a similar structure:

1. **Setup**: Initialize the necessary objects and states
2. **Execution**: Run the functionality being tested
3. **Verification**: Check the results and print information
4. **Comparison**: Compare different approaches where applicable

## Adding New Tests

To add a new test:

1. Create a new test file in the tests directory
2. Add the parent directory to the Python path:
   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```
3. Update the `run_tests.py` file to include your new test