# Quantum Semantic Formalism Tests

This directory contains tests for the quantum semantic formalism implementation.

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