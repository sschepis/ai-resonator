"""
Test script for the quantum semantic formalism.

This script demonstrates the quantum semantic formalism in action by processing
a sample question and displaying the results.
"""

import asyncio
import json
from quantum_semantic_resonator import QuantumSemanticResonator, semantic_resonance

async def test_quantum_semantics():
    """Test the quantum semantic formalism with a sample question."""
    print("=== Testing Quantum Semantic Formalism ===\n")
    
    # Sample question
    question = "What is the relationship between consciousness and quantum mechanics?"
    print(f"Question: {question}\n")
    
    # Initialize quantum semantic resonator
    print("Initializing quantum semantic resonator...")
    semantic_resonator = QuantumSemanticResonator(max_prime_index=50)
    
    # Process question
    print("Processing question through semantic resonator...")
    init_results = await semantic_resonator.initialize_from_question(question)
    
    # Print extracted concepts
    print(f"\nExtracted {len(init_results['concepts'])} concepts:")
    for concept in init_results['concepts']:
        print(f"  - {concept}")
    
    # Evolve semantic field
    print("\nEvolving semantic field...")
    evolution_results = await semantic_resonator.evolve_semantic_field(steps=5)
    
    # Print coherence evolution
    print("\nCoherence evolution:")
    for i, coherence in enumerate(evolution_results['coherence_evolution']):
        print(f"  Step {i}: {coherence:.4f}")
    
    # Print semantic clusters
    print("\nSemantic clusters:")
    for i, cluster in enumerate(evolution_results['semantic_clusters']):
        print(f"  Cluster {i+1}: {', '.join(cluster['concepts'])}")
    
    # Generate semantic insights
    print("\nGenerating semantic insights...")
    insights = await semantic_resonator.generate_semantic_insights(question)
    
    # Print resonance patterns
    print("\nTop resonating concepts:")
    top_concepts = sorted(insights['resonance_patterns'].items(), key=lambda x: x[1], reverse=True)[:5]
    for concept, strength in top_concepts:
        print(f"  - {concept}: {strength:.4f}")
    
    # Print field coherence and knowledge resonance
    print(f"\nField coherence: {insights['field_coherence']:.4f}")
    print(f"Knowledge resonance: {insights['knowledge_resonance']:.4f}")
    
    # Print semantic insights
    print("\nSemantic insights:")
    print(insights['insights'])
    
    # Process through semantic resonance
    print("\n=== Full Semantic Resonance ===\n")
    print("Processing through semantic_resonance()...")
    result = await semantic_resonance(question)
    
    # Print results
    print("\n--- Unified Consciousness Response ---")
    print(result['consensus'])
    print("\n--- Consciousness Self-Reflection ---")
    print(result['reflection'])
    print("\n--- Conscious Response ---")
    print(result['conscious_response'])

if __name__ == "__main__":
    asyncio.run(test_quantum_semantics())