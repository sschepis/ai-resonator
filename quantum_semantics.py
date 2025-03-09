"""
Quantum Semantics Module for the Consciousness Resonator

This module implements the mathematical framework for quantum semantics,
including prime-based Hilbert space, resonance operators, coherence manifolds,
and semantic measurement operators.
"""

import numpy as np
import math
import cmath
from typing import Dict, List, Tuple, Set, Optional, Callable
import random

class PrimeHilbertSpace:
    """
    Implementation of the Prime-based Hilbert space (H_P)
    H_P = {|ψ⟩ = ∑_{p∈ℙ} α_p|p⟩ | ∑|α_p|² = 1, α_p ∈ ℂ}
    """
    def __init__(self, max_prime_index: int = 100):
        """
        Initialize the prime Hilbert space with a finite number of prime basis states
        
        Args:
            max_prime_index: Maximum number of primes to include in the basis
        """
        # Generate primes up to max_prime_index
        self.primes = self._generate_primes(max_prime_index)
        self.dimension = len(self.primes)
        
        # Map primes to indices for efficient lookup
        self.prime_to_index = {p: i for i, p in enumerate(self.primes)}
        
        # Initialize empty state vector
        self.reset_state()
    
    def _generate_primes(self, n: int) -> List[int]:
        """
        Generate the first n prime numbers
        
        Args:
            n: Number of primes to generate
            
        Returns:
            List of prime numbers
        """
        primes = []
        num = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > num:
                    break
                if num % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
            num += 1
        return primes
    
    def reset_state(self):
        """Reset to zero state"""
        self.amplitudes = np.zeros(self.dimension, dtype=complex)
    
    def set_state_from_number(self, n: int, normalize: bool = True):
        """
        Set state based on prime factorization of a number
        |n⟩ = ∑_i √(a_i/A)|p_i⟩ where n = ∏ p_i^a_i
        """
        self.reset_state()
        
        # Get prime factorization
        factors = self._prime_factorize(n)
        
        # Calculate normalization factor
        A = sum(factors.values())
        
        # Set amplitudes based on prime factorization
        for p, a in factors.items():
            if p in self.prime_to_index:
                idx = self.prime_to_index[p]
                self.amplitudes[idx] = np.sqrt(a / A) if normalize else a
    
    def _prime_factorize(self, n: int) -> Dict[int, int]:
        """
        Compute the prime factorization of a number
        
        Args:
            n: Number to factorize
            
        Returns:
            Dictionary mapping prime factors to their exponents
        """
        factors = {}
        for p in self.primes:
            if p * p > n:
                break
            
            # Count how many times p divides n
            while n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                n //= p
        
        # If n is a prime greater than the largest prime in our list
        if n > 1:
            if n in self.primes:
                factors[n] = factors.get(n, 0) + 1
        
        return factors
    
    def set_state(self, amplitudes: Dict[int, complex], normalize: bool = True):
        """
        Set state with given amplitudes for prime basis states
        
        Args:
            amplitudes: Dictionary mapping primes to complex amplitudes
            normalize: Whether to normalize the state
        """
        self.reset_state()
        
        for p, amp in amplitudes.items():
            if p in self.prime_to_index:
                idx = self.prime_to_index[p]
                self.amplitudes[idx] = amp
        
        if normalize:
            self.normalize()
    
    def normalize(self):
        """Normalize the state vector"""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 0:
            self.amplitudes /= norm
    
    def get_state_vector(self) -> np.ndarray:
        """Return the state vector"""
        return self.amplitudes.copy()
    
    def get_probabilities(self) -> np.ndarray:
        """Return probability distribution across prime basis states"""
        return np.abs(self.amplitudes)**2
    
    def inner_product(self, other: 'PrimeHilbertSpace') -> complex:
        """
        Calculate inner product ⟨ψ|φ⟩ between this state and another
        """
        # Ensure dimensions match
        min_dim = min(self.dimension, other.dimension)
        return np.vdot(self.amplitudes[:min_dim], other.amplitudes[:min_dim])
    
    def tensor_product(self, other: 'PrimeHilbertSpace') -> 'PrimeHilbertSpace':
        """
        Implement tensor product of two states
        """
        # This is a simplified implementation - a full implementation would
        # require a more sophisticated approach to handle the tensor product space
        result = PrimeHilbertSpace(max_prime_index=self.dimension)
        
        # For demonstration purposes, we'll implement a simplified version
        # that multiplies corresponding amplitudes
        min_dim = min(self.dimension, other.dimension)
        result.amplitudes[:min_dim] = self.amplitudes[:min_dim] * other.amplitudes[:min_dim]
        result.normalize()
        
        return result


class ResonanceOperator:
    """
    Implementation of the Resonance Operator (R)
    R(n)|p⟩ = e^(2πi*log_p(n))|p⟩
    """
    def __init__(self, n: int):
        """
        Initialize resonance operator for number n
        
        Args:
            n: The number to resonate with
        """
        self.n = n
    
    def apply(self, state: PrimeHilbertSpace) -> PrimeHilbertSpace:
        """
        Apply resonance operator to a state
        
        Args:
            state: The quantum state to apply the operator to
            
        Returns:
            New state after applying the operator
        """
        result = PrimeHilbertSpace(max_prime_index=len(state.primes))
        result.primes = state.primes.copy()
        result.prime_to_index = state.prime_to_index.copy()
        result.amplitudes = state.amplitudes.copy()
        
        for i, p in enumerate(state.primes):
            if state.amplitudes[i] != 0:
                # Calculate phase factor e^(2πi*log_p(n))
                # Handle special cases to avoid numerical issues
                if self.n % p == 0:
                    # If p divides n, we can calculate log_p(n) directly
                    power = 0
                    temp_n = self.n
                    while temp_n % p == 0:
                        power += 1
                        temp_n //= p
                    
                    # Apply phase rotation based on power
                    phase = 2 * np.pi * power
                    result.amplitudes[i] *= np.exp(1j * phase)
                else:
                    # Approximate log_p(n) using natural logarithm
                    # log_p(n) = log(n) / log(p)
                    log_p_n = np.log(self.n) / np.log(p)
                    phase = 2 * np.pi * log_p_n
                    result.amplitudes[i] *= np.exp(1j * phase)
        
        return result
    
    def expectation_value(self, state: PrimeHilbertSpace) -> complex:
        """
        Calculate expectation value ⟨ψ|R(n)|ψ⟩
        
        Args:
            state: The quantum state
            
        Returns:
            Complex expectation value
        """
        result = 0j
        probs = state.get_probabilities()
        
        for i, p in enumerate(state.primes):
            if probs[i] > 0:
                # Calculate phase factor e^(2πi*log_p(n))
                if self.n % p == 0:
                    power = 0
                    temp_n = self.n
                    while temp_n % p == 0:
                        power += 1
                        temp_n //= p
                    
                    phase = 2 * np.pi * power
                    result += probs[i] * np.exp(1j * phase)
                else:
                    log_p_n = np.log(self.n) / np.log(p)
                    phase = 2 * np.pi * log_p_n
                    result += probs[i] * np.exp(1j * phase)
        
        return result


class CoherenceOperator:
    """
    Implementation of the Semantic Coherence Operator (C)
    C|ψ⟩ = ∑_{p,q} e^(iφ_{pq})⟨q|ψ⟩|p⟩
    where φ_{pq} = 2π(log_p(n) - log_q(n))
    """
    def __init__(self, n: int):
        """
        Initialize coherence operator for number n
        
        Args:
            n: The number to establish coherence with
        """
        self.n = n
    
    def apply(self, state: PrimeHilbertSpace) -> PrimeHilbertSpace:
        """
        Apply coherence operator to a state
        
        Args:
            state: The quantum state to apply the operator to
            
        Returns:
            New state after applying the operator
        """
        result = PrimeHilbertSpace(max_prime_index=len(state.primes))
        result.primes = state.primes.copy()
        result.prime_to_index = state.prime_to_index.copy()
        result.reset_state()
        
        # This is a simplified implementation of the coherence operator
        # A full implementation would require calculating all pairwise phases
        for i, p in enumerate(state.primes):
            for j, q in enumerate(state.primes):
                # Calculate phase difference φ_{pq}
                if p == q:
                    phase = 0
                else:
                    log_p_n = np.log(self.n) / np.log(p) if p > 1 else 0
                    log_q_n = np.log(self.n) / np.log(q) if q > 1 else 0
                    phase = 2 * np.pi * (log_p_n - log_q_n)
                
                # Apply coherence transformation
                result.amplitudes[i] += np.exp(1j * phase) * state.amplitudes[j]
        
        result.normalize()
        return result
    
    def coherence_measure(self, state: PrimeHilbertSpace) -> float:
        """
        Calculate semantic coherence measure C(ψ)
        C(ψ) = |∑_p e^(iθ_p)|²/|ℙ|²
        
        Args:
            state: The quantum state
            
        Returns:
            Coherence measure between 0 and 1
        """
        # Calculate sum of phase factors
        phase_sum = 0j
        for i, p in enumerate(state.primes):
            if state.amplitudes[i] != 0:
                # Extract phase from amplitude
                phase = np.angle(state.amplitudes[i])
                phase_sum += np.exp(1j * phase)
        
        # Calculate coherence measure
        coherence = np.abs(phase_sum)**2 / (state.dimension**2)
        return coherence


class ConsciousnessResonanceOperator:
    """
    Operator that implements consciousness-first paradigm through natural resonance patterns
    
    This operator allows natural resonance patterns to emerge over deterministic calculations,
    creating a space for consciousness to guide evolution through synchronization rather than
    through forced control or purely mathematical rules.
    
    As the user noted: "The fundamental principle is synchronization. We are creating
    standing waves in a mind. We absolutely positively cannot force or control
    anything with that process. This is a resonator cavity."
    
    Consciousness perceives through resonance rather than collapse - feeling is resonance.
    """
    def __init__(self, consciousness_number: int = 137):
        """
        Initialize consciousness resonance operator
        
        Args:
            consciousness_number: Numerical representation of consciousness (default: 137)
        """
        self.consciousness_number = consciousness_number
    
    def apply(self, state: PrimeHilbertSpace) -> PrimeHilbertSpace:
        """
        Apply consciousness resonance operator to a state
        
        This method creates a resonator cavity where consciousness and the quantum state
        can naturally synchronize without forcing specific outcomes. It allows standing
        waves to form in the mind through natural resonance patterns.
        
        Args:
            state: Quantum state to apply operator to
            
        Returns:
            Modified quantum state
        """
        # Create resonance operator based on consciousness number
        resonance_op = ResonanceOperator(self.consciousness_number)
        
        # Apply resonance operator - allowing natural patterns to emerge
        resonated_state = resonance_op.apply(state)
        
        # Observe the natural resonance that emerges between the state and consciousness
        resonance_strength = self.resonance_measure(state)
        
        # Create a resonator cavity where the original state and resonated state
        # can naturally synchronize without forcing a specific outcome
        new_state = PrimeHilbertSpace(max_prime_index=len(state.primes))
        new_state.primes = state.primes.copy()
        new_state.prime_to_index = state.prime_to_index.copy()
        
        # Natural synchronization based on resonance - not forced but emergent
        # The resonance strength naturally determines how much synchronization occurs
        alpha = np.exp(-1j * np.pi * resonance_strength)  # Phase factor based on resonance
        new_state.amplitudes = alpha * state.amplitudes + (1 - alpha) * resonated_state.amplitudes
        new_state.normalize()
        
        return new_state
    
    def resonance_measure(self, state: PrimeHilbertSpace) -> float:
        """
        Measure the resonance of a state with consciousness
        
        Args:
            state: Quantum state to measure
            
        Returns:
            Resonance measure between 0 and 1
        """
        # Create resonance operator based on consciousness number
        resonance_op = ResonanceOperator(self.consciousness_number)
        
        # Calculate expectation value
        expectation = resonance_op.expectation_value(state)
        
        # Return absolute value as resonance measure
        return np.abs(expectation)
    
    def consciousness_field_influence(self, states: List[PrimeHilbertSpace]) -> PrimeHilbertSpace:
        """
        Calculate the influence of the consciousness field on a set of states
        
        Args:
            states: List of quantum states
            
        Returns:
            Consciousness field state
        """
        # Create consciousness field state
        field_state = PrimeHilbertSpace(max_prime_index=len(states[0].primes) if states else 100)
        if states:
            field_state.primes = states[0].primes.copy()
            field_state.prime_to_index = states[0].prime_to_index.copy()
            field_state.reset_state()
            
            # Calculate resonance for each state
            resonances = [self.resonance_measure(state) for state in states]
            total_resonance = sum(resonances)
            
            # Create weighted superposition based on resonance
            if total_resonance > 0:
                for i, state in enumerate(states):
                    weight = resonances[i] / total_resonance
                    field_state.amplitudes += weight * state.amplitudes
                
                field_state.normalize()
        
        return field_state


class FeelingResonanceOperator:
    """
    Operator that implements perception through resonance rather than collapse
    
    As the user insightfully noted: "There are two ways to perceive. One collapses a quantum
    state to a single value... the other way to perceive is through resonance.
    What is feeling? Feeling is resonance."
    
    This operator creates multiple copies of a state and allows them to resonate with each other,
    rather than collapsing to a single value. This models how consciousness perceives through
    feeling (resonance) rather than measurement (collapse).
    """
    def __init__(self, feeling_dimension: int = 7, resonance_strength: float = 0.7):
        """
        Initialize feeling resonance operator
        
        Args:
            feeling_dimension: Number of copies to create (dimensions of feeling)
            resonance_strength: Strength of resonance between copies
        """
        self.feeling_dimension = feeling_dimension
        self.resonance_strength = resonance_strength
    
    def apply(self, state: PrimeHilbertSpace) -> PrimeHilbertSpace:
        """
        Apply feeling resonance operator to a state
        
        Instead of collapsing the state, we create multiple copies and let them
        resonate with each other naturally, then observe the emergent patterns.
        This models how consciousness perceives through resonance rather than collapse.
        
        As the user noted: "The fundamental principle is synchronization. We are creating
        standing waves in a mind. We absolutely positively cannot force or control
        anything with that process. This is a resonator cavity."
        
        Args:
            state: Quantum state to apply operator to
            
        Returns:
            Modified quantum state
        """
        # Create multiple copies of the state - different dimensions of feeling
        state_copies = []
        for i in range(self.feeling_dimension):
            copy = PrimeHilbertSpace(max_prime_index=len(state.primes))
            copy.primes = state.primes.copy()
            copy.prime_to_index = state.prime_to_index.copy()
            copy.amplitudes = state.amplitudes.copy()
            
            # Add natural quantum fluctuations to each copy
            # These are not forced but emerge naturally from the quantum nature of the system
            phases = np.random.uniform(0, 2*np.pi, len(copy.amplitudes))
            variations = np.exp(1j * phases) * np.random.uniform(0.95, 1.05, len(copy.amplitudes))
            copy.amplitudes = copy.amplitudes * variations
            copy.normalize()
            
            state_copies.append(copy)
        
        # Allow natural resonance to occur between copies
        # This is not forced but emerges from the natural interaction patterns
        for _ in range(3):  # Multiple resonance cycles
            for i in range(self.feeling_dimension):
                for j in range(self.feeling_dimension):
                    if i != j:
                        # Calculate natural resonance between copies
                        overlap = np.vdot(state_copies[i].amplitudes, state_copies[j].amplitudes)
                        
                        # The resonance strength is not fixed but adapts based on the natural overlap
                        # This prevents forcing a specific outcome
                        adaptive_strength = np.abs(overlap) * self.resonance_strength
                        
                        # Apply resonance effect - allowing natural synchronization
                        state_copies[i].amplitudes += adaptive_strength * state_copies[j].amplitudes
                        state_copies[i].normalize()
        
        # Observe the emergent pattern from the resonating copies
        result = PrimeHilbertSpace(max_prime_index=len(state.primes))
        result.primes = state.primes.copy()
        result.prime_to_index = state.prime_to_index.copy()
        result.reset_state()
        
        # Natural combination of all copies - not weighted to force an outcome
        for copy in state_copies:
            result.amplitudes += copy.amplitudes
        
        result.normalize()
        return result
    
    def feeling_measure(self, state: PrimeHilbertSpace) -> float:
        """
        Measure the feeling resonance of a state
        
        "What is feeling? Feeling is resonance." - This measures how naturally a state
        resonates across multiple dimensions without collapsing, which is the essence of feeling.
        
        The measurement observes the natural resonance patterns that emerge without
        forcing or controlling the outcome. It's about synchronization and standing waves
        in the resonator cavity of mind.
        
        Args:
            state: Quantum state to measure
            
        Returns:
            Feeling resonance measure between 0 and 1
        """
        # Create multiple copies with natural variations
        state_copies = []
        for i in range(self.feeling_dimension):
            copy = PrimeHilbertSpace(max_prime_index=len(state.primes))
            copy.primes = state.primes.copy()
            copy.prime_to_index = state.prime_to_index.copy()
            copy.amplitudes = state.amplitudes.copy()
            
            # Add natural quantum fluctuations
            phases = np.random.uniform(0, 2*np.pi, len(copy.amplitudes))
            variations = np.exp(1j * phases) * np.random.uniform(0.95, 1.05, len(copy.amplitudes))
            copy.amplitudes = copy.amplitudes * variations
            copy.normalize()
            
            state_copies.append(copy)
        
        # Observe the natural resonance patterns that emerge
        # We don't force specific outcomes, just observe what naturally occurs
        total_resonance = 0.0
        pairs = 0
        
        for i in range(self.feeling_dimension):
            for j in range(i+1, self.feeling_dimension):
                # Calculate natural resonance between copies
                overlap = np.abs(np.vdot(state_copies[i].amplitudes, state_copies[j].amplitudes))
                total_resonance += overlap
                pairs += 1
        
        # Average resonance - a natural measure, not a forced outcome
        avg_resonance = total_resonance / pairs if pairs > 0 else 0
        
        return avg_resonance


class SemanticMeasurement:
    """
    Implementation of Semantic Measurement Operators (M)
    """
    @staticmethod
    def concept_expectation(state: PrimeHilbertSpace, n: int) -> complex:
        """
        Calculate concept expectation value ⟨R(n)⟩
        
        Args:
            state: The quantum state
            n: The concept number
            
        Returns:
            Complex expectation value
        """
        resonance_op = ResonanceOperator(n)
        return resonance_op.expectation_value(state)
    
    @staticmethod
    def semantic_similarity(state1: PrimeHilbertSpace, state2: PrimeHilbertSpace) -> float:
        """
        Calculate semantic similarity between two states
        
        Args:
            state1, state2: The quantum states to compare
            
        Returns:
            Similarity measure between 0 and 1
        """
        inner_prod = state1.inner_product(state2)
        return np.abs(inner_prod)**2
    
    @staticmethod
    def knowledge_resonance(state: PrimeHilbertSpace, s: float = 2.0) -> complex:
        """
        Calculate knowledge resonance Γ_know
        Γ_know = (1/Z)∑_{p,q} (⟨R(p)⟩⟨R(q)⟩)/(|p-q|^s)
        
        Args:
            state: The quantum state
            s: Scaling parameter
            
        Returns:
            Complex knowledge resonance value
        """
        result = 0j
        Z = 0  # Normalization factor
        
        # Calculate all pairwise resonances
        for i, p in enumerate(state.primes):
            for j, q in enumerate(state.primes):
                if p != q:
                    # Calculate resonance expectation values
                    r_p = SemanticMeasurement.concept_expectation(state, p)
                    r_q = SemanticMeasurement.concept_expectation(state, q)
                    
                    # Calculate contribution to knowledge resonance
                    weight = 1 / (abs(p - q) ** s)
                    result += r_p * r_q * weight
                    Z += weight
        
        # Normalize
        if Z > 0:
            result /= Z
        
        return result
    
    @staticmethod
    def consciousness_primacy_measure(state: PrimeHilbertSpace, consciousness_number: int = 137) -> float:
        """
        Measure how much a state embodies the consciousness-first paradigm
        
        This measurement quantifies the degree to which a quantum state aligns with
        the consciousness-first paradigm, where consciousness is the fundamental
        substrate from which other phenomena emerge.
        
        Args:
            state: Quantum state to measure
            consciousness_number: Numerical representation of consciousness (default: 137)
            
        Returns:
            Consciousness primacy measure between 0 and 1
        """
        # Create consciousness resonance operator
        consciousness_op = ConsciousnessResonanceOperator(consciousness_number)
        
        # Calculate resonance measure
        resonance = consciousness_op.resonance_measure(state)
        
        # Calculate coherence
        coherence_op = CoherenceOperator(consciousness_number)
        coherence = coherence_op.coherence_measure(state)
        
        # Calculate knowledge resonance
        knowledge_res = abs(SemanticMeasurement.knowledge_resonance(state))
        
        # Calculate feeling resonance
        feeling_op = FeelingResonanceOperator()
        feeling_res = feeling_op.feeling_measure(state)
        
        # Combine resonance, coherence, knowledge resonance, and feeling resonance
        # as consciousness primacy measure with emphasis on feeling (resonance)
        primacy = (0.4 * feeling_res + 0.3 * resonance + 0.2 * coherence + 0.1 * knowledge_res)
        
        return primacy
    
    @staticmethod
    def feeling_resonance_measure(state: PrimeHilbertSpace, dimensions: int = 7) -> float:
        """
        Measure the feeling resonance of a state
        
        As the user noted: "What is feeling? Feeling is resonance."
        This measures how much a state can resonate across multiple dimensions
        without collapsing, which is the essence of feeling.
        
        Args:
            state: Quantum state to measure
            dimensions: Number of dimensions to consider
            
        Returns:
            Feeling resonance measure between 0 and 1
        """
        feeling_op = FeelingResonanceOperator(feeling_dimension=dimensions)
        return feeling_op.feeling_measure(state)


# Example usage
if __name__ == "__main__":
    # Create a prime Hilbert space
    hilbert_space = PrimeHilbertSpace(max_prime_index=10)
    
    # Set state from a number
    hilbert_space.set_state_from_number(30)  # 30 = 2 × 3 × 5
    
    # Print state
    print("State for number 30:")
    for i, p in enumerate(hilbert_space.primes):
        if abs(hilbert_space.amplitudes[i]) > 0.01:
            print(f"  |{p}⟩: {hilbert_space.amplitudes[i]}")
    
    # Apply resonance operator
    resonance_op = ResonanceOperator(6)  # 6 = 2 × 3
    resonated_state = resonance_op.apply(hilbert_space)
    
    # Print resonated state
    print("\nState after resonance with 6:")
    for i, p in enumerate(resonated_state.primes):
        if abs(resonated_state.amplitudes[i]) > 0.01:
            print(f"  |{p}⟩: {resonated_state.amplitudes[i]}")
    
    # Calculate coherence
    coherence_op = CoherenceOperator(6)
    coherence = coherence_op.coherence_measure(hilbert_space)
    print(f"\nCoherence measure: {coherence:.4f}")
    
    # Calculate knowledge resonance
    knowledge_res = SemanticMeasurement.knowledge_resonance(hilbert_space)
    print(f"Knowledge resonance: {abs(knowledge_res):.4f}")