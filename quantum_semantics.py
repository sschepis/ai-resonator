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