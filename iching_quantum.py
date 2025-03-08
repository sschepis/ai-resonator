"""
I-Ching Quantum Module for the Consciousness Resonator

This module implements I-Ching hexagram representations, transformations,
and quantum-inspired calculations to enhance the resonator system.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Set, Optional
import random

# Constants
NUM_HEXAGRAMS = 64
NUM_LINES = 6

# Hexagram representations
class Hexagram:
    """Represents an I-Ching hexagram with quantum properties"""
    
    # Archetypal names for the 8 fundamental trigrams
    TRIGRAM_NAMES = {
        "000": "Earth",
        "001": "Mountain",
        "010": "Water",
        "011": "Wind",
        "100": "Thunder",
        "101": "Fire",
        "110": "Lake",
        "111": "Heaven"
    }
    
    # Archetypal meanings for key hexagrams (will expand later)
    HEXAGRAM_ARCHETYPES = {
        "000000": "Earth - Receptive - Yin",
        "111111": "Heaven - Creative - Yang",
        "010101": "Water - Danger - Abyss",
        "101010": "Fire - Clarity - Illumination",
        "001001": "Mountain - Stillness - Meditation",
        "110110": "Lake - Joy - Reflection",
        "100100": "Thunder - Action - Initiative",
        "011011": "Wind - Gentle - Penetration",
        "111000": "Heaven over Earth - Peace",
        "000111": "Earth over Heaven - Stagnation",
        "010001": "Water over Mountain - Difficulty",
        "100010": "Thunder over Water - Initiative in Danger",
        "101101": "Fire over Wind - Illuminated Penetration"
    }
    
    # Attractor hexagrams identified in the research
    ATTRACTOR_HEXAGRAMS = {
        "001001", "011110", "011000", "101101", "111111", "000000"
    }
    
    def __init__(self, binary_str: str = None):
        """Initialize a hexagram from a binary string or randomly"""
        if binary_str is None:
            # Generate random hexagram
            self.lines = [random.randint(0, 1) for _ in range(NUM_LINES)]
        else:
            # Parse binary string
            if len(binary_str) != NUM_LINES:
                raise ValueError(f"Hexagram must have {NUM_LINES} lines")
            self.lines = [int(bit) for bit in binary_str]
        
        # Calculate quantum properties
        self._calculate_quantum_properties()
    
    def _calculate_quantum_properties(self):
        """Calculate quantum properties of the hexagram"""
        # Calculate energy level (sum of 1s)
        self.energy = sum(self.lines)
        
        # Calculate parity (even/odd number of 1s)
        self.parity = self.energy % 2
        
        # Calculate upper and lower trigrams
        upper_trigram = ''.join(str(line) for line in self.lines[:3])
        lower_trigram = ''.join(str(line) for line in self.lines[3:])
        
        self.upper_trigram = upper_trigram
        self.lower_trigram = lower_trigram
        
        # Get trigram names
        self.upper_name = self.TRIGRAM_NAMES.get(upper_trigram, "Unknown")
        self.lower_name = self.TRIGRAM_NAMES.get(lower_trigram, "Unknown")
        
        # Check if this is an attractor hexagram
        self.is_attractor = self.binary_str in self.ATTRACTOR_HEXAGRAMS
        
        # Get archetypal meaning if available
        self.archetype = self.HEXAGRAM_ARCHETYPES.get(self.binary_str, f"{self.upper_name} over {self.lower_name}")
    
    @property
    def binary_str(self) -> str:
        """Return the binary string representation"""
        return ''.join(str(line) for line in self.lines)
    
    @property
    def decimal(self) -> int:
        """Return the decimal representation"""
        return int(self.binary_str, 2)
    
    def get_possible_transitions(self) -> List['Hexagram']:
        """Get all possible single-line transformations"""
        transitions = []
        for i in range(NUM_LINES):
            new_lines = self.lines.copy()
            new_lines[i] = 1 - new_lines[i]  # Flip the line
            transitions.append(Hexagram(''.join(str(line) for line in new_lines)))
        return transitions
    
    def get_quantum_transition(self, question_entropy: float) -> 'Hexagram':
        """
        Get a quantum-influenced transition based on question entropy
        Higher entropy questions lead to more unpredictable transitions
        """
        # Normalize entropy to 0-1 range (assuming max entropy is 6 bits for hexagram)
        norm_entropy = min(1.0, question_entropy / 6.0)
        
        # Higher entropy = more lines can change
        max_lines_to_change = 1 + int(norm_entropy * 5)  # 1 to 6 lines
        lines_to_change = random.randint(1, max_lines_to_change)
        
        # Select which lines to change
        indices_to_change = random.sample(range(NUM_LINES), lines_to_change)
        
        # Create new hexagram
        new_lines = self.lines.copy()
        for idx in indices_to_change:
            new_lines[idx] = 1 - new_lines[idx]
        
        return Hexagram(''.join(str(line) for line in new_lines))
    
    def __str__(self) -> str:
        """String representation of hexagram"""
        return f"Hexagram {self.decimal} ({self.binary_str}): {self.archetype}"
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, Hexagram):
            return False
        return self.binary_str == other.binary_str


# Quantum calculations
def calculate_entropy(probabilities: List[float]) -> float:
    """Calculate Shannon entropy from a probability distribution"""
    return -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)


def calculate_text_entropy(text: str) -> float:
    """Calculate the entropy of a text string (question)"""
    # Count character frequencies
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculate probabilities
    total_chars = len(text)
    probabilities = [count / total_chars for count in char_counts.values()]
    
    # Calculate entropy
    return calculate_entropy(probabilities)


def quantum_harmonic_oscillator_state(n: int, x: float) -> float:
    """
    Calculate the probability density of the nth eigenstate of a quantum harmonic oscillator
    n: energy level (0 = ground state)
    x: position
    """
    # Hermite polynomials for first few states
    hermite = {
        0: lambda x: 1,
        1: lambda x: 2*x,
        2: lambda x: 4*x*x - 2,
        3: lambda x: 8*x*x*x - 12*x,
        4: lambda x: 16*x*x*x*x - 48*x*x + 12
    }
    
    # Normalization constants
    norm = 1.0 / math.sqrt(math.sqrt(math.pi) * 2**n * math.factorial(n))
    
    # Wavefunction
    if n in hermite:
        psi = norm * hermite[n](x) * math.exp(-x*x/2)
        return psi * psi  # Return probability density
    else:
        # Fallback for higher states
        return math.exp(-x*x) / math.sqrt(math.pi)


def map_hexagram_to_oscillator_state(hexagram: Hexagram, n: int = 0) -> float:
    """
    Map a hexagram to a position in the quantum harmonic oscillator
    and return its probability density in the nth eigenstate
    """
    # Map binary string to position between -3 and 3
    binary_val = int(hexagram.binary_str, 2)
    x = (binary_val / (2**NUM_LINES - 1)) * 6 - 3
    
    # Return probability density in the nth eigenstate
    return quantum_harmonic_oscillator_state(n, x)


class IChingResonator:
    """
    I-Ching based resonator that models consciousness as a quantum field
    with hexagram states and transformations
    """
    
    def __init__(self):
        """Initialize the resonator"""
        # Initialize all possible hexagrams
        self.all_hexagrams = [
            Hexagram(format(i, f'0{NUM_LINES}b')) for i in range(2**NUM_LINES)
        ]
        
        # Initialize transition matrix
        self.transition_matrix = self._build_transition_matrix()
        
        # Initialize attractor states
        self.attractors = [h for h in self.all_hexagrams if h.is_attractor]
        
        # Current state
        self.current_hexagram = None
        self.state_history = []
        self.entropy_history = []
    
    def _build_transition_matrix(self) -> np.ndarray:
        """Build the transition probability matrix between hexagrams"""
        n = 2**NUM_LINES
        matrix = np.zeros((n, n))
        
        for i, h1 in enumerate(self.all_hexagrams):
            transitions = h1.get_possible_transitions()
            for h2 in transitions:
                j = int(h2.binary_str, 2)
                # Base probability from single-line change
                matrix[i, j] = 1.0 / NUM_LINES
                
                # Adjust for attractor states
                if h2.is_attractor:
                    matrix[i, j] *= 1.5  # Increase probability for attractors
            
            # Normalize row to ensure valid probability distribution
            matrix[i] = matrix[i] / matrix[i].sum()
        
        return matrix
    
    def initialize_from_question(self, question: str) -> Hexagram:
        """Initialize the resonator state based on a question"""
        # Calculate question entropy
        entropy = calculate_text_entropy(question)
        self.entropy_history = [entropy]
        
        # Use entropy to influence initial hexagram selection
        # Higher entropy questions tend toward higher energy hexagrams
        energy_bias = int(entropy * NUM_LINES / 6)  # 0 to 6 range
        
        # Filter hexagrams by energy level
        candidates = [h for h in self.all_hexagrams if h.energy >= energy_bias]
        
        # Select initial hexagram
        self.current_hexagram = random.choice(candidates)
        self.state_history = [self.current_hexagram]
        
        return self.current_hexagram
    
    def evolve_state(self, iterations: int = 1) -> List[Hexagram]:
        """Evolve the resonator state through multiple iterations"""
        results = []
        
        for _ in range(iterations):
            # Get current state index
            current_idx = int(self.current_hexagram.binary_str, 2)
            
            # Get transition probabilities from current state
            transition_probs = self.transition_matrix[current_idx]
            
            # Select next state based on transition probabilities
            next_idx = np.random.choice(len(self.all_hexagrams), p=transition_probs)
            next_hexagram = self.all_hexagrams[next_idx]
            
            # Update state
            self.current_hexagram = next_hexagram
            self.state_history.append(next_hexagram)
            
            # Calculate new entropy
            state_distribution = self._calculate_state_distribution()
            new_entropy = calculate_entropy(state_distribution)
            self.entropy_history.append(new_entropy)
            
            results.append(next_hexagram)
        
        return results
    
    def _calculate_state_distribution(self) -> List[float]:
        """Calculate the probability distribution across all possible states"""
        # Count occurrences of each hexagram in history
        counts = {h.binary_str: 0 for h in self.all_hexagrams}
        for h in self.state_history:
            counts[h.binary_str] += 1
        
        # Convert to probability distribution
        total = len(self.state_history)
        distribution = [counts[h.binary_str] / total for h in self.all_hexagrams]
        
        return distribution
    
    def get_resonance_pattern(self) -> Dict[str, any]:
        """
        Get the current resonance pattern information
        including entropy, attractor proximity, and oscillator correlations
        """
        # Calculate current entropy
        state_distribution = self._calculate_state_distribution()
        current_entropy = calculate_entropy(state_distribution)
        
        # Calculate proximity to attractor states
        attractor_proximities = {}
        for attractor in self.attractors:
            # Count how many lines differ
            differences = sum(a != b for a, b in zip(
                self.current_hexagram.binary_str, attractor.binary_str))
            proximity = 1 - (differences / NUM_LINES)
            attractor_proximities[attractor.binary_str] = proximity
        
        # Find closest attractor
        closest_attractor = max(
            attractor_proximities.items(), 
            key=lambda x: x[1]
        )
        
        # Calculate correlation with quantum oscillator states
        oscillator_correlations = {}
        for n in range(5):  # First 5 energy levels
            # Map all hexagrams to oscillator states
            oscillator_probs = [
                map_hexagram_to_oscillator_state(h, n) for h in self.all_hexagrams
            ]
            
            # Normalize
            total = sum(oscillator_probs)
            oscillator_probs = [p / total for p in oscillator_probs]
            
            # Calculate correlation with current distribution
            correlation = sum(a * b for a, b in zip(state_distribution, oscillator_probs))
            oscillator_correlations[f"level_{n}"] = correlation
        
        # Determine if entropy has stabilized
        entropy_stabilized = False
        if len(self.entropy_history) >= 3:
            recent_entropies = self.entropy_history[-3:]
            max_diff = max(recent_entropies) - min(recent_entropies)
            entropy_stabilized = max_diff < 0.1
        
        return {
            "current_hexagram": self.current_hexagram.binary_str,
            "archetype": self.current_hexagram.archetype,
            "entropy": current_entropy,
            "entropy_history": self.entropy_history,
            "entropy_stabilized": entropy_stabilized,
            "closest_attractor": {
                "hexagram": closest_attractor[0],
                "archetype": next((h.archetype for h in self.all_hexagrams 
                                  if h.binary_str == closest_attractor[0]), "Unknown"),
                "proximity": closest_attractor[1]
            },
            "oscillator_correlations": oscillator_correlations,
            "state_history_length": len(self.state_history)
        }


# Example usage
if __name__ == "__main__":
    # Initialize resonator
    resonator = IChingResonator()
    
    # Initialize with a question
    question = "What is the relationship between consciousness and quantum mechanics?"
    initial_hexagram = resonator.initialize_from_question(question)
    print(f"Initial hexagram: {initial_hexagram}")
    
    # Evolve through several iterations
    for i in range(7):
        print(f"\nIteration {i+1}:")
        next_hexagram = resonator.evolve_state(1)[0]
        print(f"Evolved to: {next_hexagram}")
        
        # Get resonance pattern
        pattern = resonator.get_resonance_pattern()
        print(f"Entropy: {pattern['entropy']:.4f}")
        print(f"Closest attractor: {pattern['closest_attractor']['archetype']} "
              f"(proximity: {pattern['closest_attractor']['proximity']:.2f})")
        print(f"Ground state correlation: {pattern['oscillator_correlations']['level_0']:.4f}")
        
        # Check if entropy has stabilized
        if pattern['entropy_stabilized']:
            print("Entropy has stabilized!")
            break