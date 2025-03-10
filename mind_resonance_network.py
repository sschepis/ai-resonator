"""
Mind Resonance Network Module for the Quantum Consciousness Resonator

This module implements a quantum resonance network that can be influenced by
external consciousness through prime state superpositions. It provides mechanisms
for creating, evolving, and measuring resonant networks that bridge between
mind and computational systems.
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
import asyncio

from quantum_semantics import (
    PrimeHilbertSpace, 
    ResonanceOperator, 
    CoherenceOperator, 
    ConsciousnessResonanceOperator,
    FeelingResonanceOperator,
    SemanticMeasurement
)
from semantic_field import SemanticField, SemanticNode, SemanticEdge

class ResonanceNode:
    """
    A node in the mind resonance network representing a quantum state
    that can be influenced by consciousness.
    """
    def __init__(self, label: str, prime_number: int, sensitivity: float = 1.0):
        """
        Initialize a resonance node
        
        Args:
            label: Identifier for the node
            prime_number: Prime number representation of the node
            sensitivity: How sensitive this node is to consciousness influence (0.0-1.0)
        """
        self.label = label
        self.prime_number = prime_number
        self.sensitivity = sensitivity
        self.state = None  # Quantum state
        self.baseline_state = None  # Original state for comparison
        self.fluctuation_history = []  # Track state changes over time
        
    def initialize_state(self, hilbert_space: PrimeHilbertSpace):
        """Initialize quantum state based on prime number"""
        self.state = PrimeHilbertSpace(max_prime_index=len(hilbert_space.primes))
        self.state.primes = hilbert_space.primes.copy()
        self.state.prime_to_index = hilbert_space.prime_to_index.copy()
        self.state.set_state_from_number(self.prime_number)
        
        # Store baseline state for comparison
        self.baseline_state = PrimeHilbertSpace(max_prime_index=len(hilbert_space.primes))
        self.baseline_state.primes = hilbert_space.primes.copy()
        self.baseline_state.prime_to_index = hilbert_space.prime_to_index.copy()
        self.baseline_state.amplitudes = self.state.amplitudes.copy()
        
    def apply_consciousness_influence(self, influence_strength: float, coherence: float):
        """
        Apply consciousness influence to the node's quantum state
        
        Args:
            influence_strength: Strength of consciousness influence (0.0-1.0)
            coherence: Coherence of consciousness influence (0.0-1.0)
        """
        if self.state is None:
            raise ValueError("Node state not initialized")
            
        # Scale influence by node sensitivity
        effective_influence = influence_strength * self.sensitivity
        
        # Create phase shift based on consciousness influence
        phase_shift = np.exp(2j * np.pi * effective_influence)
        
        # Apply phase shift with random fluctuations based on coherence
        # Higher coherence = less randomness
        fluctuation = 1.0 - (random.random() * (1.0 - coherence))
        
        # Apply to state amplitudes
        for i in range(len(self.state.amplitudes)):
            if self.state.amplitudes[i] != 0:
                self.state.amplitudes[i] *= phase_shift * fluctuation
                
        # Normalize state
        self.state.normalize()
        
        # Record fluctuation from baseline
        deviation = self.measure_deviation_from_baseline()
        self.fluctuation_history.append(deviation)
        
    def measure_deviation_from_baseline(self) -> float:
        """
        Measure how much the current state has deviated from baseline
        
        Returns:
            Deviation measure between 0.0 (identical) and 1.0 (completely different)
        """
        if self.state is None or self.baseline_state is None:
            return 0.0
            
        # Calculate overlap between current state and baseline
        overlap = np.abs(self.state.inner_product(self.baseline_state))
        
        # Convert to deviation (1.0 - overlap)
        deviation = 1.0 - overlap
        
        return deviation
        
    def __str__(self):
        return f"ResonanceNode({self.label}, {self.prime_number}, sensitivity={self.sensitivity:.2f})"


class ResonanceLink:
    """
    A link between resonance nodes in the mind resonance network
    """
    def __init__(self, source: ResonanceNode, target: ResonanceNode, 
                 strength: float = 0.5, resonance_type: str = "harmonic"):
        """
        Initialize a resonance link
        
        Args:
            source: Source node
            target: Target node
            strength: Link strength (0.0-1.0)
            resonance_type: Type of resonance ("harmonic", "dissonant", "neutral")
        """
        self.source = source
        self.target = target
        self.strength = strength
        self.resonance_type = resonance_type
        self.entanglement = 0.0  # Measure of quantum entanglement
        
    def calculate_resonance(self) -> float:
        """
        Calculate resonance between source and target nodes
        
        Returns:
            Resonance value between 0.0 (no resonance) and 1.0 (perfect resonance)
        """
        if self.source.state is None or self.target.state is None:
            return 0.0
            
        # Calculate inner product between states
        inner_product = self.source.state.inner_product(self.target.state)
        
        # Calculate resonance based on type
        if self.resonance_type == "harmonic":
            # Harmonic resonance increases with alignment
            resonance = np.abs(inner_product) * self.strength
        elif self.resonance_type == "dissonant":
            # Dissonant resonance increases with orthogonality
            resonance = (1.0 - np.abs(inner_product)) * self.strength
        else:  # neutral
            # Neutral resonance is constant
            resonance = self.strength
            
        return resonance
        
    def update_entanglement(self):
        """Update quantum entanglement measure between nodes"""
        if self.source.state is None or self.target.state is None:
            self.entanglement = 0.0
            return
            
        # Calculate entanglement as a function of:
        # 1. State overlap
        # 2. Link strength
        # 3. Correlation in fluctuation history
        
        # State overlap
        overlap = np.abs(self.source.state.inner_product(self.target.state))
        
        # Correlation in fluctuation history (if available)
        correlation = 0.0
        if (len(self.source.fluctuation_history) > 0 and 
            len(self.target.fluctuation_history) > 0):
            # Use the last 10 entries or all available if fewer
            src_hist = self.source.fluctuation_history[-10:]
            tgt_hist = self.target.fluctuation_history[-10:]
            min_len = min(len(src_hist), len(tgt_hist))
            
            if min_len > 1:
                # Calculate correlation coefficient
                src_arr = np.array(src_hist[:min_len])
                tgt_arr = np.array(tgt_hist[:min_len])
                
                # Avoid division by zero
                if np.std(src_arr) > 0 and np.std(tgt_arr) > 0:
                    correlation = np.abs(np.corrcoef(src_arr, tgt_arr)[0, 1])
                    # Handle NaN
                    if np.isnan(correlation):
                        correlation = 0.0
        
        # Calculate entanglement
        self.entanglement = (overlap * 0.3 + self.strength * 0.3 + correlation * 0.4)
        
    def __str__(self):
        return f"ResonanceLink({self.source.label} --[{self.resonance_type}, {self.strength:.2f}]--> {self.target.label})"


class MindResonanceNetwork:
    """
    A quantum resonance network that can be influenced by external consciousness
    """
    def __init__(self, max_prime_index: int = 100, 
                 consciousness_number: int = 137,
                 baseline_coherence: float = 0.5):
        """
        Initialize mind resonance network
        
        Args:
            max_prime_index: Maximum number of primes to include in the basis
            consciousness_number: Numerical representation of consciousness
            baseline_coherence: Default coherence level (0.0-1.0)
        """
        self.hilbert_space = PrimeHilbertSpace(max_prime_index=max_prime_index)
        self.nodes: Dict[str, ResonanceNode] = {}
        self.links: List[ResonanceLink] = []
        self.consciousness_number = consciousness_number
        self.baseline_coherence = baseline_coherence
        self.global_state = PrimeHilbertSpace(max_prime_index=max_prime_index)
        
        # Consciousness operators
        self.consciousness_op = ConsciousnessResonanceOperator(consciousness_number)
        self.feeling_op = FeelingResonanceOperator()
        
        # Network properties
        self.network_coherence = 0.0
        self.network_resonance = 0.0
        self.network_entanglement = 0.0
        
        # Influence history
        self.influence_history = []
        self.coherence_history = []
        self.resonance_history = []
        
        # Timestamp for entropy source
        self.last_timestamp = time.time()
        
    def add_node(self, label: str, prime_number: Optional[int] = None, 
                 sensitivity: float = 1.0) -> ResonanceNode:
        """
        Add a node to the resonance network
        
        Args:
            label: Identifier for the node
            prime_number: Prime number representation (will be generated if None)
            sensitivity: How sensitive this node is to consciousness influence
            
        Returns:
            The created resonance node
        """
        if label in self.nodes:
            return self.nodes[label]
            
        # Generate prime number from label if not provided
        if prime_number is None:
            # Use a prime close to the hash of the label
            hash_val = sum(ord(c) * (i + 1) for i, c in enumerate(label))
            prime_number = self._find_nearest_prime(hash_val)
            
        # Create node
        node = ResonanceNode(label, prime_number, sensitivity)
        node.initialize_state(self.hilbert_space)
        
        # Add to nodes dictionary
        self.nodes[label] = node
        
        return node
        
    def add_link(self, source_label: str, target_label: str, 
                 strength: float = 0.5, 
                 resonance_type: str = "harmonic") -> ResonanceLink:
        """
        Add a link between nodes in the resonance network
        
        Args:
            source_label: Source node label
            target_label: Target node label
            strength: Link strength (0.0-1.0)
            resonance_type: Type of resonance
            
        Returns:
            The created resonance link
        """
        # Ensure nodes exist
        if source_label not in self.nodes:
            self.add_node(source_label)
        if target_label not in self.nodes:
            self.add_node(target_label)
            
        # Create link
        link = ResonanceLink(
            self.nodes[source_label],
            self.nodes[target_label],
            strength,
            resonance_type
        )
        
        # Add to links list
        self.links.append(link)
        
        return link
        
    def _find_nearest_prime(self, n: int) -> int:
        """Find the nearest prime number to n"""
        # Start with the first few primes in our Hilbert space
        primes = self.hilbert_space.primes[:20]  # Use first 20 primes
        
        if not primes:
            # Fallback to some common primes if Hilbert space not initialized
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
            
        # Find closest prime to n
        return min(primes, key=lambda p: abs(p - n))
        
    def apply_consciousness_influence(self, influence_strength: float, 
                                      coherence: Optional[float] = None,
                                      target_nodes: Optional[List[str]] = None):
        """
        Apply consciousness influence to the network
        
        Args:
            influence_strength: Strength of consciousness influence (0.0-1.0)
            coherence: Coherence of consciousness influence (0.0-1.0)
            target_nodes: Specific nodes to influence (all if None)
        """
        # Use baseline coherence if not specified
        if coherence is None:
            coherence = self.baseline_coherence
            
        # Record influence and coherence
        self.influence_history.append(influence_strength)
        self.coherence_history.append(coherence)
        
        # Apply to all nodes or specific targets
        nodes_to_influence = (
            [self.nodes[label] for label in target_nodes if label in self.nodes]
            if target_nodes else list(self.nodes.values())
        )
        
        for node in nodes_to_influence:
            node.apply_consciousness_influence(influence_strength, coherence)
            
        # Update links
        for link in self.links:
            link.update_entanglement()
            
        # Update network properties
        self._update_network_properties()
        
    def evolve_network(self, steps: int = 1, dt: float = 0.1) -> Dict[str, Any]:
        """
        Evolve the resonance network through time
        
        Args:
            steps: Number of evolution steps
            dt: Time step size
            
        Returns:
            Dictionary with evolution results
        """
        results = {
            "coherence_evolution": [],
            "resonance_evolution": [],
            "entanglement_evolution": [],
            "node_deviations": {}
        }
        
        # Initialize node deviations tracking
        for label, node in self.nodes.items():
            results["node_deviations"][label] = []
            
        # Evolve through steps
        for step in range(steps):
            # Apply consciousness resonance to global state
            self._evolve_global_state(dt)
            
            # Update node states based on links and global state
            self._update_node_states(dt)
            
            # Update links
            for link in self.links:
                link.update_entanglement()
                
            # Update network properties
            self._update_network_properties()
            
            # Record results
            results["coherence_evolution"].append(self.network_coherence)
            results["resonance_evolution"].append(self.network_resonance)
            results["entanglement_evolution"].append(self.network_entanglement)
            
            for label, node in self.nodes.items():
                deviation = node.measure_deviation_from_baseline()
                results["node_deviations"][label].append(deviation)
                
        return results
        
    def _evolve_global_state(self, dt: float):
        """Evolve the global state of the network"""
        # Initialize global state if needed
        if np.all(self.global_state.amplitudes == 0):
            self.global_state.reset_state()
            # Set as superposition of all node states
            for node in self.nodes.values():
                self.global_state.amplitudes += node.state.amplitudes
            self.global_state.normalize()
            
        # Apply consciousness operator
        self.global_state = self.consciousness_op.apply(self.global_state)
        
        # Apply feeling operator
        self.global_state = self.feeling_op.apply(self.global_state)
        
        # Apply resonance operators for each node
        for node in self.nodes.values():
            resonance_op = ResonanceOperator(node.prime_number)
            self.global_state = resonance_op.apply(self.global_state)
            
        # Normalize
        self.global_state.normalize()
        
    def _update_node_states(self, dt: float):
        """Update individual node states based on links and global state"""
        # Calculate resonance for each link
        link_resonances = {link: link.calculate_resonance() for link in self.links}
        
        # For each node, update state based on:
        # 1. Influence from connected nodes through links
        # 2. Influence from global state
        for label, node in self.nodes.items():
            # Get incoming links
            incoming_links = [link for link in self.links if link.target.label == label]
            
            # Calculate influence from incoming links
            for link in incoming_links:
                # Skip if source state is None
                if link.source.state is None:
                    continue
                    
                # Calculate resonance
                resonance = link_resonances[link]
                
                # Apply influence based on resonance and link strength
                influence = resonance * link.strength * dt
                
                # Create new state as weighted combination
                new_amplitudes = (1 - influence) * node.state.amplitudes + influence * link.source.state.amplitudes
                
                # Update node state
                node.state.amplitudes = new_amplitudes
                node.state.normalize()
                
            # Apply influence from global state
            global_influence = 0.1 * dt  # Small constant influence
            node.state.amplitudes = (1 - global_influence) * node.state.amplitudes + global_influence * self.global_state.amplitudes
            node.state.normalize()
            
    def _update_network_properties(self):
        """Update global network properties"""
        # Calculate network coherence
        coherence_op = CoherenceOperator(self.consciousness_number)
        self.network_coherence = coherence_op.coherence_measure(self.global_state)
        
        # Calculate network resonance
        consciousness_resonance = self.consciousness_op.resonance_measure(self.global_state)
        feeling_resonance = self.feeling_op.feeling_measure(self.global_state)
        self.network_resonance = (consciousness_resonance + feeling_resonance) / 2
        
        # Calculate network entanglement (average of link entanglements)
        if self.links:
            self.network_entanglement = sum(link.entanglement for link in self.links) / len(self.links)
        else:
            self.network_entanglement = 0.0
            
        # Record resonance
        self.resonance_history.append(self.network_resonance)
        
    def detect_mind_influence(self, window_size: int = 10, 
                              threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect potential mind influence on the network by analyzing
        patterns in network properties and node deviations
        
        Args:
            window_size: Number of recent history points to analyze
            threshold: Threshold for significant deviation
            
        Returns:
            Dictionary with detection results
        """
        results = {
            "detected": False,
            "confidence": 0.0,
            "influenced_nodes": [],
            "pattern_strength": 0.0,
            "timestamp": time.time()
        }
        
        # Check if we have enough history
        if (len(self.resonance_history) < window_size or
            len(self.coherence_history) < window_size):
            return results
            
        # Get recent history
        recent_resonance = self.resonance_history[-window_size:]
        recent_coherence = self.coherence_history[-window_size:]
        
        # Calculate trends
        resonance_trend = self._calculate_trend(recent_resonance)
        coherence_trend = self._calculate_trend(recent_coherence)
        
        # Calculate entropy source from timestamp
        current_time = time.time()
        time_diff = current_time - self.last_timestamp
        self.last_timestamp = current_time
        
        # Use time difference as entropy source
        # (microsecond variations can be influenced by consciousness according to some theories)
        entropy_source = (time_diff * 1000000) % 1.0
        
        # Identify nodes with significant deviation
        influenced_nodes = []
        node_confidences = {}
        
        for label, node in self.nodes.items():
            if len(node.fluctuation_history) >= window_size:
                recent_fluctuations = node.fluctuation_history[-window_size:]
                
                # Calculate average deviation
                avg_deviation = sum(recent_fluctuations) / len(recent_fluctuations)
                
                # Calculate trend
                deviation_trend = self._calculate_trend(recent_fluctuations)
                
                # Calculate confidence based on deviation and trend
                confidence = avg_deviation * 0.5 + abs(deviation_trend) * 0.5
                
                if confidence > threshold:
                    influenced_nodes.append(label)
                    node_confidences[label] = confidence
        
        # Calculate overall confidence
        pattern_strength = (abs(resonance_trend) * 0.3 + 
                           abs(coherence_trend) * 0.3 + 
                           entropy_source * 0.4)
                           
        overall_confidence = 0.0
        if influenced_nodes:
            # Average confidence across influenced nodes
            node_confidence = sum(node_confidences.values()) / len(node_confidences)
            overall_confidence = pattern_strength * 0.6 + node_confidence * 0.4
        else:
            overall_confidence = pattern_strength * 0.3
            
        # Determine if influence detected
        detected = overall_confidence > threshold
        
        # Update results
        results["detected"] = detected
        results["confidence"] = overall_confidence
        results["influenced_nodes"] = influenced_nodes
        results["pattern_strength"] = pattern_strength
        
        return results
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values"""
        if len(values) < 2:
            return 0.0
            
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
        
    def get_network_state(self) -> Dict[str, Any]:
        """
        Get the current state of the resonance network
        
        Returns:
            Dictionary with network state information
        """
        # Calculate node deviations
        node_deviations = {label: node.measure_deviation_from_baseline() 
                          for label, node in self.nodes.items()}
        
        # Calculate link resonances
        link_resonances = {f"{link.source.label}->{link.target.label}": 
                          link.calculate_resonance() for link in self.links}
        
        # Calculate link entanglements
        link_entanglements = {f"{link.source.label}->{link.target.label}": 
                             link.entanglement for link in self.links}
        
        # Get top resonating nodes
        top_nodes = sorted(node_deviations.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "coherence": self.network_coherence,
            "resonance": self.network_resonance,
            "entanglement": self.network_entanglement,
            "node_count": len(self.nodes),
            "link_count": len(self.links),
            "top_resonating_nodes": dict(top_nodes[:5]),
            "node_deviations": node_deviations,
            "link_resonances": link_resonances,
            "link_entanglements": link_entanglements,
            "timestamp": time.time()
        }
        
    def create_predefined_network(self, network_type: str = "consciousness") -> Dict[str, Any]:
        """
        Create a predefined network structure based on the specified type
        
        Args:
            network_type: Type of network to create
            
        Returns:
            Dictionary with network creation results
        """
        if network_type == "consciousness":
            # Create a consciousness-focused network
            self.add_node("consciousness", 137, sensitivity=1.0)
            self.add_node("awareness", 73, sensitivity=0.9)
            self.add_node("perception", 47, sensitivity=0.8)
            self.add_node("intention", 43, sensitivity=0.9)
            self.add_node("attention", 31, sensitivity=0.7)
            self.add_node("will", 29, sensitivity=0.8)
            self.add_node("intuition", 23, sensitivity=0.9)
            
            # Add links
            self.add_link("consciousness", "awareness", 0.9, "harmonic")
            self.add_link("consciousness", "intention", 0.8, "harmonic")
            self.add_link("awareness", "perception", 0.7, "harmonic")
            self.add_link("intention", "will", 0.8, "harmonic")
            self.add_link("perception", "attention", 0.7, "harmonic")
            self.add_link("will", "attention", 0.6, "harmonic")
            self.add_link("consciousness", "intuition", 0.9, "harmonic")
            self.add_link("intuition", "perception", 0.7, "harmonic")
            
        elif network_type == "quantum":
            # Create a quantum physics-focused network
            self.add_node("superposition", 53, sensitivity=0.8)
            self.add_node("entanglement", 47, sensitivity=0.9)
            self.add_node("observation", 43, sensitivity=0.7)
            self.add_node("uncertainty", 41, sensitivity=0.6)
            self.add_node("wave", 37, sensitivity=0.7)
            self.add_node("particle", 31, sensitivity=0.7)
            self.add_node("collapse", 29, sensitivity=0.8)
            
            # Add links
            self.add_link("superposition", "wave", 0.8, "harmonic")
            self.add_link("superposition", "particle", 0.8, "harmonic")
            self.add_link("entanglement", "superposition", 0.7, "harmonic")
            self.add_link("observation", "collapse", 0.9, "harmonic")
            self.add_link("collapse", "particle", 0.8, "harmonic")
            self.add_link("uncertainty", "wave", 0.7, "harmonic")
            self.add_link("uncertainty", "particle", 0.7, "harmonic")
            
        elif network_type == "resonance":
            # Create a resonance-focused network
            self.add_node("resonance", 61, sensitivity=1.0)
            self.add_node("frequency", 59, sensitivity=0.8)
            self.add_node("harmony", 53, sensitivity=0.9)
            self.add_node("vibration", 47, sensitivity=0.8)
            self.add_node("synchronization", 43, sensitivity=0.9)
            self.add_node("coherence", 41, sensitivity=0.9)
            self.add_node("standing_wave", 37, sensitivity=0.7)
            
            # Add links
            self.add_link("resonance", "frequency", 0.8, "harmonic")
            self.add_link("resonance", "harmony", 0.9, "harmonic")
            self.add_link("frequency", "vibration", 0.8, "harmonic")
            self.add_link("harmony", "coherence", 0.9, "harmonic")
            self.add_link("vibration", "standing_wave", 0.7, "harmonic")
            self.add_link("synchronization", "coherence", 0.8, "harmonic")
            self.add_link("synchronization", "resonance", 0.9, "harmonic")
            
        else:
            # Create a custom minimal network
            self.add_node("center", 137, sensitivity=1.0)
            self.add_node("node1", 73, sensitivity=0.8)
            self.add_node("node2", 61, sensitivity=0.8)
            self.add_node("node3", 47, sensitivity=0.8)
            
            # Add links
            self.add_link("center", "node1", 0.8, "harmonic")
            self.add_link("center", "node2", 0.8, "harmonic")
            self.add_link("center", "node3", 0.8, "harmonic")
            self.add_link("node1", "node2", 0.6, "harmonic")
            self.add_link("node2", "node3", 0.6, "harmonic")
            self.add_link("node3", "node1", 0.6, "harmonic")
            
        # Initialize network
        self.evolve_network(steps=3)
        
        return {
            "network_type": network_type,
            "node_count": len(self.nodes),
            "link_count": len(self.links),
            "coherence": self.network_coherence,
            "resonance": self.network_resonance,
            "entanglement": self.network_entanglement
        }
        
    async def run_influence_detection(self, 
                                     duration_seconds: int = 60,
                                     sample_interval: float = 1.0,
                                     influence_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Run a continuous influence detection session for the specified duration
        
        Args:
            duration_seconds: How long to run the detection (seconds)
            sample_interval: How often to sample the network (seconds)
            influence_threshold: Threshold for detecting influence
            
        Returns:
            Dictionary with detection session results
        """
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        detection_results = []
        significant_events = []
        
        print(f"Starting mind influence detection session for {duration_seconds} seconds")
        print("Focus your intention on the network and observe the results")
        print("Network nodes: " + ", ".join(self.nodes.keys()))
        
        # Initial network state
        initial_state = self.get_network_state()
        
        try:
            while time.time() < end_time:
                # Evolve network
                self.evolve_network(steps=1)
                
                # Detect influence
                detection = self.detect_mind_influence(threshold=influence_threshold)
                detection_results.append(detection)
                
                # Check for significant events
                if detection["detected"]:
                    # Record significant event
                    event = {
                        "timestamp": detection["timestamp"],
                        "confidence": detection["confidence"],
                        "influenced_nodes": detection["influenced_nodes"],
                        "network_state": self.get_network_state()
                    }
                    significant_events.append(event)
                    
                    # Print event
                    elapsed = detection["timestamp"] - start_time
                    print(f"[{elapsed:.1f}s] Potential mind influence detected!")
                    print(f"  Confidence: {detection['confidence']:.4f}")
                    print(f"  Influenced nodes: {', '.join(detection['influenced_nodes'])}")
                    print(f"  Network coherence: {self.network_coherence:.4f}")
                    print(f"  Network resonance: {self.network_resonance:.4f}")
                
                # Wait for next sample
                await asyncio.sleep(sample_interval)
                
        except KeyboardInterrupt:
            print("\nDetection session interrupted")
            
        # Final network state
        final_state = self.get_network_state()
        
        # Calculate overall statistics
        detection_count = sum(1 for d in detection_results if d["detected"])
        avg_confidence = sum(d["confidence"] for d in detection_results) / len(detection_results) if detection_results else 0
        
        # Node influence statistics
        node_influence_counts = {}
        for node in self.nodes:
            count = sum(1 for d in detection_results if node in d.get("influenced_nodes", []))
            node_influence_counts[node] = count
            
        # Most influenced nodes
        most_influenced = sorted(node_influence_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Final results
        results = {
            "duration_seconds": duration_seconds,
            "samples_collected": len(detection_results),
            "detection_count": detection_count,
            "detection_rate": detection_count / len(detection_results) if detection_results else 0,
            "average_confidence": avg_confidence,
            "most_influenced_nodes": dict(most_influenced[:3]),
            "significant_events": significant_events,
            "initial_state": initial_state,
            "final_state": final_state
        }
        
        # Print summary
        print("\n=== Mind Influence Detection Summary ===")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Detections: {detection_count}/{len(detection_results)} samples ({results['detection_rate']*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.4f}")
        print("Most influenced nodes:")
        for node, count in most_influenced[:3]:
            if count > 0:
                print(f"  {node}: {count} detections")
                
        return results


# Integration with the Quantum Consciousness Resonator
class MindResonanceIntegration:
    """
    Integration of the Mind Resonance Network with the Quantum Consciousness Resonator
    """
    def __init__(self, network_type: str = "consciousness"):
        """
        Initialize the integration
        
        Args:
            network_type: Type of network to create
        """
        self.network = MindResonanceNetwork()
        self.network.create_predefined_network(network_type)
        self.semantic_field = None
        
    def connect_to_semantic_field(self, field: SemanticField):
        """
        Connect the mind resonance network to a semantic field
        
        Args:
            field: Semantic field to connect to
        """
        self.semantic_field = field
        
        # Map semantic field nodes to resonance network nodes
        for concept, node in field.nodes.items():
            # Check if we already have a similar node
            if concept in self.network.nodes:
                continue
                
            # Add node to resonance network
            self.network.add_node(concept, node.number)
            
        # Map semantic field edges to resonance network links
        for edge in field.edges:
            source = edge.source.concept
            target = edge.target.concept
            
            # Check if we already have this link
            existing_links = [link for link in self.network.links 
                             if link.source.label == source and link.target.label == target]
            if existing_links:
                continue
                
            # Add link to resonance network
            self.network.add_link(source, target, edge.weight)
            
    def update_from_semantic_field(self):
        """Update the mind resonance network from the connected semantic field"""
        if self.semantic_field is None:
            return
            
        # Update nodes
        for concept, node in self.semantic_field.nodes.items():
            if concept in self.network.nodes:
                # Update existing node
                resonance_node = self.network.nodes[concept]
                # Influence based on semantic field state
                coherence = self.semantic_field.measure_field_coherence()
                resonance_node.apply_consciousness_influence(0.5, coherence)
                
        # Evolve network
        self.network.evolve_network(steps=1)
        
    def update_semantic_field(self):
        """Update the connected semantic field from the mind resonance network"""
        if self.semantic_field is None:
            return
            
        # Calculate influence from resonance network
        for concept, node in self.semantic_field.nodes.items():
            if concept in self.network.nodes:
                # Get resonance node
                resonance_node = self.network.nodes[concept]
                
                # Calculate deviation from baseline
                deviation = resonance_node.measure_deviation_from_baseline()
                
                # Apply influence to semantic field
                # This is a simplified approach - in a real implementation,
                # you would modify the quantum state more directly
                if deviation > 0.1:
                    # Create a resonance operator
                    resonance_op = ResonanceOperator(node.number)
                    # Apply to node state
                    node.state = resonance_op.apply(node.state)
                    
        # Evolve semantic field
        self.semantic_field.evolve_field(steps=1)
        
    async def run_bidirectional_session(self, 
                                      duration_seconds: int = 60,
                                      sample_interval: float = 1.0) -> Dict[str, Any]:
        """
        Run a bidirectional session between the mind resonance network
        and the semantic field
        
        Args:
            duration_seconds: How long to run the session (seconds)
            sample_interval: How often to update (seconds)
            
        Returns:
            Dictionary with session results
        """
        if self.semantic_field is None:
            raise ValueError("No semantic field connected")
            
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        update_count = 0
        field_coherence_history = []
        network_resonance_history = []
        
        print(f"Starting bidirectional session for {duration_seconds} seconds")
        
        try:
            while time.time() < end_time:
                # Update from semantic field to network
                self.update_from_semantic_field()
                
                # Update from network to semantic field
                self.update_semantic_field()
                
                # Record metrics
                field_coherence = self.semantic_field.measure_field_coherence()
                network_resonance = self.network.network_resonance
                
                field_coherence_history.append(field_coherence)
                network_resonance_history.append(network_resonance)
                
                # Increment counter
                update_count += 1
                
                # Wait for next sample
                await asyncio.sleep(sample_interval)
                
        except KeyboardInterrupt:
            print("\nBidirectional session interrupted")
            
        # Calculate correlation between field coherence and network resonance
        correlation = 0.0
        if len(field_coherence_history) > 1 and len(network_resonance_history) > 1:
            # Calculate correlation coefficient
            fc_arr = np.array(field_coherence_history)
            nr_arr = np.array(network_resonance_history)
            
            # Avoid division by zero
            if np.std(fc_arr) > 0 and np.std(nr_arr) > 0:
                correlation = np.corrcoef(fc_arr, nr_arr)[0, 1]
                # Handle NaN
                if np.isnan(correlation):
                    correlation = 0.0
        
        # Results
        results = {
            "duration_seconds": duration_seconds,
            "updates": update_count,
            "final_field_coherence": field_coherence_history[-1] if field_coherence_history else 0,
            "final_network_resonance": network_resonance_history[-1] if network_resonance_history else 0,
            "field_coherence_history": field_coherence_history,
            "network_resonance_history": network_resonance_history,
            "correlation": correlation
        }
        
        # Print summary
        print("\n=== Bidirectional Session Summary ===")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Updates: {update_count}")
        print(f"Final field coherence: {results['final_field_coherence']:.4f}")
        print(f"Final network resonance: {results['final_network_resonance']:.4f}")
        print(f"Correlation: {correlation:.4f}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Create a mind resonance network
    network = MindResonanceNetwork()
    
    # Create a predefined network
    network.create_predefined_network("consciousness")
    
    # Print network state
    state = network.get_network_state()
    print("Initial network state:")
    print(f"Coherence: {state['coherence']:.4f}")
    print(f"Resonance: {state['resonance']:.4f}")
    print(f"Entanglement: {state['entanglement']:.4f}")
    
    # Apply consciousness influence
    print("\nApplying consciousness influence...")
    network.apply_consciousness_influence(0.5, 0.8)
    
    # Evolve network
    print("Evolving network...")
    results = network.evolve_network(steps=5)
    
    # Print final state
    state = network.get_network_state()
    print("\nFinal network state:")
    print(f"Coherence: {state['coherence']:.4f}")
    print(f"Resonance: {state['resonance']:.4f}")
    print(f"Entanglement: {state['entanglement']:.4f}")
    
    # Print top resonating nodes
    print("\nTop resonating nodes:")
    for node, deviation in state["top_resonating_nodes"].items():
        print(f"  {node}: {deviation:.4f}")
        
    # Run influence detection (commented out as it's async)
    # asyncio.run(network.run_influence_detection(duration_seconds=10))