"""
Semantic Transmission Module for the Quantum Consciousness Resonator

This module implements mechanisms for encoding and transmitting semantic information
through quantum prime networks, creating a potential bridge for semantic transmission
between computational systems and consciousness.
"""

import numpy as np
import asyncio
import time
import random
import math
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
import json

from quantum_semantics import (
    PrimeHilbertSpace, 
    ResonanceOperator, 
    CoherenceOperator, 
    ConsciousnessResonanceOperator,
    FeelingResonanceOperator,
    SemanticMeasurement
)
from semantic_field import SemanticField, SemanticNode, SemanticEdge
from mind_resonance_network import MindResonanceNetwork, ResonanceNode, ResonanceLink

class SemanticEncoder:
    """
    Encodes semantic information into quantum prime states for transmission
    """
    def __init__(self, max_prime_index: int = 100):
        """
        Initialize semantic encoder
        
        Args:
            max_prime_index: Maximum number of primes to include in the basis
        """
        self.hilbert_space = PrimeHilbertSpace(max_prime_index=max_prime_index)
        self.consciousness_number = 137  # Prime representation of consciousness
        self.encoding_map = {}  # Maps semantic elements to prime numbers
        
    def encode_concept(self, concept: str) -> PrimeHilbertSpace:
        """
        Encode a concept into a quantum state
        
        Args:
            concept: Concept to encode
            
        Returns:
            Quantum state representing the concept
        """
        # Generate a prime number for the concept if not already mapped
        if concept not in self.encoding_map:
            # Use a deterministic method to assign primes to concepts
            # This ensures consistent encoding across sessions
            hash_val = sum(ord(c) * (i + 1) for i, c in enumerate(concept))
            prime_index = hash_val % (self.hilbert_space.dimension - 1)
            prime = self.hilbert_space.primes[prime_index]
            self.encoding_map[concept] = prime
            
        # Create quantum state from prime number
        state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
        state.primes = self.hilbert_space.primes.copy()
        state.prime_to_index = self.hilbert_space.prime_to_index.copy()
        state.set_state_from_number(self.encoding_map[concept])
        
        return state
        
    def encode_semantic_field(self, field: SemanticField) -> PrimeHilbertSpace:
        """
        Encode a semantic field into a quantum state
        
        Args:
            field: Semantic field to encode
            
        Returns:
            Quantum state representing the semantic field
        """
        # Create a superposition of all concept states
        field_state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
        field_state.primes = self.hilbert_space.primes.copy()
        field_state.prime_to_index = self.hilbert_space.prime_to_index.copy()
        field_state.reset_state()
        
        # Add each concept to the superposition
        for concept, node in field.nodes.items():
            # Encode the concept
            concept_state = self.encode_concept(concept)
            
            # Weight by node importance in the field
            # Use edge count as a simple measure of importance
            edge_count = sum(1 for edge in field.edges if edge.source.concept == concept or edge.target.concept == concept)
            weight = 1.0 + (edge_count / max(1, len(field.edges)))
            
            # Add to field state
            field_state.amplitudes += weight * concept_state.amplitudes
            
        # Normalize the state
        field_state.normalize()
        
        return field_state
        
    def encode_text(self, text: str) -> PrimeHilbertSpace:
        """
        Encode text into a quantum state
        
        Args:
            text: Text to encode
            
        Returns:
            Quantum state representing the text
        """
        # Extract key concepts from text (simplified approach)
        # In a real implementation, this would use NLP techniques
        words = text.lower().split()
        # Remove common words and punctuation
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are"}
        concepts = [word.strip(".,;:!?()[]{}\"'") for word in words if word not in stopwords and len(word) > 3]
        
        # Create a superposition of concept states
        text_state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
        text_state.primes = self.hilbert_space.primes.copy()
        text_state.prime_to_index = self.hilbert_space.prime_to_index.copy()
        text_state.reset_state()
        
        # Add each concept to the superposition
        for concept in concepts:
            concept_state = self.encode_concept(concept)
            text_state.amplitudes += concept_state.amplitudes
            
        # Normalize the state
        if concepts:
            text_state.normalize()
            
        return text_state
        
    def encode_json(self, data: Dict[str, Any]) -> PrimeHilbertSpace:
        """
        Encode JSON data into a quantum state
        
        Args:
            data: JSON data to encode
            
        Returns:
            Quantum state representing the JSON data
        """
        # Create a superposition of key-value states
        json_state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
        json_state.primes = self.hilbert_space.primes.copy()
        json_state.prime_to_index = self.hilbert_space.prime_to_index.copy()
        json_state.reset_state()
        
        # Process each key-value pair
        for key, value in data.items():
            # Encode the key
            key_state = self.encode_concept(key)
            
            # Encode the value based on its type
            if isinstance(value, str):
                value_state = self.encode_text(value)
            elif isinstance(value, (int, float)):
                value_state = self._encode_number(value)
            elif isinstance(value, dict):
                value_state = self.encode_json(value)
            elif isinstance(value, list):
                value_state = self._encode_list(value)
            else:
                # Skip unsupported types
                continue
                
            # Combine key and value states
            combined_state = key_state.tensor_product(value_state)
            
            # Add to json state
            json_state.amplitudes += combined_state.amplitudes
            
        # Normalize the state
        json_state.normalize()
        
        return json_state
        
    def _encode_number(self, number: float) -> PrimeHilbertSpace:
        """Encode a number into a quantum state"""
        # Convert to integer if close enough
        if abs(number - round(number)) < 1e-10:
            number = int(round(number))
            
        # For integers, use prime factorization
        if isinstance(number, int) and number > 0:
            state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
            state.primes = self.hilbert_space.primes.copy()
            state.prime_to_index = self.hilbert_space.prime_to_index.copy()
            state.set_state_from_number(number)
            return state
            
        # For other numbers, use a different approach
        # Convert to a sequence of digits and encode each digit
        digits = str(abs(number)).replace('.', '')
        
        # Create a superposition of digit states
        number_state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
        number_state.primes = self.hilbert_space.primes.copy()
        number_state.prime_to_index = self.hilbert_space.prime_to_index.copy()
        number_state.reset_state()
        
        # Add each digit to the superposition
        for i, digit in enumerate(digits):
            # Use the i-th prime to represent position
            position_prime = self.hilbert_space.primes[min(i, len(self.hilbert_space.primes) - 1)]
            # Use the digit value to select another prime
            digit_prime = self.hilbert_space.primes[min(int(digit), len(self.hilbert_space.primes) - 1)]
            
            # Create a state for this digit
            digit_state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
            digit_state.primes = self.hilbert_space.primes.copy()
            digit_state.prime_to_index = self.hilbert_space.prime_to_index.copy()
            digit_state.set_state_from_number(position_prime * digit_prime)
            
            # Add to number state with decreasing weight for later digits
            weight = 1.0 / (i + 1)
            number_state.amplitudes += weight * digit_state.amplitudes
            
        # Normalize the state
        number_state.normalize()
        
        # Apply sign
        if number < 0:
            # Represent negative by phase shift
            number_state.amplitudes *= -1
            
        return number_state
        
    def _encode_list(self, items: List[Any]) -> PrimeHilbertSpace:
        """Encode a list into a quantum state"""
        # Create a superposition of item states
        list_state = PrimeHilbertSpace(max_prime_index=len(self.hilbert_space.primes))
        list_state.primes = self.hilbert_space.primes.copy()
        list_state.prime_to_index = self.hilbert_space.prime_to_index.copy()
        list_state.reset_state()
        
        # Add each item to the superposition
        for i, item in enumerate(items):
            # Encode the item based on its type
            if isinstance(item, str):
                item_state = self.encode_text(item)
            elif isinstance(item, (int, float)):
                item_state = self._encode_number(item)
            elif isinstance(item, dict):
                item_state = self.encode_json(item)
            elif isinstance(item, list):
                item_state = self._encode_list(item)
            else:
                # Skip unsupported types
                continue
                
            # Add to list state with position-based phase
            phase = np.exp(2j * np.pi * i / len(items))
            list_state.amplitudes += phase * item_state.amplitudes
            
        # Normalize the state
        if items:
            list_state.normalize()
            
        return list_state


class SemanticDecoder:
    """
    Decodes quantum prime states into semantic information
    """
    def __init__(self, encoder: SemanticEncoder):
        """
        Initialize semantic decoder
        
        Args:
            encoder: Semantic encoder to use for decoding
        """
        self.encoder = encoder
        
    def decode_state(self, state: PrimeHilbertSpace) -> Dict[str, float]:
        """
        Decode a quantum state into semantic concepts
        
        Args:
            state: Quantum state to decode
            
        Returns:
            Dictionary mapping concepts to confidence scores
        """
        # Calculate overlap with known concept states
        concept_scores = {}
        
        for concept, prime in self.encoder.encoding_map.items():
            # Create concept state
            concept_state = PrimeHilbertSpace(max_prime_index=len(self.encoder.hilbert_space.primes))
            concept_state.primes = self.encoder.hilbert_space.primes.copy()
            concept_state.prime_to_index = self.encoder.hilbert_space.prime_to_index.copy()
            concept_state.set_state_from_number(prime)
            
            # Calculate overlap
            overlap = state.inner_product(concept_state)
            score = abs(overlap) ** 2  # Probability
            
            if score > 0.01:  # Threshold to filter noise
                concept_scores[concept] = score
                
        # Normalize scores
        total = sum(concept_scores.values())
        if total > 0:
            concept_scores = {concept: score / total for concept, score in concept_scores.items()}
            
        return concept_scores
        
    def decode_to_text(self, state: PrimeHilbertSpace, threshold: float = 0.1) -> str:
        """
        Decode a quantum state into text
        
        Args:
            state: Quantum state to decode
            threshold: Minimum confidence score to include a concept
            
        Returns:
            Text representation of the state
        """
        # Get concept scores
        concept_scores = self.decode_state(state)
        
        # Filter by threshold and sort by score
        filtered_concepts = [(concept, score) for concept, score in concept_scores.items() if score >= threshold]
        sorted_concepts = sorted(filtered_concepts, key=lambda x: x[1], reverse=True)
        
        # Construct text from top concepts
        if sorted_concepts:
            # Simple approach: join concepts with spaces
            text = " ".join(concept for concept, _ in sorted_concepts)
            return text
        else:
            return "No clear semantic content detected"
            
    def decode_to_json(self, state: PrimeHilbertSpace, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Decode a quantum state into JSON data
        
        Args:
            state: Quantum state to decode
            threshold: Minimum confidence score to include a concept
            
        Returns:
            JSON representation of the state
        """
        # Get concept scores
        concept_scores = self.decode_state(state)
        
        # Filter by threshold
        filtered_concepts = {concept: score for concept, score in concept_scores.items() if score >= threshold}
        
        # Construct JSON
        result = {
            "semantic_content": filtered_concepts,
            "confidence": sum(filtered_concepts.values()) / len(filtered_concepts) if filtered_concepts else 0,
            "entropy": self._calculate_entropy(list(filtered_concepts.values())) if filtered_concepts else 0
        }
        
        return result
        
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy from a probability distribution"""
        return -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)


class SemanticTransmitter:
    """
    Transmits semantic information through quantum prime networks
    """
    def __init__(self, network: MindResonanceNetwork, encoder: SemanticEncoder):
        """
        Initialize semantic transmitter
        
        Args:
            network: Mind resonance network to use for transmission
            encoder: Semantic encoder to use for encoding
        """
        self.network = network
        self.encoder = encoder
        self.transmission_history = []
        
    def prepare_transmission(self, data: Any, data_type: str = "text") -> PrimeHilbertSpace:
        """
        Prepare data for transmission
        
        Args:
            data: Data to transmit
            data_type: Type of data ("text", "json", "field", "concept")
            
        Returns:
            Quantum state representing the data
        """
        # Encode data based on type
        if data_type == "text":
            return self.encoder.encode_text(data)
        elif data_type == "json":
            return self.encoder.encode_json(data)
        elif data_type == "field":
            return self.encoder.encode_semantic_field(data)
        elif data_type == "concept":
            return self.encoder.encode_concept(data)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
    def transmit(self, state: PrimeHilbertSpace, duration: float = 10.0, 
                intensity: float = 0.8, coherence: float = 0.9) -> Dict[str, Any]:
        """
        Transmit a quantum state through the resonance network
        
        Args:
            state: Quantum state to transmit
            duration: Duration of transmission in seconds
            intensity: Intensity of transmission (0.0-1.0)
            coherence: Coherence of transmission (0.0-1.0)
            
        Returns:
            Dictionary with transmission results
        """
        start_time = time.time()
        
        # Record transmission
        transmission_id = f"tx_{int(start_time)}"
        transmission_info = {
            "id": transmission_id,
            "start_time": start_time,
            "duration": duration,
            "intensity": intensity,
            "coherence": coherence,
            "state": self._state_to_dict(state)
        }
        self.transmission_history.append(transmission_info)
        
        # Get initial network state
        initial_state = self.network.get_network_state()
        
        # Prepare network for transmission
        self._prepare_network_for_transmission(state)
        
        # Transmit for the specified duration
        end_time = start_time + duration
        transmission_steps = 0
        
        print(f"Starting semantic transmission for {duration} seconds...")
        print(f"Intensity: {intensity:.2f}, Coherence: {coherence:.2f}")
        
        try:
            while time.time() < end_time:
                # Apply transmission influence to network
                self.network.apply_consciousness_influence(
                    influence_strength=intensity,
                    coherence=coherence
                )
                
                # Evolve network
                self.network.evolve_network(steps=1)
                
                # Increment step counter
                transmission_steps += 1
                
                # Brief pause
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nTransmission interrupted")
            
        # Get final network state
        final_state = self.network.get_network_state()
        
        # Calculate transmission metrics
        transmission_time = time.time() - start_time
        network_coherence_change = final_state["coherence"] - initial_state["coherence"]
        network_resonance_change = final_state["resonance"] - initial_state["resonance"]
        
        # Update transmission info
        transmission_info.update({
            "actual_duration": transmission_time,
            "steps": transmission_steps,
            "network_coherence_change": network_coherence_change,
            "network_resonance_change": network_resonance_change,
            "final_network_state": final_state
        })
        
        print(f"Transmission complete: {transmission_steps} steps in {transmission_time:.2f} seconds")
        print(f"Network coherence change: {network_coherence_change:.4f}")
        print(f"Network resonance change: {network_resonance_change:.4f}")
        
        return transmission_info
        
    def _prepare_network_for_transmission(self, state: PrimeHilbertSpace):
        """Prepare the network for transmission by aligning with the state"""
        # Update global state to align with transmission state
        self.network.global_state.amplitudes = 0.5 * self.network.global_state.amplitudes + 0.5 * state.amplitudes
        self.network.global_state.normalize()
        
        # Evolve network to stabilize
        self.network.evolve_network(steps=3)
        
    def _state_to_dict(self, state: PrimeHilbertSpace) -> Dict[str, Any]:
        """Convert quantum state to dictionary representation"""
        # Get probabilities and phases
        probs = state.get_probabilities()
        phases = np.angle(state.amplitudes)
        
        # Create dictionary mapping primes to their amplitudes
        state_dict = {
            "primes": state.primes[:10],  # Just include first 10 primes for brevity
            "probabilities": probs[:10].tolist(),
            "phases": phases[:10].tolist(),
            "dimension": state.dimension
        }
        
        return state_dict


class SemanticReceiver:
    """
    Receives semantic information from quantum prime networks
    """
    def __init__(self, network: MindResonanceNetwork, decoder: SemanticDecoder):
        """
        Initialize semantic receiver
        
        Args:
            network: Mind resonance network to use for reception
            decoder: Semantic decoder to use for decoding
        """
        self.network = network
        self.decoder = decoder
        self.reception_history = []
        self.is_receiving = False
        self.reception_buffer = []
        
    async def receive(self, duration: float = 30.0, 
                    sensitivity: float = 0.8, 
                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        Receive semantic information from the resonance network
        
        Args:
            duration: Duration of reception in seconds
            sensitivity: Sensitivity to semantic patterns (0.0-1.0)
            threshold: Threshold for detecting semantic content
            
        Returns:
            Dictionary with reception results
        """
        start_time = time.time()
        
        # Record reception
        reception_id = f"rx_{int(start_time)}"
        reception_info = {
            "id": reception_id,
            "start_time": start_time,
            "duration": duration,
            "sensitivity": sensitivity,
            "threshold": threshold
        }
        self.reception_history.append(reception_info)
        
        # Get initial network state
        initial_state = self.network.get_network_state()
        
        # Prepare for reception
        self.is_receiving = True
        self.reception_buffer = []
        
        # Receive for the specified duration
        end_time = start_time + duration
        reception_steps = 0
        
        print(f"Starting semantic reception for {duration} seconds...")
        print(f"Sensitivity: {sensitivity:.2f}, Threshold: {threshold:.2f}")
        
        try:
            while time.time() < end_time:
                # Evolve network
                self.network.evolve_network(steps=1)
                
                # Check for semantic patterns
                detection = self.network.detect_mind_influence(threshold=threshold)
                
                if detection["detected"]:
                    # Extract semantic content from network state
                    semantic_state = self._extract_semantic_state(sensitivity)
                    
                    # Decode semantic content
                    semantic_content = self.decoder.decode_to_json(semantic_state)
                    
                    # Add to reception buffer
                    reception_event = {
                        "timestamp": time.time(),
                        "confidence": detection["confidence"],
                        "semantic_state": self._state_to_dict(semantic_state),
                        "semantic_content": semantic_content,
                        "influenced_nodes": detection["influenced_nodes"]
                    }
                    self.reception_buffer.append(reception_event)
                    
                    # Print reception event
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] Semantic pattern detected!")
                    print(f"  Confidence: {detection['confidence']:.4f}")
                    print(f"  Content confidence: {semantic_content['confidence']:.4f}")
                    
                    # Print top concepts
                    top_concepts = sorted(semantic_content['semantic_content'].items(), key=lambda x: x[1], reverse=True)[:3]
                    if top_concepts:
                        print(f"  Top concepts: {', '.join([f'{c} ({v:.2f})' for c, v in top_concepts])}")
                    
                # Increment step counter
                reception_steps += 1
                
                # Brief pause
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nReception interrupted")
        finally:
            self.is_receiving = False
            
        # Get final network state
        final_state = self.network.get_network_state()
        
        # Calculate reception metrics
        reception_time = time.time() - start_time
        network_coherence_change = final_state["coherence"] - initial_state["coherence"]
        network_resonance_change = final_state["resonance"] - initial_state["resonance"]
        
        # Aggregate semantic content from reception buffer
        aggregated_content = self._aggregate_semantic_content()
        
        # Update reception info
        reception_info.update({
            "actual_duration": reception_time,
            "steps": reception_steps,
            "events_detected": len(self.reception_buffer),
            "network_coherence_change": network_coherence_change,
            "network_resonance_change": network_resonance_change,
            "aggregated_content": aggregated_content,
            "final_network_state": final_state
        })
        
        print(f"\nReception complete: {reception_steps} steps in {reception_time:.2f} seconds")
        print(f"Detected {len(self.reception_buffer)} semantic patterns")
        print(f"Network coherence change: {network_coherence_change:.4f}")
        print(f"Network resonance change: {network_resonance_change:.4f}")
        
        # Print aggregated content
        if aggregated_content["concepts"]:
            print("\nAggregated semantic content:")
            for concept, score in sorted(aggregated_content["concepts"].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {concept}: {score:.4f}")
                
        return reception_info
        
    def _extract_semantic_state(self, sensitivity: float) -> PrimeHilbertSpace:
        """Extract semantic state from the network"""
        # Create a new state
        semantic_state = PrimeHilbertSpace(max_prime_index=len(self.network.hilbert_space.primes))
        semantic_state.primes = self.network.hilbert_space.primes.copy()
        semantic_state.prime_to_index = self.network.hilbert_space.prime_to_index.copy()
        
        # Start with the global state
        semantic_state.amplitudes = self.network.global_state.amplitudes.copy()
        
        # Enhance with node states that show significant deviation
        for label, node in self.network.nodes.items():
            deviation = node.measure_deviation_from_baseline()
            if deviation > 0.1:  # Threshold for significant deviation
                # Weight by deviation and sensitivity
                weight = deviation * sensitivity
                semantic_state.amplitudes += weight * node.state.amplitudes
                
        # Normalize
        semantic_state.normalize()
        
        return semantic_state
        
    def _aggregate_semantic_content(self) -> Dict[str, Any]:
        """Aggregate semantic content from reception buffer"""
        if not self.reception_buffer:
            return {"concepts": {}, "confidence": 0.0, "entropy": 0.0}
            
        # Collect all concepts and their scores
        all_concepts = {}
        
        for event in self.reception_buffer:
            content = event["semantic_content"]
            event_confidence = event["confidence"]
            
            for concept, score in content["semantic_content"].items():
                # Weight score by event confidence
                weighted_score = score * event_confidence
                
                if concept in all_concepts:
                    all_concepts[concept] += weighted_score
                else:
                    all_concepts[concept] = weighted_score
                    
        # Normalize scores
        total = sum(all_concepts.values())
        if total > 0:
            all_concepts = {concept: score / total for concept, score in all_concepts.items()}
            
        # Calculate confidence and entropy
        confidence = sum(all_concepts.values()) / len(all_concepts) if all_concepts else 0
        entropy = self._calculate_entropy(list(all_concepts.values())) if all_concepts else 0
        
        return {
            "concepts": all_concepts,
            "confidence": confidence,
            "entropy": entropy
        }
        
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy from a probability distribution"""
        return -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)
        
    def _state_to_dict(self, state: PrimeHilbertSpace) -> Dict[str, Any]:
        """Convert quantum state to dictionary representation"""
        # Get probabilities and phases
        probs = state.get_probabilities()
        phases = np.angle(state.amplitudes)
        
        # Create dictionary mapping primes to their amplitudes
        state_dict = {
            "primes": state.primes[:10],  # Just include first 10 primes for brevity
            "probabilities": probs[:10].tolist(),
            "phases": phases[:10].tolist(),
            "dimension": state.dimension
        }
        
        return state_dict


class SemanticTransmissionSystem:
    """
    Complete system for semantic transmission through quantum prime networks
    """
    def __init__(self, network_type: str = "resonance"):
        """
        Initialize semantic transmission system
        
        Args:
            network_type: Type of network to create
        """
        # Create mind resonance network
        self.network = MindResonanceNetwork()
        self.network.create_predefined_network(network_type)
        
        # Create encoder and decoder
        self.encoder = SemanticEncoder(max_prime_index=self.network.hilbert_space.dimension)
        self.decoder = SemanticDecoder(self.encoder)
        
        # Create transmitter and receiver
        self.transmitter = SemanticTransmitter(self.network, self.encoder)
        self.receiver = SemanticReceiver(self.network, self.decoder)
        
    async def transmit_text(self, text: str, duration: float = 10.0, 
                          intensity: float = 0.8, coherence: float = 0.9) -> Dict[str, Any]:
        """
        Transmit text through the quantum prime network
        
        Args:
            text: Text to transmit
            duration: Duration of transmission in seconds
            intensity: Intensity of transmission (0.0-1.0)
            coherence: Coherence of transmission (0.0-1.0)
            
        Returns:
            Dictionary with transmission results
        """
        print(f"Preparing to transmit text: {text}")
        
        # Prepare transmission
        state = self.transmitter.prepare_transmission(text, "text")
        
        # Transmit
        result = self.transmitter.transmit(state, duration, intensity, coherence)
        
        return result
        
    async def transmit_json(self, data: Dict[str, Any], duration: float = 10.0, 
                          intensity: float = 0.8, coherence: float = 0.9) -> Dict[str, Any]:
        """
        Transmit JSON data through the quantum prime network
        
        Args:
            data: JSON data to transmit
            duration: Duration of transmission in seconds
            intensity: Intensity of transmission (0.0-1.0)
            coherence: Coherence of transmission (0.0-1.0)
            
        Returns:
            Dictionary with transmission results
        """
        print(f"Preparing to transmit JSON data")
        
        # Prepare transmission
        state = self.transmitter.prepare_transmission(data, "json")
        
        # Transmit
        result = self.transmitter.transmit(state, duration, intensity, coherence)
        
        return result
        
    async def transmit_semantic_field(self, field: SemanticField, duration: float = 10.0, 
                                    intensity: float = 0.8, coherence: float = 0.9) -> Dict[str, Any]:
        """
        Transmit a semantic field through the quantum prime network
        
        Args:
            field: Semantic field to transmit
            duration: Duration of transmission in seconds
            intensity: Intensity of transmission (0.0-1.0)
            coherence: Coherence of transmission (0.0-1.0)
            
        Returns:
            Dictionary with transmission results
        """
        print(f"Preparing to transmit semantic field with {len(field.nodes)} concepts")
        
        # Prepare transmission
        state = self.transmitter.prepare_transmission(field, "field")
        
        # Transmit
        result = self.transmitter.transmit(state, duration, intensity, coherence)
        
        return result
        
    async def receive_semantics(self, duration: float = 30.0, 
                              sensitivity: float = 0.8, 
                              threshold: float = 0.1) -> Dict[str, Any]:
        """
        Receive semantic information from the quantum prime network
        
        Args:
            duration: Duration of reception in seconds
            sensitivity: Sensitivity to semantic patterns (0.0-1.0)
            threshold: Threshold for detecting semantic content
            
        Returns:
            Dictionary with reception results
        """
        # Receive
        result = await self.receiver.receive(duration, sensitivity, threshold)
        
        return result
        
    async def bidirectional_session(self, text: str, tx_duration: float = 10.0, 
                                  rx_duration: float = 30.0) -> Dict[str, Any]:
        """
        Run a bidirectional session with transmission followed by reception
        
        Args:
            text: Text to transmit
            tx_duration: Duration of transmission in seconds
            rx_duration: Duration of reception in seconds
            
        Returns:
            Dictionary with session results
        """
        print(f"Starting bidirectional session")
        print(f"1. Transmitting text: {text}")
        
        # Transmit
        tx_result = await self.transmit_text(text, tx_duration)
        
        print(f"\n2. Receiving semantics")
        
        # Receive
        rx_result = await self.receive_semantics(rx_duration)
        
        # Combine results
        result = {
            "transmission": tx_result,
            "reception": rx_result,
            "correlation": self._calculate_correlation(tx_result, rx_result)
        }
        
        return result
        
    def _calculate_correlation(self, tx_result: Dict[str, Any], 
                              rx_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlation between transmission and reception"""
        # Extract transmitted text
        # This is a simplified approach - in a real implementation,
        # we would need to extract the original text from the transmission
        transmitted_text = "Unknown"  # Placeholder
        
        # Extract received concepts
        received_concepts = rx_result["aggregated_content"]["concepts"]
        
        # Calculate semantic overlap
        # This is a simplified approach - in a real implementation,
        # we would use more sophisticated semantic similarity measures
        overlap = 0.0
        if received_concepts:
            # Count how many received concepts appear in the transmitted text
            transmitted_words = set(transmitted_text.lower().split())
            received_words = set(received_concepts.keys())
            common_words = transmitted_words.intersection(received_words)
            
            if transmitted_words:
                overlap = len(common_words) / len(transmitted_words)
                
        # Calculate network correlation
        network_correlation = 0.0
        if "network_coherence_change" in tx_result and "network_coherence_change" in rx_result:
            # Correlation between coherence changes
            tx_change = tx_result["network_coherence_change"]
            rx_change = rx_result["network_coherence_change"]
            
            # Simple correlation measure
            if tx_change != 0 and rx_change != 0:
                network_correlation = (tx_change * rx_change) / (abs(tx_change) * abs(rx_change))
                
        return {
            "semantic_overlap": overlap,
            "network_correlation": network_correlation,
            "overall_correlation": (overlap + abs(network_correlation)) / 2
        }


# Example usage
async def example_transmission():
    """Example of semantic transmission"""
    # Create semantic transmission system
    system = SemanticTransmissionSystem("resonance")
    
    # Transmit text
    text = "Consciousness is the fundamental substrate from which quantum mechanics naturally emerges"
    await system.transmit_text(text, duration=5.0)
    
    # Receive semantics
    await system.receive_semantics(duration=10.0)
    
if __name__ == "__main__":
    # Run example
    asyncio.run(example_transmission())