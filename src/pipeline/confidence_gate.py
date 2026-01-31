"""
MUST++ Confidence Gate Module

Implements Step 4 of the MUST++ Pipeline:
- Confidence evaluation using max probability, entropy, and tokenization coverage
- Decision to accept transformer result or trigger fallback
"""

import numpy as np
from typing import Dict, List, Tuple, NamedTuple, Optional
from dataclasses import dataclass
from enum import Enum


class GateDecision(Enum):
    """Gate decision outcomes"""
    ACCEPT = "ACCEPT"               # Accept transformer result
    FALLBACK_INDIC = "FALLBACK_INDIC"     # Trigger Indic specialist
    FALLBACK_SAFETY = "FALLBACK_SAFETY"   # Trigger safety reasoning
    FALLBACK_RULE = "FALLBACK_RULE"       # Trigger rule-based escalation


@dataclass
class ConfidenceMetrics:
    """Confidence evaluation metrics"""
    max_probability: float
    entropy: float
    tokenization_coverage: float
    prediction_class: int
    class_probabilities: Dict[str, float]
    is_reliable: bool
    gate_decision: GateDecision
    reasoning: str


class ConfidenceGate:
    """
    Confidence Gate for transformer predictions.
    
    Evaluates reliability using:
    - Max probability (threshold: 0.75)
    - Prediction entropy (low = confident)
    - Tokenization coverage (unknown token ratio)
    
    Decision Rule:
    - If confidence >= 0.75 AND entropy is low: accept
    - Else: trigger fallback
    """
    
    # Thresholds
    CONFIDENCE_THRESHOLD = 0.75
    ENTROPY_LOW_THRESHOLD = 0.5      # Below this = low entropy
    ENTROPY_HIGH_THRESHOLD = 1.0     # Above this = very uncertain
    COVERAGE_THRESHOLD = 0.80        # Minimum known token ratio
    
    # Class labels
    LABELS = ["neutral", "offensive", "hate"]
    
    def __init__(
        self,
        confidence_threshold: float = 0.75,
        entropy_threshold: float = 0.5,
        coverage_threshold: float = 0.80
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.coverage_threshold = coverage_threshold
    
    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate Shannon entropy of probability distribution.
        
        Lower entropy = more confident prediction.
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Entropy value (0 = certain, higher = uncertain)
        """
        # Avoid log(0) by clipping
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)
    
    def calculate_tokenization_coverage(
        self,
        tokens: List[str],
        unknown_token: str = "[UNK]"
    ) -> float:
        """
        Calculate ratio of known tokens.
        
        Args:
            tokens: List of tokenized tokens
            unknown_token: The unknown token marker
            
        Returns:
            Coverage ratio (0-1)
        """
        if not tokens:
            return 0.0
        
        known_count = sum(1 for t in tokens if t != unknown_token and not t.startswith("##"))
        return known_count / len(tokens)
    
    def evaluate(
        self,
        probabilities: np.ndarray,
        tokens: Optional[List[str]] = None,
        is_romanized: bool = False,
        is_indic_dominant: bool = False,
        is_short_text: bool = False
    ) -> ConfidenceMetrics:
        """
        Evaluate confidence and determine gate decision.
        
        Args:
            probabilities: Model output probabilities [neutral, offensive, hate]
            tokens: Tokenized input (for coverage calculation)
            is_romanized: Whether text is Romanized Indian language
            is_indic_dominant: Whether Tamil/Hindi is dominant
            is_short_text: Whether text is very short (<5 words)
            
        Returns:
            ConfidenceMetrics with gate decision
        """
        # Ensure probabilities is numpy array
        probs = np.array(probabilities).flatten()
        if len(probs) != 3:
            # Pad or truncate to 3 classes
            probs = np.pad(probs, (0, max(0, 3 - len(probs))))[:3]
        
        # Normalize probabilities
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.array([1/3, 1/3, 1/3])
        
        max_prob = float(np.max(probs))
        predicted_class = int(np.argmax(probs))
        entropy = self.calculate_entropy(probs)
        
        # Calculate coverage if tokens provided
        coverage = 1.0
        if tokens:
            coverage = self.calculate_tokenization_coverage(tokens)
        
        # Build class probability dict
        class_probs = {
            self.LABELS[i]: float(probs[i]) 
            for i in range(len(self.LABELS))
        }
        
        # Determine reliability and gate decision
        is_reliable, decision, reasoning = self._make_decision(
            max_prob=max_prob,
            entropy=entropy,
            coverage=coverage,
            predicted_class=predicted_class,
            is_romanized=is_romanized,
            is_indic_dominant=is_indic_dominant,
            is_short_text=is_short_text
        )
        
        return ConfidenceMetrics(
            max_probability=max_prob,
            entropy=entropy,
            tokenization_coverage=coverage,
            prediction_class=predicted_class,
            class_probabilities=class_probs,
            is_reliable=is_reliable,
            gate_decision=decision,
            reasoning=reasoning
        )
    
    def _make_decision(
        self,
        max_prob: float,
        entropy: float,
        coverage: float,
        predicted_class: int,
        is_romanized: bool,
        is_indic_dominant: bool,
        is_short_text: bool
    ) -> Tuple[bool, GateDecision, str]:
        """
        Make gate decision based on metrics.
        
        Returns:
            Tuple of (is_reliable, decision, reasoning)
        """
        reasons = []
        
        # Check primary confidence criteria
        high_confidence = max_prob >= self.confidence_threshold
        low_entropy = entropy <= self.entropy_threshold
        good_coverage = coverage >= self.coverage_threshold
        
        # ACCEPT condition: high confidence AND low entropy
        if high_confidence and low_entropy:
            if good_coverage or not is_romanized:
                return (
                    True,
                    GateDecision.ACCEPT,
                    f"High confidence ({max_prob:.2f}) with low entropy ({entropy:.2f}). "
                    f"Transformer prediction is reliable."
                )
        
        # Determine fallback type
        reasons.append(f"Max prob: {max_prob:.2f}, Entropy: {entropy:.2f}, Coverage: {coverage:.2f}")
        
        # FALLBACK_INDIC: For Romanized or Indic-dominant text
        if is_romanized or is_indic_dominant:
            reasons.append("Romanized/Indic text requires specialist verification")
            return (
                False,
                GateDecision.FALLBACK_INDIC,
                " | ".join(reasons)
            )
        
        # FALLBACK_SAFETY: For short or fragmentary text with high uncertainty
        if is_short_text or entropy > self.ENTROPY_HIGH_THRESHOLD:
            reasons.append("Short/fragmentary text with high uncertainty")
            return (
                False,
                GateDecision.FALLBACK_SAFETY,
                " | ".join(reasons)
            )
        
        # Low confidence triggers safety fallback
        if not high_confidence:
            reasons.append("Low confidence requires safety fallback")
            return (
                False,
                GateDecision.FALLBACK_SAFETY,
                " | ".join(reasons)
            )
        
        # High entropy with moderate confidence
        if not low_entropy:
            reasons.append("High entropy indicates uncertainty")
            return (
                False,
                GateDecision.FALLBACK_SAFETY,
                " | ".join(reasons)
            )
        
        # Default fallback
        return (
            False,
            GateDecision.FALLBACK_SAFETY,
            "Confidence criteria not met, defaulting to safety fallback"
        )
    
    def should_trigger_fallback(self, metrics: ConfidenceMetrics) -> bool:
        """Check if fallback should be triggered."""
        return metrics.gate_decision != GateDecision.ACCEPT
    
    def get_fallback_tier(self, metrics: ConfidenceMetrics) -> int:
        """
        Get fallback tier number.
        
        Returns:
            2 = Indic Specialist
            3 = Classical Safety
            4 = Rule-Based (handled separately)
        """
        if metrics.gate_decision == GateDecision.FALLBACK_INDIC:
            return 2
        elif metrics.gate_decision == GateDecision.FALLBACK_SAFETY:
            return 3
        elif metrics.gate_decision == GateDecision.FALLBACK_RULE:
            return 4
        return 0
