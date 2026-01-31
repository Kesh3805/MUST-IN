"""
MUST++ Decision Resolver Module

Implements Step 6 of the MUST++ Pipeline:
- Combines signals conservatively
- Priority order: hate > offensive > neutral
- Resolves ambiguity between hate and offensive
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .confidence_gate import ConfidenceMetrics, GateDecision
from .fallback_logic import FallbackResult, PredictionLabel


@dataclass
class ResolvedDecision:
    """Final resolved classification decision"""
    label: str                          # "neutral", "offensive", "hate"
    confidence: float                   # 0.0 - 1.0
    languages_detected: Dict[str, float]
    fallback_used: bool
    key_harm_tokens: List[str]
    explanation: str
    
    # Detailed metadata
    transformer_prediction: Optional[str]
    transformer_confidence: float
    fallback_tier: Optional[int]
    escalation_triggered: bool
    identity_groups: List[str]
    rejection_reasons: Dict[str, str]   # Why weaker labels were rejected


class DecisionResolver:
    """
    Resolves final classification by combining transformer and fallback signals.
    
    Priority Order:
    - hate > offensive > neutral
    
    Ambiguity Rule:
    - If ambiguity exists between hate and offensive: choose hate
    - Explicitly explain why
    """
    
    LABEL_PRIORITY = {
        "hate": 3,
        "offensive": 2,
        "neutral": 1
    }
    
    def __init__(self):
        self.labels = ["neutral", "offensive", "hate"]
    
    def resolve(
        self,
        transformer_probs: List[float],
        transformer_prediction: int,
        confidence_metrics: ConfidenceMetrics,
        fallback_result: Optional[FallbackResult],
        language_proportions: Dict[str, float],
        original_text: str
    ) -> ResolvedDecision:
        """
        Resolve final classification decision.
        
        Args:
            transformer_probs: Probabilities from transformer [neutral, offensive, hate]
            transformer_prediction: Argmax of transformer output
            confidence_metrics: Confidence gate evaluation
            fallback_result: Result from fallback logic (if triggered)
            language_proportions: Detected language proportions
            original_text: Original input text
            
        Returns:
            ResolvedDecision with final label and explanation
        """
        rejection_reasons = {}
        explanation_parts = []
        
        # Determine if fallback was used
        fallback_used = fallback_result is not None and not confidence_metrics.is_reliable
        
        # Get transformer label
        trans_label = self.labels[transformer_prediction] if transformer_prediction < 3 else "neutral"
        trans_conf = confidence_metrics.max_probability
        
        if fallback_used and fallback_result:
            # Combine transformer and fallback
            final_label, final_conf, explanation = self._combine_predictions(
                trans_label=trans_label,
                trans_conf=trans_conf,
                fallback_label=fallback_result.prediction.value,
                fallback_conf=fallback_result.confidence,
                escalation=fallback_result.escalation_triggered,
                harm_tokens=fallback_result.harm_tokens,
                rejection_reasons=rejection_reasons
            )
            
            explanation_parts.append(explanation)
            
            # Build harm tokens list
            harm_tokens = [t['token'] for t in fallback_result.harm_tokens[:10]]
            identity_groups = fallback_result.identity_groups_detected
            fallback_tier = fallback_result.tier_used.value
            escalation_triggered = fallback_result.escalation_triggered
            
        else:
            # Use transformer prediction directly
            final_label = trans_label
            final_conf = trans_conf
            
            explanation_parts.append(
                f"Transformer prediction accepted with high confidence ({trans_conf:.2f}). "
                f"Predicted label: {trans_label}."
            )
            
            # Build rejection reasons for non-selected labels
            if trans_label == "hate":
                rejection_reasons["offensive"] = (
                    f"Hate classification overrides offensive due to severity "
                    f"(confidence: {transformer_probs[2]:.2f} > {transformer_probs[1]:.2f})"
                )
                rejection_reasons["neutral"] = (
                    f"Content shows clear harmful intent with high probability "
                    f"({transformer_probs[2]:.2f})"
                )
            elif trans_label == "offensive":
                rejection_reasons["hate"] = (
                    f"Insufficient evidence for hate classification "
                    f"(hate prob: {transformer_probs[2]:.2f} < threshold)"
                )
                rejection_reasons["neutral"] = (
                    f"Offensive content detected with confidence {transformer_probs[1]:.2f}"
                )
            else:
                rejection_reasons["hate"] = (
                    f"No hate indicators detected (hate prob: {transformer_probs[2]:.2f})"
                )
                rejection_reasons["offensive"] = (
                    f"No offensive content detected (offensive prob: {transformer_probs[1]:.2f})"
                )
            
            harm_tokens = []
            identity_groups = []
            fallback_tier = None
            escalation_triggered = False
        
        return ResolvedDecision(
            label=final_label,
            confidence=final_conf,
            languages_detected=language_proportions,
            fallback_used=fallback_used,
            key_harm_tokens=harm_tokens,
            explanation=" | ".join(explanation_parts),
            transformer_prediction=trans_label,
            transformer_confidence=trans_conf,
            fallback_tier=fallback_tier,
            escalation_triggered=escalation_triggered,
            identity_groups=identity_groups,
            rejection_reasons=rejection_reasons
        )
    
    def _combine_predictions(
        self,
        trans_label: str,
        trans_conf: float,
        fallback_label: str,
        fallback_conf: float,
        escalation: bool,
        harm_tokens: List[Dict],
        rejection_reasons: Dict[str, str]
    ) -> Tuple[str, float, str]:
        """
        Combine transformer and fallback predictions conservatively.
        
        Priority: hate > offensive > neutral
        
        Returns:
            Tuple of (final_label, confidence, explanation)
        """
        # If escalation triggered, fallback overrides
        if escalation:
            rejection_reasons["neutral"] = (
                "Rule-based escalation triggered due to critical content"
            )
            if fallback_label == "hate":
                rejection_reasons["offensive"] = (
                    "Escalated to hate due to dehumanization, violence, or identity targeting"
                )
            
            explanation = (
                f"ESCALATION: Fallback triggered automatic escalation to {fallback_label}. "
                f"Harm tokens: {[t['token'] for t in harm_tokens[:5]]}. "
                f"Transformer prediction ({trans_label}) overridden."
            )
            return fallback_label, fallback_conf, explanation
        
        # Get priority scores
        trans_priority = self.LABEL_PRIORITY.get(trans_label, 1)
        fallback_priority = self.LABEL_PRIORITY.get(fallback_label, 1)
        
        # Conservative combination: take higher severity
        if fallback_priority > trans_priority:
            # Fallback is more severe
            final_label = fallback_label
            final_conf = (fallback_conf * 0.6 + trans_conf * 0.4)
            
            explanation = (
                f"Fallback ({fallback_label}, conf: {fallback_conf:.2f}) "
                f"elevated severity from transformer ({trans_label}). "
                f"Conservative decision toward safety."
            )
            
            # Rejection reasons
            if final_label == "hate":
                rejection_reasons["offensive"] = (
                    f"Fallback analysis elevated to hate based on harm patterns"
                )
                rejection_reasons["neutral"] = (
                    f"Both transformer and fallback indicate harmful content"
                )
            elif final_label == "offensive":
                rejection_reasons["neutral"] = (
                    f"Fallback detected offensive content missed by transformer"
                )
                rejection_reasons["hate"] = (
                    f"Insufficient evidence for hate escalation"
                )
                
        elif trans_priority > fallback_priority:
            # Transformer is more severe - trust it
            final_label = trans_label
            final_conf = trans_conf * 0.8  # Slight reduction due to fallback disagreement
            
            explanation = (
                f"Transformer ({trans_label}, conf: {trans_conf:.2f}) "
                f"more severe than fallback ({fallback_label}). "
                f"Retaining transformer prediction."
            )
            
            if final_label == "hate":
                rejection_reasons["offensive"] = (
                    f"Transformer confidence in hate ({trans_conf:.2f}) is authoritative"
                )
                rejection_reasons["neutral"] = (
                    f"Strong hate indicators in transformer analysis"
                )
            elif final_label == "offensive":
                rejection_reasons["neutral"] = (
                    f"Transformer detected offensive content"
                )
                rejection_reasons["hate"] = (
                    f"Fallback did not confirm hate escalation"
                )
        else:
            # Agreement
            final_label = trans_label
            final_conf = (trans_conf * 0.5 + fallback_conf * 0.5)
            
            explanation = (
                f"Agreement between transformer ({trans_label}) "
                f"and fallback ({fallback_label}). "
                f"Combined confidence: {final_conf:.2f}."
            )
            
            if final_label == "hate":
                rejection_reasons["offensive"] = "Both systems agree on hate classification"
                rejection_reasons["neutral"] = "Strong agreement on harmful content"
            elif final_label == "offensive":
                rejection_reasons["neutral"] = "Both systems agree on offensive content"
                rejection_reasons["hate"] = "Neither system found sufficient hate evidence"
            else:
                rejection_reasons["offensive"] = "Both systems found no offensive content"
                rejection_reasons["hate"] = "Both systems found no hate indicators"
        
        return final_label, final_conf, explanation
    
    def resolve_ambiguity(
        self,
        label_a: str,
        conf_a: float,
        label_b: str,
        conf_b: float,
        harm_tokens: List[Dict]
    ) -> Tuple[str, float, str]:
        """
        Resolve ambiguity between two labels.
        
        Rule: If ambiguity between hate and offensive, choose hate.
        
        Returns:
            Tuple of (chosen_label, confidence, reason)
        """
        # Check for hate vs offensive ambiguity
        labels = {label_a, label_b}
        
        if labels == {"hate", "offensive"}:
            # Ambiguity between hate and offensive
            # RULE: Choose hate, explain why
            
            # Check if there are identity-targeting tokens
            identity_tokens = [t for t in harm_tokens if t.get('is_identity_targeting')]
            
            if identity_tokens:
                reason = (
                    f"Ambiguity between hate and offensive resolved to HATE: "
                    f"Identity-targeting content detected ({[t['token'] for t in identity_tokens[:3]]})"
                )
            else:
                reason = (
                    f"Ambiguity between hate ({conf_a:.2f}) and offensive ({conf_b:.2f}) "
                    f"resolved to HATE per safety-first policy. "
                    f"When uncertain, bias toward safety, not neutrality."
                )
            
            return "hate", max(conf_a, conf_b), reason
        
        # For other ambiguities, choose higher priority
        priority_a = self.LABEL_PRIORITY.get(label_a, 1)
        priority_b = self.LABEL_PRIORITY.get(label_b, 1)
        
        if priority_a >= priority_b:
            return label_a, conf_a, f"Higher severity label chosen: {label_a}"
        else:
            return label_b, conf_b, f"Higher severity label chosen: {label_b}"
