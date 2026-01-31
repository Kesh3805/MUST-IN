"""
MUST++ Explainability Module

Implements Step 7 of the MUST++ Pipeline:
- Text-grounded explanations
- Harm token highlighting
- Label rejection reasoning
- LIME integration for feature importance
"""

from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ExplanationResult:
    """Structured explanation result"""
    label: str
    confidence: float
    harm_tokens: List[str]
    why_label_applies: str
    why_weaker_rejected: Dict[str, str]
    lime_features: List[tuple]  # (word, weight) pairs
    identity_groups: List[str]
    raw_text_evidence: List[str]


class HateSpeechExplainer:
    """
    MUST++ Explainability Engine
    
    Requirements:
    - Every output must include harm-contributing tokens
    - Explain why label applies
    - Explain why weaker labels were rejected
    - Reference actual text evidence
    - Avoid abstract moral language
    - Avoid speculation
    """
    
    LABELS = ['neutral', 'offensive', 'hate']
    LABELS_DISPLAY = ['Neutral', 'Offensive', 'Hate']
    
    def __init__(self, class_names=None):
        """
        Args:
            class_names: List of labels (default: ['Neutral', 'Offensive', 'Hate'])
        """
        if class_names is None:
            class_names = self.LABELS_DISPLAY
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names)

    def explain_traditional(self, pipeline, text_instance, num_features=10):
        """
        Explain a traditional sklearn pipeline prediction using LIME.
        
        Args:
            pipeline: Sklearn pipeline with predict_proba
            text_instance: Text to explain
            num_features: Number of top features to return
            
        Returns:
            LIME explanation object
        """
        exp = self.explainer.explain_instance(
            text_instance, 
            pipeline.predict_proba, 
            num_features=num_features
        )
        return exp

    def explain_transformer(self, model, tokenizer, text_instance, num_features=10):
        """
        Explain a HuggingFace Transformer prediction using LIME.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            text_instance: Text to explain
            num_features: Number of top features
            
        Returns:
            LIME explanation object
        """
        def predictor(texts):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.to(device)
            model.eval()
            
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs

        exp = self.explainer.explain_instance(
            text_instance, 
            predictor, 
            num_features=num_features
        )
        return exp
    
    def generate_must_explanation(
        self,
        text: str,
        label: str,
        confidence: float,
        harm_tokens: List[str],
        identity_groups: List[str],
        fallback_used: bool,
        fallback_tier: Optional[int],
        escalation_triggered: bool,
        transformer_prediction: str,
        transformer_confidence: float
    ) -> ExplanationResult:
        """
        Generate MUST++ compliant explanation.
        
        Requirements:
        - Reference actual text
        - No abstract moral language
        - No speculation beyond evidence
        
        Args:
            text: Original text
            label: Final classification
            confidence: Final confidence
            harm_tokens: Detected harm tokens
            identity_groups: Detected identity groups
            fallback_used: Whether fallback was triggered
            fallback_tier: Which tier was used
            escalation_triggered: Whether rule escalation happened
            transformer_prediction: Primary prediction
            transformer_confidence: Primary confidence
            
        Returns:
            ExplanationResult with structured explanation
        """
        # Build why_label_applies
        why_parts = []
        
        if label == "hate":
            if escalation_triggered:
                why_parts.append(
                    f"ESCALATION TRIGGERED: Rule-based escalation due to critical content."
                )
            if harm_tokens:
                why_parts.append(
                    f"Hate-indicating tokens detected: {harm_tokens[:5]}"
                )
            if identity_groups:
                why_parts.append(
                    f"Content targets identity group(s): {identity_groups[:3]}"
                )
            if not harm_tokens and not identity_groups:
                why_parts.append(
                    f"Transformer classification with confidence {confidence:.2f}"
                )
                
        elif label == "offensive":
            if harm_tokens:
                why_parts.append(
                    f"Offensive language detected: {harm_tokens[:5]}"
                )
            else:
                why_parts.append(
                    f"Offensive patterns in text structure"
                )
            why_parts.append(
                f"No identity-targeting escalation detected"
            )
            
        else:  # neutral
            why_parts.append("No significant harm indicators detected")
            if transformer_confidence > 0.8:
                why_parts.append(
                    f"High confidence neutral classification ({transformer_confidence:.2f})"
                )
        
        why_label_applies = " | ".join(why_parts)
        
        # Build rejection reasons
        why_weaker_rejected = {}
        
        if label == "hate":
            why_weaker_rejected["offensive"] = self._build_rejection_reason(
                "offensive", harm_tokens, identity_groups, "hate"
            )
            why_weaker_rejected["neutral"] = self._build_rejection_reason(
                "neutral", harm_tokens, identity_groups, "hate"
            )
        elif label == "offensive":
            why_weaker_rejected["hate"] = self._build_rejection_reason(
                "hate", harm_tokens, identity_groups, "offensive"
            )
            why_weaker_rejected["neutral"] = self._build_rejection_reason(
                "neutral", harm_tokens, identity_groups, "offensive"
            )
        else:
            why_weaker_rejected["hate"] = self._build_rejection_reason(
                "hate", harm_tokens, identity_groups, "neutral"
            )
            why_weaker_rejected["offensive"] = self._build_rejection_reason(
                "offensive", harm_tokens, identity_groups, "neutral"
            )
        
        # Extract text evidence
        raw_evidence = self._extract_text_evidence(text, harm_tokens)
        
        return ExplanationResult(
            label=label,
            confidence=confidence,
            harm_tokens=harm_tokens,
            why_label_applies=why_label_applies,
            why_weaker_rejected=why_weaker_rejected,
            lime_features=[],  # To be populated by LIME if called
            identity_groups=identity_groups,
            raw_text_evidence=raw_evidence
        )
    
    def _build_rejection_reason(
        self,
        rejected_label: str,
        harm_tokens: List[str],
        identity_groups: List[str],
        chosen_label: str
    ) -> str:
        """Build rejection reason for a label."""
        
        if rejected_label == "hate" and chosen_label == "offensive":
            if not identity_groups:
                return "No identity group targeting detected to escalate to hate"
            return "Harm level insufficient for hate classification"
            
        elif rejected_label == "hate" and chosen_label == "neutral":
            return f"No hate indicators found in text"
            
        elif rejected_label == "offensive" and chosen_label == "hate":
            return f"Severity elevated beyond offensive due to: {harm_tokens[:3] or identity_groups[:2]}"
            
        elif rejected_label == "offensive" and chosen_label == "neutral":
            return "No offensive content markers detected"
            
        elif rejected_label == "neutral" and chosen_label in ["offensive", "hate"]:
            if harm_tokens:
                return f"Cannot be neutral: harm tokens present ({harm_tokens[:3]})"
            return "Content indicators prevent neutral classification"
            
        return "Classification evidence supports chosen label"
    
    def _extract_text_evidence(
        self,
        text: str,
        harm_tokens: List[str]
    ) -> List[str]:
        """Extract relevant evidence snippets from text."""
        evidence = []
        
        # Add harm token contexts
        for token in harm_tokens[:5]:
            # Find token in text with context
            idx = text.lower().find(token.lower())
            if idx >= 0:
                start = max(0, idx - 20)
                end = min(len(text), idx + len(token) + 20)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."
                evidence.append(snippet)
        
        return evidence[:5]
    
    def explain_with_lime(
        self,
        model,
        tokenizer,
        text: str,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation with feature importance.
        
        Args:
            model: Classification model
            tokenizer: Tokenizer (for transformers)
            text: Text to explain
            num_features: Number of features
            
        Returns:
            Dict with LIME explanation data
        """
        exp = self.explain_transformer(model, tokenizer, text, num_features)
        
        # Get prediction
        probs = exp.predict_proba
        predicted_class = int(np.argmax(probs))
        
        # Get feature weights for predicted class
        feature_weights = exp.as_list(label=predicted_class)
        
        # Separate positive and negative contributors
        positive_features = [
            (word, weight) for word, weight in feature_weights if weight > 0
        ]
        negative_features = [
            (word, weight) for word, weight in feature_weights if weight < 0
        ]
        
        return {
            "predicted_label": self.class_names[predicted_class],
            "probabilities": {
                self.class_names[i]: float(probs[i]) 
                for i in range(len(self.class_names))
            },
            "positive_contributors": positive_features,
            "negative_contributors": negative_features,
            "all_features": feature_weights,
            "explanation_html": exp.as_html()
        }
    
    def format_explanation_text(self, result: ExplanationResult) -> str:
        """
        Format explanation as human-readable text.
        
        Args:
            result: ExplanationResult object
            
        Returns:
            Formatted string
        """
        lines = [
            f"CLASSIFICATION: {result.label.upper()}",
            f"Confidence: {result.confidence:.2%}",
            "",
            "WHY THIS LABEL APPLIES:",
            result.why_label_applies,
            "",
            "HARM-CONTRIBUTING TOKENS:",
            ", ".join(result.harm_tokens) if result.harm_tokens else "(none detected)",
            "",
            "IDENTITY GROUPS MENTIONED:",
            ", ".join(result.identity_groups) if result.identity_groups else "(none detected)",
            "",
            "WHY OTHER LABELS WERE REJECTED:"
        ]
        
        for label, reason in result.why_weaker_rejected.items():
            lines.append(f"  - {label}: {reason}")
        
        if result.raw_text_evidence:
            lines.append("")
            lines.append("TEXT EVIDENCE:")
            for evidence in result.raw_text_evidence:
                lines.append(f'  "{evidence}"')
        
        return "\n".join(lines)
