"""
MUST++ Fallback Logic Module

Implements Step 5 of the MUST++ Pipeline:
- Tier 2: Indic Specialist Reasoning
- Tier 3: Classical Safety Reasoning
- Tier 4: Rule-Based Escalation

Fallback is MANDATORY when confidence gate fails.
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import re

from .hate_lexicon import HateLexicon, SeverityLevel, HarmCategory
from .script_detector import ScriptDetector, LanguageProfile, Language


class FallbackTier(Enum):
    """Fallback tier levels"""
    TIER_2_INDIC = 2        # Indic specialist reasoning
    TIER_3_SAFETY = 3       # Classical safety reasoning
    TIER_4_RULE = 4         # Rule-based escalation


class PredictionLabel(Enum):
    """Classification labels"""
    NEUTRAL = "neutral"
    OFFENSIVE = "offensive"
    HATE = "hate"


@dataclass
class FallbackResult:
    """Result from fallback reasoning"""
    tier_used: FallbackTier
    prediction: PredictionLabel
    confidence: float
    harm_tokens: List[Dict]
    reasoning: str
    escalation_triggered: bool
    identity_groups_detected: List[str]


class FallbackManager:
    """
    Manages fallback reasoning paths for uncertain predictions.
    
    Tier Priority:
    - Tier 4 (Rule-Based) is checked FIRST for automatic escalation
    - Tier 2 (Indic) for Tamil/Hindi dominant or Romanized text
    - Tier 3 (Safety) for short/fragmentary text or high uncertainty
    """
    
    def __init__(self):
        self.lexicon = HateLexicon()
        self.script_detector = ScriptDetector()
        
        # Intensity modifiers that amplify harm
        self.intensity_modifiers = [
            'fucking', 'damn', 'bloody', 'fucking', 'goddamn',
            'sala', 'saala', 'besharam', 'nikamma',
            'romba', 'mokka', 'mairu',
        ]
        
        # Dismissive patterns that indicate derogation
        self.dismissive_patterns = [
            r'\b(these|those|all)\s+(people|guys|idiots?|fools?|morons?)',
            r'\b(typical|always|never)\s+\w+\s+(people|community|group)',
            r'\b(go back|get out|leave)\s+(to|from)',
        ]
    
    def execute_fallback(
        self,
        text: str,
        language_profile: LanguageProfile,
        transformer_prediction: Optional[int] = None,
        transformer_confidence: float = 0.0,
        tier: FallbackTier = FallbackTier.TIER_3_SAFETY
    ) -> FallbackResult:
        """
        Execute fallback reasoning for uncertain prediction.
        
        Args:
            text: Input text
            language_profile: Detected language profile
            transformer_prediction: Original transformer prediction (0/1/2)
            transformer_confidence: Transformer confidence score
            tier: Which fallback tier to use
            
        Returns:
            FallbackResult with final decision
        """
        # ALWAYS check Tier 4 first for automatic escalation
        rule_result = self._tier_4_rule_based(text, language_profile)
        if rule_result.escalation_triggered:
            return rule_result
        
        # Execute appropriate tier
        if tier == FallbackTier.TIER_2_INDIC:
            return self._tier_2_indic_specialist(
                text, language_profile, transformer_prediction, transformer_confidence
            )
        elif tier == FallbackTier.TIER_3_SAFETY:
            return self._tier_3_classical_safety(
                text, language_profile, transformer_prediction, transformer_confidence
            )
        else:
            # Default to safety reasoning
            return self._tier_3_classical_safety(
                text, language_profile, transformer_prediction, transformer_confidence
            )
    
    def _tier_4_rule_based(
        self,
        text: str,
        language_profile: LanguageProfile
    ) -> FallbackResult:
        """
        Tier 4: Rule-Based Escalation
        
        IMMEDIATELY classify as hate if text includes:
        - Dehumanization
        - Calls for violence
        - Collective identity targeting with harm intent
        
        Rules OVERRIDE model confidence.
        """
        harm_tokens = self.lexicon.get_harm_tokens(text)
        identity_groups = self.lexicon.check_identity_targeting(text)
        
        # Check for critical content (auto-escalate)
        has_critical, critical_reasons = self.lexicon.has_critical_content(text)
        
        if has_critical:
            return FallbackResult(
                tier_used=FallbackTier.TIER_4_RULE,
                prediction=PredictionLabel.HATE,
                confidence=0.95,
                harm_tokens=harm_tokens,
                reasoning=f"RULE ESCALATION: {'; '.join(critical_reasons)}",
                escalation_triggered=True,
                identity_groups_detected=identity_groups
            )
        
        # Check for violence + identity combination
        violence_patterns = self.lexicon.check_violence(text)
        if violence_patterns and identity_groups:
            return FallbackResult(
                tier_used=FallbackTier.TIER_4_RULE,
                prediction=PredictionLabel.HATE,
                confidence=0.92,
                harm_tokens=harm_tokens,
                reasoning=f"RULE ESCALATION: Violence ({violence_patterns[:2]}) targeting "
                         f"identity group ({identity_groups[:2]})",
                escalation_triggered=True,
                identity_groups_detected=identity_groups
            )
        
        # Check for dehumanization
        dehuman = self.lexicon.check_dehumanization(text)
        if dehuman:
            return FallbackResult(
                tier_used=FallbackTier.TIER_4_RULE,
                prediction=PredictionLabel.HATE,
                confidence=0.90,
                harm_tokens=harm_tokens,
                reasoning=f"RULE ESCALATION: Dehumanization detected ({dehuman[:2]})",
                escalation_triggered=True,
                identity_groups_detected=identity_groups
            )
        
        # No escalation triggered
        return FallbackResult(
            tier_used=FallbackTier.TIER_4_RULE,
            prediction=PredictionLabel.NEUTRAL,  # Placeholder, won't be used
            confidence=0.0,
            harm_tokens=harm_tokens,
            reasoning="No rule-based escalation triggered",
            escalation_triggered=False,
            identity_groups_detected=identity_groups
        )
    
    def _tier_2_indic_specialist(
        self,
        text: str,
        language_profile: LanguageProfile,
        transformer_prediction: Optional[int],
        transformer_confidence: float
    ) -> FallbackResult:
        """
        Tier 2: Indic Specialist Reasoning
        
        Use when:
        - Tamil or Hindi dominance
        - Heavy slang or Romanization
        
        Focus:
        - Cultural insults
        - Caste, religion, gender targeting
        """
        harm_tokens = self.lexicon.get_harm_tokens(text)
        identity_groups = self.lexicon.check_identity_targeting(text)
        
        # Determine primary language for focused analysis
        primary_lang = language_profile.primary_language.value.lower()
        
        # Analyze harm patterns
        severity = self.lexicon.get_max_severity(text)
        
        # Check for caste-based discrimination
        caste_tokens = [
            t for t in harm_tokens 
            if t.get('category') == HarmCategory.CASTE.value
        ]
        
        # Check for religious targeting
        religious_tokens = [
            t for t in harm_tokens 
            if t.get('category') == HarmCategory.RELIGIOUS.value
        ]
        
        # Check for gender-based abuse
        gender_tokens = [
            t for t in harm_tokens 
            if t.get('category') == HarmCategory.GENDER.value
        ]
        
        # Decision logic
        reasoning_parts = []
        
        # Caste-based = auto HATE
        if caste_tokens:
            reasoning_parts.append(
                f"Caste-based discrimination: {[t['token'] for t in caste_tokens[:3]]}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_2_INDIC,
                prediction=PredictionLabel.HATE,
                confidence=0.90,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=True,
                identity_groups_detected=identity_groups
            )
        
        # Religious + slur = HATE
        if religious_tokens and severity.value >= SeverityLevel.HIGH.value:
            reasoning_parts.append(
                f"Religious targeting with severe language: "
                f"{[t['token'] for t in religious_tokens[:3]]}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_2_INDIC,
                prediction=PredictionLabel.HATE,
                confidence=0.88,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=True,
                identity_groups_detected=identity_groups
            )
        
        # Multiple severe slurs = at least OFFENSIVE
        high_severity_tokens = [
            t for t in harm_tokens 
            if t.get('severity', 0) >= SeverityLevel.HIGH.value
        ]
        
        if len(high_severity_tokens) >= 2:
            if identity_groups:
                reasoning_parts.append(
                    f"Multiple severe slurs targeting identity: "
                    f"{[t['token'] for t in high_severity_tokens[:3]]}"
                )
                return FallbackResult(
                    tier_used=FallbackTier.TIER_2_INDIC,
                    prediction=PredictionLabel.HATE,
                    confidence=0.85,
                    harm_tokens=harm_tokens,
                    reasoning="; ".join(reasoning_parts),
                    escalation_triggered=True,
                    identity_groups_detected=identity_groups
                )
            else:
                reasoning_parts.append(
                    f"Multiple severe slurs: {[t['token'] for t in high_severity_tokens[:3]]}"
                )
                return FallbackResult(
                    tier_used=FallbackTier.TIER_2_INDIC,
                    prediction=PredictionLabel.OFFENSIVE,
                    confidence=0.82,
                    harm_tokens=harm_tokens,
                    reasoning="; ".join(reasoning_parts),
                    escalation_triggered=False,
                    identity_groups_detected=identity_groups
                )
        
        # Single slur = OFFENSIVE
        if harm_tokens:
            reasoning_parts.append(
                f"Indic slur detected: {[t['token'] for t in harm_tokens[:3]]}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_2_INDIC,
                prediction=PredictionLabel.OFFENSIVE,
                confidence=0.78,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=False,
                identity_groups_detected=identity_groups
            )
        
        # Combine with transformer if available
        if transformer_prediction is not None:
            labels = [PredictionLabel.NEUTRAL, PredictionLabel.OFFENSIVE, PredictionLabel.HATE]
            trans_label = labels[transformer_prediction] if transformer_prediction < 3 else PredictionLabel.NEUTRAL
            
            reasoning_parts.append(
                f"No Indic-specific harm detected; deferring to transformer: {trans_label.value}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_2_INDIC,
                prediction=trans_label,
                confidence=transformer_confidence * 0.9,  # Slight reduction
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=False,
                identity_groups_detected=identity_groups
            )
        
        # Default neutral
        return FallbackResult(
            tier_used=FallbackTier.TIER_2_INDIC,
            prediction=PredictionLabel.NEUTRAL,
            confidence=0.60,
            harm_tokens=harm_tokens,
            reasoning="No significant harm indicators in Indic analysis",
            escalation_triggered=False,
            identity_groups_detected=identity_groups
        )
    
    def _tier_3_classical_safety(
        self,
        text: str,
        language_profile: LanguageProfile,
        transformer_prediction: Optional[int],
        transformer_confidence: float
    ) -> FallbackResult:
        """
        Tier 3: Classical Safety Reasoning
        
        Use when:
        - Text is short
        - Fragmentary abuse
        - Transformer uncertainty is high
        
        Focus:
        - Hate lexicon presence
        - Identity-based slurs
        """
        harm_tokens = self.lexicon.get_harm_tokens(text)
        identity_groups = self.lexicon.check_identity_targeting(text)
        severity = self.lexicon.get_max_severity(text)
        
        reasoning_parts = []
        
        # Check intensity modifiers
        has_intensity = any(
            mod in text.lower() for mod in self.intensity_modifiers
        )
        
        # Check dismissive patterns
        has_dismissive = any(
            re.search(pattern, text, re.IGNORECASE) 
            for pattern in self.dismissive_patterns
        )
        
        # Decision logic based on lexicon and patterns
        
        # Critical severity = HATE
        if severity == SeverityLevel.CRITICAL:
            reasoning_parts.append(
                f"Critical severity lexicon match: "
                f"{[t['token'] for t in harm_tokens if t.get('severity', 0) == 4][:3]}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.HATE,
                confidence=0.88,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=True,
                identity_groups_detected=identity_groups
            )
        
        # High severity + identity = HATE
        if severity == SeverityLevel.HIGH and identity_groups:
            reasoning_parts.append(
                f"High severity targeting identity group: {identity_groups[:2]}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.HATE,
                confidence=0.85,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=True,
                identity_groups_detected=identity_groups
            )
        
        # Identity-targeting slurs = HATE
        identity_slurs = [
            t for t in harm_tokens if t.get('is_identity_targeting', False)
        ]
        if identity_slurs:
            reasoning_parts.append(
                f"Identity-targeting slurs: {[t['token'] for t in identity_slurs[:3]]}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.HATE,
                confidence=0.83,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=True,
                identity_groups_detected=identity_groups
            )
        
        # High/medium severity = OFFENSIVE
        if severity.value >= SeverityLevel.MEDIUM.value:
            if has_intensity:
                reasoning_parts.append("Intensity modifiers amplify harm")
            reasoning_parts.append(
                f"Lexicon severity {severity.value}: "
                f"{[t['token'] for t in harm_tokens[:3]]}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.OFFENSIVE,
                confidence=0.80,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=False,
                identity_groups_detected=identity_groups
            )
        
        # Low severity = OFFENSIVE (bias toward safety)
        if harm_tokens:
            reasoning_parts.append(
                f"Low severity markers: {[t['token'] for t in harm_tokens[:3]]}"
            )
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.OFFENSIVE,
                confidence=0.65,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=False,
                identity_groups_detected=identity_groups
            )
        
        # Check for English insults if no lexicon matches
        english_insults = self.lexicon.check_english_insults(text)
        if english_insults:
            insult_tokens = [match for match, sev in english_insults]
            reasoning_parts.append(
                f"English insults detected: {insult_tokens[:3]}"
            )
            # Add to harm tokens for output
            for match, sev in english_insults:
                harm_tokens.append({
                    'token': match,
                    'original': match,
                    'category': 'SLUR',
                    'severity': sev.value,
                    'language': 'English',
                    'is_identity_targeting': False
                })
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.OFFENSIVE,
                confidence=0.70,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=False,
                identity_groups_detected=identity_groups
            )
        
        # Check for threat emojis
        threat_emojis = self.lexicon.check_threat_emojis(text)
        if threat_emojis:
            reasoning_parts.append(f"Threat emojis detected: {threat_emojis[:3]}")
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.OFFENSIVE,
                confidence=0.72,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=False,
                identity_groups_detected=identity_groups
            )
        
        # Check for mockery emojis with identity groups
        mockery_emojis = self.lexicon.check_mockery_emojis(text)
        if mockery_emojis and identity_groups:
            reasoning_parts.append(f"Mockery emojis {mockery_emojis[:3]} targeting {identity_groups[:2]}")
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.OFFENSIVE,
                confidence=0.75,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=False,
                identity_groups_detected=identity_groups
            )
        
        # Dismissive patterns without slurs
        if has_dismissive:
            reasoning_parts.append("Dismissive/derogatory patterns detected")
            return FallbackResult(
                tier_used=FallbackTier.TIER_3_SAFETY,
                prediction=PredictionLabel.OFFENSIVE,
                confidence=0.60,
                harm_tokens=harm_tokens,
                reasoning="; ".join(reasoning_parts),
                escalation_triggered=False,
                identity_groups_detected=identity_groups
            )
        
        # No harm detected
        # If transformer was uncertain, bias toward offensive over neutral (SAFETY BIAS)
        if transformer_prediction is not None and transformer_confidence < 0.5:
            # When very uncertain, check text length
            word_count = len(text.split())
            if word_count < 5:
                reasoning_parts.append(
                    "Short text with low transformer confidence; "
                    "insufficient evidence for harm classification"
                )
                return FallbackResult(
                    tier_used=FallbackTier.TIER_3_SAFETY,
                    prediction=PredictionLabel.NEUTRAL,
                    confidence=0.55,
                    harm_tokens=harm_tokens,
                    reasoning="; ".join(reasoning_parts),
                    escalation_triggered=False,
                    identity_groups_detected=identity_groups
                )
        
        return FallbackResult(
            tier_used=FallbackTier.TIER_3_SAFETY,
            prediction=PredictionLabel.NEUTRAL,
            confidence=0.65,
            harm_tokens=harm_tokens,
            reasoning="No significant harm indicators in safety analysis",
            escalation_triggered=False,
            identity_groups_detected=identity_groups
        )
    
    def get_tier_for_context(
        self,
        language_profile: LanguageProfile,
        is_romanized: bool,
        is_short_text: bool
    ) -> FallbackTier:
        """
        Determine appropriate fallback tier based on context.
        
        Args:
            language_profile: Detected language profile
            is_romanized: Whether text is Romanized
            is_short_text: Whether text is very short
            
        Returns:
            Appropriate FallbackTier
        """
        # Indic dominant or Romanized = Tier 2
        if language_profile.primary_language in [Language.TAMIL, Language.HINDI]:
            return FallbackTier.TIER_2_INDIC
        
        if is_romanized:
            return FallbackTier.TIER_2_INDIC
        
        # Short or uncertain = Tier 3
        if is_short_text:
            return FallbackTier.TIER_3_SAFETY
        
        # Default to safety
        return FallbackTier.TIER_3_SAFETY
