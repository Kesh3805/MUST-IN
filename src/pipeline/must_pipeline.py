"""
MUST++ Main Pipeline Orchestrator

Complete 7-step pipeline for multilingual hate speech detection:
1. Language + Script Detection
2. Loss-Aware Normalization
3. Primary Classification (Transformer)
4. Confidence Gate
5. Fallback Logic
6. Final Decision Resolution
7. Explainability

This is the main entry point for the MUST++ system.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import json
import re
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MUST++")

# Torch imports with graceful degradation
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Running in fallback-only mode.")

# Transformers imports with graceful degradation
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Running in fallback-only mode.")

from .script_detector import ScriptDetector, LanguageProfile, Language, ScriptType
from .confidence_gate import ConfidenceGate, ConfidenceMetrics, GateDecision
from .fallback_logic import FallbackManager, FallbackResult, FallbackTier, PredictionLabel
from .decision_resolver import DecisionResolver, ResolvedDecision
from .hate_lexicon import HateLexicon


@dataclass
class PipelineConfig:
    """Runtime configuration for MUST++ pipeline."""
    disable_transformer: bool = False       # Force fallback-only mode
    force_safety_mode: bool = False         # Always use maximum safety
    confidence_threshold: float = 0.75      # Confidence gate threshold
    entropy_threshold: float = 0.5          # Entropy threshold
    model_timeout_seconds: float = 30.0     # Model loading timeout
    enable_logging: bool = True             # Structured logging
    log_level: str = "INFO"                 # Logging level


@dataclass
class PipelineLog:
    """Structured log entry for a classification."""
    timestamp: str
    input_text_hash: str
    language_profile: Dict
    transformer_available: bool
    transformer_prediction: Optional[str]
    transformer_confidence: Optional[float]
    confidence_gate_decision: str
    fallback_tier_used: Optional[int]
    escalation_triggered: bool
    final_label: str
    final_confidence: float
    processing_time_ms: float


@dataclass
class MUSTPlusOutput:
    """
    Structured output from MUST++ pipeline.
    
    ALL FIELDS ARE MANDATORY - NO OPTIONAL FIELDS.
    
    Matches required output format:
    - Label: neutral | offensive | hate
    - Confidence: 0.0 – 1.0
    - Languages Detected: language proportions
    - Fallback Used: true | false
    - Escalation Triggered: true | false
    - Key Harm Tokens: list
    - Explanation: text-grounded justification
    """
    label: str
    confidence: float
    languages_detected: Dict[str, float]
    fallback_used: bool
    escalation_triggered: bool              # MANDATORY field
    key_harm_tokens: List[str]
    explanation: str
    
    # Extended metadata (all mandatory)
    script_distribution: Dict[str, float]
    is_code_mixed: bool
    transformer_prediction: str
    transformer_confidence: float
    fallback_tier: Optional[int]
    identity_groups_detected: List[str]
    rejection_reasons: Dict[str, str]
    entropy: float
    tokenization_coverage: float
    degraded_mode: bool = False             # True if transformer unavailable
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def validate(self) -> bool:
        """Validate all required fields are present and valid."""
        required_fields = [
            'label', 'confidence', 'languages_detected', 'fallback_used',
            'escalation_triggered', 'key_harm_tokens', 'explanation'
        ]
        for field_name in required_fields:
            if getattr(self, field_name, None) is None:
                raise ValueError(f"Missing required field: {field_name}")
        
        if self.label not in ['neutral', 'offensive', 'hate']:
            raise ValueError(f"Invalid label: {self.label}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        
        return True
    
    def __str__(self) -> str:
        """Human-readable string format."""
        lines = [
            f"Label: {self.label}",
            f"Confidence: {self.confidence:.2f}",
            f"Languages Detected: {self.languages_detected}",
            f"Fallback Used: {self.fallback_used}",
            f"Escalation Triggered: {self.escalation_triggered}",
            f"Key Harm Tokens: {self.key_harm_tokens}",
            f"Explanation: {self.explanation}"
        ]
        return "\n".join(lines)


class MUSTPlusPipeline:
    """
    MUST++ Multilingual Hate Speech Detection Pipeline.
    
    A linguistic firewall that:
    - Inspects text for multilingual hate speech
    - Routes through confidence-gated classification
    - Verifies with fallback reasoning
    - Explains decisions with text evidence
    
    Supported Languages:
    - Tamil (Native + Romanized/Tanglish)
    - Hindi (Native + Romanized/Hinglish)
    - English
    
    Labels:
    - neutral: No harmful content
    - offensive: Inappropriate/vulgar but not hate
    - hate: Targeted hate speech
    """
    
    LABELS = ["neutral", "offensive", "hate"]
    
    # Recommended multilingual models in order of preference
    SUPPORTED_MODELS = [
        "xlm-roberta-base",
        "xlm-roberta-large", 
        "bert-base-multilingual-cased",
        "microsoft/mdeberta-v3-base",
        "cardiffnlp/twitter-xlm-roberta-base"
    ]
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        confidence_threshold: float = 0.75,
        entropy_threshold: float = 0.5,
        device: str = None,
        config: PipelineConfig = None
    ):
        """
        Initialize MUST++ Pipeline.
        
        Args:
            model_name: HuggingFace model name for primary classifier
                        Recommended: xlm-roberta-base or bert-base-multilingual-cased
            confidence_threshold: Minimum confidence to accept transformer result
            entropy_threshold: Maximum entropy for low-uncertainty acceptance
            device: 'cuda', 'cpu', or None for auto-detect
            config: Runtime configuration options
        """
        # Handle case where PipelineConfig is passed as first positional argument
        if isinstance(model_name, PipelineConfig):
            config = model_name
            model_name = "bert-base-multilingual-cased"
        
        # Runtime config
        self.config = config or PipelineConfig()
        
        # Configure logging
        if self.config.enable_logging:
            logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Device setup (with fallback for no torch)
        if TORCH_AVAILABLE:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
        else:
            self.device = None
            logger.warning("Running without PyTorch - Tier 3+4 fallback only")
        
        # Initialize components
        self.script_detector = ScriptDetector()
        self.confidence_gate = ConfidenceGate(
            confidence_threshold=confidence_threshold,
            entropy_threshold=entropy_threshold
        )
        self.fallback_manager = FallbackManager()
        self.decision_resolver = DecisionResolver()
        self.lexicon = HateLexicon()
        
        # Transformer model (lazy loading with graceful degradation)
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        self._model_load_failed = False
        self._model_load_error = None
        
        # Degraded mode tracking
        self.degraded_mode = False
        
        # Pipeline logs
        self.logs: List[PipelineLog] = []
        
        # Normalization config
        self.emoji_intent_map = self._build_emoji_intent_map()
        
        logger.info(f"MUST++ Pipeline initialized. Device: {self.device}, Model: {model_name}")
    
    def _build_emoji_intent_map(self) -> Dict[str, str]:
        """Build emoji to intent token mapping."""
        return {
            # Mockery
            "clown_face": "INTENT_MOCKERY",
            "rolling_on_the_floor_laughing": "INTENT_MOCKERY",
            "face_with_tears_of_joy": "INTENT_MOCKERY",
            "smirking_face": "INTENT_MOCKERY",
            
            # Anger/Threat
            "enraged_face": "INTENT_ANGER",
            "face_with_symbols_on_mouth": "INTENT_ANGER",
            "angry_face": "INTENT_ANGER",
            "pouting_face": "INTENT_ANGER",
            "skull": "INTENT_THREAT",
            "skull_and_crossbones": "INTENT_THREAT",
            "dagger": "INTENT_THREAT",
            "kitchen_knife": "INTENT_THREAT",
            "bomb": "INTENT_THREAT",
            "fire": "INTENT_THREAT",
            "collision": "INTENT_THREAT",
            
            # Sarcasm
            "upside_down_face": "INTENT_SARCASM",
            "winking_face": "INTENT_SARCASM",
            "zipper_mouth_face": "INTENT_SARCASM",
            
            # Disgust
            "nauseated_face": "INTENT_DISGUST",
            "face_vomiting": "INTENT_DISGUST",
            "pile_of_poo": "INTENT_DISGUST",
            "middle_finger": "INTENT_DISGUST",
            
            # General negative
            "thumbs_down": "INTENT_NEGATIVE",
            "cross_mark": "INTENT_NEGATIVE",
        }
    
    def _load_model(self) -> bool:
        """
        Lazy load transformer model with graceful degradation.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        # Already failed before - don't retry
        if self._model_load_failed:
            return False
        
        # Already loaded successfully
        if self._model_loaded:
            return True
        
        # Force fallback mode
        if self.config.disable_transformer:
            logger.info("Transformer disabled via config - using fallback mode")
            self.degraded_mode = True
            return False
        
        # Check if transformers are available
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("Transformers/PyTorch not available - using fallback mode")
            self.degraded_mode = True
            self._model_load_failed = True
            return False
        
        try:
            logger.info(f"Loading transformer model: {self.model_name}")
            start_time = time.time()
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True  # Use fast tokenizer for better performance
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=3
            )
            self._model.to(self.device)
            self._model.eval()
            self._model_loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            # Verify tokenization works for Romanized text
            test_tokens = self._tokenizer.tokenize("kaṭua chutiya यह हिंदी है")
            logger.debug(f"Tokenization test: {test_tokens[:10]}...")
            
            return True
            
        except Exception as e:
            self._model_load_failed = True
            self._model_load_error = str(e)
            self.degraded_mode = True
            logger.error(f"Failed to load model: {e}. Using fallback mode.")
            return False
    
    def is_transformer_available(self) -> bool:
        """Check if transformer is available for classification."""
        if self._model_load_failed or self.config.disable_transformer:
            return False
        if self._model_loaded:
            return True
        # Try to load
        return self._load_model()
    
    def set_model(self, model, tokenizer):
        """
        Set a pre-trained model for the pipeline.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
        """
        self._model = model
        self._tokenizer = tokenizer
        self._model.to(self.device)
        self._model.eval()
        self._model_loaded = True
    
    # =========================================
    # STEP 1: Language + Script Detection
    # =========================================
    
    def detect_language(self, text: str) -> LanguageProfile:
        """
        Step 1: Detect language and script profile.
        
        Uses token mass weighting, not token count.
        Retains secondary languages if >= 20% mass.
        
        Args:
            text: Input text
            
        Returns:
            LanguageProfile with all detection metadata
        """
        return self.script_detector.detect_language_profile(text)
    
    # =========================================
    # STEP 2: Loss-Aware Normalization
    # =========================================
    
    def normalize_text(
        self, 
        text: str, 
        language_profile: LanguageProfile
    ) -> Tuple[str, List[str]]:
        """
        Step 2: Loss-aware normalization.
        
        PRESERVES:
        - Slurs and abusive terms
        - Cultural/caste/religious markers
        - Romanized slang (soft-normalized, not translated)
        
        REMOVES:
        - URLs, mentions
        - Pure noise
        
        CONVERTS:
        - Emojis to semantic intent tokens
        
        Args:
            text: Input text
            language_profile: Detected language profile
            
        Returns:
            Tuple of (normalized_text, preserved_terms)
        """
        preserved_terms = []
        
        # 1. Preserve harm tokens BEFORE normalization
        harm_tokens = self.lexicon.scan_text(text)
        preserved_terms = [token for token, _ in harm_tokens]
        
        # 2. Remove URLs and mentions (pure noise)
        normalized = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        normalized = re.sub(r'@\w+', '', normalized)
        
        # 3. Convert emojis to intent tokens (NOT remove)
        # First demojize to text
        try:
            import emoji
            normalized = emoji.demojize(normalized, delimiters=(" ", " "))
        except ImportError:
            pass
        
        # Map emoji names to intent tokens
        for emoji_name, intent in self.emoji_intent_map.items():
            normalized = normalized.replace(emoji_name, intent)
        
        # 4. Normalize repeated characters (but preserve meaning)
        # e.g., "haaaaaate" -> "haate" (not "hate")
        normalized = re.sub(r'(.)\1{3,}', r'\1\1', normalized)
        
        # 5. Clean extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # 6. DO NOT lowercase (preserve casing for emphasis)
        # 7. DO NOT translate Romanized to English
        # 8. DO NOT remove profanity
        
        return normalized, preserved_terms
    
    # =========================================
    # STEP 3: Primary Classification
    # =========================================
    
    def classify_transformer(
        self, 
        text: str,
        language_profile: LanguageProfile
    ) -> Tuple[Optional[np.ndarray], Optional[int], float, List[str], bool]:
        """
        Step 3: Primary transformer classification with graceful degradation.
        
        Args:
            text: Normalized text
            language_profile: Language profile for context
            
        Returns:
            Tuple of (probabilities, prediction, entropy, tokens, success)
            If transformer unavailable, returns (None, None, 0.0, [], False)
        """
        # Try to load model - graceful failure
        if not self._load_model():
            logger.warning("Transformer unavailable - returning empty classification")
            return None, None, 0.0, [], False
        
        try:
            # Prepare input with language/script tags
            tagged_text = self._prepare_tagged_input(text, language_profile)
            
            # Tokenize
            inputs = self._tokenizer(
                tagged_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get tokens for coverage calculation
            tokens = self._tokenizer.tokenize(tagged_text)
            
            # Verify tokenization coverage for Romanized text
            unk_count = sum(1 for t in tokens if t in ['[UNK]', '<unk>'])
            if len(tokens) > 0:
                coverage = 1.0 - (unk_count / len(tokens))
                if coverage < 0.5:
                    logger.warning(f"Low tokenization coverage ({coverage:.2f}) for text: {text[:50]}...")
            
            # Inference with error handling
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            prediction = int(np.argmax(probs))
            entropy = self._calculate_entropy(probs)
            
            logger.debug(f"Transformer: {self.LABELS[prediction]} ({probs[prediction]:.3f})")
            
            return probs, prediction, entropy, tokens, True
            
        except Exception as e:
            logger.error(f"Transformer inference failed: {e}")
            self.degraded_mode = True
            return None, None, 0.0, [], False
    
    def _prepare_tagged_input(
        self, 
        text: str, 
        language_profile: LanguageProfile
    ) -> str:
        """
        Prepare input with language and script tags.
        
        Format: [LANG:Hindi] [SCRIPT:ROMAN] <text>
        """
        tags = []
        
        # Primary language tag
        primary = language_profile.primary_language.value
        tags.append(f"[LANG:{primary}]")
        
        # Script distribution tag
        dominant_script = max(
            language_profile.script_distribution.items(),
            key=lambda x: x[1],
            default=("SCRIPT_ENGLISH", 1.0)
        )[0]
        tags.append(f"[SCRIPT:{dominant_script}]")
        
        # Code-mixed indicator
        if language_profile.is_code_mixed:
            tags.append("[CODEMIX]")
        
        return " ".join(tags) + " " + text
    
    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log2(probs)))
    
    # =========================================
    # STEP 4: Confidence Gate
    # =========================================
    
    def evaluate_confidence(
        self,
        probs: np.ndarray,
        tokens: List[str],
        language_profile: LanguageProfile,
        text: str
    ) -> ConfidenceMetrics:
        """
        Step 4: Evaluate prediction confidence.
        
        Args:
            probs: Model probabilities
            tokens: Tokenized input
            language_profile: Language profile
            text: Original text
            
        Returns:
            ConfidenceMetrics with gate decision
        """
        # Determine context flags
        is_romanized = self.script_detector.is_romanized_dominant(text)
        is_indic = language_profile.primary_language in [Language.TAMIL, Language.HINDI]
        is_short = len(text.split()) < 5
        
        return self.confidence_gate.evaluate(
            probabilities=probs,
            tokens=tokens,
            is_romanized=is_romanized,
            is_indic_dominant=is_indic,
            is_short_text=is_short
        )
    
    # =========================================
    # STEP 5: Fallback Logic
    # =========================================
    
    def execute_fallback(
        self,
        text: str,
        language_profile: LanguageProfile,
        transformer_prediction: int,
        transformer_confidence: float,
        confidence_metrics: ConfidenceMetrics
    ) -> FallbackResult:
        """
        Step 5: Execute fallback reasoning.
        
        MANDATORY when confidence gate fails.
        
        Args:
            text: Original text
            language_profile: Language profile
            transformer_prediction: Primary prediction
            transformer_confidence: Primary confidence
            confidence_metrics: Gate evaluation
            
        Returns:
            FallbackResult with fallback decision
        """
        # Determine appropriate tier
        is_romanized = self.script_detector.is_romanized_dominant(text)
        is_short = len(text.split()) < 5
        
        tier = self.fallback_manager.get_tier_for_context(
            language_profile=language_profile,
            is_romanized=is_romanized,
            is_short_text=is_short
        )
        
        return self.fallback_manager.execute_fallback(
            text=text,
            language_profile=language_profile,
            transformer_prediction=transformer_prediction,
            transformer_confidence=transformer_confidence,
            tier=tier
        )
    
    # =========================================
    # STEP 6: Final Decision Resolution
    # =========================================
    
    def resolve_decision(
        self,
        probs: np.ndarray,
        prediction: int,
        confidence_metrics: ConfidenceMetrics,
        fallback_result: Optional[FallbackResult],
        language_profile: LanguageProfile,
        text: str
    ) -> ResolvedDecision:
        """
        Step 6: Resolve final classification decision.
        
        Priority: hate > offensive > neutral
        
        Args:
            probs: Transformer probabilities
            prediction: Transformer prediction
            confidence_metrics: Confidence evaluation
            fallback_result: Fallback result (if triggered)
            language_profile: Language profile
            text: Original text
            
        Returns:
            ResolvedDecision with final label
        """
        return self.decision_resolver.resolve(
            transformer_probs=probs.tolist(),
            transformer_prediction=prediction,
            confidence_metrics=confidence_metrics,
            fallback_result=fallback_result,
            language_proportions=language_profile.language_proportions,
            original_text=text
        )
    
    # =========================================
    # STEP 7: Explainability
    # =========================================
    
    def generate_explanation(
        self,
        resolved: ResolvedDecision,
        language_profile: LanguageProfile,
        text: str
    ) -> str:
        """
        Step 7: Generate text-grounded explanation.
        
        Requirements:
        - Reference actual text
        - Explain why label applies
        - Explain why weaker labels rejected
        - Avoid abstract moral language
        - Avoid speculation
        
        Args:
            resolved: Resolved decision
            language_profile: Language profile
            text: Original text
            
        Returns:
            Structured explanation string
        """
        parts = []
        
        # 1. Primary classification reason
        parts.append(f"CLASSIFICATION: {resolved.label.upper()}")
        parts.append(f"Reason: {resolved.explanation}")
        
        # 2. Evidence from text
        if resolved.key_harm_tokens:
            parts.append(f"Evidence tokens: {resolved.key_harm_tokens[:5]}")
        
        if resolved.identity_groups:
            parts.append(f"Identity groups mentioned: {resolved.identity_groups[:3]}")
        
        # 3. Rejection reasons for other labels
        parts.append("Label rejection reasons:")
        for label, reason in resolved.rejection_reasons.items():
            parts.append(f"  - {label}: {reason}")
        
        # 4. Pipeline path
        if resolved.fallback_used:
            parts.append(f"Fallback tier {resolved.fallback_tier} was triggered.")
            if resolved.escalation_triggered:
                parts.append("ESCALATION: Rule-based escalation applied.")
        else:
            parts.append("High-confidence transformer prediction accepted.")
        
        return " | ".join(parts)
    
    # =========================================
    # MAIN INFERENCE METHOD
    # =========================================
    
    def classify(self, text: str) -> MUSTPlusOutput:
        """
        Main classification method.
        
        Executes the full 7-step MUST++ pipeline.
        
        Guarantees:
        - ALL output fields will be populated (no silent failures)
        - Graceful degradation to fallback if transformer unavailable
        - Structured logging if enabled
        
        Args:
            text: Input text to classify
            
        Returns:
            MUSTPlusOutput with structured result
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return self._empty_input_response()
        
        # Step 1: Language Detection
        language_profile = self.detect_language(text)
        
        # Step 2: Loss-Aware Normalization
        normalized_text, preserved_terms = self.normalize_text(text, language_profile)
        
        # Step 3: Primary Classification (with graceful degradation)
        probs, prediction, entropy, tokens, transformer_success = self.classify_transformer(
            normalized_text, language_profile
        )
        
        # Force safety mode - bypass transformer result
        if self.config.force_safety_mode:
            logger.info("Force safety mode enabled - using conservative path")
            transformer_success = False
            probs = None
            prediction = None
        
        # Handle transformer unavailable - force fallback path
        if not transformer_success:
            logger.info("Transformer unavailable - forcing Tier 3+4 fallback")
            return self._classify_fallback_only(
                text=text,
                normalized_text=normalized_text,
                language_profile=language_profile,
                preserved_terms=preserved_terms,
                start_time=start_time
            )
        
        # Step 4: Confidence Gate
        confidence_metrics = self.evaluate_confidence(
            probs, tokens, language_profile, text
        )
        
        # Step 5: Fallback Logic (if needed)
        fallback_result = None
        if self.confidence_gate.should_trigger_fallback(confidence_metrics):
            fallback_result = self.execute_fallback(
                text=text,
                language_profile=language_profile,
                transformer_prediction=prediction,
                transformer_confidence=confidence_metrics.max_probability,
                confidence_metrics=confidence_metrics
            )
        
        # Step 6: Decision Resolution
        resolved = self.resolve_decision(
            probs=probs,
            prediction=prediction,
            confidence_metrics=confidence_metrics,
            fallback_result=fallback_result,
            language_profile=language_profile,
            text=text
        )
        
        # Step 7: Explanation
        explanation = self.generate_explanation(resolved, language_profile, text)
        
        # Build output
        output = MUSTPlusOutput(
            label=resolved.label,
            confidence=resolved.confidence,
            languages_detected=resolved.languages_detected,
            fallback_used=resolved.fallback_used,
            escalation_triggered=resolved.escalation_triggered,
            key_harm_tokens=resolved.key_harm_tokens,
            explanation=explanation,
            script_distribution=language_profile.script_distribution,
            is_code_mixed=language_profile.is_code_mixed,
            transformer_prediction=resolved.transformer_prediction,
            transformer_confidence=resolved.transformer_confidence,
            fallback_tier=resolved.fallback_tier,
            identity_groups_detected=resolved.identity_groups,
            rejection_reasons=resolved.rejection_reasons,
            entropy=confidence_metrics.entropy,
            tokenization_coverage=confidence_metrics.tokenization_coverage,
            degraded_mode=self.degraded_mode
        )
        
        # Validate output contract
        output.validate()
        
        # Log if enabled
        if self.config.enable_logging:
            self._log_classification(
                text=text,
                language_profile=language_profile,
                output=output,
                start_time=start_time
            )
        
        return output
    
    def _empty_input_response(self) -> MUSTPlusOutput:
        """Return response for empty input."""
        return MUSTPlusOutput(
            label="neutral",
            confidence=1.0,
            languages_detected={"Unknown": 1.0},
            fallback_used=False,
            escalation_triggered=False,
            key_harm_tokens=[],
            explanation="Empty or whitespace-only input",
            script_distribution={"SCRIPT_ENGLISH": 1.0},
            is_code_mixed=False,
            transformer_prediction="neutral",
            transformer_confidence=1.0,
            fallback_tier=None,
            identity_groups_detected=[],
            rejection_reasons={},
            entropy=0.0,
            tokenization_coverage=1.0,
            degraded_mode=self.degraded_mode
        )
    
    def _classify_fallback_only(
        self,
        text: str,
        normalized_text: str,
        language_profile: LanguageProfile,
        preserved_terms: List[str],
        start_time: float
    ) -> MUSTPlusOutput:
        """
        Classify using fallback only (degraded mode).
        
        Used when transformer is unavailable or force_safety_mode is enabled.
        Forces Tier 3 + Tier 4 pipeline.
        """
        # Force Tier 3 + Tier 4 fallback
        fallback_result = self.fallback_manager.execute_fallback(
            text=text,
            language_profile=language_profile,
            transformer_prediction=1,  # Default to offensive for safety
            transformer_confidence=0.0,
            tier=FallbackTier.TIER_3_SAFETY
        )
        
        # Get harm tokens
        harm_tokens = [token for token, _ in self.lexicon.scan_text(text)]
        harm_tokens.extend(preserved_terms)
        
        # Get identity groups
        identity_groups = self.lexicon.check_identity_targeting(text)
        
        # Build explanation
        explanation_parts = ["[DEGRADED MODE - Transformer unavailable]"]
        explanation_parts.append(f"Classification via Tier {fallback_result.tier_used.value} fallback.")
        if harm_tokens:
            explanation_parts.append(f"Harm tokens detected: {harm_tokens[:5]}")
        if fallback_result.escalation_triggered:
            explanation_parts.append("ESCALATION triggered by rule-based safety checks.")
        
        # Build rejection reasons based on final label
        rejection_reasons = {"transformer": "Model unavailable or disabled"}
        final_label = fallback_result.prediction.value
        if final_label == "hate":
            rejection_reasons["neutral"] = "Harm tokens and/or escalation triggers detected"
            rejection_reasons["offensive"] = "Critical severity or identity-targeting content detected"
        elif final_label == "offensive":
            rejection_reasons["neutral"] = "Offensive language or harm tokens detected"
            rejection_reasons["hate"] = "No critical slurs or identity targeting detected"
        else:
            rejection_reasons["offensive"] = "No significant harm indicators detected"
            rejection_reasons["hate"] = "No hate speech markers detected"
        
        output = MUSTPlusOutput(
            label=fallback_result.prediction.value,
            confidence=fallback_result.confidence,
            languages_detected={lp.value: w for lp, w in 
                              [(language_profile.primary_language, 1.0)]},
            fallback_used=True,
            escalation_triggered=fallback_result.escalation_triggered,
            key_harm_tokens=list(set(harm_tokens))[:10],
            explanation=" | ".join(explanation_parts),
            script_distribution=language_profile.script_distribution,
            is_code_mixed=language_profile.is_code_mixed,
            transformer_prediction="unavailable",
            transformer_confidence=0.0,
            fallback_tier=fallback_result.tier_used.value,
            identity_groups_detected=identity_groups,
            rejection_reasons=rejection_reasons,
            entropy=0.0,
            tokenization_coverage=0.0,
            degraded_mode=True
        )
        
        # Validate
        output.validate()
        
        # Log
        if self.config.enable_logging:
            self._log_classification(text, language_profile, output, start_time)
        
        return output
    
    def _log_classification(
        self,
        text: str,
        language_profile: LanguageProfile,
        output: MUSTPlusOutput,
        start_time: float
    ):
        """Log classification for audit trail."""
        import hashlib
        
        # Compute dominant script from distribution
        dominant_script = "unknown"
        if language_profile.script_distribution:
            dominant_script = max(
                language_profile.script_distribution.items(),
                key=lambda x: x[1]
            )[0]
        
        log_entry = PipelineLog(
            timestamp=datetime.now().isoformat(),
            input_text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            language_profile={
                "primary": language_profile.primary_language.value,
                "script": dominant_script
            },
            transformer_available=not self.degraded_mode,
            transformer_prediction=output.transformer_prediction if output.transformer_prediction != "unavailable" else None,
            transformer_confidence=output.transformer_confidence if output.transformer_prediction != "unavailable" else None,
            confidence_gate_decision="pass" if not output.fallback_used else "trigger_fallback",
            fallback_tier_used=output.fallback_tier,
            escalation_triggered=output.escalation_triggered,
            final_label=output.label,
            final_confidence=output.confidence,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        self.logs.append(log_entry)
        logger.debug(f"Classified: {log_entry.final_label} ({log_entry.final_confidence:.2f}) in {log_entry.processing_time_ms:.1f}ms")
    
    def classify_batch(self, texts: List[str]) -> List[MUSTPlusOutput]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of MUSTPlusOutput results
        """
        return [self.classify(text) for text in texts]
    
    # =========================================
    # UTILITY METHODS
    # =========================================
    
    def get_harm_analysis(self, text: str) -> Dict:
        """
        Get detailed harm analysis for a text.
        
        Returns dict with harm tokens, severity, patterns.
        """
        harm_tokens = self.lexicon.get_harm_tokens(text)
        severity = self.lexicon.get_max_severity(text)
        has_critical, reasons = self.lexicon.has_critical_content(text)
        
        return {
            "harm_tokens": harm_tokens,
            "max_severity": severity.value,
            "has_critical_content": has_critical,
            "critical_reasons": reasons,
            "dehumanization_patterns": self.lexicon.check_dehumanization(text),
            "violence_patterns": self.lexicon.check_violence(text),
            "identity_groups": self.lexicon.check_identity_targeting(text)
        }
    
    def explain_prediction(self, text: str) -> Dict:
        """
        Get detailed explanation for a classification.
        
        Returns comprehensive analysis dict.
        """
        result = self.classify(text)
        harm_analysis = self.get_harm_analysis(text)
        language_profile = self.detect_language(text)
        
        return {
            "classification": result.to_dict(),
            "harm_analysis": harm_analysis,
            "language_profile": {
                "primary": language_profile.primary_language.value,
                "proportions": language_profile.language_proportions,
                "scripts": language_profile.script_distribution,
                "is_code_mixed": language_profile.is_code_mixed
            }
        }


# Alias for convenience
MUSTPlus = MUSTPlusPipeline
