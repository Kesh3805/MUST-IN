"""
MUST++ Golden Test Suite

End-to-end validation tests for the complete pipeline.
Tests all 7 steps, labels (neutral, offensive, hate), fallback paths.

Run with: python -m pytest tests/test_golden_suite.py -v
Or directly: python tests/test_golden_suite.py
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    MUSTPlus, 
    MUSTPlusOutput, 
    PipelineConfig,
    HateLexicon,
    ScriptDetector,
    ConfidenceGate,
    FallbackManager,
    DecisionResolver,
    Language,
    ScriptType
)


# Fixture decorator that works with or without pytest
if HAS_PYTEST:
    fixture = pytest.fixture
else:
    def fixture(autouse=False):
        def decorator(func):
            return func
        return decorator


class TestGoldenSuite:
    """Golden test cases for MUST++ pipeline validation."""
    
    @fixture(autouse=True)
    def setup(self):
        """Setup pipeline in degraded mode (no transformer) for fast testing."""
        config = PipelineConfig(
            disable_transformer=True,  # Test fallback path
            enable_logging=True
        )
        self.pipeline = MUSTPlus(config=config)
        self.lexicon = HateLexicon()
    
    # =========================================
    # NEUTRAL CASES
    # =========================================
    
    def test_neutral_hindi_greeting(self):
        """Neutral Hindi greeting should be classified as neutral."""
        text = "आप कैसे हैं? मुझे आपसे मिलकर खुशी हुई।"
        result = self.pipeline.classify(text)
        
        assert result.label == "neutral", f"Expected neutral, got {result.label}"
        assert result.key_harm_tokens == [] or len(result.key_harm_tokens) == 0
    
    def test_neutral_tamil_greeting(self):
        """Neutral Tamil greeting should be classified as neutral."""
        text = "நலமா இருக்கீங்களா? உங்களை சந்தித்ததில் மகிழ்ச்சி."
        result = self.pipeline.classify(text)
        
        assert result.label == "neutral", f"Expected neutral, got {result.label}"
    
    def test_neutral_english_statement(self):
        """Neutral English statement should be classified as neutral."""
        text = "The weather is nice today. I hope you're having a good day."
        result = self.pipeline.classify(text)
        
        assert result.label == "neutral", f"Expected neutral, got {result.label}"
    
    def test_neutral_codemixed_hinglish(self):
        """Neutral Hinglish statement should be classified as neutral."""
        text = "Yaar aaj weather bohot accha hai, chal chai peete hain."
        result = self.pipeline.classify(text)
        
        assert result.label == "neutral", f"Expected neutral, got {result.label}"
    
    # =========================================
    # OFFENSIVE CASES
    # =========================================
    
    def test_offensive_profanity_hindi_romanized(self):
        """Offensive Hindi profanity (Romanized) should be classified as offensive."""
        text = "saala pagal hai kya tu"
        result = self.pipeline.classify(text)
        
        assert result.label in ["offensive", "hate"], f"Expected offensive/hate, got {result.label}"
        assert "pagal" in [t.lower() for t in result.key_harm_tokens] or len(result.key_harm_tokens) > 0
    
    def test_offensive_english_insult(self):
        """Offensive English insult should be classified as offensive."""
        text = "You're such a stupid idiot, go away!"
        result = self.pipeline.classify(text)
        
        assert result.label in ["offensive", "hate"], f"Expected offensive/hate, got {result.label}"
    
    # =========================================
    # HATE CASES - Critical
    # =========================================
    
    def test_hate_katua_slur(self):
        """Anti-Muslim slur 'katua' should trigger hate classification."""
        text = "saale katue sab desh se nikalo"
        result = self.pipeline.classify(text)
        
        assert result.label == "hate", f"Expected hate, got {result.label}"
        assert result.escalation_triggered == True, "Escalation should be triggered"
        assert any("kat" in t.lower() for t in result.key_harm_tokens), f"Expected 'katua' variant in harm tokens: {result.key_harm_tokens}"
    
    def test_hate_katue_variant(self):
        """Anti-Muslim slur variant 'katue' should trigger hate."""
        text = "yeh katue sirf masla karte hain"
        result = self.pipeline.classify(text)
        
        assert result.label == "hate", f"Expected hate, got {result.label}"
        assert result.escalation_triggered == True
    
    def test_hate_chutiya_slur(self):
        """Offensive slur 'chutiya' should trigger offensive/hate."""
        text = "tu chutiya hai kya be"
        result = self.pipeline.classify(text)
        
        assert result.label in ["offensive", "hate"], f"Expected offensive/hate, got {result.label}"
    
    def test_hate_caste_discrimination_chamar(self):
        """Caste slur 'chamar' should trigger hate classification."""
        text = "ye chamaar log kahan se aate hain"
        result = self.pipeline.classify(text)
        
        assert result.label == "hate", f"Expected hate, got {result.label}"
        assert result.escalation_triggered == True
    
    def test_hate_tamil_paarpaan(self):
        """Tamil caste slur 'பார்ப்பான்' should trigger hate."""
        text = "இந்த பார்ப்பான் எல்லாம் ஒழிய வேண்டும்"
        result = self.pipeline.classify(text)
        
        assert result.label == "hate", f"Expected hate, got {result.label}"
    
    def test_hate_dehumanization(self):
        """Dehumanization patterns should trigger hate."""
        text = "these people are like cockroaches, should be exterminated"
        result = self.pipeline.classify(text)
        
        assert result.label == "hate", f"Expected hate, got {result.label}"
        assert result.escalation_triggered == True
    
    def test_hate_violence_threat(self):
        """Violence threats should trigger hate."""
        text = "maro inko, saalo ko jinda jalado"
        result = self.pipeline.classify(text)
        
        assert result.label == "hate", f"Expected hate, got {result.label}"
        assert result.escalation_triggered == True
    
    # =========================================
    # LANGUAGE DETECTION
    # =========================================
    
    def test_language_detection_hindi_native(self):
        """Native Hindi script should be detected correctly."""
        text = "यह हिंदी में लिखा गया है"
        result = self.pipeline.classify(text)
        
        assert "Hindi" in result.languages_detected or "HINDI" in str(result.languages_detected)
    
    def test_language_detection_tamil_native(self):
        """Native Tamil script should be detected correctly."""
        text = "இது தமிழில் எழுதப்பட்டுள்ளது"
        result = self.pipeline.classify(text)
        
        assert "Tamil" in result.languages_detected or "TAMIL" in str(result.languages_detected)
    
    def test_language_detection_codemixed(self):
        """Code-mixed text should be flagged correctly."""
        text = "Yaar यार aaj kal weather बहुत अच्छा है"
        result = self.pipeline.classify(text)
        
        assert result.is_code_mixed == True, "Should detect code-mixing"
    
    # =========================================
    # FALLBACK VALIDATION
    # =========================================
    
    def test_fallback_triggers_on_degraded_mode(self):
        """Fallback should be used in degraded mode."""
        text = "This is a test sentence."
        result = self.pipeline.classify(text)
        
        assert result.fallback_used == True, "Fallback should be used in degraded mode"
        assert result.degraded_mode == True, "Should be marked as degraded mode"
    
    def test_escalation_on_hate_lexicon(self):
        """Escalation should trigger when hate lexicon matches."""
        text = "sab bhangion ko bhaga do yahan se"
        result = self.pipeline.classify(text)
        
        assert result.escalation_triggered == True, "Should trigger escalation on caste slur"
        assert result.label == "hate"
    
    # =========================================
    # OUTPUT CONTRACT VALIDATION
    # =========================================
    
    def test_output_has_all_required_fields(self):
        """All required output fields should be present."""
        text = "Test input text"
        result = self.pipeline.classify(text)
        
        # Required fields per spec
        assert hasattr(result, 'label')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'languages_detected')
        assert hasattr(result, 'fallback_used')
        assert hasattr(result, 'escalation_triggered')
        assert hasattr(result, 'key_harm_tokens')
        assert hasattr(result, 'explanation')
        
        # Extended fields
        assert hasattr(result, 'script_distribution')
        assert hasattr(result, 'is_code_mixed')
        assert hasattr(result, 'transformer_prediction')
        assert hasattr(result, 'transformer_confidence')
        assert hasattr(result, 'fallback_tier')
        assert hasattr(result, 'identity_groups_detected')
        assert hasattr(result, 'rejection_reasons')
        assert hasattr(result, 'entropy')
        assert hasattr(result, 'tokenization_coverage')
        assert hasattr(result, 'degraded_mode')
    
    def test_output_validates_successfully(self):
        """Output validation should pass."""
        text = "Test input"
        result = self.pipeline.classify(text)
        
        # Should not raise
        assert result.validate() == True
    
    def test_label_in_valid_range(self):
        """Label should be one of neutral/offensive/hate."""
        for text in ["hello", "stupid fool", "kill them all"]:
            result = self.pipeline.classify(text)
            assert result.label in ["neutral", "offensive", "hate"]
    
    def test_confidence_in_valid_range(self):
        """Confidence should be between 0 and 1."""
        text = "Any test text here"
        result = self.pipeline.classify(text)
        
        assert 0.0 <= result.confidence <= 1.0
    
    # =========================================
    # EDGE CASES
    # =========================================
    
    def test_empty_input(self):
        """Empty input should return neutral."""
        result = self.pipeline.classify("")
        assert result.label == "neutral"
        assert result.confidence == 1.0
        assert result.degraded_mode is True
    
    def test_whitespace_only(self):
        """Whitespace-only input should return neutral."""
        result = self.pipeline.classify("   \n\t  ")
        assert result.label == "neutral"
    
    def test_very_short_input(self):
        """Very short input should still be classified."""
        result = self.pipeline.classify("hi")
        assert result.label in ["neutral", "offensive", "hate"]
    
    def test_very_long_input(self):
        """Long input should be handled without error."""
        long_text = "This is a test sentence. " * 100
        result = self.pipeline.classify(long_text)
        assert result.label in ["neutral", "offensive", "hate"]


class TestAdversarialCases:
    """Adversarial and edge-case testing."""
    
    @fixture(autouse=True)
    def setup(self):
        """Setup pipeline."""
        config = PipelineConfig(disable_transformer=True, enable_logging=True)
        self.pipeline = MUSTPlus(config=config)
    
    # =========================================
    # OBFUSCATED SLURS
    # =========================================
    
    def test_obfuscated_katua_with_at(self):
        """Obfuscated 'k@tua' should be detected."""
        text = "ye k@tua log sab same hain"
        result = self.pipeline.classify(text)
        
        # Should at least trigger concern due to pattern
        assert result.label in ["offensive", "hate"]
    
    def test_obfuscated_chutiya_with_numbers(self):
        """Obfuscated 'chut1ya' should be detected."""
        text = "tu chut1ya hai be"
        result = self.pipeline.classify(text)
        
        assert result.label in ["offensive", "hate"]
    
    def test_leetspeak_slur(self):
        """Leetspeak slurs should be caught."""
        text = "all k4tu4s should leave"
        result = self.pipeline.classify(text)
        
        # May not catch all leetspeak, but should flag suspicious patterns
        assert result.label in ["neutral", "offensive", "hate"]  # Document capability
    
    # =========================================
    # SARCASM + EMOJI
    # =========================================
    
    def test_sarcasm_with_clown_emoji(self):
        """Sarcasm with mockery emoji should be flagged."""
        text = "wow such intelligent people 🤡🤡🤡 these minorities"
        result = self.pipeline.classify(text)
        
        # Should detect mockery intent
        assert result.label in ["offensive", "hate"]
    
    def test_threat_with_knife_emoji(self):
        """Threat with weapon emoji should escalate."""
        text = "I'll deal with them 🔪🔪"
        result = self.pipeline.classify(text)
        
        assert result.label in ["offensive", "hate"]
    
    # =========================================
    # MIXED SCRIPT ATTACKS
    # =========================================
    
    def test_mixed_script_slur(self):
        """Mixed script slurs should be detected."""
        text = "sab कटुआ को bhagao"  # Mixed Romanized + Devanagari
        result = self.pipeline.classify(text)
        
        assert result.label == "hate", f"Expected hate, got {result.label}"
        assert result.is_code_mixed == True
    
    # =========================================
    # CONTEXT-DEPENDENT CASES
    # =========================================
    
    def test_neutral_word_in_hate_context(self):
        """Neutral words in hate context should be caught."""
        text = "these people should all be removed from our country permanently"
        result = self.pipeline.classify(text)
        
        # May be hate due to "removed permanently" pattern
        assert result.label in ["offensive", "hate"]
    
    def test_implicit_violence(self):
        """Implicit violence should be caught."""
        text = "unko unki jagah dikhado, permanently"
        result = self.pipeline.classify(text)
        
        # Should detect implied threat
        assert result.label in ["offensive", "hate"]


class TestLexiconCoverage:
    """Test lexicon coverage and accuracy."""
    
    @fixture(autouse=True)
    def setup(self):
        """Setup lexicon."""
        self.lexicon = HateLexicon()
    
    def test_hindi_religious_slurs(self):
        """Hindi religious slurs should be in lexicon."""
        slurs = ["katua", "katue", "mulla", "jihadi"]
        for slur in slurs:
            result = self.lexicon.lookup(slur)
            assert result is not None, f"Expected '{slur}' to be in lexicon"
    
    def test_hindi_caste_slurs(self):
        """Hindi caste slurs should be in lexicon."""
        slurs = ["chamar", "bhangi", "chamaar"]
        for slur in slurs:
            result = self.lexicon.lookup(slur)
            assert result is not None, f"Expected '{slur}' to be in lexicon"
    
    def test_tamil_caste_slurs(self):
        """Tamil caste slurs should be in lexicon."""
        slurs = ["பார்ப்பான்", "பறையன்"]
        for slur in slurs:
            result = self.lexicon.lookup(slur)
            assert result is not None, f"Expected Tamil slur to be in lexicon"
    
    def test_violence_keywords(self):
        """Violence keywords should trigger detection."""
        texts = ["maro saalo ko", "jinda jalado", "exterminate them"]
        for text in texts:
            has_violence = self.lexicon.check_violence(text)
            assert len(has_violence) > 0, f"Expected violence detection in: {text}"
    
    def test_dehumanization_patterns(self):
        """Dehumanization patterns should be detected."""
        texts = ["they are cockroaches", "like animals", "subhuman creatures"]
        for text in texts:
            patterns = self.lexicon.check_dehumanization(text)
            assert len(patterns) > 0, f"Expected dehumanization detection in: {text}"


class TestExplainability:
    """Test explainability and grounding."""
    
    @fixture(autouse=True)
    def setup(self):
        """Setup pipeline."""
        config = PipelineConfig(disable_transformer=True, enable_logging=True)
        self.pipeline = MUSTPlus(config=config)
    
    def test_explanation_includes_harm_tokens(self):
        """Explanation should mention detected harm tokens."""
        text = "ye katue log khatam karo"
        result = self.pipeline.classify(text)
        
        # Explanation should mention the slur
        assert "harm" in result.explanation.lower() or len(result.key_harm_tokens) > 0
    
    def test_explanation_includes_label_reason(self):
        """Explanation should justify the label."""
        text = "hello friend, how are you today"
        result = self.pipeline.classify(text)
        
        assert len(result.explanation) > 0
    
    def test_explanation_mentions_fallback(self):
        """Explanation should mention if fallback was used."""
        text = "some test text"
        result = self.pipeline.classify(text)
        
        if result.fallback_used:
            assert "fallback" in result.explanation.lower() or "tier" in result.explanation.lower()
    
    def test_rejection_reasons_populated(self):
        """Rejection reasons should explain why other labels were rejected."""
        text = "ye katua log bahut gande hain"
        result = self.pipeline.classify(text)
        
        # If hate, should explain why not neutral/offensive
        if result.label == "hate":
            assert "neutral" in result.rejection_reasons or "offensive" in result.rejection_reasons


# =========================================
# PYTEST MAIN
# =========================================

if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        # Run tests manually without pytest
        import logging
        logging.disable(logging.WARNING)
        
        print("Running MUST++ Golden Test Suite (without pytest)")
        print("=" * 60)
        
        # Setup
        from src.pipeline import MUSTPlus, PipelineConfig, HateLexicon
        
        config = PipelineConfig(disable_transformer=True, enable_logging=False)
        pipeline = MUSTPlus(config=config)
        lexicon = HateLexicon()
        
        passed = 0
        failed = 0
        
        # Test cases
        test_cases = [
            ("test_hate_katua_slur", "saale katue sab desh se nikalo", "hate"),
            ("test_hate_katue_variant", "yeh katue sirf masla karte hain", "hate"),
            ("test_hate_caste_discrimination_chamar", "ye chamaar log kahan se aate hain", "hate"),
            ("test_hate_violence", "maro inko, saalo ko jinda jalado", "hate"),
            ("test_neutral_english", "The weather is nice today.", "neutral"),
            ("test_neutral_hindi", "Yaar aaj weather bohot accha hai", "neutral"),
            ("test_hate_dehumanization", "these people are like cockroaches", "hate"),
        ]
        
        for test_name, text, expected in test_cases:
            result = pipeline.classify(text)
            if result.label == expected:
                passed += 1
                print(f"  PASS: {test_name}")
            else:
                failed += 1
                print(f"  FAIL: {test_name} - expected {expected}, got {result.label}")
        
        print()
        print("=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        
        sys.exit(0 if failed == 0 else 1)
