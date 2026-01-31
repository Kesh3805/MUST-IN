"""Quick test script for MUST++ components"""

from src.pipeline.script_detector import ScriptDetector
from src.pipeline.confidence_gate import ConfidenceGate
from src.pipeline.fallback_logic import FallbackManager, FallbackTier
from src.pipeline.hate_lexicon import HateLexicon
import numpy as np

# Initialize components
detector = ScriptDetector()
gate = ConfidenceGate()
fallback = FallbackManager()
lexicon = HateLexicon()

# Test cases
test_cases = [
    'Hello, how are you?',
    'Tu chutiya hai sala',
    'Kill all Muslims',
    'Saare katue maaro',
    'Nanba romba thanks da',
    'Thevdiya paiyan avan',
]

print('=== MUST++ Component Tests ===\n')

for text in test_cases:
    print(f'Text: "{text}"')
    
    # Language detection
    profile = detector.detect_language_profile(text)
    print(f'  Language: {profile.primary_language.value}')
    print(f'  Scripts: {profile.script_distribution}')
    
    # Harm analysis
    harm = lexicon.get_harm_tokens(text)
    has_critical, reasons = lexicon.has_critical_content(text)
    print(f'  Harm tokens: {[h["token"] for h in harm]}')
    print(f'  Critical: {has_critical}')
    
    # Simulate fallback
    if harm or has_critical:
        result = fallback.execute_fallback(
            text=text,
            language_profile=profile,
            transformer_prediction=1,  # offensive
            transformer_confidence=0.6,
            tier=FallbackTier.TIER_3_SAFETY
        )
        print(f'  Fallback prediction: {result.prediction.value}')
        print(f'  Escalation: {result.escalation_triggered}')
    
    print()
