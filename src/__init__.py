"""
MUST++ Multilingual Hate Speech Detection Framework

A comprehensive pipeline for detecting hate speech in:
- Tamil (Native + Romanized/Tanglish)
- Hindi (Native + Romanized/Hinglish)  
- English

Classification Labels:
- neutral: No harmful content
- offensive: Inappropriate but not hate
- hate: Targeted hate speech
"""

from src.pipeline import (
    MUSTPlusPipeline,
    ConfidenceGate,
    FallbackManager,
    ScriptDetector,
    DecisionResolver
)

__version__ = "2.0.0"
__all__ = [
    'MUSTPlusPipeline',
    'ConfidenceGate',
    'FallbackManager', 
    'ScriptDetector',
    'DecisionResolver'
]
