"""
MUST++ Script and Language Detection Module

Implements Step 1 of the MUST++ Pipeline:
- Probabilistic language profile inference
- Token-level script tagging
- Multi-language detection with token mass weighting
"""

import re
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum


class ScriptType(Enum):
    """Script classification tags"""
    SCRIPT_NATIVE = "SCRIPT_NATIVE"      # Native script (Devanagari, Tamil)
    SCRIPT_ROMAN = "SCRIPT_ROMAN"        # Romanized Indian language
    SCRIPT_ENGLISH = "SCRIPT_ENGLISH"    # Pure English


class Language(Enum):
    """Supported languages"""
    TAMIL = "Tamil"
    HINDI = "Hindi"
    ENGLISH = "English"
    UNKNOWN = "Unknown"


@dataclass
class TokenInfo:
    """Token with script and language metadata"""
    token: str
    script: ScriptType
    language: Language
    confidence: float


@dataclass
class LanguageProfile:
    """Probabilistic language profile for a text"""
    primary_language: Language
    language_proportions: Dict[str, float]
    script_distribution: Dict[str, float]
    tokens: List[TokenInfo]
    is_code_mixed: bool
    secondary_languages: List[Language]


class ScriptDetector:
    """
    Advanced script and language detection for Indian multilingual text.
    
    Key Features:
    - Token mass weighting (not just count)
    - Multi-language detection with 20% threshold for secondary
    - Script tagging for each token
    - Never relies on user-declared language
    """
    
    # Unicode ranges for script detection
    DEVANAGARI_RANGE = (0x0900, 0x097F)
    TAMIL_RANGE = (0x0B80, 0x0BFF)
    
    # Common Romanized Hindi patterns
    HINGLISH_PATTERNS = [
        r'\b(kya|kaise|kyun|kab|kaun|kaha|hai|hain|ho|hun|tha|thi|the|kar|karo|karna|raha|rahe|rahi)\b',
        r'\b(nahi|nahin|nhi|ni|mat|mujhe|tujhe|usko|isko|usse|isse|aur|ya|lekin|phir)\b',
        r'\b(accha|achha|bahut|bohot|thoda|zyada|jyada|bilkul|sirf|sab|kuch|kitna|itna)\b',
        r'\b(bhai|behen|yaar|dost|beta|baap|maa|ghar|kaam|log|paisa|time)\b',
        r'\b(chutiya|bhenchod|madarchod|gaand|laude|lund|bc|mc|sala|saala|kamina|harami)\b',
        r'\b(jaake|dekho|bolo|suno|chalo|jao|aao|lao|do|lo|khao|pio|baitho)\b',
    ]
    
    # Common Romanized Tamil patterns
    TANGLISH_PATTERNS = [
        r'\b(enna|epdi|enga|evlo|yaar|yaaru|yen|en|un|avan|aval|adu|idhu|adhu)\b',
        r'\b(illa|illai|iruku|irukku|irundha|panna|pannu|pannunga|sollu|sollungo)\b',
        r'\b(nalla|nallaa|periya|chinna|romba|konjam|thaan|mattum|vera|innum)\b',
        r'\b(nanba|nanban|machan|machaan|da|di|pa|ma|thala|anna|akka|thambi)\b',
        r'\b(thevdiya|otha|punda|sunni|baadu|pool|koothi|oombu|lavada)\b',
        r'\b(podu|paar|vaa|po|va|seri|okay|okke|aprom|appuram|mudiyala|mudiyum)\b',
    ]
    
    # English common words (to distinguish from Romanized Indian)
    ENGLISH_COMMON = {
        'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
        'could', 'should', 'may', 'might', 'must', 'shall', 'can',
        'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'once', 'here', 'there', 'where',
        'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'now', 'new', 'first',
        'you', 'your', 'he', 'she', 'it', 'we', 'they', 'i', 'me', 'him',
        'her', 'us', 'them', 'my', 'his', 'its', 'our', 'their', 'this',
        'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'hate', 'kill', 'die', 'dead', 'death', 'attack', 'destroy',
        'fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'bastard',
    }
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.hinglish_regex = re.compile(
            '|'.join(self.HINGLISH_PATTERNS), 
            re.IGNORECASE | re.UNICODE
        )
        self.tanglish_regex = re.compile(
            '|'.join(self.TANGLISH_PATTERNS), 
            re.IGNORECASE | re.UNICODE
        )
    
    def detect_token_script(self, token: str) -> Tuple[ScriptType, Language, float]:
        """
        Detect script and language for a single token.
        
        Returns:
            Tuple of (ScriptType, Language, confidence)
        """
        if not token or not token.strip():
            return ScriptType.SCRIPT_ENGLISH, Language.UNKNOWN, 0.0
        
        token = token.strip()
        
        # Check for native scripts first
        devanagari_count = sum(
            1 for c in token 
            if self.DEVANAGARI_RANGE[0] <= ord(c) <= self.DEVANAGARI_RANGE[1]
        )
        tamil_count = sum(
            1 for c in token 
            if self.TAMIL_RANGE[0] <= ord(c) <= self.TAMIL_RANGE[1]
        )
        
        total_alpha = sum(1 for c in token if c.isalpha())
        if total_alpha == 0:
            return ScriptType.SCRIPT_ENGLISH, Language.UNKNOWN, 0.0
        
        # Native Devanagari (Hindi)
        if devanagari_count / total_alpha > 0.5:
            return ScriptType.SCRIPT_NATIVE, Language.HINDI, 0.95
        
        # Native Tamil
        if tamil_count / total_alpha > 0.5:
            return ScriptType.SCRIPT_NATIVE, Language.TAMIL, 0.95
        
        # Check for Romanized patterns
        token_lower = token.lower()
        
        # Check Hinglish
        if self.hinglish_regex.search(token_lower):
            return ScriptType.SCRIPT_ROMAN, Language.HINDI, 0.85
        
        # Check Tanglish
        if self.tanglish_regex.search(token_lower):
            return ScriptType.SCRIPT_ROMAN, Language.TAMIL, 0.85
        
        # Check if it's common English
        if token_lower in self.ENGLISH_COMMON:
            return ScriptType.SCRIPT_ENGLISH, Language.ENGLISH, 0.90
        
        # Default to English for Latin script with uncertainty
        if all(c.isascii() for c in token if c.isalpha()):
            return ScriptType.SCRIPT_ENGLISH, Language.ENGLISH, 0.60
        
        return ScriptType.SCRIPT_ENGLISH, Language.UNKNOWN, 0.30
    
    def detect_language_profile(self, text: str) -> LanguageProfile:
        """
        Detect probabilistic language profile for entire text.
        
        Uses token mass (weighted by length) not just token count.
        Retains secondary languages if >= 20% token mass.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LanguageProfile with all detection metadata
        """
        if not text or not text.strip():
            return LanguageProfile(
                primary_language=Language.UNKNOWN,
                language_proportions={"Unknown": 1.0},
                script_distribution={"SCRIPT_ENGLISH": 1.0},
                tokens=[],
                is_code_mixed=False,
                secondary_languages=[]
            )
        
        # Tokenize
        tokens = re.findall(r'\b[\w\u0900-\u097F\u0B80-\u0BFF]+\b', text, re.UNICODE)
        
        token_infos: List[TokenInfo] = []
        language_mass: Dict[Language, float] = {lang: 0.0 for lang in Language}
        script_mass: Dict[ScriptType, float] = {script: 0.0 for script in ScriptType}
        
        total_mass = 0.0
        
        for token in tokens:
            script, language, confidence = self.detect_token_script(token)
            
            # Token mass = length (weighted by confidence)
            mass = len(token) * confidence
            
            token_infos.append(TokenInfo(
                token=token,
                script=script,
                language=language,
                confidence=confidence
            ))
            
            language_mass[language] += mass
            script_mass[script] += mass
            total_mass += mass
        
        if total_mass == 0:
            total_mass = 1.0  # Avoid division by zero
        
        # Calculate proportions
        language_proportions = {
            lang.value: round(mass / total_mass, 3) 
            for lang, mass in language_mass.items() 
            if mass > 0
        }
        
        script_distribution = {
            script.value: round(mass / total_mass, 3) 
            for script, mass in script_mass.items() 
            if mass > 0
        }
        
        # Determine primary language
        primary = max(language_mass.items(), key=lambda x: x[1])[0]
        
        # Find secondary languages (>= 20% threshold)
        secondary = [
            lang for lang, mass in language_mass.items()
            if lang != primary and mass / total_mass >= 0.20
        ]
        
        # Determine if code-mixed
        significant_languages = [
            lang for lang, mass in language_mass.items()
            if mass / total_mass >= 0.20
        ]
        is_code_mixed = len(significant_languages) > 1
        
        return LanguageProfile(
            primary_language=primary,
            language_proportions=language_proportions,
            script_distribution=script_distribution,
            tokens=token_infos,
            is_code_mixed=is_code_mixed,
            secondary_languages=secondary
        )
    
    def get_script_tags(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Get script tags for each token in text.
        
        Returns:
            List of (token, script_tag, language) tuples
        """
        profile = self.detect_language_profile(text)
        return [
            (ti.token, ti.script.value, ti.language.value) 
            for ti in profile.tokens
        ]
    
    def is_romanized_dominant(self, text: str) -> bool:
        """Check if text is predominantly Romanized Indian language."""
        profile = self.detect_language_profile(text)
        roman_mass = profile.script_distribution.get('SCRIPT_ROMAN', 0.0)
        return roman_mass >= 0.40
    
    def has_native_script(self, text: str) -> bool:
        """Check if text contains native Indian script."""
        profile = self.detect_language_profile(text)
        return profile.script_distribution.get('SCRIPT_NATIVE', 0.0) > 0.0
