import re
import emoji
import string
from typing import Tuple, List, Dict
from src.utils.config import (
    NORMALIZATION_DICT_HINDI, 
    NORMALIZATION_DICT_TAMIL,
    EMOJI_INTENT_MAP,
    INTENSITY_MODIFIERS
)


class TextPreprocessor:
    """
    MUST++ Loss-Aware Normalization Module
    
    Implements Step 2 of the MUST++ Pipeline:
    - Normalize without destroying harm signals
    - Convert emojis to semantic intent tokens
    - Soft-normalize Romanized slang
    - Remove only pure noise (URLs, mentions)
    
    CRITICAL: Never remove slurs, never translate, never over-clean
    """
    
    def __init__(self):
        # Intent mapping for emojis
        self.emoji_intent_map = EMOJI_INTENT_MAP
        
        # Harm-preserving patterns (DO NOT NORMALIZE)
        self._compile_harm_patterns()
    
    def _compile_harm_patterns(self):
        """Compile patterns for harm tokens that must be preserved."""
        # These are patterns that should NEVER be normalized away
        self.harm_preserve_patterns = [
            # Hindi slurs
            r'\b(chutiya|bhenchod|madarchod|gaandu|harami|kamina|bc|mc)\b',
            r'\b(katua|mulla|jihadi|chamar|bhangi)\b',
            # Tamil slurs
            r'\b(thevdiya|otha|punda|koothi|sunni|loosu|naaye|panni)\b',
            r'\b(pariah|pallar|chakkiliyar|thulukan)\b',
            # English
            r'\b(fuck|shit|bitch|bastard|whore|slut|terrorist)\b',
            r'\b(kill|murder|die|exterminate|genocide)\b',
            # Dehumanization
            r'\b(cockroach|vermin|animal|rat|pig|dog)\b',
        ]
        self.harm_regex = [
            re.compile(p, re.IGNORECASE | re.UNICODE) 
            for p in self.harm_preserve_patterns
        ]

    def remove_special_characters(self, text):
        """\
        Paper preprocessing: Special character removal.

        Removes characters like @, #, $, %, ^, &, etc. while preserving
        multilingual scripts (Devanagari, Tamil, Latin) and whitespace.

        Strategy: keep Unicode word characters (letters/digits/underscore)
        and whitespace; strip everything else.
        """
        if not isinstance(text, str):
            return ""

        # Keep letters/numbers from all scripts + whitespace. Drop symbols/punct.
        # Note: underscores are treated as word chars; we remove them after.
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_noise(self, text):
        """
        Section 2.1: Noise Removal
        - URLs
        - User mentions
        - Extra whitespaces
        - Repeated punctuation
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove User mentions (@user)
        text = re.sub(r'@\w+', '', text)
        
        # Remove Repeated punctuation (e.g., "!!" -> "!")
        # We replace any punctuation character followed by itself one or more times with the single character
        for char in string.punctuation:
            pattern = re.escape(char) + r"+"
            text = re.sub(pattern, lambda m, c=char: c, text)
            
        # Extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def handle_emojis(self, text):
        """
        MUST++ Emoji Handling (Loss-Aware)
        
        Convert emojis to SEMANTIC INTENT TOKENS, not just text descriptions.
        This preserves the emotional/threatening intent of emoji usage.
        
        Categories:
        - INTENT_MOCKERY: Laughing/mocking emojis
        - INTENT_ANGER: Angry face emojis
        - INTENT_THREAT: Skull, knife, bomb emojis
        - INTENT_SARCASM: Upside-down, winking emojis
        - INTENT_DISGUST: Vomiting, poop emojis
        - INTENT_NEGATIVE: Thumbs down, cross mark
        """
        # First demojize to text form
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Map emoji names to intent tokens
        for emoji_name, intent in self.emoji_intent_map.items():
            # Handle both with and without colons
            text = text.replace(f":{emoji_name}:", f" {intent} ")
            text = text.replace(emoji_name, f" {intent} ")
        
        # Clean up underscores from emoji names
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def handle_emojis_preserve(self, text):
        """
        Alternative: Keep emoji text descriptions (for models that can use them)
        """
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_transliteration(self, text, language_script):
        """
        MUST++ Soft Normalization for Romanized Text
        
        CRITICAL RULES:
        - Many-to-one mapping only (normalize variants)
        - NEVER translate to English
        - NEVER remove slurs or profanity
        - Preserve original if not in dictionary
        """
        if language_script not in ['Hindi_Romanized', 'Tamil_Romanized', 'Hinglish', 'Tanglish']:
            return text
            
        words = text.split()
        normalized_words = []
        
        normalization_dict = {}
        if language_script in ['Hindi_Romanized', 'Hinglish']:
            normalization_dict = NORMALIZATION_DICT_HINDI
        elif language_script in ['Tamil_Romanized', 'Tanglish']:
            normalization_dict = NORMALIZATION_DICT_TAMIL
            
        for word in words:
            lower_word = word.lower()
            
            # Check if it's a harm token - PRESERVE AS-IS
            is_harm = any(regex.search(word) for regex in self.harm_regex)
            if is_harm:
                normalized_words.append(word)
                continue
            
            # Soft normalize if in dictionary
            if lower_word in normalization_dict:
                normalized_words.append(normalization_dict[lower_word])
            else:
                normalized_words.append(word)
                
        return " ".join(normalized_words)
    
    def extract_harm_tokens(self, text: str) -> List[str]:
        """
        Extract all harm-contributing tokens from text.
        
        Returns:
            List of harm tokens found in text
        """
        harm_tokens = []
        for regex in self.harm_regex:
            matches = regex.findall(text)
            harm_tokens.extend(matches)
        return list(set(harm_tokens))

    def preprocess(self, text, language_script, uncased=False):
        """
        MUST++ Loss-Aware Preprocessing Pipeline
        
        Order of operations:
        1. Noise removal (URLs, mentions, pure noise ONLY)
        2. Emoji handling (convert to intent tokens)
        3. Special character removal (preserve scripts)
        4. Transliteration normalization (soft, harm-preserving)
        5. Case handling (optional)
        
        NEVER:
        - Remove slurs or profanity
        - Translate to English
        - Over-clean the text
        """
        # 1. Noise Removal (conservative)
        text = self.remove_noise(text)
        
        # 2. Emoji Handling (intent-aware)
        text = self.handle_emojis(text)

        # 3. Special Character Removal (preserve scripts)
        text = self.remove_special_characters(text)
        
        # 4. Transliteration Normalization (soft)
        text = self.normalize_transliteration(text, language_script)
        
        # 5. Case Handling (optional - preserve by default)
        if uncased:
            text = text.lower()
            
        return text
    
    def preprocess_for_must(
        self, 
        text: str, 
        language_script: str = None,
        preserve_case: bool = True
    ) -> Tuple[str, List[str]]:
        """
        MUST++ specific preprocessing with harm token extraction.
        
        Args:
            text: Input text
            language_script: Detected language/script
            preserve_case: Whether to preserve case (default True for MUST++)
            
        Returns:
            Tuple of (preprocessed_text, harm_tokens)
        """
        # Extract harm tokens BEFORE preprocessing
        harm_tokens = self.extract_harm_tokens(text)
        
        # Preprocess
        processed = self.preprocess(
            text, 
            language_script, 
            uncased=not preserve_case
        )
        
        return processed, harm_tokens
