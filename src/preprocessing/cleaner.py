import re
import emoji
import string
from src.utils.config import NORMALIZATION_DICT_HINDI, NORMALIZATION_DICT_TAMIL

class TextPreprocessor:
    """
    Implements Section 2: Data Preprocessing Pipeline
    """
    
    def __init__(self):
        pass

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
        Section 2.2: Emoji Handling
        - Convert emojis into textual meaning
        - Keep them as tokens
        """
        # emoji.demojize converts 😡 to :enraged_face:
        # We use space delimiters to keep them as tokens: " enraged_face "
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_transliteration(self, text, language_script):
        """
        Section 2.3: Transliteration Normalization
        - Dictionary-based normalization for Romanized Hindi and Tamil
        """
        if language_script not in ['Hindi_Romanized', 'Tamil_Romanized']:
            return text
            
        words = text.split()
        normalized_words = []
        
        normalization_dict = {}
        if language_script == 'Hindi_Romanized':
            normalization_dict = NORMALIZATION_DICT_HINDI
        elif language_script == 'Tamil_Romanized':
            normalization_dict = NORMALIZATION_DICT_TAMIL
            
        for word in words:
            # Simple case-insensitive lookup
            lower_word = word.lower()
            if lower_word in normalization_dict:
                normalized_words.append(normalization_dict[lower_word])
            else:
                normalized_words.append(word)
                
        return " ".join(normalized_words)

    def preprocess(self, text, language_script, uncased=False):
        """
        Orchestrate the full pipeline
        Section 2.4: Case Handling
        """
        # 1. Noise Removal
        text = self.remove_noise(text)
        
        # 2. Emoji Handling
        text = self.handle_emojis(text)

        # 2b. Special Character Removal (paper)
        # Note: remove_noise already handles URLs/mentions; this strips remaining symbols.
        text = self.remove_special_characters(text)
        
        # 3. Transliteration Normalization
        text = self.normalize_transliteration(text, language_script)
        
        # 4. Case Handling
        if uncased:
            text = text.lower()
            
        return text
