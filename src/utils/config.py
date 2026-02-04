# Configuration and Constants for MUST++ Pipeline

# ============================================
# MUST++ CORE LABELS
# ============================================
LABELS = ["neutral", "offensive", "hate"]
LABELS_DISPLAY = ["Neutral", "Offensive", "Hate"]

# Label priority for conservative resolution
LABEL_PRIORITY = {
    "hate": 3,
    "offensive": 2,
    "neutral": 1
}

# ============================================
# SUPPORTED LANGUAGES
# ============================================
LANGUAGES = ["Hindi", "Tamil", "English"]
LANGUAGE_FORMS = ["Native", "Romanized"]

LANGUAGE_BASE_MAP = {
    "Hindi": "Hindi",
    "Hindi_Romanized": "Hindi",
    "Hinglish": "Hindi",
    "Tamil": "Tamil",
    "Tamil_Romanized": "Tamil",
    "Tanglish": "Tamil",
    "English": "English",
}

# ============================================
# SCRIPT TYPES
# ============================================
SCRIPT_TYPES = {
    "SCRIPT_NATIVE": "Native script (Devanagari, Tamil)",
    "SCRIPT_ROMAN": "Romanized Indian language",
    "SCRIPT_ENGLISH": "English/Latin script"
}

# ============================================
# CONFIDENCE GATE THRESHOLDS
# ============================================
CONFIDENCE_THRESHOLD = 0.75        # Minimum to accept transformer
ENTROPY_LOW_THRESHOLD = 0.5        # Below = low entropy (confident)
ENTROPY_HIGH_THRESHOLD = 1.0       # Above = very uncertain
COVERAGE_THRESHOLD = 0.80          # Minimum known token ratio

# ============================================
# FALLBACK TIER CONFIGURATION
# ============================================
FALLBACK_TIERS = {
    2: "Indic Specialist Reasoning",
    3: "Classical Safety Reasoning",
    4: "Rule-Based Escalation"
}

# Auto-escalation triggers (Tier 4)
ESCALATION_TRIGGERS = [
    "dehumanization",
    "violence_incitement",
    "collective_identity_targeting_with_harm"
]

# ============================================
# EMOJI TO INTENT MAPPING
# ============================================
EMOJI_INTENT_MAP = {
    # Mockery
    "clown_face": "INTENT_MOCKERY",
    "rolling_on_the_floor_laughing": "INTENT_MOCKERY",
    "face_with_tears_of_joy": "INTENT_MOCKERY",
    "smirking_face": "INTENT_MOCKERY",
    "grinning_squinting_face": "INTENT_MOCKERY",
    
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
    "crossed_swords": "INTENT_THREAT",
    "gun": "INTENT_THREAT",
    
    # Sarcasm
    "upside_down_face": "INTENT_SARCASM",
    "winking_face": "INTENT_SARCASM",
    "zipper_mouth_face": "INTENT_SARCASM",
    "face_with_raised_eyebrow": "INTENT_SARCASM",
    
    # Disgust
    "nauseated_face": "INTENT_DISGUST",
    "face_vomiting": "INTENT_DISGUST",
    "pile_of_poo": "INTENT_DISGUST",
    "middle_finger": "INTENT_DISGUST",
    "face_with_rolling_eyes": "INTENT_DISGUST",
    
    # Negative
    "thumbs_down": "INTENT_NEGATIVE",
    "cross_mark": "INTENT_NEGATIVE",
    "no_entry": "INTENT_NEGATIVE",
    "prohibited": "INTENT_NEGATIVE",
}

# ============================================
# ROMANIZED NORMALIZATION DICTIONARIES
# ============================================
# Hindi Romanized (Hinglish) - Many-to-one mapping
NORMALIZATION_DICT_HINDI = {
    # Pronouns/determiners
    "kese": "kaise",
    "kya": "kya",
    "kab": "kab",
    "kaun": "kaun",
    "kyun": "kyun",
    "kyu": "kyun",
    "thik": "theek",
    "bohot": "bahut",
    "boht": "bahut",
    "ni": "nahi",
    "nhi": "nahi",
    "nahin": "nahi",
    "ye": "yeh",
    "wo": "woh",
    "tum": "tum",
    "aap": "aap",
    
    # Common verbs
    "kro": "karo",
    "krna": "karna",
    "krte": "karte",
    "hoga": "hoga",
    "tha": "tha",
    "thi": "thi",
    "hai": "hai",
    "hain": "hain",
    "ho": "ho",
    
    # Slang normalization (preserve harm)
    "bc": "bhenchod",
    "mc": "madarchod",
    "sala": "saala",
    "kamina": "kameena",
    
    # Common words
    "accha": "achha",
    "acha": "achha",
    "kuch": "kuch",
    "kuchh": "kuch",
}

# Tamil Romanized (Tanglish) - Many-to-one mapping  
NORMALIZATION_DICT_TAMIL = {
    # Common words
    "nanba": "nanba",
    "machaan": "machan",
    "machan": "machan",
    "loosu": "loosu",
    "losu": "loosu",
    "thala": "thala",
    "thalai": "thala",
    "da": "da",
    "di": "di",
    "pa": "pa",
    "ma": "ma",
    
    # Pronouns
    "en": "en",
    "un": "un",
    "avan": "avan",
    "aval": "aval",
    "adhu": "adhu",
    "idhu": "idhu",
    
    # Common verbs
    "panna": "pannu",
    "pannunga": "pannunga",
    "sollu": "sollu",
    "sollungo": "sollungo",
    "vaa": "vaa",
    "poo": "po",
    "po": "po",
    
    # Slang normalization (preserve harm)
    "otha": "otha",
    "punda": "punda",
    "thevdiya": "thevdiya",
    "thevidiya": "thevdiya",
}

# Mixed token normalization
MIXED_TOKEN_NORMALIZATION = {
    # Common code-mixed patterns
    "very": "very",
    "ok": "ok",
    "okay": "ok",
    "bro": "bro",
    "dude": "dude",
}

# ============================================
# FOREIGN TERM TRANSLATION
# ============================================
# Optional dictionary-based normalization for mixed-language tokens.
# Keep this list small and domain-specific to avoid over-normalization.
FOREIGN_TERM_TRANSLATIONS = {
    "fb": "facebook",
    "yt": "youtube",
    "ig": "instagram",
    "kpk": "khyber pakhtoon khaa",
}

# ============================================
# INTENSITY MODIFIERS
# ============================================
INTENSITY_MODIFIERS = [
    # English
    "fucking", "damn", "bloody", "goddamn", "freaking",
    # Hindi
    "sala", "saala", "besharam", "nikamma", "bewakoof",
    # Tamil
    "romba", "mokka", "mairu", "poda", "podi",
]

# ============================================
# RANDOM SEED
# ============================================
SEED = 42

# ============================================
# MODEL CONFIGURATION
# ============================================
DEFAULT_MODEL = "bert-base-multilingual-cased"
MAX_SEQUENCE_LENGTH = 128
BATCH_SIZE = 8

# ============================================
# OUTPUT FORMAT TEMPLATE
# ============================================
OUTPUT_TEMPLATE = """
Label: {label}
Confidence: {confidence:.2f}
Languages Detected: {languages}
Fallback Used: {fallback_used}
Key Harm Tokens: {harm_tokens}
Explanation: {explanation}
"""
