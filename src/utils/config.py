# Configuration and Constants

# Section 2.3: Dictionary-based normalization
# This is a sample dictionary. In a real scenario, this would be much larger.
NORMALIZATION_DICT_HINDI = {
    "kese": "kaise",
    "kya": "kya", # identity
    "thik": "theek",
    "bohot": "bahut",
    "ni": "nahi",
}

NORMALIZATION_DICT_TAMIL = {
    "nanba": "nanba", # friend
    "loosu": "loosu", # idiot/fool
    "fraud": "fraud",
}

# Optional: dictionary-based normalization for commonly mixed tokens.
# Keep this limited to Hindi/Tamil/English and their romanized derivatives unless you
# explicitly decide to support additional languages.
MIXED_TOKEN_NORMALIZATION = {
    # Example: "kpk": "khyber pakhtunkhwa"  # (paper example is Urdu-specific; leave disabled)
}

# Section 1.3 Labels
LABELS = ["Neutral", "Offensive", "Hate"]

# Section 3.2 Output Labels for Language ID
LANGUAGES = ["Hindi", "Tamil", "English"]

LANGUAGE_BASE_MAP = {
    "Hindi": "Hindi",
    "Hindi_Romanized": "Hindi",
    "Tamil": "Tamil",
    "Tamil_Romanized": "Tamil",
    "English": "English",
}

# Random Seed for reproducibility
SEED = 42
