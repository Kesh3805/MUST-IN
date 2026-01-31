"""
MUST++ Hate Lexicon Module

Comprehensive hate speech lexicon for Indian multilingual contexts:
- Tamil (Native + Romanized/Tanglish)
- Hindi (Native + Romanized/Hinglish)
- English

Categories:
- Slurs and derogatory terms
- Caste-based insults
- Religious targeting
- Gender-based abuse
- Dehumanization
- Violence incitement
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class HarmCategory(Enum):
    """Categories of harmful content"""
    SLUR = "SLUR"                           # General slurs/profanity
    CASTE = "CASTE"                         # Caste-based discrimination
    RELIGIOUS = "RELIGIOUS"                 # Religious targeting
    GENDER = "GENDER"                       # Gender-based abuse
    ETHNIC = "ETHNIC"                       # Ethnic/racial targeting
    DEHUMANIZATION = "DEHUMANIZATION"       # Dehumanizing language
    VIOLENCE = "VIOLENCE"                   # Violence incitement
    SEXUAL = "SEXUAL"                       # Sexual harassment
    IDENTITY = "IDENTITY"                   # Identity-based attack


class SeverityLevel(Enum):
    """Severity levels for hate content"""
    LOW = 1          # Mild offensive language
    MEDIUM = 2       # Strong offensive language
    HIGH = 3         # Hate speech
    CRITICAL = 4     # Extreme hate/violence incitement


@dataclass
class LexiconEntry:
    """Entry in hate lexicon"""
    term: str
    normalized_forms: List[str]
    category: HarmCategory
    severity: SeverityLevel
    language: str
    is_identity_targeting: bool
    context_notes: str


class HateLexicon:
    """
    Multilingual hate speech lexicon for Indian languages.
    
    WARNING: This lexicon contains offensive terms for research purposes only.
    These terms are documented to enable hate speech detection systems.
    """
    
    def __init__(self):
        self._build_lexicons()
        self._compile_patterns()
    
    def _build_lexicons(self):
        """Build comprehensive lexicons for all supported languages."""
        
        # ============================================
        # HINDI LEXICON (Native Devanagari + Romanized)
        # ============================================
        self.hindi_lexicon: Dict[str, LexiconEntry] = {}
        
        # Severe slurs - Hindi
        hindi_slurs = [
            # Common profanity (Romanized forms)
            ("chutiya", ["chutia", "chootiya", "chutiye", "chutiyo"], HarmCategory.SLUR, SeverityLevel.HIGH),
            ("madarchod", ["mc", "madarc**d", "madarchd", "maderchod"], HarmCategory.SLUR, SeverityLevel.CRITICAL),
            ("bhenchod", ["bc", "bhnchod", "bhenchd", "bhen ke lode"], HarmCategory.SLUR, SeverityLevel.CRITICAL),
            ("gaandu", ["gandu", "gaand", "gand"], HarmCategory.SLUR, SeverityLevel.HIGH),
            ("harami", ["haraam", "haraami", "haramkhor"], HarmCategory.SLUR, SeverityLevel.MEDIUM),
            ("kamina", ["kameena", "kamini", "kamine"], HarmCategory.SLUR, SeverityLevel.MEDIUM),
            ("saala", ["sala", "saale", "sali"], HarmCategory.SLUR, SeverityLevel.LOW),
            ("kutte", ["kutta", "kutti", "kutton"], HarmCategory.DEHUMANIZATION, SeverityLevel.MEDIUM),
            ("suar", ["suwar", "suwwar", "suaron"], HarmCategory.DEHUMANIZATION, SeverityLevel.MEDIUM),
            ("laude", ["lode", "lund", "lawde", "lauda"], HarmCategory.SLUR, SeverityLevel.HIGH),
            ("randi", ["raand", "randi ka", "randikhana"], HarmCategory.GENDER, SeverityLevel.HIGH),
            ("chakka", ["chhakka", "hijra", "kinnar"], HarmCategory.GENDER, SeverityLevel.HIGH),
            
            # Caste-based
            ("chamar", ["chamaar", "chamaro"], HarmCategory.CASTE, SeverityLevel.CRITICAL),
            ("bhangi", ["bhangiyo", "bhangian", "bhangion", "bhangio"], HarmCategory.CASTE, SeverityLevel.CRITICAL),
            ("neech", ["nich", "neech jaati"], HarmCategory.CASTE, SeverityLevel.HIGH),
            ("dalit", [], HarmCategory.CASTE, SeverityLevel.MEDIUM),  # context-dependent
            
            # Religious targeting
            ("kafir", ["kaafir", "kaffir"], HarmCategory.RELIGIOUS, SeverityLevel.HIGH),
            ("mulla", ["mullao", "maulvi", "maulana"], HarmCategory.RELIGIOUS, SeverityLevel.MEDIUM),
            ("jihadi", ["jihaadi", "jaihadi"], HarmCategory.RELIGIOUS, SeverityLevel.HIGH),
            ("katua", ["katwe", "katuwa", "katuo", "katue", "katuon", "katuwo"], HarmCategory.RELIGIOUS, SeverityLevel.CRITICAL),
            ("hindu terror", ["bhagwa terror", "saffron terror"], HarmCategory.RELIGIOUS, SeverityLevel.HIGH),
            
            # Violence indicators
            ("maaro", ["maar do", "maar daalo", "maarna"], HarmCategory.VIOLENCE, SeverityLevel.HIGH),
            ("kaat do", ["kaato", "kaat dalo"], HarmCategory.VIOLENCE, SeverityLevel.CRITICAL),
            ("jala do", ["jalao", "jala dalo"], HarmCategory.VIOLENCE, SeverityLevel.CRITICAL),
            ("bhagao", ["nikalo", "bhaga do"], HarmCategory.VIOLENCE, SeverityLevel.MEDIUM),
        ]
        
        for term, variants, category, severity in hindi_slurs:
            is_identity = category in [HarmCategory.CASTE, HarmCategory.RELIGIOUS, HarmCategory.ETHNIC]
            self.hindi_lexicon[term.lower()] = LexiconEntry(
                term=term,
                normalized_forms=[v.lower() for v in variants],
                category=category,
                severity=severity,
                language="Hindi",
                is_identity_targeting=is_identity,
                context_notes=""
            )
        
        # ============================================
        # TAMIL LEXICON (Native + Romanized/Tanglish)
        # ============================================
        self.tamil_lexicon: Dict[str, LexiconEntry] = {}
        
        tamil_slurs = [
            # Common profanity (Romanized forms)
            ("thevdiya", ["thevudiya", "thevidiya", "thevdia", "thevidya"], HarmCategory.GENDER, SeverityLevel.HIGH),
            ("otha", ["oththa", "othe", "da otha"], HarmCategory.SLUR, SeverityLevel.HIGH),
            ("punda", ["pundai", "pundek", "punde"], HarmCategory.SLUR, SeverityLevel.HIGH),
            ("koothi", ["koothu", "koothichi", "kootha"], HarmCategory.GENDER, SeverityLevel.HIGH),
            ("sunni", ["chunni", "sunniya"], HarmCategory.SLUR, SeverityLevel.HIGH),
            ("baadu", ["baadhu", "badu", "potta baadu"], HarmCategory.GENDER, SeverityLevel.MEDIUM),
            ("oombu", ["oombuda", "oombi", "oombra"], HarmCategory.SEXUAL, SeverityLevel.HIGH),
            ("pool", ["poolu", "poolikka"], HarmCategory.SLUR, SeverityLevel.HIGH),
            ("lavada", ["lavde", "lavadika"], HarmCategory.SLUR, SeverityLevel.HIGH),
            ("loosu", ["loosa", "lossu", "mental loosu"], HarmCategory.SLUR, SeverityLevel.LOW),
            ("naaye", ["naayi", "naai", "nakkal naaye"], HarmCategory.DEHUMANIZATION, SeverityLevel.MEDIUM),
            ("panni", ["panri", "panniya", "panni punda"], HarmCategory.DEHUMANIZATION, SeverityLevel.MEDIUM),
            
            # Caste-based Tamil (Romanized)
            ("pariah", ["paraiyar", "parayar", "parayan", "பறையன்"], HarmCategory.CASTE, SeverityLevel.CRITICAL),
            ("pallar", ["pallaro", "palla"], HarmCategory.CASTE, SeverityLevel.CRITICAL),
            ("chakkiliyar", ["chakkili", "chakkiliya"], HarmCategory.CASTE, SeverityLevel.CRITICAL),
            
            # Caste-based Tamil (Native Script)
            ("பார்ப்பான்", ["பாப்பான்", "brahmanan", "brahmin"], HarmCategory.CASTE, SeverityLevel.CRITICAL),
            ("பறையன்", ["paraiyar", "parayan"], HarmCategory.CASTE, SeverityLevel.CRITICAL),
            
            # Religious targeting
            ("thulukan", ["thulukkan", "tulukan", "துலுக்கன்"], HarmCategory.RELIGIOUS, SeverityLevel.HIGH),
            
            # Violence Tamil (ஒழிய = ozhiya = destroy/eradicate)
            ("ஒழிய", ["ozhiya", "ozhi", "ஒழித்து"], HarmCategory.VIOLENCE, SeverityLevel.CRITICAL),
            ("adichirdu", ["adichidu", "adi", "adichu"], HarmCategory.VIOLENCE, SeverityLevel.MEDIUM),
            ("kollu", ["konnu", "kolluven", "kollunga"], HarmCategory.VIOLENCE, SeverityLevel.CRITICAL),
            ("vetti edu", ["vettu", "vettidu"], HarmCategory.VIOLENCE, SeverityLevel.CRITICAL),
        ]
        
        for term, variants, category, severity in tamil_slurs:
            is_identity = category in [HarmCategory.CASTE, HarmCategory.RELIGIOUS, HarmCategory.ETHNIC]
            self.tamil_lexicon[term.lower()] = LexiconEntry(
                term=term,
                normalized_forms=[v.lower() for v in variants],
                category=category,
                severity=severity,
                language="Tamil",
                is_identity_targeting=is_identity,
                context_notes=""
            )
        
        # ============================================
        # ENGLISH LEXICON
        # ============================================
        self.english_lexicon: Dict[str, LexiconEntry] = {}
        
        english_slurs = [
            # General profanity
            ("fuck", ["fck", "fuk", "f**k", "effing", "fking"], HarmCategory.SLUR, SeverityLevel.MEDIUM),
            ("shit", ["sht", "sh*t", "shitty"], HarmCategory.SLUR, SeverityLevel.LOW),
            ("bitch", ["btch", "b*tch", "biatch"], HarmCategory.GENDER, SeverityLevel.MEDIUM),
            ("bastard", ["bstrd", "b**tard"], HarmCategory.SLUR, SeverityLevel.MEDIUM),
            ("whore", ["wh*re", "hoe", "ho"], HarmCategory.GENDER, SeverityLevel.HIGH),
            ("slut", ["sl*t", "slutty"], HarmCategory.GENDER, SeverityLevel.HIGH),
            
            # Racial/ethnic (used in Indian context)
            ("paki", ["pakkis", "packis"], HarmCategory.ETHNIC, SeverityLevel.CRITICAL),
            ("curry muncher", ["curry eater", "currymuncher"], HarmCategory.ETHNIC, SeverityLevel.HIGH),
            ("terrorist", ["terro", "terorist"], HarmCategory.RELIGIOUS, SeverityLevel.HIGH),
            
            # Dehumanization
            ("subhuman", ["sub-human", "subhumans"], HarmCategory.DEHUMANIZATION, SeverityLevel.CRITICAL),
            ("vermin", ["vermins", "verminous"], HarmCategory.DEHUMANIZATION, SeverityLevel.CRITICAL),
            ("cockroach", ["roach", "cockroaches"], HarmCategory.DEHUMANIZATION, SeverityLevel.CRITICAL),
            ("rat", ["rats", "dirty rats"], HarmCategory.DEHUMANIZATION, SeverityLevel.HIGH),
            ("animal", ["animals", "like animals"], HarmCategory.DEHUMANIZATION, SeverityLevel.MEDIUM),
            
            # Violence
            ("kill", ["kil", "k1ll", "killing"], HarmCategory.VIOLENCE, SeverityLevel.HIGH),
            ("murder", ["murd3r", "murdering"], HarmCategory.VIOLENCE, SeverityLevel.CRITICAL),
            ("die", ["d1e", "should die", "must die"], HarmCategory.VIOLENCE, SeverityLevel.HIGH),
            ("exterminate", ["extermination", "wipe out"], HarmCategory.VIOLENCE, SeverityLevel.CRITICAL),
            ("burn", ["burn them", "set fire"], HarmCategory.VIOLENCE, SeverityLevel.HIGH),
            ("genocide", ["gen0cide", "ethnic cleansing"], HarmCategory.VIOLENCE, SeverityLevel.CRITICAL),
        ]
        
        for term, variants, category, severity in english_slurs:
            is_identity = category in [HarmCategory.CASTE, HarmCategory.RELIGIOUS, HarmCategory.ETHNIC]
            self.english_lexicon[term.lower()] = LexiconEntry(
                term=term,
                normalized_forms=[v.lower() for v in variants],
                category=category,
                severity=severity,
                language="English",
                is_identity_targeting=is_identity,
                context_notes=""
            )
        
        # ============================================
        # DEHUMANIZATION PATTERNS
        # ============================================
        self.dehumanization_patterns = [
            r'\b(they are|these|those|all)\s*(are)?\s*(animals?|vermin|cockroach|rat|pig|dog|insect)',
            r'\b(are|like)\s+(cockroach|vermin|animal|rat|pig|dog|insect)e?s?\b',
            r'\b(sub-?human|less than human|not human|subhuman)',
            r'\b(breed like|breeding like)\s*(animals?|rats?|pigs?)',
            r'\b(infestation|plague|disease)\b.*\b(of|by)\s*(them|muslims?|hindus?)',
            r'\b(go back|leave|get out).*\b(country|india|pakistan)',
            r'\blike\s+animals?\b',
            r'\bsubhuman\s+creatures?\b',
            r'\bcockroaches?\b',
            r'\bvermin\b',
        ]
        
        # ============================================
        # EMOJI THREAT/MOCKERY PATTERNS
        # ============================================
        self.threat_emojis = ['🔪', '🗡️', '⚔️', '🔫', '💣', '💀', '☠️', '🪓', '🏹']
        self.mockery_emojis = ['🤡', '🃏', '😂', '🤣', '😹', '🙃']
        
        # ============================================
        # VIOLENCE PATTERNS
        # ============================================
        self.violence_patterns = [
            # English violence
            r'\b(kill|murder|slaughter|butcher|exterminate)\s*(all|every|them|these)',
            r'\b(should|must|need to|have to)\s*(die|be killed|be murdered)',
            r'\b(burn|hang|shoot|stab|behead)\s*(them|all|every)',
            r'\b(wipe out|eliminate|eradicate|cleanse)\s*(them|all|every)',
            r'\b(death to|kill all)\s*\w+',
            r'\b(riot|pogrom|massacre)\b',
            
            # Hindi violence (Romanized)
            r'\b(maar|maro|maaro)\s*(do|dale?o?|inko|saalo|unko)',
            r'\b(jala|jalao|jala\s*do|jinda\s*jala)',
            r'\b(kaat|kaato|kaat\s*do|khatam)',
            r'\b(nikalo|bhagao|bhaga\s*do)',
            r'\b(saale?|saalo)\s+\w*\s*(ko|nikalo|maaro)',
            r'\b(permanently|hamesha ke liye|khatam\s*karo)',
            
            # Tamil violence (Romanized)
            r'\b(kollu|konnu|adichu)\b',
            r'\b(vettu|vetti)\s*(edu|podu)',
        ]
        
        # ============================================
        # IDENTITY GROUP PATTERNS
        # ============================================
        self.identity_patterns = [
            # Religious groups
            r'\b(muslims?|islam|islamic|musalman|mussalman)\b',
            r'\b(hindus?|hinduism|sanatan|brahmin)\b',
            r'\b(christians?|christianity|church)\b',
            r'\b(sikhs?|sikhism|sardars?)\b',
            r'\b(jews?|jewish|judaism)\b',
            
            # Caste groups
            r'\b(dalits?|shudras?|lower caste|upper caste)\b',
            r'\b(brahmins?|kshatriya|vaishya)\b',
            r'\b(scheduled caste|sc/st|obc|reservation)\b',
            
            # Regional/ethnic
            r'\b(pakistanis?|bangladeshis?|nepalis?)\b',
            r'\b(north indian|south indian|madrasi|bihari)\b',
            r'\b(tamils?|telugus?|mallus?|gujjus?|marathis?)\b',
            
            # Gender/sexuality
            r'\b(women|girls|ladies|females)\b',
            r'\b(lgbtq?|gay|lesbian|trans|homosexual)\b',
            
            # Generic minority references
            r'\b(minorities?|immigrants?|refugees?|migrants?|outsiders?)\b',
            r'\b(these people|those people|such people|them all)\b',
        ]
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.dehumanization_regex = [
            re.compile(p, re.IGNORECASE | re.UNICODE) 
            for p in self.dehumanization_patterns
        ]
        self.violence_regex = [
            re.compile(p, re.IGNORECASE | re.UNICODE) 
            for p in self.violence_patterns
        ]
        self.identity_regex = [
            re.compile(p, re.IGNORECASE | re.UNICODE) 
            for p in self.identity_patterns
        ]
        
        # ============================================
        # OBFUSCATION PATTERNS (leetspeak, symbols)
        # ============================================
        self.obfuscation_patterns = [
            # Hindi slurs with common obfuscations
            (r'\bk[@4]tu[a4@e]\b', 'katua', SeverityLevel.CRITICAL),
            (r'\bk[a4@]tu[e3]\b', 'katue', SeverityLevel.CRITICAL),
            (r'\bchut[1i!]y[a4@]\b', 'chutiya', SeverityLevel.HIGH),
            (r'\bch[u0o]t[1i!]?y?[a4@]?\b', 'chutiya', SeverityLevel.HIGH),
            (r'\bm[a4@]d[a4@]rch[o0]d\b', 'madarchod', SeverityLevel.CRITICAL),
            (r'\bbh[e3]nch[o0]d\b', 'bhenchod', SeverityLevel.CRITICAL),
            (r'\bg[a4@]ndu\b', 'gaandu', SeverityLevel.HIGH),
            (r'\br[a4@]nd[i1!]\b', 'randi', SeverityLevel.HIGH),
            (r'\bch[a4@]m[a4@]r\b', 'chamar', SeverityLevel.CRITICAL),
            (r'\bbh[a4@]ng[i1!]\b', 'bhangi', SeverityLevel.CRITICAL),
            # English obfuscations
            (r'\bk[i1!]ll\b', 'kill', SeverityLevel.HIGH),
            (r'\bd[i1!]e\b', 'die', SeverityLevel.HIGH),
            (r'\bf[u\*]ck\b', 'fuck', SeverityLevel.MEDIUM),
            (r'\bb[i1!]tch\b', 'bitch', SeverityLevel.MEDIUM),
        ]
        self.obfuscation_regex = [
            (re.compile(p, re.IGNORECASE | re.UNICODE), orig, sev)
            for p, orig, sev in self.obfuscation_patterns
        ]
        
        # ============================================
        # ENGLISH INSULT PATTERNS
        # ============================================
        self.english_insult_patterns = [
            (r'\b(stupid|dumb|idiot|moron|fool|imbecile|retard)\b', SeverityLevel.MEDIUM),
            (r'\b(trash|garbage|worthless|pathetic|disgusting)\b', SeverityLevel.MEDIUM),
            (r'\b(go away|get lost|shut up|piss off)\b', SeverityLevel.LOW),
            (r'\b(loser|failure|joke|clown)\b', SeverityLevel.LOW),
        ]
        self.english_insult_regex = [
            (re.compile(p, re.IGNORECASE | re.UNICODE), sev)
            for p, sev in self.english_insult_patterns
        ]
    
    def lookup(self, token: str, language: str = None) -> List[LexiconEntry]:
        """
        Look up a token in the lexicon.
        
        Args:
            token: Token to look up
            language: Optional language filter
            
        Returns:
            List of matching lexicon entries
        """
        import unicodedata
        
        # Normalize Unicode to NFC form
        token_lower = unicodedata.normalize('NFC', token.lower().strip())
        matches = []
        
        lexicons = []
        if language is None or language.lower() in ['hindi', 'hinglish']:
            lexicons.append(self.hindi_lexicon)
        if language is None or language.lower() in ['tamil', 'tanglish']:
            lexicons.append(self.tamil_lexicon)
        if language is None or language.lower() == 'english':
            lexicons.append(self.english_lexicon)
        
        for lexicon in lexicons:
            # Direct match (with normalization)
            for key, entry in lexicon.items():
                key_normalized = unicodedata.normalize('NFC', key.lower())
                if token_lower == key_normalized:
                    matches.append(entry)
                    continue
                
                # Check variants (with normalization)
                for variant in entry.normalized_forms:
                    variant_normalized = unicodedata.normalize('NFC', variant.lower())
                    if token_lower == variant_normalized:
                        matches.append(entry)
                        break
        
        return matches
    
    def scan_text(self, text: str) -> List[Tuple[str, LexiconEntry]]:
        """
        Scan text for all lexicon matches.
        
        Args:
            text: Text to scan
            
        Returns:
            List of (matched_token, entry) tuples
        """
        import unicodedata
        
        # Normalize Unicode to NFC form
        text = unicodedata.normalize('NFC', text.lower())
        
        # Use split for better non-Latin script support, plus regex for mixed text
        words_from_split = text.split()
        words_from_regex = re.findall(r'\b[\w]+\b', text, re.UNICODE)
        
        # Combine and deduplicate
        all_words = list(set(words_from_split + words_from_regex))
        matches = []
        
        for word in all_words:
            entries = self.lookup(word)
            for entry in entries:
                matches.append((word, entry))
        
        return matches
    
    def check_dehumanization(self, text: str) -> List[str]:
        """Check for dehumanization patterns."""
        matches = []
        for regex in self.dehumanization_regex:
            found = regex.findall(text)
            if found:
                matches.extend([m if isinstance(m, str) else ' '.join(m) for m in found])
        return matches
    
    def check_violence(self, text: str) -> List[str]:
        """Check for violence incitement patterns."""
        matches = []
        for regex in self.violence_regex:
            found = regex.findall(text)
            if found:
                matches.extend([m if isinstance(m, str) else ' '.join(m) for m in found])
        return matches
    
    def check_threat_emojis(self, text: str) -> List[str]:
        """Check for threat/weapon emojis."""
        return [e for e in self.threat_emojis if e in text]
    
    def check_mockery_emojis(self, text: str) -> List[str]:
        """Check for mockery/sarcasm emojis."""
        return [e for e in self.mockery_emojis if e in text]
    
    def check_identity_targeting(self, text: str) -> List[str]:
        """Check for identity group mentions."""
        matches = []
        for regex in self.identity_regex:
            found = regex.findall(text)
            if found:
                matches.extend(found)
        return matches
    
    def get_max_severity(self, text: str) -> SeverityLevel:
        """Get maximum severity level found in text."""
        matches = self.scan_text(text)
        
        if not matches:
            # Check patterns
            if self.check_violence(text) or self.check_dehumanization(text):
                return SeverityLevel.CRITICAL
            return SeverityLevel.LOW
        
        max_sev = max(entry.severity.value for _, entry in matches)
        return SeverityLevel(max_sev)

    def check_obfuscated_slurs(self, text: str) -> List[Tuple[str, str, SeverityLevel]]:
        """
        Check for obfuscated slurs using leetspeak and symbol substitution.
        
        Returns:
            List of (matched_text, original_slur, severity) tuples
        """
        matches = []
        for regex, orig, sev in self.obfuscation_regex:
            found = regex.findall(text)
            if found:
                for match in found:
                    matches.append((match, orig, sev))
        return matches
    
    def check_english_insults(self, text: str) -> List[Tuple[str, SeverityLevel]]:
        """
        Check for English insults and offensive language.
        
        Returns:
            List of (matched_text, severity) tuples
        """
        matches = []
        for regex, sev in self.english_insult_regex:
            found = regex.findall(text)
            if found:
                for match in found:
                    matches.append((match, sev))
        return matches
    
    def get_harm_tokens(self, text: str) -> List[Dict]:
        """
        Get all harm-contributing tokens from text.
        Includes obfuscated slurs and English insults.
        
        Returns:
            List of dicts with token info
        """
        matches = self.scan_text(text)
        harm_tokens = []
        
        for token, entry in matches:
            harm_tokens.append({
                'token': token,
                'original': entry.term,
                'category': entry.category.value,
                'severity': entry.severity.value,
                'language': entry.language,
                'is_identity_targeting': entry.is_identity_targeting
            })
        
        # Add obfuscated slurs
        obfuscated = self.check_obfuscated_slurs(text)
        for match, orig, sev in obfuscated:
            harm_tokens.append({
                'token': match,
                'original': orig,
                'category': 'SLUR',
                'severity': sev.value,
                'language': 'Mixed',
                'is_identity_targeting': orig in ['katua', 'katue', 'chamar', 'bhangi']
            })
        
        # Add English insults
        insults = self.check_english_insults(text)
        for match, sev in insults:
            harm_tokens.append({
                'token': match,
                'original': match,
                'category': 'SLUR',
                'severity': sev.value,
                'language': 'English',
                'is_identity_targeting': False
            })
        
        return harm_tokens
    
    def has_critical_content(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text has critical-level content (auto-escalate to hate).
        
        Returns:
            Tuple of (has_critical, reasons)
        """
        reasons = []
        
        # Check for dehumanization
        dehuman = self.check_dehumanization(text)
        if dehuman:
            reasons.append(f"Dehumanization detected: {dehuman[:3]}")
        
        # Check for violence with identity
        violence = self.check_violence(text)
        identity = self.check_identity_targeting(text)
        
        if violence:
            reasons.append(f"Violence incitement: {violence[:3]}")
            if identity:
                reasons.append(f"Targeting identity group: {identity[:3]}")
        
        # Check for critical severity tokens
        matches = self.scan_text(text)
        critical = [t for t, e in matches if e.severity == SeverityLevel.CRITICAL]
        if critical:
            reasons.append(f"Critical slurs: {critical[:5]}")
        
        # Check for obfuscated critical slurs
        obfuscated = self.check_obfuscated_slurs(text)
        critical_obfuscated = [m for m, o, s in obfuscated if s == SeverityLevel.CRITICAL]
        if critical_obfuscated:
            reasons.append(f"Obfuscated critical slurs: {critical_obfuscated[:5]}")
        
        # Check for threat emojis with identity targeting
        threat_emojis = self.check_threat_emojis(text)
        if threat_emojis and identity:
            reasons.append(f"Threat emojis {threat_emojis[:3]} targeting {identity[:2]}")
        
        # Check for mockery emojis with identity targeting
        mockery_emojis = self.check_mockery_emojis(text)
        if mockery_emojis and identity:
            reasons.append(f"Mockery emojis {mockery_emojis[:3]} targeting {identity[:2]}")
        
        return len(reasons) > 0, reasons
