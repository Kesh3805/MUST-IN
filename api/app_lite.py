"""
MUST++ Lightweight API Server

A lightweight API server that uses only the fallback components
(no transformer/torch loading) for fast startup and frontend development.

This server demonstrates the full UI with the safety-first fallback system.
"""

import sys
import os
import time
import hashlib
import re
import unicodedata
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from functools import lru_cache
from collections import defaultdict
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ==================== Logging Setup ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
# Enable CORS with explicit configuration for local development
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})


# ==================== Rate Limiting ====================
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        self.requests[key].append(now)
        return True
    
    def get_remaining(self, key: str) -> int:
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        valid_requests = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        return max(0, self.max_requests - len(valid_requests))


# ==================== Analysis Cache ====================
class AnalysisCache:
    """LRU cache for analysis results with TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
    
    def _make_key(self, text: str, language_hint: Optional[str]) -> str:
        key_str = f"{text}|{language_hint or ''}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, text: str, language_hint: Optional[str] = None):
        key = self._make_key(text, language_hint)
        
        if key not in self.cache:
            return None
        
        # Check TTL
        if datetime.now() - self.access_times[key] > timedelta(seconds=self.ttl_seconds):
            del self.cache[key]
            del self.access_times[key]
            return None
        
        self.access_times[key] = datetime.now()
        return self.cache[key]
    
    def set(self, text: str, language_hint: Optional[str], result: dict):
        key = self._make_key(text, language_hint)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = result
        self.access_times[key] = datetime.now()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()


# Initialize rate limiter and cache
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
analysis_cache = AnalysisCache(max_size=500, ttl_seconds=1800)


# ==================== Request Stats ====================
class RequestStats:
    """Track request statistics."""
    
    def __init__(self):
        self.total_requests = 0
        self.label_counts = defaultdict(int)
        self.avg_processing_time_ms = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = datetime.now()
    
    def record_request(self, label: str, processing_time_ms: float, cached: bool):
        self.total_requests += 1
        self.label_counts[label] += 1
        
        # Running average
        self.avg_processing_time_ms = (
            (self.avg_processing_time_ms * (self.total_requests - 1) + processing_time_ms)
            / self.total_requests
        )
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_stats(self) -> dict:
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "total_requests": self.total_requests,
            "label_distribution": dict(self.label_counts),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "cache_hit_rate": (
                round(self.cache_hits / (self.cache_hits + self.cache_misses), 3)
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
            "uptime_seconds": round(uptime, 0),
            "requests_per_minute": round(self.total_requests / (uptime / 60), 2) if uptime > 0 else 0
        }


stats = RequestStats()


# ==================== Lightweight Components ====================

class ScriptType(Enum):
    LATIN = "latin"
    DEVANAGARI = "devanagari"
    TAMIL = "tamil"
    UNKNOWN = "unknown"


@dataclass
class AnalysisResult:
    """Result from text analysis."""
    label: str
    confidence: float
    languages_detected: Dict[str, float]
    fallback_used: bool
    escalation_triggered: bool
    key_harm_tokens: List[str]
    explanation: str
    script_distribution: Dict[str, float]
    is_code_mixed: bool
    transformer_prediction: str
    transformer_confidence: float
    fallback_tier: Optional[int]
    identity_groups_detected: List[str]
    rejection_reasons: Dict[str, str]
    entropy: float
    tokenization_coverage: float
    degraded_mode: bool = True


class LightweightAnalyzer:
    """
    Lightweight text analyzer using only lexicon-based detection.
    For demo purposes - uses the safety-first fallback system.
    """
    
    def __init__(self):
        # Load hate lexicon
        self.lexicon = self._build_lexicon()
        
    def _build_lexicon(self) -> Dict[str, Dict]:
        """Build comprehensive multilingual hate lexicon."""
        return {
            # ============================================================
            # === HATE SPEECH (Targeted slurs and dehumanizing terms) ===
            # ============================================================
            
            # --- Hindi Caste Slurs ---
            "bhangi": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "bhangion": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "bhangiyon": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "chamar": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "chamaar": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "chamari": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "chuhra": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "chura": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "dom": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "dhobi": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "kanjar": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "kanjari": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "shudra": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "neech": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "neechi jaati": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "achhoot": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "achoot": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            
            # --- Religious Slurs ---
            "katua": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "katue": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "katwe": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "katuon": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "jihadi": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "jihadis": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "mullah": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "mullahs": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "landya": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "sulla": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "sullas": {"severity": "hate", "category": "religious", "groups": ["muslim"]},
            "porkistan": {"severity": "hate", "category": "religious", "groups": ["muslim", "pakistani"]},
            "hindu terrorist": {"severity": "hate", "category": "religious", "groups": ["hindu"]},
            "sanghi": {"severity": "offensive", "category": "religious", "groups": ["hindu"]},
            "bhakt": {"severity": "offensive", "category": "religious", "groups": ["hindu"]},
            "librandus": {"severity": "offensive", "category": "political", "groups": []},
            "liberandu": {"severity": "offensive", "category": "political", "groups": []},
            
            # --- Tamil Caste Slurs (Romanized) ---
            "paraiyan": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "paraiyar": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "parayan": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "paraiya": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "pallar": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "pallan": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "chakkiliyar": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "chakkiliyan": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "parpan": {"severity": "hate", "category": "caste", "groups": ["brahmin"]},
            "parpaan": {"severity": "hate", "category": "caste", "groups": ["brahmin"]},
            "parppan": {"severity": "hate", "category": "caste", "groups": ["brahmin"]},
            "pappaan": {"severity": "hate", "category": "caste", "groups": ["brahmin"]},
            "koothichi": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "kooththi": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "koothi": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "arunthathiyar": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            
            # --- Tamil Slurs (Native Script) ---
            "பார்ப்பான்": {"severity": "hate", "category": "caste", "groups": ["brahmin"]},
            "பறையன்": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "பள்ளர்": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "சக்கிலியன்": {"severity": "hate", "category": "caste", "groups": ["dalit"]},
            "ஒழிய": {"severity": "hate", "category": "violence", "groups": []},
            "கொல்லு": {"severity": "hate", "category": "violence", "groups": []},
            "தேவடியா": {"severity": "offensive", "category": "vulgar", "groups": []},
            "புண்டை": {"severity": "offensive", "category": "vulgar", "groups": []},
            "சுன்னி": {"severity": "offensive", "category": "vulgar", "groups": []},
            
            # --- Violence Terms (Hindi) ---
            "maro": {"severity": "hate", "category": "violence", "groups": []},
            "maaro": {"severity": "hate", "category": "violence", "groups": []},
            "maar do": {"severity": "hate", "category": "violence", "groups": []},
            "maar dalo": {"severity": "hate", "category": "violence", "groups": []},
            "jala do": {"severity": "hate", "category": "violence", "groups": []},
            "jala de": {"severity": "hate", "category": "violence", "groups": []},
            "jalao": {"severity": "hate", "category": "violence", "groups": []},
            "kaat do": {"severity": "hate", "category": "violence", "groups": []},
            "kaato": {"severity": "hate", "category": "violence", "groups": []},
            "goli maaro": {"severity": "hate", "category": "violence", "groups": []},
            "desh nikala": {"severity": "hate", "category": "violence", "groups": []},
            "pakistan bhejo": {"severity": "hate", "category": "violence", "groups": ["muslim"]},
            "jala": {"severity": "offensive", "category": "violence", "groups": []},
            "kaat": {"severity": "offensive", "category": "violence", "groups": []},
            "nikalo": {"severity": "offensive", "category": "violence", "groups": []},
            "khatam": {"severity": "offensive", "category": "violence", "groups": []},
            "khatam karo": {"severity": "hate", "category": "violence", "groups": []},
            
            # --- Violence Terms (Tamil) ---
            "kollu": {"severity": "hate", "category": "violence", "groups": []},
            "kollanum": {"severity": "hate", "category": "violence", "groups": []},
            "adipom": {"severity": "hate", "category": "violence", "groups": []},
            "adippom": {"severity": "hate", "category": "violence", "groups": []},
            "vettu": {"severity": "offensive", "category": "violence", "groups": []},
            "vettalam": {"severity": "hate", "category": "violence", "groups": []},
            "ozhikanum": {"severity": "hate", "category": "violence", "groups": []},
            
            # --- English Slurs ---
            "faggot": {"severity": "hate", "category": "homophobia", "groups": ["lgbtq"]},
            "fag": {"severity": "hate", "category": "homophobia", "groups": ["lgbtq"]},
            "faggots": {"severity": "hate", "category": "homophobia", "groups": ["lgbtq"]},
            "dyke": {"severity": "hate", "category": "homophobia", "groups": ["lgbtq"]},
            "tranny": {"severity": "hate", "category": "transphobia", "groups": ["lgbtq"]},
            "shemale": {"severity": "hate", "category": "transphobia", "groups": ["lgbtq"]},
            "nigger": {"severity": "hate", "category": "racism", "groups": ["black"]},
            "nigga": {"severity": "hate", "category": "racism", "groups": ["black"]},
            "niggers": {"severity": "hate", "category": "racism", "groups": ["black"]},
            "coon": {"severity": "hate", "category": "racism", "groups": ["black"]},
            "spic": {"severity": "hate", "category": "racism", "groups": ["latino"]},
            "wetback": {"severity": "hate", "category": "racism", "groups": ["latino"]},
            "chink": {"severity": "hate", "category": "racism", "groups": ["asian"]},
            "gook": {"severity": "hate", "category": "racism", "groups": ["asian"]},
            "paki": {"severity": "hate", "category": "racism", "groups": ["pakistani"]},
            "kike": {"severity": "hate", "category": "antisemitism", "groups": ["jewish"]},
            "kyke": {"severity": "hate", "category": "antisemitism", "groups": ["jewish"]},
            "yid": {"severity": "hate", "category": "antisemitism", "groups": ["jewish"]},
            "sandnigger": {"severity": "hate", "category": "racism", "groups": ["arab"]},
            "towelhead": {"severity": "hate", "category": "racism", "groups": ["arab", "muslim"]},
            "raghead": {"severity": "hate", "category": "racism", "groups": ["arab", "muslim"]},
            "terrorist": {"severity": "offensive", "category": "religious", "groups": []},
            "retard": {"severity": "hate", "category": "ableism", "groups": ["disabled"]},
            "retards": {"severity": "hate", "category": "ableism", "groups": ["disabled"]},
            "retarded": {"severity": "hate", "category": "ableism", "groups": ["disabled"]},
            
            # --- Dehumanizing Terms ---
            "cockroach": {"severity": "hate", "category": "dehumanizing", "groups": []},
            "cockroaches": {"severity": "hate", "category": "dehumanizing", "groups": []},
            "vermin": {"severity": "hate", "category": "dehumanizing", "groups": []},
            "termites": {"severity": "hate", "category": "dehumanizing", "groups": []},
            "subhuman": {"severity": "hate", "category": "dehumanizing", "groups": []},
            "animals": {"severity": "offensive", "category": "dehumanizing", "groups": []},
            
            # --- Gender/Misogynistic Slurs ---
            "slut": {"severity": "hate", "category": "misogyny", "groups": ["women"]},
            "whore": {"severity": "hate", "category": "misogyny", "groups": ["women"]},
            "bitch": {"severity": "offensive", "category": "misogyny", "groups": ["women"]},
            "cunt": {"severity": "hate", "category": "misogyny", "groups": ["women"]},
            "randi": {"severity": "hate", "category": "misogyny", "groups": ["women"]},
            "raand": {"severity": "hate", "category": "misogyny", "groups": ["women"]},
            "besharmi": {"severity": "offensive", "category": "misogyny", "groups": ["women"]},
            "thevdiya munda": {"severity": "hate", "category": "misogyny", "groups": ["women"]},
            "veshya": {"severity": "hate", "category": "misogyny", "groups": ["women"]},
            
            # ============================================================
            # === OFFENSIVE (Vulgar but not identity-targeted) ===
            # ============================================================
            
            # --- Hindi Vulgar Terms ---
            "chutiya": {"severity": "offensive", "category": "vulgar", "groups": []},
            "chutiye": {"severity": "offensive", "category": "vulgar", "groups": []},
            "chutiyon": {"severity": "offensive", "category": "vulgar", "groups": []},
            "chut": {"severity": "offensive", "category": "vulgar", "groups": []},
            "choot": {"severity": "offensive", "category": "vulgar", "groups": []},
            "gaand": {"severity": "offensive", "category": "vulgar", "groups": []},
            "gaandu": {"severity": "offensive", "category": "vulgar", "groups": []},
            "gandu": {"severity": "offensive", "category": "vulgar", "groups": []},
            "madarchod": {"severity": "offensive", "category": "vulgar", "groups": []},
            "madarjaat": {"severity": "offensive", "category": "vulgar", "groups": []},
            "behenchod": {"severity": "offensive", "category": "vulgar", "groups": []},
            "bhosdike": {"severity": "offensive", "category": "vulgar", "groups": []},
            "bhosdi": {"severity": "offensive", "category": "vulgar", "groups": []},
            "bhosdiwale": {"severity": "offensive", "category": "vulgar", "groups": []},
            "lund": {"severity": "offensive", "category": "vulgar", "groups": []},
            "loda": {"severity": "offensive", "category": "vulgar", "groups": []},
            "lauda": {"severity": "offensive", "category": "vulgar", "groups": []},
            "lavda": {"severity": "offensive", "category": "vulgar", "groups": []},
            "jhant": {"severity": "offensive", "category": "vulgar", "groups": []},
            "harami": {"severity": "offensive", "category": "vulgar", "groups": []},
            "haramzaade": {"severity": "offensive", "category": "vulgar", "groups": []},
            "kutte": {"severity": "offensive", "category": "vulgar", "groups": []},
            "kutta": {"severity": "offensive", "category": "vulgar", "groups": []},
            "kutiya": {"severity": "offensive", "category": "vulgar", "groups": []},
            "suar": {"severity": "offensive", "category": "vulgar", "groups": []},
            "bc": {"severity": "offensive", "category": "vulgar", "groups": []},
            "mc": {"severity": "offensive", "category": "vulgar", "groups": []},
            "bsdk": {"severity": "offensive", "category": "vulgar", "groups": []},
            
            # --- Tamil Vulgar Terms ---
            "thevidiya": {"severity": "offensive", "category": "vulgar", "groups": []},
            "thevdiya": {"severity": "offensive", "category": "vulgar", "groups": []},
            "thevidiyaa": {"severity": "offensive", "category": "vulgar", "groups": []},
            "thevdiyaa": {"severity": "offensive", "category": "vulgar", "groups": []},
            "thevidiya paiyan": {"severity": "offensive", "category": "vulgar", "groups": []},
            "thevadiya": {"severity": "offensive", "category": "vulgar", "groups": []},
            "thevudiya": {"severity": "offensive", "category": "vulgar", "groups": []},
            "otha": {"severity": "offensive", "category": "vulgar", "groups": []},
            "ottha": {"severity": "offensive", "category": "vulgar", "groups": []},
            "oththaa": {"severity": "offensive", "category": "vulgar", "groups": []},
            "poda": {"severity": "offensive", "category": "vulgar", "groups": []},
            "podi": {"severity": "offensive", "category": "vulgar", "groups": []},
            "podaa": {"severity": "offensive", "category": "vulgar", "groups": []},
            "podii": {"severity": "offensive", "category": "vulgar", "groups": []},
            "po da": {"severity": "offensive", "category": "vulgar", "groups": []},
            "pundai": {"severity": "offensive", "category": "vulgar", "groups": []},
            "punda": {"severity": "offensive", "category": "vulgar", "groups": []},
            "pundek": {"severity": "offensive", "category": "vulgar", "groups": []},
            "sunni": {"severity": "offensive", "category": "vulgar", "groups": []},
            "sunnii": {"severity": "offensive", "category": "vulgar", "groups": []},
            "baadu": {"severity": "offensive", "category": "vulgar", "groups": []},
            "baaduu": {"severity": "offensive", "category": "vulgar", "groups": []},
            "loosu": {"severity": "offensive", "category": "vulgar", "groups": []},
            "loosa": {"severity": "offensive", "category": "vulgar", "groups": []},
            "mayiru": {"severity": "offensive", "category": "vulgar", "groups": []},
            "mayir": {"severity": "offensive", "category": "vulgar", "groups": []},
            "molai": {"severity": "offensive", "category": "vulgar", "groups": []},
            "thayoli": {"severity": "offensive", "category": "vulgar", "groups": []},
            "ommala": {"severity": "offensive", "category": "vulgar", "groups": []},
            "oombu": {"severity": "offensive", "category": "vulgar", "groups": []},
            "oombuda": {"severity": "offensive", "category": "vulgar", "groups": []},
            "naaye": {"severity": "offensive", "category": "vulgar", "groups": []},
            "naay": {"severity": "offensive", "category": "vulgar", "groups": []},
            "panni": {"severity": "offensive", "category": "vulgar", "groups": []},
            "pannii": {"severity": "offensive", "category": "vulgar", "groups": []},
            "kena": {"severity": "offensive", "category": "vulgar", "groups": []},
            "kenaa": {"severity": "offensive", "category": "vulgar", "groups": []},
            "koodhi": {"severity": "offensive", "category": "vulgar", "groups": []},
            "sothappal": {"severity": "offensive", "category": "vulgar", "groups": []},
            "olunga": {"severity": "offensive", "category": "vulgar", "groups": []},
            
            # --- English Vulgar Terms ---
            "fuck": {"severity": "offensive", "category": "vulgar", "groups": []},
            "fucking": {"severity": "offensive", "category": "vulgar", "groups": []},
            "fucker": {"severity": "offensive", "category": "vulgar", "groups": []},
            "fucked": {"severity": "offensive", "category": "vulgar", "groups": []},
            "fck": {"severity": "offensive", "category": "vulgar", "groups": []},
            "shit": {"severity": "offensive", "category": "vulgar", "groups": []},
            "shitty": {"severity": "offensive", "category": "vulgar", "groups": []},
            "bullshit": {"severity": "offensive", "category": "vulgar", "groups": []},
            "asshole": {"severity": "offensive", "category": "vulgar", "groups": []},
            "ass": {"severity": "offensive", "category": "vulgar", "groups": []},
            "bastard": {"severity": "offensive", "category": "vulgar", "groups": []},
            "bastards": {"severity": "offensive", "category": "vulgar", "groups": []},
            "dick": {"severity": "offensive", "category": "vulgar", "groups": []},
            "dickhead": {"severity": "offensive", "category": "vulgar", "groups": []},
            "damn": {"severity": "offensive", "category": "vulgar", "groups": []},
            "crap": {"severity": "offensive", "category": "vulgar", "groups": []},
            "idiot": {"severity": "offensive", "category": "vulgar", "groups": []},
            "moron": {"severity": "offensive", "category": "vulgar", "groups": []},
            "dumbass": {"severity": "offensive", "category": "vulgar", "groups": []},
            "stupid": {"severity": "offensive", "category": "vulgar", "groups": []},
            "stfu": {"severity": "offensive", "category": "vulgar", "groups": []},
            "wtf": {"severity": "offensive", "category": "vulgar", "groups": []},
        }
    
    def detect_scripts(self, text: str) -> Dict[str, float]:
        """Detect script distribution in text."""
        if not text:
            return {}
        
        scripts = {"latin": 0, "devanagari": 0, "tamil": 0, "other": 0}
        total = 0
        
        for char in text:
            if char.isspace() or char in '.,!?;:\'"()-':
                continue
            total += 1
            
            code = ord(char)
            if 0x0041 <= code <= 0x007A:  # Basic Latin
                scripts["latin"] += 1
            elif 0x0900 <= code <= 0x097F:  # Devanagari
                scripts["devanagari"] += 1
            elif 0x0B80 <= code <= 0x0BFF:  # Tamil
                scripts["tamil"] += 1
            else:
                scripts["other"] += 1
        
        if total == 0:
            return {}
        
        return {k: v / total for k, v in scripts.items() if v > 0}
    
    def detect_language(self, text: str, scripts: Dict[str, float]) -> Dict[str, float]:
        """Infer language from script distribution."""
        if not scripts:
            return {"unknown": 1.0}
        
        languages = {}
        
        if scripts.get("tamil", 0) > 0.3:
            languages["tamil"] = scripts.get("tamil", 0)
        if scripts.get("devanagari", 0) > 0.3:
            languages["hindi"] = scripts.get("devanagari", 0)
        if scripts.get("latin", 0) > 0.3:
            # Could be English, Romanized Tamil, or Romanized Hindi
            # Check for language-specific patterns
            text_lower = text.lower()
            if any(w in text_lower for w in ["the", "is", "are", "was", "were", "have", "has"]):
                languages["english"] = scripts.get("latin", 0) * 0.7
            else:
                languages["english"] = scripts.get("latin", 0) * 0.5
                languages["romanized"] = scripts.get("latin", 0) * 0.5
        
        if not languages:
            languages["unknown"] = 1.0
        
        # Normalize
        total = sum(languages.values())
        return {k: v / total for k, v in languages.items()}
    
    def scan_text(self, text: str) -> tuple:
        """
        Scan text for harmful content.
        Returns: (matches, severity, groups)
        """
        # Normalize Unicode (important for Tamil)
        text_normalized = unicodedata.normalize('NFC', text)
        text_lower = text_normalized.lower()
        
        matches = []
        max_severity = "neutral"
        groups = set()
        
        severity_order = {"neutral": 0, "offensive": 1, "hate": 2}
        
        # Word tokenization
        words = re.findall(r'\b\w+\b', text_lower)
        # Also check full text for non-Latin scripts
        words.extend(text_normalized.split())
        
        for word in words:
            word_normalized = unicodedata.normalize('NFC', word.lower())
            
            if word_normalized in self.lexicon:
                entry = self.lexicon[word_normalized]
                matches.append(word)
                groups.update(entry.get("groups", []))
                
                if severity_order.get(entry["severity"], 0) > severity_order.get(max_severity, 0):
                    max_severity = entry["severity"]
        
        return list(set(matches)), max_severity, list(groups)
    
    def analyze(self, text: str) -> AnalysisResult:
        """Analyze text for harmful content."""
        # Detect scripts and languages
        scripts = self.detect_scripts(text)
        languages = self.detect_language(text, scripts)
        
        # Scan for harmful content
        harm_tokens, severity, identity_groups = self.scan_text(text)
        
        # Determine label and confidence
        if severity == "hate":
            label = "hate"
            confidence = 0.95
            explanation = f"Text contains hate speech targeting {', '.join(identity_groups) if identity_groups else 'individuals/groups'}."
            rejection_reasons = {
                "offensive": "Escalated to hate due to targeted slurs.",
                "neutral": "Hate speech detected."
            }
            escalation = True
        elif severity == "offensive":
            label = "offensive"
            confidence = 0.85
            explanation = "Text contains offensive or vulgar language."
            rejection_reasons = {
                "neutral": "Vulgar language detected."
            }
            escalation = False
        else:
            label = "neutral"
            confidence = 0.70
            explanation = "No harmful content detected."
            rejection_reasons = {}
            escalation = False
        
        is_code_mixed = len([s for s, p in scripts.items() if p > 0.1]) > 1
        
        return AnalysisResult(
            label=label,
            confidence=confidence,
            languages_detected=languages,
            fallback_used=True,
            escalation_triggered=escalation,
            key_harm_tokens=harm_tokens,
            explanation=explanation,
            script_distribution=scripts,
            is_code_mixed=is_code_mixed,
            transformer_prediction="(unavailable)",
            transformer_confidence=0.0,
            fallback_tier=3,  # Safety-first fallback
            identity_groups_detected=identity_groups,
            rejection_reasons=rejection_reasons,
            entropy=0.5,
            tokenization_coverage=1.0,
            degraded_mode=True
        )


# Initialize analyzer
analyzer = LightweightAnalyzer()


# ==================== Middleware ====================

@app.before_request
def log_request():
    """Log incoming requests."""
    if request.path not in ['/', '/health', '/favicon.ico'] and not request.path.startswith('/static'):
        logger.info(f"→ {request.method} {request.path}")


@app.after_request
def add_headers(response):
    """Add security and caching headers."""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # CORS headers (already handled by flask-cors, but explicit)
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    return response


# ==================== Routes ====================

@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/test')
def test_page():
    """Serve the API test page."""
    return send_from_directory(app.static_folder, 'test_api.html')


@app.route('/health', methods=['GET'])
def health():
    """System health check."""
    return jsonify({
        "status": "healthy",
        "transformer": "unavailable (lightweight mode)",
        "degraded_mode": True,
        "fallback_mode_intentional": True,
        "supported_languages": ["tamil", "hindi", "english"],
        "supported_scripts": ["tamil", "devanagari", "latin"],
        "version": "1.1.0-lightweight",
        "stats": stats.get_stats()
    })


@app.route('/config', methods=['GET'])
def get_config():
    """Get system configuration."""
    return jsonify({
        "confidence_threshold": 0.75,
        "entropy_threshold": 0.5,
        "transformer_available": False,
        "labels": ["neutral", "offensive", "hate"],
        "safety_priority": "hate > offensive > neutral",
        "mode": "lightweight",
        "rate_limit": {
            "max_requests": rate_limiter.max_requests,
            "window_seconds": rate_limiter.window_seconds
        },
        "cache": {
            "enabled": True,
            "ttl_seconds": analysis_cache.ttl_seconds
        }
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics."""
    return jsonify(stats.get_stats())


@app.route('/detect-script', methods=['POST'])
def detect_script():
    """Real-time script detection for UI indicator."""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"scripts": {}, "primary_script": None, "is_mixed": False})
    
    scripts = analyzer.detect_scripts(text)
    
    primary = None
    if scripts:
        primary = max(scripts, key=scripts.get)
    
    is_mixed = len([s for s, p in scripts.items() if p > 0.1]) > 1
    
    return jsonify({
        "scripts": {k: round(v, 3) for k, v in scripts.items()},
        "primary_script": primary,
        "is_mixed": is_mixed
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Main classification endpoint with rate limiting and caching."""
    start_time = time.time()
    
    # Get client IP for rate limiting
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        remaining = rate_limiter.get_remaining(client_ip)
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return jsonify({
            "error": "Rate limit exceeded",
            "error_type": "rate_limit",
            "retry_after_seconds": rate_limiter.window_seconds,
            "remaining_requests": remaining
        }), 429
    
    data = request.get_json()
    
    # Validate request
    if not data:
        return jsonify({
            "error": "Invalid JSON",
            "error_type": "validation"
        }), 400
    
    text = data.get('text', '').strip()
    language_hint = data.get('language_hint')
    
    if not text:
        return jsonify({
            "error": "No text provided",
            "error_type": "validation",
            "safe_default": {
                "label": "neutral",
                "confidence": 0.0,
                "explanation": "No text was provided for analysis."
            }
        }), 400
    
    # Check text length
    if len(text) > 10000:
        return jsonify({
            "error": "Text too long (max 10,000 characters)",
            "error_type": "validation"
        }), 400
    
    try:
        # Check cache first
        cached_result = analysis_cache.get(text, language_hint)
        if cached_result:
            processing_time_ms = (time.time() - start_time) * 1000
            cached_result['metadata']['processing_time_ms'] = round(processing_time_ms, 2)
            cached_result['metadata']['cached'] = True
            
            stats.record_request(cached_result['label'], processing_time_ms, cached=True)
            logger.info(f"← Cache hit: {cached_result['label']} ({processing_time_ms:.0f}ms)")
            
            return jsonify(cached_result)
        
        # Run analysis
        result = analyzer.analyze(text)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build response
        response = {
            # === Decision Layer ===
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "safety_badge": _compute_safety_badge(result),
            
            # === Explanation Layer ===
            "explanation": {
                "summary": result.explanation,
                "key_harm_tokens": result.key_harm_tokens,
                "identity_groups": result.identity_groups_detected,
                "rejection_reasons": result.rejection_reasons,
                "label_justification": _generate_label_justification(result),
                "weaker_labels_rejected": _explain_rejected_labels(result)
            },
            
            # === System Trace Layer ===
            "system_trace": {
                "languages_detected": result.languages_detected,
                "script_distribution": result.script_distribution,
                "is_code_mixed": result.is_code_mixed,
                "transformer_used": False,
                "transformer_prediction": result.transformer_prediction,
                "transformer_confidence": result.transformer_confidence,
                "confidence_gate_decision": "bypassed_degraded_mode",
                "fallback_used": result.fallback_used,
                "fallback_tier": result.fallback_tier,
                "escalation_triggered": result.escalation_triggered,
                "entropy": result.entropy,
                "tokenization_coverage": result.tokenization_coverage,
                "degraded_mode": result.degraded_mode
            },
            
            # === Metadata ===
            "metadata": {
                "processing_time_ms": round(processing_time_ms, 2),
                "text_length": len(text),
                "text_hash": hashlib.md5(text.encode()).hexdigest()[:12],
                "language_hint_provided": language_hint is not None,
                "language_hint_value": language_hint,
                "cached": False,
                "api_version": "1.1.0"
            }
        }
        
        # Cache result
        analysis_cache.set(text, language_hint, response)
        
        # Record stats
        stats.record_request(result.label, processing_time_ms, cached=False)
        
        # Log result
        log_level = logging.WARNING if result.label == "hate" else logging.INFO
        logger.log(log_level, f"← {result.label.upper()} (conf={result.confidence:.2f}, {processing_time_ms:.0f}ms)")
        
        return jsonify(response)
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Analysis error: {str(e)}")
        
        return jsonify({
            "error": str(e),
            "error_type": "system",
            "safe_default": {
                "label": "neutral",
                "confidence": 0.0,
                "explanation": "System error. Returning safe default.",
                "fallback_used": True,
                "escalation_triggered": True
            },
            "metadata": {
                "processing_time_ms": round(processing_time_ms, 2),
                "text_length": len(text),
                "degraded_mode": True
            }
        }), 500


def _compute_safety_badge(result: AnalysisResult) -> dict:
    """Compute safety badge."""
    if result.escalation_triggered:
        return {
            "type": "rule_escalation",
            "label": "Rule Escalation",
            "tooltip": "Safety rules triggered escalation due to detected harm signals."
        }
    elif result.fallback_used:
        return {
            "type": "fallback_used",
            "label": "Fallback Mode",
            "tooltip": f"Classification using safety-first fallback (Tier {result.fallback_tier})."
        }
    else:
        return {
            "type": "normal",
            "label": "Normal",
            "tooltip": "Classification completed with primary classifier."
        }


def _generate_label_justification(result: AnalysisResult) -> str:
    """Generate human-readable label justification."""
    if result.label == "hate":
        if result.key_harm_tokens:
            tokens = ", ".join(f'"{t}"' for t in result.key_harm_tokens[:3])
            return f"Classified as hate due to presence of harmful terms: {tokens}"
        return "Classified as hate based on detected hate speech patterns."
    elif result.label == "offensive":
        return "Classified as offensive due to vulgar or inappropriate language."
    else:
        return "No harmful content detected. Text appears neutral."


def _explain_rejected_labels(result: AnalysisResult) -> list:
    """Explain why weaker labels were rejected."""
    rejected = []
    
    if result.label == "hate":
        rejected.append({
            "label": "offensive",
            "reason": result.rejection_reasons.get("offensive", "Escalated to hate.")
        })
        rejected.append({
            "label": "neutral",
            "reason": result.rejection_reasons.get("neutral", "Harmful content detected.")
        })
    elif result.label == "offensive":
        rejected.append({
            "label": "neutral",
            "reason": result.rejection_reasons.get("neutral", "Vulgar language detected.")
        })
    
    return rejected


if __name__ == '__main__':
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          MUST++ Lightweight API Server v1.1.0             ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print("║  Mode: LIGHTWEIGHT (no torch/transformers)                ║")
    print("║  Fallback: Safety-first lexicon system                    ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print("║  Features:                                                ║")
    print("║    ✓ Rate limiting (100 req/min)                          ║")
    print("║    ✓ Response caching (30 min TTL)                        ║")
    print("║    ✓ Request logging                                      ║")
    print("║    ✓ Usage statistics (/stats)                            ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print("║  Languages: Tamil, Hindi, English, Code-mixed             ║")
    print("║  Labels: neutral, offensive, hate                         ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print("║  Server: http://localhost:8080                            ║")
    print("║  Health: http://localhost:8080/health                     ║")
    print("║  Stats:  http://localhost:8080/stats                      ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    
    logger.info("Starting MUST++ API server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
