"""
MUST++ API Server

REST API for the multilingual hate speech detection pipeline.
Designed to serve the operator-grade frontend.

Endpoints:
- POST /analyze - Classify text
- GET /health - System health check
- GET /config - Get system configuration
"""

import sys
import os
import time
import hashlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import MUST++ pipeline
from src.pipeline import (
    MUSTPlusPipeline, 
    MUSTPlusOutput, 
    PipelineConfig,
    ScriptDetector,
    ScriptType,
    Language
)
from src.utils.env import get_env_bool, get_env_str
from src.utils.model_download import preload_models

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Optional: preload paper-specified transformer models
if get_env_bool("MUST_PRELOAD_MODELS", default=False):
    preload_models()

# Initialize pipeline with config - defer model loading
config = PipelineConfig(
    enable_logging=True,
    model_timeout_seconds=30.0,
    disable_transformer=get_env_bool("MUST_DISABLE_TRANSFORMER", default=True)
)

model_name = get_env_str("MUST_MODEL_NAME", default="bert-base-multilingual-cased")
pipeline = MUSTPlusPipeline(model_name=model_name, config=config)
script_detector = ScriptDetector()

# Flag to track if we've tried to enable transformer
_transformer_init_attempted = False


def detect_scripts_realtime(text: str) -> dict:
    """
    Detect scripts in text for real-time indicator.
    Returns script distribution for UI display.
    """
    if not text:
        return {"scripts": {}, "primary_script": None, "is_mixed": False}

    # Use MUST++ ScriptDetector output, but map into frontend-facing categories:
    # latin | devanagari | tamil
    profile = script_detector.detect_language_profile(text)

    mass = {"latin": 0.0, "devanagari": 0.0, "tamil": 0.0}
    total_mass = 0.0

    for token_info in profile.tokens:
        token = token_info.token or ""
        token_mass = float(len(token)) if token else 0.0
        if token_mass <= 0:
            continue

        total_mass += token_mass

        if token_info.script == ScriptType.SCRIPT_NATIVE:
            if token_info.language == Language.HINDI:
                mass["devanagari"] += token_mass
            elif token_info.language == Language.TAMIL:
                mass["tamil"] += token_mass
            else:
                mass["latin"] += token_mass
        else:
            mass["latin"] += token_mass

    if total_mass <= 0:
        return {"scripts": {}, "primary_script": None, "is_mixed": False}

    scripts = {
        script: round(script_mass / total_mass, 3)
        for script, script_mass in mass.items()
        if script_mass / total_mass > 0.01
    }

    primary_script = max(scripts, key=scripts.get) if scripts else None
    is_mixed = len([s for s, p in scripts.items() if p > 0.1]) > 1

    return {"scripts": scripts, "primary_script": primary_script, "is_mixed": is_mixed}


@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/health', methods=['GET'])
def health():
    """System health check."""
    # In fallback mode, transformer is intentionally disabled
    transformer_available = not config.disable_transformer and pipeline.is_transformer_available()
    transformer_status = "available" if transformer_available else "unavailable (fallback mode)"
    
    return jsonify({
        "status": "healthy",
        "transformer": transformer_status,
        "degraded_mode": not transformer_available,
        "fallback_mode_intentional": config.disable_transformer,
        "supported_languages": ["tamil", "hindi", "english"],
        "supported_scripts": ["tamil", "devanagari", "latin"],
        "version": "1.0.0"
    })


@app.route('/config', methods=['GET'])
def get_config():
    """Get system configuration."""
    return jsonify({
        "confidence_threshold": config.confidence_threshold,
        "entropy_threshold": config.entropy_threshold,
        "transformer_available": pipeline.is_transformer_available(),
        "labels": ["neutral", "offensive", "hate"],
        "safety_priority": "hate > offensive > neutral"
    })


@app.route('/detect-script', methods=['POST'])
def detect_script():
    """Real-time script detection for UI indicator."""
    data = request.get_json()
    text = (data or {}).get('text', '')
    
    result = detect_scripts_realtime(text)
    return jsonify(result)


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main classification endpoint.
    
    Request:
        {
            "text": "string",
            "language_hint": "string" (optional)
        }
    
    Response:
        Complete MUSTPlusOutput with all fields for frontend rendering.
    """
    start_time = time.time()
    
    data = request.get_json()
    text = data.get('text', '').strip()
    language_hint = data.get('language_hint')  # Optional, not fully trusted
    
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
    
    try:
        # Run classification
        result: MUSTPlusOutput = pipeline.classify(text)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build comprehensive response for frontend
        response = {
            # === Decision Layer (Primary View) ===
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
                "script_distribution": detect_scripts_realtime(text).get("scripts", {}),
                "is_code_mixed": result.is_code_mixed,
                "transformer_used": not result.degraded_mode,
                "transformer_prediction": result.transformer_prediction,
                "transformer_confidence": round(result.transformer_confidence, 4),
                "confidence_gate_decision": _infer_gate_decision(result),
                "fallback_used": result.fallback_used,
                "fallback_tier": result.fallback_tier,
                "escalation_triggered": result.escalation_triggered,
                "entropy": round(result.entropy, 4),
                "tokenization_coverage": round(result.tokenization_coverage, 4),
                "degraded_mode": result.degraded_mode
            },
            
            # === Metadata ===
            "metadata": {
                "processing_time_ms": round(processing_time_ms, 2),
                "text_length": len(text),
                "text_hash": hashlib.md5(text.encode()).hexdigest()[:12],
                "language_hint_provided": language_hint is not None,
                "language_hint_value": language_hint
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Never silently fail - return safe default with explanation
        processing_time_ms = (time.time() - start_time) * 1000
        
        return jsonify({
            "error": str(e),
            "error_type": "system",
            "safe_default": {
                "label": "neutral",
                "confidence": 0.0,
                "explanation": "System encountered an error. Returning safe default. Please retry.",
                "fallback_used": True,
                "escalation_triggered": True
            },
            "metadata": {
                "processing_time_ms": round(processing_time_ms, 2),
                "text_length": len(text),
                "degraded_mode": True
            }
        }), 500


def _compute_safety_badge(result: MUSTPlusOutput) -> dict:
    """Compute safety badge for Decision Layer."""
    if result.escalation_triggered:
        return {
            "type": "rule_escalation",
            "label": "Rule Escalation",
            "tooltip": "Safety rules triggered escalation due to detected harm signals."
        }
    elif result.fallback_used:
        tier_desc = {
            1: "Primary classifier",
            2: "Indic NLP fallback",
            3: "Safety-first fallback",
            4: "Rule-based fallback"
        }
        return {
            "type": "fallback_used",
            "label": "Fallback Used",
            "tooltip": f"Classification used {tier_desc.get(result.fallback_tier, 'fallback')} (Tier {result.fallback_tier})."
        }
    else:
        return {
            "type": "normal",
            "label": "Normal",
            "tooltip": "Classification completed with primary classifier."
        }


def _generate_label_justification(result: MUSTPlusOutput) -> str:
    """Generate human-readable label justification."""
    if result.label == "hate":
        if result.key_harm_tokens:
            tokens = ", ".join(f'"{t}"' for t in result.key_harm_tokens[:3])
            return f"Classified as hate due to presence of harmful terms: {tokens}"
        return "Classified as hate based on detected hate speech patterns."
    elif result.label == "offensive":
        return "Classified as offensive due to vulgar or inappropriate language, but no targeted hate detected."
    else:
        return "No harmful content detected. Text appears neutral."


def _explain_rejected_labels(result: MUSTPlusOutput) -> list:
    """Explain why weaker labels were rejected."""
    rejected = []
    
    if result.label == "hate":
        if result.rejection_reasons.get("offensive"):
            rejected.append({
                "label": "offensive",
                "reason": result.rejection_reasons["offensive"]
            })
        else:
            rejected.append({
                "label": "offensive",
                "reason": "Escalated to hate due to targeted slurs or identity-based attacks."
            })
        rejected.append({
            "label": "neutral",
            "reason": "Harmful content detected."
        })
    elif result.label == "offensive":
        if result.rejection_reasons.get("neutral"):
            rejected.append({
                "label": "neutral",
                "reason": result.rejection_reasons["neutral"]
            })
        else:
            rejected.append({
                "label": "neutral",
                "reason": "Vulgar or inappropriate language detected."
            })
    
    return rejected


def _format_script_distribution(script_dist: dict) -> dict:
    """Format script distribution for frontend display."""
    formatted = {}
    for script, proportion in script_dist.items():
        # Handle both enum and string keys
        script_name = script.value if hasattr(script, 'value') else str(script)
        formatted[script_name] = round(proportion, 3)
    return formatted


def _infer_gate_decision(result: MUSTPlusOutput) -> str:
    """Infer confidence gate decision for display."""
    if result.degraded_mode:
        return "bypassed_degraded_mode"
    elif result.fallback_used:
        if result.entropy > 0.5:
            return "uncertain_high_entropy"
        elif result.transformer_confidence < 0.75:
            return "low_confidence"
        else:
            return "fallback_override"
    else:
        return "accepted"


if __name__ == '__main__':
    print("=" * 60)
    print("MUST++ API Server")
    print("=" * 60)
    print("Running in fallback-only mode (transformer disabled for quick startup)")
    print("To enable transformer, set disable_transformer=False in config")
    print("=" * 60)
    print("Starting server on http://localhost:8080")
    print("Open http://localhost:8080 in your browser")
    print("=" * 60)
    
    debug = get_env_bool("MUST_API_DEBUG", default=False)
    app.run(host='0.0.0.0', port=8080, debug=debug, use_reloader=debug)
