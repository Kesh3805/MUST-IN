# MUST++ System Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION                                 │
│                                                                           │
│  Browser → http://localhost:8080                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        WEB INTERFACE                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  │  Input   │  │ Decision │  │ Explain  │  │  System  │           │  │
│  │  │  Layer   │  │  Layer   │  │  Layer   │  │  Trace   │           │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │  │
│  │                                                                     │  │
│  │  Features:                                                          │  │
│  │  • Real-time script detection                                       │  │
│  │  • Multilingual input (Tamil/Hindi/English/Mixed)                   │  │
│  │  • Classification with confidence                                   │  │
│  │  • Harm token highlighting                                          │  │
│  │  • History tracking (50 items)                                      │  │
│  │  • Dark mode, keyboard shortcuts                                    │  │
│  │                                                                     │  │
│  │  Files: index.html (408 lines)                                      │  │
│  │         app.js (1,085 lines)                                        │  │
│  │         styles.css (1,481 lines)                                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                              HTTP/JSON API
                                     │
┌───────────────────────────────────────────────────────────────────────────┐
│                           FLASK API SERVER                                 │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  API Endpoints:                                                    │   │
│  │  • GET  /           → Serve frontend                               │   │
│  │  • GET  /health     → System health check                          │   │
│  │  • GET  /config     → Configuration info                           │   │
│  │  • POST /detect-script → Real-time script detection                │   │
│  │  • POST /analyze    → Main classification endpoint                 │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  Two Server Modes:                                                        │
│  ┌─────────────────────┐           ┌─────────────────────┐              │
│  │   app_lite.py       │           │   app.py            │              │
│  │   (Lightweight)     │           │   (Full Pipeline)   │              │
│  │                     │           │                     │              │
│  │  • Fast startup     │           │  • With transformers│              │
│  │    (2-3 seconds)    │           │    (30-60 seconds)  │              │
│  │  • Fallback only    │           │  • High accuracy    │              │
│  │  • Demo/Development │           │  • Production ready │              │
│  │                     │           │                     │              │
│  │  942 lines          │           │  372 lines          │              │
│  └─────────────────────┘           └─────────────────────┘              │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                             Calls Pipeline
                                     │
┌───────────────────────────────────────────────────────────────────────────┐
│                      MUST++ PIPELINE (7 Steps)                             │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: Script Detection                                          │  │
│  │  • Detect Tamil (தமிழ்), Hindi (हिंदी), English, Mixed             │  │
│  │  • Script distribution percentages                                  │  │
│  │  • Code-mixed indicator                                             │  │
│  │                                                                     │  │
│  │  Module: src/pipeline/script_detector.py                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: Language Identification                                   │  │
│  │  • Identify Hindi, Tamil, English, Romanized variants              │  │
│  │  • Language confidence scores                                       │  │
│  │  • Multi-language detection                                         │  │
│  │                                                                     │  │
│  │  Module: src/models/language_id.py                                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: Transformer Classification                                │  │
│  │  • mBERT or XLM-RoBERTa (if enabled)                               │  │
│  │  • Initial prediction: Neutral/Offensive/Hate                      │  │
│  │  • Confidence score                                                 │  │
│  │                                                                     │  │
│  │  Module: src/models/classifiers.py                                 │  │
│  │  Models: bert-base-multilingual-cased, xlm-roberta-base            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: Confidence Gating                                         │  │
│  │  • Check confidence threshold (0.75)                                │  │
│  │  • Check entropy threshold (0.5)                                    │  │
│  │  • Decide: Accept or trigger fallback                              │  │
│  │                                                                     │  │
│  │  Module: src/pipeline/confidence_gate.py                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 5: Multi-Tier Fallback                                       │  │
│  │  • Tier 1: Primary classifier (transformer)                        │  │
│  │  • Tier 2: Indic NLP fallback                                      │  │
│  │  • Tier 3: Traditional ML (SVM, Naive Bayes, Random Forest)       │  │
│  │  • Tier 4: Rule-based safety net                                   │  │
│  │                                                                     │  │
│  │  Module: src/pipeline/fallback_logic.py                            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 6: Rule-Based Escalation                                     │  │
│  │  • Hate lexicon matching (Hindi, Tamil, English)                   │  │
│  │  • Identity group detection                                         │  │
│  │  • Pattern-based escalation                                         │  │
│  │  • Override to higher severity if needed                            │  │
│  │                                                                     │  │
│  │  Module: src/pipeline/hate_lexicon.py                              │  │
│  │  Module: src/pipeline/decision_resolver.py                         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 7: Explainability                                            │  │
│  │  • LIME-based explanation                                           │  │
│  │  • Highlight harm tokens                                            │  │
│  │  • Generate justification                                           │  │
│  │  • Explain rejected labels                                          │  │
│  │                                                                     │  │
│  │  Module: src/xai/explainer.py                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  Output: MUSTPlusOutput                                                   │
│  • label (neutral/offensive/hate)                                         │
│  • confidence (0.0 - 1.0)                                                 │
│  • explanation (human-readable)                                           │
│  • key_harm_tokens (detected harmful words)                               │
│  • identity_groups_detected (targeted groups)                             │
│  • system_trace (languages, scripts, fallback status)                     │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                              Returns JSON
                                     │
┌───────────────────────────────────────────────────────────────────────────┐
│                         RESPONSE TO FRONTEND                               │
│                                                                           │
│  {                                                                        │
│    "label": "offensive",                                                  │
│    "confidence": 0.87,                                                    │
│    "safety_badge": {                                                      │
│      "type": "normal",                                                    │
│      "label": "Normal"                                                    │
│    },                                                                     │
│    "explanation": {                                                       │
│      "summary": "Classified as offensive...",                             │
│      "key_harm_tokens": ["बेवकूफ"],                                      │
│      "label_justification": "Vulgar language detected"                   │
│    },                                                                     │
│    "system_trace": {                                                      │
│      "languages_detected": ["hindi", "english"],                          │
│      "script_distribution": {"devanagari": 0.4, "latin": 0.6},          │
│      "is_code_mixed": true,                                               │
│      "transformer_used": false,                                           │
│      "fallback_used": true,                                               │
│      "processing_time_ms": 125.3                                          │
│    }                                                                      │
│  }                                                                        │
│                                                                           │
│  Frontend displays this in 4 layers with full detail                      │
└───────────────────────────────────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════════
SUPPORTED LANGUAGES & SCRIPTS
════════════════════════════════════════════════════════════════════════════

┌─────────────┬──────────────┬──────────────┬─────────────────────────────┐
│  Language   │    Script    │   Example    │          Variants           │
├─────────────┼──────────────┼──────────────┼─────────────────────────────┤
│   Tamil     │    Tamil     │ இது சோதனை   │ Native + Tanglish (Roman)   │
│   Hindi     │  Devanagari  │ यह परीक्षण   │ Native + Hinglish (Roman)   │
│   English   │    Latin     │ This is test │ Native only                 │
│   Mixed     │   Multiple   │ This is தமிழ் │ Any combination             │
└─────────────┴──────────────┴──────────────┴─────────────────────────────┘


════════════════════════════════════════════════════════════════════════════
CLASSIFICATION LABELS
════════════════════════════════════════════════════════════════════════════

• NEUTRAL    - No harmful content detected
• OFFENSIVE  - Vulgar or inappropriate language, no targeted hate
• HATE       - Targeted hate speech with identity-based attacks


════════════════════════════════════════════════════════════════════════════
SYSTEM FEATURES
════════════════════════════════════════════════════════════════════════════

✅ Real-time script detection     ✅ History tracking (50 items)
✅ Multilingual classification     ✅ Dark mode support
✅ Confidence scoring              ✅ Keyboard shortcuts
✅ Harm token highlighting         ✅ Copy/export results
✅ Full explainability             ✅ System health monitoring
✅ Fallback system (4 tiers)       ✅ Accessibility (ARIA, color-blind)
✅ Rule-based escalation           ✅ Production-ready API


════════════════════════════════════════════════════════════════════════════
QUICK START
════════════════════════════════════════════════════════════════════════════

1. Run: scripts\start_server.bat
2. Open: http://localhost:8080
3. Test: Use examples from TESTING_EXAMPLES.html

For detailed guide, see: LAUNCH_GUIDE.md
```
