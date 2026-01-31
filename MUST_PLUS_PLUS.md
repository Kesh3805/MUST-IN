# MUST++ System Documentation

## Overview

MUST++ is a multilingual hate speech detection engine designed for low-resource, code-mixed Indian language environments. It operates as a **linguistic firewall** that inspects, routes, verifies, and explains text classifications.

### Supported Languages
- **Tamil** (Native script + Romanized/Tanglish)
- **Hindi** (Native script + Romanized/Hinglish)
- **English**

### Classification Labels
| Label | Description |
|-------|-------------|
| `neutral` | No harmful content detected |
| `offensive` | Inappropriate/vulgar but not hate speech |
| `hate` | Targeted hate speech against identity groups |

---

## Pipeline Architecture

The MUST++ pipeline follows a strict 7-step classification flow:

### Step 1: Language + Script Detection
```
Input: Raw text
Output: LanguageProfile with:
  - Primary language
  - Language proportions (token mass weighted)
  - Script distribution (SCRIPT_NATIVE, SCRIPT_ROMAN, SCRIPT_ENGLISH)
  - Code-mixing indicator
  - Secondary languages (if >= 20% token mass)
```

**Key Rules:**
- Use token mass (length × confidence), not token count
- Never rely on user-declared language
- Assign script tags to each token

### Step 2: Loss-Aware Normalization
```
PRESERVES:
  ✓ Slurs and abusive terms
  ✓ Cultural/caste/religious markers
  ✓ Romanized slang (soft-normalized only)

REMOVES:
  ✗ URLs and mentions
  ✗ Pure noise

CONVERTS:
  • Emojis → Semantic intent tokens (INTENT_MOCKERY, INTENT_ANGER, etc.)
```

**Critical**: Never translate to English, never over-clean profanity.

### Step 3: Primary Classification (Transformer)
```
Model: Multilingual transformer (e.g., mBERT)
Input: Tagged text with language/script markers
Output:
  - Probability for each class [neutral, offensive, hate]
  - Prediction entropy
  - Tokenization for coverage analysis
```

### Step 4: Confidence Gate
```
Evaluation metrics:
  - Max probability (threshold: 0.75)
  - Entropy (low < 0.5 < high)
  - Tokenization coverage (threshold: 0.80)

Decision:
  IF confidence >= 0.75 AND entropy is low:
      → ACCEPT transformer result
  ELSE:
      → TRIGGER fallback logic
```

### Step 5: Fallback Logic (Mandatory)

| Tier | Name | When Used | Focus |
|------|------|-----------|-------|
| **Tier 4** | Rule-Based Escalation | Always checked first | Dehumanization, violence, identity targeting |
| **Tier 2** | Indic Specialist | Tamil/Hindi dominant, Romanized | Cultural insults, caste/religion/gender |
| **Tier 3** | Classical Safety | Short text, high uncertainty | Hate lexicon, identity slurs |

**Tier 4 Auto-Escalation to HATE:**
- Dehumanization detected
- Calls for violence
- Collective identity targeting with harm intent

Rules OVERRIDE model confidence.

### Step 6: Final Decision Resolution
```
Priority order: hate > offensive > neutral

Ambiguity rule:
  IF uncertain between hate and offensive:
      → Choose HATE
      → Explicitly explain why
```

### Step 7: Explainability
Every output includes:
- Harm-contributing tokens/phrases
- Why the label applies
- Why weaker labels were rejected
- Text evidence (no speculation)

---

## Output Format

```json
{
  "label": "offensive",
  "confidence": 0.85,
  "languages_detected": {"Hindi": 0.75, "English": 0.25},
  "fallback_used": true,
  "key_harm_tokens": ["chutiya", "sala"],
  "explanation": "Offensive language detected: ['chutiya', 'sala'] | No identity-targeting escalation"
}
```

---

## Usage

### Command Line

```bash
# Single text
python inference.py --text "Your text here"

# Interactive mode
python inference.py --interactive

# Batch processing
python inference.py --file texts.txt --output results.json

# Demo with samples
python inference.py --demo
```

### Python API

```python
from src.pipeline import MUSTPlusPipeline

pipeline = MUSTPlusPipeline(
    model_name="bert-base-multilingual-cased",
    confidence_threshold=0.75
)

# Classify
result = pipeline.classify("text to analyze")
print(result.label)       # "neutral", "offensive", or "hate"
print(result.confidence)  # 0.0 - 1.0
print(result.explanation) # Text-grounded justification

# Detailed analysis
analysis = pipeline.get_harm_analysis("text")
explanation = pipeline.explain_prediction("text")
```

---

## Configuration

### Confidence Thresholds
```python
CONFIDENCE_THRESHOLD = 0.75    # Minimum to accept transformer
ENTROPY_LOW_THRESHOLD = 0.5    # Below = confident prediction
ENTROPY_HIGH_THRESHOLD = 1.0   # Above = very uncertain
COVERAGE_THRESHOLD = 0.80      # Minimum known token ratio
```

### Emoji Intent Mapping
```python
EMOJI_INTENT_MAP = {
    "enraged_face": "INTENT_ANGER",
    "skull": "INTENT_THREAT",
    "clown_face": "INTENT_MOCKERY",
    ...
}
```

---

## Module Reference

### `src/pipeline/must_pipeline.py`
Main orchestrator implementing the 7-step pipeline.

### `src/pipeline/script_detector.py`
Language and script detection with token mass weighting.

### `src/pipeline/confidence_gate.py`
Confidence evaluation using entropy and coverage metrics.

### `src/pipeline/fallback_logic.py`
Tiered fallback reasoning (Indic, Safety, Rule-based).

### `src/pipeline/hate_lexicon.py`
Comprehensive hate speech lexicon for Tamil, Hindi, English.

### `src/pipeline/decision_resolver.py`
Conservative decision combination with priority ordering.

### `src/preprocessing/cleaner.py`
Loss-aware text normalization.

### `src/xai/explainer.py`
MUST++ explainability with LIME integration.

---

## Safety Principles

1. **Never invent intent** beyond linguistic evidence
2. **When uncertain, bias toward safety**, not neutrality
3. **Explanations are mandatory** and must be text-grounded
4. **A single model is never trusted blindly**
5. **Rule-based escalation overrides** model confidence

---

## Forbidden Behaviors

The system must NOT:
- ❌ Collapse labels into binary toxicity
- ❌ Ignore Romanized language
- ❌ Skip fallback logic
- ❌ Produce explanations without text grounding
- ❌ Claim intent not expressed in text

---

## Failure Handling

- If evidence is weak, do NOT default to neutral
- If identity groups are targeted, elevate severity
- If unsure, explain uncertainty explicitly
