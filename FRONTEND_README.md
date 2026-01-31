# MUST++ Frontend

Operator-grade UI for multilingual hate speech detection.

## Quick Start

### Option 1: Run the batch script (Windows)
```
scripts\start_server.bat
```

### Option 2: Run directly with Python
```bash
python api/app_lite.py
```

Then open your browser to: **http://localhost:8080**

## Features

### 4-Layer Information Architecture

1. **Input Layer**
   - Multi-script text input (Tamil, Hindi, English, mixed)
   - Real-time script detection indicator
   - Character counter
   - Optional language hint

2. **Decision Layer** (Primary View)
   - Final label: Neutral | Offensive | Hate
   - Confidence bar (numeric + visual)
   - Safety badge: Normal | Fallback Used | Rule Escalation

3. **Explanation Layer** (Expandable)
   - Highlighted harm tokens in original text
   - Why this label was chosen
   - Why weaker labels were rejected

4. **System Trace Layer** (Advanced)
   - Languages and scripts detected
   - Transformer/fallback status
   - Confidence gate decision
   - Entropy and coverage metrics

## Design Principles

- **Never hide uncertainty** - All confidence levels visible
- **Never oversimplify harm** - Full explanation available
- **Safety explanations are inspectable** - Nothing hidden
- **Multilingual users understood** - Original text preserved, no translation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/health` | GET | System health check |
| `/config` | GET | System configuration |
| `/detect-script` | POST | Real-time script detection |
| `/analyze` | POST | Main classification endpoint |

### Example Request

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### Response Structure

```json
{
  "label": "neutral",
  "confidence": 0.7,
  "safety_badge": {
    "type": "fallback_used",
    "label": "Fallback Mode",
    "tooltip": "..."
  },
  "explanation": {
    "summary": "...",
    "key_harm_tokens": [],
    "label_justification": "...",
    "weaker_labels_rejected": []
  },
  "system_trace": {
    "languages_detected": {"english": 1.0},
    "script_distribution": {"latin": 1.0},
    "fallback_used": true,
    "escalation_triggered": false,
    "degraded_mode": true
  },
  "metadata": {
    "processing_time_ms": 5.2,
    "text_length": 11
  }
}
```

## Testing

Run the test suite:
```bash
# First, start the server in one terminal
python api/app_lite.py

# Then, in another terminal:
python api/test_api.py
```

## Files

```
frontend/
  ├── index.html      # Main HTML structure
  ├── styles.css      # Clinical, accessible styles
  └── app.js          # UI logic and API integration

api/
  ├── app_lite.py     # Lightweight API server (no transformer)
  ├── app.py          # Full API server (with transformer)
  └── test_api.py     # API test suite
```

## Accessibility

- Color-blind safe palette
- Full keyboard navigation
- ARIA labels and roles
- High contrast text
- Copyable explanations

## Visual Design

- **Calm**: No flashing, no drama
- **Clinical**: Professional moderation tool
- **Neutral**: No emojis, no playful language

### Color Coding

| Label | Color |
|-------|-------|
| Neutral | Muted gray |
| Offensive | Amber |
| Hate | Red (solid, no flash) |

## Safety Badges

| Badge | Meaning |
|-------|---------|
| Normal | Primary classifier succeeded |
| Fallback Used | Safety fallback system used |
| Rule Escalation | Hard rules triggered escalation |
