# MUST++ Frontend

Operator-grade UI for multilingual hate speech detection with full support for **Tamil**, **Hindi**, **English**, and code-mixed text.

## 🚀 Quick Start

### Option 1: Run the batch script (Windows) - RECOMMENDED
```bash
scripts\start_server.bat
```
This will prompt you to choose between:
- **[1] Lightweight mode** - Quick startup (2-3 seconds), uses fallback classifiers
- **[2] Full pipeline mode** - Includes transformer models (30-60 seconds startup)

### Option 2: Run directly with Python

**Lightweight mode (fast startup):**
```bash
python api/app_lite.py
```

**Full pipeline mode (with transformers):**
```bash
python api/app.py
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

### Multilingual Test Examples

The system supports **Tamil**, **Hindi**, **English**, and **code-mixed** text. Test with these examples:

#### English Examples
```
Neutral: "This is a test message"
Offensive: "You are an idiot"
```

#### Hindi Examples (Devanagari script)
```
Neutral: "यह एक परीक्षण संदेश है"
Offensive: "तुम बेवकूफ हो"
```

#### Tamil Examples (Tamil script)
```
Neutral: "இது ஒரு சோதனை செய்தி"
Offensive: "நீ முட்டாள்"
```

#### Hinglish (Romanized Hindi)
```
Neutral: "Kaise ho aap?"
Offensive: "Tu pagal hai"
```

#### Tanglish (Romanized Tamil)
```
Neutral: "Naan Chennai la irukken"
Offensive: "Nee ooru loosu"
```

#### Code-Mixed Examples
```
Mixed English-Hindi: "You are a बेवकूफ person"
Mixed English-Tamil: "This is வேஸ்ட் content"
```

### Testing Instructions

1. **Start the server**:
   ```bash
   scripts\\start_server.bat
   ```
   Choose option [1] for quick testing or [2] for full pipeline.

2. **Open browser**: Navigate to http://localhost:8080

3. **Enter test text**: Copy any example above into the text input

4. **Click "Analyze"**: The system will:
   - Detect the script (Tamil/Devanagari/Latin/Mixed)
   - Identify the language(s)
   - Classify as Neutral/Offensive/Hate
   - Show confidence score and explanation

5. **Explore layers**:
   - **Decision Layer**: See the final classification
   - **Explanation Layer**: View detected harm tokens (if any)
   - **System Trace**: See technical details (languages, scripts, fallback status)

### Run Automated Test Suite
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
