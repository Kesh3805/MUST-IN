# 🚀 MUST++ Launch Guide

Complete guide to launching and testing the MUST++ multilingual hate speech detection system.

## 📋 Prerequisites

✅ Python 3.10 or higher  
✅ All dependencies installed: `pip install -r requirements.txt`  
✅ Modern web browser (Chrome, Firefox, Edge)

## 🎯 Quick Launch (2 minutes)

### Step 1: Start the Server

**Windows:**
```bash
scripts\start_server.bat
```

**Linux/Mac:**
```bash
python api/app_lite.py
```

When prompted, choose:
- **[1] Lightweight mode** - Fast startup (2-3 seconds), uses fallback classifiers
- **[2] Full pipeline mode** - With transformers (30-60 seconds startup)

### Step 2: Open Browser

Navigate to: **http://localhost:8080**

You should see the MUST++ interface with:
- Text input area
- Real-time script detector
- "Analyze" button

### Step 3: Test with Multilingual Examples

Try these examples to test different languages:

#### 🇬🇧 English
```
This is a neutral test message
```

#### 🇮🇳 Hindi (Devanagari)
```
यह एक परीक्षण संदेश है
```

#### 🇮🇳 Tamil (Tamil script)
```
இது ஒரு சோதனை செய்தி
```

#### 🔀 Code-Mixed (Hinglish)
```
Aaj ka din bahut achha hai
```

#### 🔀 Code-Mixed (Tanglish)
```
Naan Chennai la work panren
```

#### 🌐 Multi-Script Mixed
```
This is a test with हिंदी and தமிழ் scripts
```

## 🧪 Testing the System

### Test 1: Script Detection

1. Type in the text input field
2. Watch the "Script:" indicator update in real-time
3. It should detect: **Latin**, **Devanagari**, **Tamil**, or **Mixed**

### Test 2: Classification

1. Enter any text from the examples above
2. Click "Analyze" or press `Ctrl+Enter`
3. View the **Decision Layer**:
   - Label: Neutral / Offensive / Hate
   - Confidence score (0.00 - 1.00)
   - Safety badge status

### Test 3: Explanation Layer

1. After classification, click on "Explanation" to expand
2. You'll see:
   - **Highlighted text** - Original text with harm tokens marked (if any)
   - **Why this label was chosen** - Human-readable justification
   - **Why weaker labels were rejected** - Reasoning for classification
   - **Key harm tokens** - Detected harmful words (if any)
   - **Identity groups detected** - Targeted groups (if hate speech)

### Test 4: System Trace (Advanced)

1. Click on "System Trace" to see technical details:
   - Languages detected (e.g., "Hindi: 0.6, English: 0.4")
   - Script distribution (percentage breakdown)
   - Code-mixed indicator (Yes/No)
   - Transformer status (Available/Unavailable)
   - Confidence gate decision
   - Fallback tier used (if applicable)
   - Processing time in milliseconds

### Test 5: History Feature

1. Analyze multiple texts
2. Click the "📋 History" button in the header
3. Your past analyses appear in a slide-out panel
4. Click any history item to reload it

## ⚙️ Configuration Options

### .env Configuration

Edit `.env` file to customize behavior:

```bash
# Preload transformer models at startup (slower startup, faster first request)
MUST_PRELOAD_MODELS=false

# Disable transformer models (use fallback-only mode)
MUST_DISABLE_TRANSFORMER=true

# Choose transformer model (if enabled)
MUST_MODEL_NAME=bert-base-multilingual-cased
```

**Model options:**
- `bert-base-multilingual-cased` (default, best for Hindi/Tamil)
- `bert-base-multilingual-uncased`
- `xlm-roberta-base` (best overall, requires training)

### Server Modes

**Lightweight Mode (app_lite.py):**
- ✅ Fast startup (2-3 seconds)
- ✅ No dependencies on PyTorch/Transformers
- ✅ Uses fallback classifiers (SVM, Naive Bayes)
- ⚠️ Lower accuracy than transformer models
- 💡 Best for: Development, testing, quick demos

**Full Pipeline Mode (app.py):**
- ✅ High accuracy with transformers
- ✅ Full MUST++ pipeline with confidence gating
- ✅ Supports all languages and scripts
- ⚠️ Slower startup (30-60 seconds for model loading)
- 💡 Best for: Production, accuracy-critical use cases

## 🎨 UI Features

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Analyze text |
| `/` | Focus input field |
| `Esc` | Clear input |
| `Ctrl+D` | Toggle dark mode |
| `Ctrl+H` | Open history panel |
| `?` | Show all shortcuts |

### Theme Modes

Click the 🌙/☀️ button to cycle through:
1. **System** - Follows OS preference
2. **Light** - Always light theme
3. **Dark** - Always dark theme

### History Management

- Stores up to 50 recent analyses
- Saved in browser localStorage
- Click "Clear" to remove all history
- Click any item to reload the text

## 🔧 Troubleshooting

### Server won't start

**Error: `ModuleNotFoundError`**
```bash
# Install dependencies
pip install -r requirements.txt
```

**Error: `Port 8080 already in use`**
```bash
# Find and kill the process using port 8080
# Windows:
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8080 | xargs kill -9
```

### Frontend shows "System error"

1. Check if server is running: http://localhost:8080/health
2. Expected response:
```json
{
  "status": "healthy",
  "transformer": "unavailable (fallback mode)",
  "degraded_mode": true,
  "supported_languages": ["tamil", "hindi", "english"],
  "supported_scripts": ["tamil", "devanagari", "latin"]
}
```

### Script detection not working

1. Make sure you're typing non-English characters
2. Script detection debounces for 300ms - wait briefly after typing
3. Very short texts (<3 characters) may not trigger detection

### Classification seems wrong

**In Lightweight Mode (app_lite.py):**
- Uses fallback classifiers (lower accuracy)
- Limited training data (49 samples in demo dataset)
- Expected behavior for demo purposes

**To improve accuracy:**
1. Switch to Full Pipeline mode (app.py)
2. Train on larger dataset (replace data/processed/dataset_cleaned.csv)
3. Run training: `python main.py --run-xlm`

## 📊 API Testing

### Health Check
```bash
curl http://localhost:8080/health
```

### Script Detection
```bash
curl -X POST http://localhost:8080/detect-script \
  -H "Content-Type: application/json" \
  -d '{"text": "यह एक परीक्षण है"}'
```

### Classification
```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message", "language_hint": "english"}'
```

## 🎯 Next Steps

### For Development
1. Add more training data to `data/raw/`
2. Run training: `python main.py`
3. Test with: `python inference.py "your text here"`
4. View results in `results/`

### For Production
1. Train with full dataset (not the 49-sample demo)
2. Run full pipeline: `python api/app.py`
3. Set `.env`: `MUST_DISABLE_TRANSFORMER=false`
4. Deploy with proper WSGI server (gunicorn, waitress)

### For Research
1. Read `MUST_PLUS_PLUS.md` for architecture details
2. Check `VALIDATION_SUMMARY.md` for evaluation results
3. Explore `results/` for experiment outputs
4. Run golden test suite: `python tests/test_golden_suite.py`

## 📚 Documentation

- **Frontend Guide**: `FRONTEND_README.md`
- **Quick Start**: `QUICKSTART.md`
- **Architecture**: `MUST_PLUS_PLUS.md`
- **Main README**: `README.md`
- **Validation Results**: `VALIDATION_SUMMARY.md`

## ✅ System Check

Verify everything works:

```bash
# 1. Start server
python api/app_lite.py

# 2. In another terminal, run health check
curl http://localhost:8080/health

# 3. Run test suite
python api/test_api.py

# 4. Open browser
# Navigate to: http://localhost:8080
# Test with multilingual examples above
```

If all tests pass ✅, your system is ready!

## 🆘 Support

If you encounter issues:
1. Check the console/terminal for error messages
2. Verify Python version: `python --version` (should be 3.10+)
3. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
4. Try lightweight mode first: `python api/app_lite.py`

---

**Ready to launch?** Run `scripts\start_server.bat` and test with the examples above! 🚀
