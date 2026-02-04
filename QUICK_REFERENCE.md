# 🚀 MUST++ Quick Reference Card

## Launch Commands

```bash
# Windows (Interactive)
scripts\start_server.bat

# Windows/Linux/Mac (Lightweight)
python api/app_lite.py

# Windows/Linux/Mac (Full Pipeline)
python api/app.py
```

**Access:** http://localhost:8080

---

## Test Examples (Copy-Paste Ready)

### Neutral Examples
```
English:  This is a test message
Hindi:    यह एक परीक्षण संदेश है
Tamil:    இது ஒரு சோதனை செய்தி
Hinglish: Aaj ka din bahut achha hai
Tanglish: Naan Chennai la irukken
Mixed:    This is एक test with தமிழ்
```

### Offensive Examples
```
English:  You are an idiot
Hindi:    तुम बेवकूफ हो
Tamil:    நீ முட்டாள்
Hinglish: Tu pagal hai
Tanglish: Nee ooru loosu da
Mixed:    You are a बेवकूफ person
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Frontend UI |
| `/health` | GET | Health check |
| `/detect-script` | POST | Script detection |
| `/analyze` | POST | Classification |

### Example cURL
```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test"}'
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Analyze text |
| `/` | Focus input |
| `Esc` | Clear input |
| `Ctrl+D` | Toggle dark mode |
| `Ctrl+H` | Open history |
| `?` | Show shortcuts |

---

## File Structure

```
MUST-IN/
├── frontend/              # Web UI (HTML/CSS/JS)
├── api/                   # Flask servers
│   ├── app.py            # Full pipeline
│   └── app_lite.py       # Lightweight
├── src/                   # Core pipeline
│   ├── pipeline/         # MUST++ implementation
│   ├── models/           # Classifiers
│   └── xai/              # Explainability
├── scripts/               # Launcher scripts
├── LAUNCH_GUIDE.md       # Complete tutorial
├── FRONTEND_README.md    # UI documentation
├── PROJECT_STATUS.md     # Status & checklist
└── TESTING_EXAMPLES.html # Interactive tests
```

---

## Documentation Guide

| Need to... | Read... |
|------------|---------|
| Launch system | `LAUNCH_GUIDE.md` |
| Understand UI | `FRONTEND_README.md` |
| Check status | `PROJECT_STATUS.md` |
| Test examples | `TESTING_EXAMPLES.html` |
| Architecture | `ARCHITECTURE_DIAGRAM.md` |
| Quick start | `QUICKSTART.md` |

---

## Supported Languages

✅ **Tamil** - தமிழ் (Native) + Tanglish (Romanized)  
✅ **Hindi** - हिंदी (Native) + Hinglish (Romanized)  
✅ **English** - Latin script  
✅ **Mixed** - Multiple scripts in one text

---

## Classification Labels

| Label | Description |
|-------|-------------|
| **NEUTRAL** | No harmful content |
| **OFFENSIVE** | Vulgar language, no targeted hate |
| **HATE** | Targeted hate speech |

---

## Server Modes

### Lightweight (app_lite.py)
- ⚡ Fast startup (2-3 seconds)
- 🔧 No transformer dependencies
- 📝 Uses fallback classifiers
- 🎯 Perfect for testing/demos

### Full Pipeline (app.py)
- 🎯 High accuracy
- 🤖 With transformer models
- ⏱️ Slower startup (30-60 seconds)
- 🚀 Production-ready

---

## Troubleshooting

### Server won't start
```bash
pip install -r requirements.txt
```

### Port 8080 in use
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8080 | xargs kill -9
```

### Health check
```bash
curl http://localhost:8080/health
```

---

## Configuration (.env)

```bash
# Preload models at startup
MUST_PRELOAD_MODELS=false

# Disable transformer (use fallback only)
MUST_DISABLE_TRANSFORMER=true

# Transformer model name
MUST_MODEL_NAME=bert-base-multilingual-cased
```

**Models:**
- `bert-base-multilingual-cased`
- `bert-base-multilingual-uncased`
- `xlm-roberta-base`

---

## Common Tasks

### Start lightweight server
```bash
python api/app_lite.py
```

### Start full pipeline
```bash
python api/app.py
```

### Run tests
```bash
python api/test_api.py
```

### Train models
```bash
python main.py
```

### Run inference CLI
```bash
python inference.py "your text here"
```

---

## Quick Links

🌐 **Web Interface:** http://localhost:8080  
🧪 **Testing Page:** Open `TESTING_EXAMPLES.html`  
📖 **Full Guide:** `LAUNCH_GUIDE.md`  
✅ **Status:** `PROJECT_STATUS.md`  

---

## System Status

✅ Frontend complete (3 files, 2,974 lines)  
✅ Backend complete (2 servers)  
✅ Pipeline complete (7 steps)  
✅ Documentation complete (7 guides)  
✅ Testing tools complete  

**Ready to use!** 🎉

---

## Emergency Quick Start

```bash
# 1. Launch
scripts\start_server.bat

# 2. Open browser
http://localhost:8080

# 3. Test
Type: "This is a test"
Click: Analyze
```

**Done!** 🚀

---

**Questions?** See `LAUNCH_GUIDE.md` for detailed help.
