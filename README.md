<p align="center">
  <img src="https://img.shields.io/badge/MUST++-Multilingual_Safety-667eea?style=for-the-badge&logo=shield&logoColor=white" alt="MUST++"/>
</p>

<h1 align="center">
  🛡️ MUST++
</h1>

<h3 align="center">
  <em>Multilingual Hate Speech Detection • Explainable AI • Safety-First</em>
</h3>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-api">API</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Transformers-🤗_HuggingFace-yellow?style=flat-square" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Languages-Tamil_•_Hindi_•_English-764abc?style=flat-square" alt="Languages"/>
  <img src="https://img.shields.io/badge/Scripts-Native_•_Romanized_•_Mixed-f97316?style=flat-square" alt="Scripts"/>
</p>

---

<div align="center">

### 🎯 **Zero Tolerance for Missed Hate Speech**

*A linguistic firewall that catches what others miss—across languages, scripts, and cultural contexts.*

</div>

---

## ✨ What Makes MUST++ Different?

<table>
<tr>
<td width="50%">

### 🌍 **True Multilingual**
Not just translation—native understanding of:
- **Tamil** (தமிழ்) + Tanglish
- **Hindi** (हिंदी) + Hinglish  
- **English** + Code-mixed variants

</td>
<td width="50%">

### 🔍 **Explainable by Design**
Every decision is auditable:
- Harm tokens highlighted
- Confidence with uncertainty
- Fallback reasoning visible

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ **Safety-First Architecture**
When in doubt, escalate:
- Confidence-gated classification
- Multi-tier fallback system
- Zero silent failures

</td>
<td width="50%">

### ⚡ **Production Ready**
Operator-grade tooling:
- REST API with full contract
- 4-layer UI for moderators
- Graceful degradation

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 30-Second Setup

```bash
# Clone and install
git clone https://github.com/your-org/must-in.git
cd must-in
pip install -r requirements.txt

# Launch the UI
python api/app_lite.py
```

**Open your browser to** → [http://localhost:8080](http://localhost:8080)

<p align="center">
  <img src="https://img.shields.io/badge/Ready_in-30_seconds-00c853?style=for-the-badge" alt="Ready"/>
</p>

---

## 🎨 The Interface

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│   MUST++                                    ● System Ready      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Enter text in Tamil, Hindi, English, or mixed...        │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│  Script: Latin                              142 characters      │
│                                                                 │
│  Language hint: [Auto-detect ▼]              [ Analyze ]        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DECISION                                                       │
│  ┌────────────┐                                                 │
│  │   HATE     │  ████████████████░░░░  0.95                     │
│  └────────────┘  confidence                                     │
│                                           [Rule Escalation]     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   EXPLANATION                                                   │
│   SYSTEM TRACE                                [Advanced]        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

</div>

### 4-Layer Information Architecture

| Layer | Purpose | Visibility |
|-------|---------|------------|
| **Input** | Multi-script text entry with real-time script detection | Always visible |
| **Decision** | Label + Confidence + Safety Badge | Primary view |
| **Explanation** | Highlighted tokens, justification, rejected alternatives | Expandable |
| **System Trace** | Languages, fallback tier, entropy, processing time | Advanced toggle |

---

## 🧠 The Pipeline

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   MUST++ PIPELINE                   │
                    └─────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
            ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
            │   STEP 1     │      │   STEP 2     │      │   STEP 3     │
            │  Language    │──────│  Normalize   │──────│  Classify    │
            │  Detection   │      │  (Loss-Aware)│      │ (Transformer)│
            └──────────────┘      └──────────────┘      └──────────────┘
                                                                │
                    ┌───────────────────────────────────────────┘
                    ▼
            ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
            │   STEP 4     │      │   STEP 5     │      │   STEP 6     │
            │  Confidence  │──────│  Fallback    │──────│  Decision    │
            │    Gate      │      │   Logic      │      │  Resolver    │
            └──────────────┘      └──────────────┘      └──────────────┘
                                                                │
                    ┌───────────────────────────────────────────┘
                    ▼
            ┌──────────────────────────────────────────────────────────┐
            │                      STEP 7: EXPLAIN                      │
            │  "This text was classified as HATE because it contains   │
            │   the slur 'X' targeting group 'Y'. Confidence: 0.95"    │
            └──────────────────────────────────────────────────────────┘
```

---

## 📊 Labels & Severity

<div align="center">

| Label | Color | Description | Action |
|:-----:|:-----:|-------------|--------|
| ![Neutral](https://img.shields.io/badge/NEUTRAL-gray?style=flat-square) | Gray | No harmful content | Pass |
| ![Offensive](https://img.shields.io/badge/OFFENSIVE-orange?style=flat-square) | Amber | Vulgar/inappropriate | Review |
| ![Hate](https://img.shields.io/badge/HATE-red?style=flat-square) | Red | Targeted hate speech | **Escalate** |

</div>

### Safety Priority

```
HATE > OFFENSIVE > NEUTRAL
```

When signals conflict, always escalate to the safer (more severe) label.

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/config` | Configuration details |
| `POST` | `/detect-script` | Real-time script detection |
| `POST` | `/analyze` | **Main classification** |

### Example Request

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "உங்கள் கருத்து என்ன?"}'
```

### Response Contract

```json
{
  "label": "neutral",
  "confidence": 0.85,
  "safety_badge": {
    "type": "normal",
    "label": "Normal",
    "tooltip": "Primary classifier succeeded"
  },
  "explanation": {
    "summary": "No harmful content detected.",
    "key_harm_tokens": [],
    "label_justification": "Text appears neutral.",
    "weaker_labels_rejected": []
  },
  "system_trace": {
    "languages_detected": {"tamil": 1.0},
    "script_distribution": {"tamil": 1.0},
    "fallback_used": false,
    "escalation_triggered": false,
    "transformer_used": true,
    "processing_time_ms": 45.2
  }
}
```

---

## 📁 Project Structure

```
MUST-IN/
├── 🎨 frontend/                 # Operator-grade UI
│   ├── index.html              # 4-layer interface
│   ├── styles.css              # Color-blind safe design
│   └── app.js                  # Interactive logic
│
├── 🔌 api/                      # REST API
│   ├── app.py                  # Full server (with transformer)
│   ├── app_lite.py             # Lightweight server (fast startup)
│   └── test_api.py             # API test suite
│
├── 🧠 src/
│   ├── pipeline/               # 7-step classification pipeline
│   │   ├── must_pipeline.py    # Main orchestrator
│   │   ├── script_detector.py  # Language/script detection
│   │   ├── confidence_gate.py  # Uncertainty handling
│   │   ├── fallback_logic.py   # Safety-first fallbacks
│   │   ├── decision_resolver.py# Final label resolution
│   │   └── hate_lexicon.py     # Multilingual harm dictionary
│   │
│   ├── preprocessing/          # Text normalization
│   ├── features/               # TF-IDF, BERT embeddings
│   ├── models/                 # ML classifiers
│   ├── evaluation/             # Metrics & validation
│   └── xai/                    # LIME explainability
│
├── 📊 data/
│   ├── raw/                    # Source datasets
│   └── processed/              # Cleaned data
│
├── 📈 results/                  # Outputs & reports
├── 🧪 tests/                    # Test suites
└── 📜 scripts/                  # Utility scripts
```

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for dev tools)
- Git

### Full Installation

```bash
# 1. Clone repository
git clone https://github.com/your-org/must-in.git
cd must-in

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install Node dependencies (for Husky hooks)
npm install

# 5. Verify installation
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
```

### 🐶 Git Hooks (Husky)

This project uses **Husky v9** + **lint-staged** for automated code quality:

| Hook | Action |
|------|--------|
| `pre-commit` | Lint & format staged files |
| `commit-msg` | Validate commit message format |
| `pre-push` | Run tests before pushing |

```bash
# Hooks run automatically on commit:
# ✓ Ruff (fast linting)
# ✓ Black (formatting)
# ✓ isort (import sorting)
# ✓ Type checking (mypy)
```

---

## 🧪 Testing

### Run All Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Golden test suite (safety validation)
python tests/test_golden_suite.py

# API tests (start server first)
python api/app_lite.py &
python api/test_api.py
```

### Validation Metrics

| Metric | Target | Current |
|--------|--------|---------|
| **Hate FNR** | 0% | ✅ 0% |
| **Hate Recall** | 100% | ✅ 100% |
| **Hate Precision** | >90% | ✅ 100% |
| **Offensive Recall** | >80% | ⚠️ Degraded mode limited |

---

## 🎯 Roadmap

<details>
<summary><b>Phase 1: Core Pipeline</b> ✅</summary>

- [x] 7-step classification pipeline
- [x] Multilingual lexicon
- [x] Confidence gating
- [x] Fallback system
- [x] Explainability layer

</details>

<details>
<summary><b>Phase 2: Frontend</b> ✅</summary>

- [x] 4-layer UI architecture
- [x] Real-time script detection
- [x] Expandable explanations
- [x] System trace panel
- [x] Color-blind safe design

</details>

<details>
<summary><b>Phase 3: Hardening</b> 🚧</summary>

- [x] Validation test suite
- [x] Safety-first fallbacks
- [ ] Load testing
- [ ] Adversarial robustness
- [ ] Obfuscation detection

</details>

<details>
<summary><b>Phase 4: Scale</b> 📋</summary>

- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Batch processing API
- [ ] Model versioning
- [ ] A/B testing framework

</details>

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

```bash
# 1. Fork the repository

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes (Husky will auto-format)
git add .
git commit -m "feat: add amazing feature"

# 4. Push and create PR
git push origin feature/amazing-feature
```

### Commit Convention

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `style` | Formatting |
| `refactor` | Code restructure |
| `test` | Tests |
| `chore` | Maintenance |

---

## 📚 References

<details>
<summary>Papers & Resources</summary>

- [LIME: "Why Should I Trust You?"](https://arxiv.org/abs/1602.04938)
- [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md)
- [XLM-RoBERTa](https://arxiv.org/abs/1911.02116)
- [Hate Speech Detection Survey](https://arxiv.org/abs/2004.04287)

</details>

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

<div align="center">

### Built with ❤️ for a safer internet

**[⬆ Back to top](#-mustpp)**

</div>

---

<p align="center">
  <sub>
    🛡️ MUST++ • Protecting conversations across languages and cultures
  </sub>
</p>
