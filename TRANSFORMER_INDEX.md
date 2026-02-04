# 🚀 Transformer Setup & Training - Complete Index

**All resources for downloading, training, and deploying transformer models in MUST-IN**

---

## 📚 Documentation Files

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[TRANSFORMER_GUIDE.md](TRANSFORMER_GUIDE.md)** | Complete guide to models, download, training | 20 mins |
| **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)** | Step-by-step training workflow | 15 mins |
| **This File** | Quick index and command reference | 5 mins |

---

## 🎯 Quick Start

### Option 1: Interactive Setup (Easiest)

```bash
# Windows
scripts\quickstart_transformers.bat

# The script will:
# 1. Check Python environment
# 2. Download transformer models
# 3. Validate setup
# 4. Start training (optional)
```

### Option 2: Command Line (Advanced)

```bash
# 1. Download models
python scripts/download_transformers.py --all

# 2. Validate dataset
python scripts/validate_dataset.py --data data/raw/sample_dataset.csv

# 3. Train models
python main.py --run-dl --save-models --generate-report
```

### Option 3: Python API

```python
# 1. Download models
from src.utils.model_download import preload_models
preload_models()

# 2. Train
from src.models.classifiers import TransformerClassifier
classifier = TransformerClassifier("bert-base-multilingual-cased", num_labels=3)
classifier.train(X_train, y_train, X_val, y_val, epochs=3)
```

---

## 🛠️ Available Scripts

### Download & Verification

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_transformers.py` | Download models | `python scripts/download_transformers.py --all` |
| `validate_dataset.py` | Validate training data | `python scripts/validate_dataset.py --data <path>` |
| `quickstart_transformers.bat` | Interactive setup | `scripts\quickstart_transformers.bat` |

### Training & Evaluation

| Script | Purpose | Usage |
|--------|---------|-------|
| `main.py` | Main training script | `python main.py --run-dl --save-models` |
| `validate_must_pipeline.py` | Test pipeline | `python validate_must_pipeline.py` |
| `test_components.py` | Unit tests | `python test_components.py` |

---

## 📥 Download Commands Reference

### Download All Models

```bash
# Recommended: Download all paper models
python scripts/download_transformers.py --all

# Models downloaded:
# - bert-base-multilingual-cased (~400MB)
# - bert-base-multilingual-uncased (~400MB)  
# - xlm-roberta-base (~1GB)
# Total: ~1.8GB
```

### Download Specific Models

```bash
# mBERT cased only (default, fastest)
python scripts/download_transformers.py --model mbert-cased

# XLM-RoBERTa only (most accurate)
python scripts/download_transformers.py --model xlm-roberta

# Multiple models
python scripts/download_transformers.py \
    --model mbert-cased \
    --model xlm-roberta
```

### List Available Models

```bash
python scripts/download_transformers.py --list
```

### Verify Downloaded Models

```bash
# Verify all downloaded models
python scripts/download_transformers.py --verify

# Verify specific model
python scripts/download_transformers.py --model mbert-cased --verify
```

---

## 🎓 Training Commands Reference

### Quick Test (30 minutes)

```bash
# Train mBERT with 1 epoch (testing)
python main.py --run-dl

# Use when:
# - Testing setup
# - Quick validation
# - Limited time
```

### Standard Training (2-3 hours)

```bash
# Train mBERT with 3 epochs
python main.py --run-dl --save-models --generate-report

# Use when:
# - Production training
# - Balanced speed/accuracy
# - First time training
```

### Advanced Training (4-5 hours)

```bash
# Train XLM-RoBERTa with 3 epochs
python main.py --run-xlm --save-models --generate-report

# Use when:
# - Maximum accuracy needed
# - Research purposes
# - Have GPU available
```

### Complete Benchmark (6-8 hours)

```bash
# Train all models + traditional ML
python main.py \
    --run-dl \
    --run-xlm \
    --run-uncased \
    --save-models \
    --generate-report

# Use when:
# - Full evaluation needed
# - Model comparison
# - Research paper
```

---

## 🤖 Available Transformer Models

### Primary Models (Paper-Specified)

| Model | Alias | Size | Languages | F1-Score |
|-------|-------|------|-----------|----------|
| bert-base-multilingual-cased | `mbert-cased` | 110M | 104 | 0.88 |
| bert-base-multilingual-uncased | `mbert-uncased` | 110M | 104 | 0.85 |
| xlm-roberta-base | `xlm-roberta` | 270M | 100 | 0.90 |

### Extended Models (Optional)

| Model | Alias | Size | Languages | F1-Score |
|-------|-------|------|-----------|----------|
| xlm-roberta-large | `xlm-roberta-large` | 550M | 100 | 0.92 |
| ai4bharat/indic-bert | `indic-bert` | 110M | 12 | 0.89 |
| google/muril-base-cased | `muril` | 237M | 17 | 0.87 |

---

## 🎯 Model Selection Guide

### Choose **mBERT-cased** when:
- ✅ Balanced performance needed
- ✅ Fast inference required
- ✅ Limited GPU memory
- ✅ Production deployment
- ✅ Code-mixed text (Tanglish, Hinglish)

### Choose **XLM-RoBERTa** when:
- ✅ Maximum accuracy required
- ✅ Research purposes
- ✅ Have GPU available
- ✅ Cross-lingual transfer
- ✅ Low-resource languages

### Choose **IndIC-BERT** when:
- ✅ Tamil/Hindi only
- ✅ Indic language focus
- ✅ Devanagari/Tamil script
- ✅ Regional language expertise

---

## 📊 Training Configuration

### Default Settings (main.py)

```python
# Hyperparameters (Section 5.3 of paper)
epochs = 3                    # Training epochs
batch_size = 8                # Batch size (reduce if OOM)
learning_rate = 2e-5          # Learning rate
warmup_steps = 100            # Warmup steps
weight_decay = 0.01           # L2 regularization
max_length = 128              # Max sequence length

# Data split
train_ratio = 0.8             # 80% training
val_ratio = 0.1               # 10% validation (from training)
test_ratio = 0.2              # 20% test (held out)
```

### Customize Training

Edit `src/models/classifiers.py`:

```python
# For faster training (less accurate)
epochs = 1
batch_size = 16

# For better accuracy (slower)
epochs = 5
batch_size = 4
```

---

## 🚀 Deployment Commands

### Enable Transformers in API

```bash
# Edit .env file
MUST_DISABLE_TRANSFORMER=false
MUST_MODEL_NAME=bert-base-multilingual-cased
```

### Start API with Transformers

```bash
# Option 1: Use launcher
scripts\start_server.bat
# Select: 2 (Full Pipeline)

# Option 2: Direct
python api/app.py
```

### Test API

```bash
# Test endpoint
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"यह बहुत गंदी बात है\"}"

# Check transformer status
curl http://localhost:8080/health
```

---

## 📈 Performance Benchmarks

### Training Time (10K samples)

| Model | CPU | GPU (8GB) | GPU (16GB) |
|-------|-----|-----------|------------|
| mBERT | 2-3h | 30-45m | 20-30m |
| XLM-R | 4-5h | 45-60m | 30-40m |
| IndIC-BERT | 3-4h | 35-50m | 25-35m |

### Inference Speed

| Model | CPU | GPU |
|-------|-----|-----|
| mBERT | 50-100/s | 500-1000/s |
| XLM-R | 30-50/s | 300-500/s |
| IndIC-BERT | 40-70/s | 400-700/s |

### Accuracy (Test Set)

| Model | Tamil | Hindi | English | Code-Mixed |
|-------|-------|-------|---------|------------|
| mBERT | 0.87 | 0.88 | 0.90 | 0.86 |
| XLM-R | 0.90 | 0.89 | 0.92 | 0.89 |
| IndIC-BERT | 0.91 | 0.90 | 0.85 | 0.88 |

---

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **OOM Error** | Reduce batch size: `batch_size=4` |
| **Slow Training** | Enable GPU or reduce epochs |
| **Low Accuracy** | Use XLM-R or increase epochs |
| **Import Error** | Install: `pip install transformers torch` |
| **Download Failed** | Check internet/disk space |

### Debug Commands

```bash
# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
dir  # Windows

# Test model loading
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-multilingual-cased')"
```

---

## 📚 Documentation Cross-Reference

### For Model Details
→ See **[TRANSFORMER_GUIDE.md](TRANSFORMER_GUIDE.md)**
- Model comparison
- Use cases
- Download instructions
- Performance benchmarks

### For Training Steps
→ See **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)**
- Step-by-step guide
- Training recipes
- Monitoring
- Deployment

### For API Usage
→ See **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)**
- Starting server
- API endpoints
- Frontend integration

### For Pipeline Details
→ See **[MUST_PLUS_PLUS.md](MUST_PLUS_PLUS.md)**
- 7-step pipeline
- Architecture
- Confidence gate
- Fallback logic

---

## ✅ Setup Checklist

### Before Training
- [ ] Python 3.8+ installed
- [ ] PyTorch installed (`pip install torch`)
- [ ] Transformers installed (`pip install transformers`)
- [ ] Models downloaded (`python scripts/download_transformers.py --all`)
- [ ] Dataset validated (`python scripts/validate_dataset.py`)
- [ ] GPU available (optional but recommended)

### During Training
- [ ] Training started (`python main.py --run-dl --save-models`)
- [ ] Monitor progress (watch terminal output)
- [ ] Check GPU usage (if using GPU)
- [ ] Validation metrics improving

### After Training
- [ ] Results generated (`results/results_summary.html`)
- [ ] Models saved (`saved_models/`)
- [ ] Test inference (`python validate_must_pipeline.py`)
- [ ] Deploy API (`python api/app.py`)

---

## 📞 Need Help?

### Documentation Files
1. **TRANSFORMER_GUIDE.md** - Complete model guide
2. **TRAINING_WORKFLOW.md** - Training steps
3. **LAUNCH_GUIDE.md** - Deployment guide
4. **README.md** - Project overview
5. **QUICKSTART.md** - Quick start guide

### Example Commands
```bash
# Get help for any script
python scripts/download_transformers.py --help
python main.py --help

# List available models
python scripts/download_transformers.py --list

# Verify setup
python scripts/download_transformers.py --verify
python scripts/validate_dataset.py
```

---

**Last Updated:** February 4, 2026  
**Version:** 1.0.0  
**Status:** ✅ Ready for Use
