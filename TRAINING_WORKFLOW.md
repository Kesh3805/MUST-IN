# 🎓 MUST-IN Training Workflow

**Step-by-step guide for training transformer models from scratch**

---

## 🎯 Training Objectives

- Train multilingual transformer models (mBERT, XLM-R)
- Achieve 85%+ F1-score on test set
- Support Tamil, Hindi, English, and code-mixed text
- Deploy trained models for real-time inference

---

## 📋 Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] GPU with 8GB+ VRAM (optional but recommended)
- [ ] 20GB free disk space
- [ ] Training dataset prepared
- [ ] Dependencies installed

---

## 🔧 Phase 1: Environment Setup (15 minutes)

### Step 1: Install Dependencies

```bash
# Install PyTorch
# CPU only:
pip install torch==2.0.0

# GPU (CUDA 11.8):
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Verify GPU (Optional)

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only')"
```

### Step 3: Configure Environment

```bash
# Create .env file
cp .env.example .env

# Edit .env
MUST_PRELOAD_MODELS=true
MUST_DISABLE_TRANSFORMER=false
MUST_MODEL_NAME=bert-base-multilingual-cased
```

---

## 📥 Phase 2: Download Models (30-60 minutes)

### Option 1: Download All Models (Recommended)

```bash
python scripts/download_transformers.py --all
```

### Option 2: Download Specific Models

```bash
# Download mBERT only
python scripts/download_transformers.py --model bert-base-multilingual-cased

# Download XLM-R only
python scripts/download_transformers.py --model xlm-roberta-base

# Download both
python scripts/download_transformers.py \
    --model bert-base-multilingual-cased \
    --model xlm-roberta-base
```

### Option 3: Manual Download (Python)

```python
from src.utils.model_download import preload_models

# Download all paper models
downloaded = preload_models()
print(f"Downloaded: {downloaded}")
```

### Verify Downloads

```bash
# Check cached models
python scripts/verify_models.py
```

---

## 📊 Phase 3: Prepare Data (10 minutes)

### Step 1: Check Dataset Format

Your dataset should have:
- **text** column: Input text (multilingual)
- **label** column: 0=neutral, 1=offensive, 2=hate
- **language** column (optional): Language metadata

Example CSV:
```csv
text,label,language
"यह बहुत अच्छी बात है",0,Hindi
"तुम बेवकूफ हो",1,Hindi
"இவன் பார்ப்பான்",2,Tamil
"The weather is nice",0,English
```

### Step 2: Validate Dataset

```bash
python scripts/validate_dataset.py --data data/raw/sample_dataset.csv
```

Expected output:
```
✓ Dataset loaded: 10000 samples
✓ Required columns present: text, label
✓ Label distribution:
    - Neutral: 5000 (50%)
    - Offensive: 3000 (30%)
    - Hate: 2000 (20%)
✓ Language distribution:
    - Hindi: 3500
    - Tamil: 3000
    - English: 2500
    - Code-mixed: 1000
✓ Dataset ready for training
```

### Step 3: Split Dataset (Automatic)

The training script automatically splits:
- **80%** Training
- **10%** Validation (from training)
- **20%** Test (held out)

---

## 🚀 Phase 4: Train Models (2-5 hours)

### Quick Start: Train All Models

```bash
# Train mBERT + XLM-R with all options
python main.py \
    --data data/raw/sample_dataset.csv \
    --run-dl \
    --run-xlm \
    --save-models \
    --generate-report
```

### Train mBERT Only (Faster)

```bash
# Train only mBERT (cased)
python main.py \
    --data data/raw/sample_dataset.csv \
    --run-dl \
    --save-models
```

### Train XLM-RoBERTa Only

```bash
# Train only XLM-R
python main.py \
    --data data/raw/sample_dataset.csv \
    --run-xlm \
    --save-models
```

### Custom Training Script

```python
from src.models.classifiers import TransformerClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('data/raw/sample_dataset.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# Further validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

# Initialize model
classifier = TransformerClassifier(
    model_name="bert-base-multilingual-cased",
    num_labels=3,
    output_dir="./saved_models/mbert"
)

# Train
classifier.train(
    X_train=X_train.tolist(),
    y_train=y_train.tolist(),
    X_val=X_val.tolist(),
    y_val=y_val.tolist(),
    epochs=3,
    batch_size=8  # Reduce if OOM
)

# Save
classifier.model.save_pretrained("./saved_models/mbert_final")
classifier.tokenizer.save_pretrained("./saved_models/mbert_final")
print("✓ Model saved to ./saved_models/mbert_final")
```

---

## 📈 Phase 5: Monitor Training

### Training Progress

```
Training Transformer Model: bert-base-multilingual-cased...

Epoch 1/3:
  - Training Loss: 0.8234
  - Validation Loss: 0.6123
  - Validation Accuracy: 0.78
  - Validation F1: 0.76
  
Epoch 2/3:
  - Training Loss: 0.4512
  - Validation Loss: 0.4234
  - Validation Accuracy: 0.85
  - Validation F1: 0.84
  
Epoch 3/3:
  - Training Loss: 0.2891
  - Validation Loss: 0.3456
  - Validation Accuracy: 0.89
  - Validation F1: 0.88

Training Complete.
Model saved to: saved_models/mbert_final/
```

### Monitor GPU Usage (Optional)

```bash
# In separate terminal
watch -n 1 nvidia-smi
```

### Check Training Logs

```bash
# Real-time monitoring
tail -f logs/training.log
```

---

## ✅ Phase 6: Evaluate Models

### Automatic Evaluation

After training completes, results are automatically saved to:
- `results/model_comparison.csv` - Performance metrics
- `results/results_summary.html` - Visual report
- `results/best_model_report.txt` - Best model details

### Manual Evaluation

```python
from src.evaluation.metrics import Evaluator
from src.models.classifiers import TransformerClassifier
import pandas as pd

# Load trained model
classifier = TransformerClassifier(
    model_name="./saved_models/mbert_final",
    num_labels=3
)

# Load test data
df = pd.read_csv('data/raw/sample_dataset.csv')
X_test = df['text'].tolist()
y_test = df['label'].tolist()

# Predict
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

# Evaluate
evaluator = Evaluator()
evaluator.evaluate(y_test, y_pred, y_prob, class_names=['neutral', 'offensive', 'hate'])
evaluator.print_report()
```

### View Results

```bash
# Open results summary
start results/results_summary.html  # Windows
open results/results_summary.html   # Mac
xdg-open results/results_summary.html  # Linux
```

---

## 🚀 Phase 7: Deploy Trained Models

### Step 1: Update Configuration

```bash
# Edit .env
MUST_DISABLE_TRANSFORMER=false
MUST_MODEL_NAME=./saved_models/mbert_final
```

### Step 2: Test Model Loading

```python
from src.pipeline.must_pipeline import MUSTPlusPipeline, PipelineConfig

# Load your trained model
pipeline = MUSTPlusPipeline(
    model_name="./saved_models/mbert_final",
    config=PipelineConfig(disable_transformer=False)
)

# Test inference
result = pipeline.classify("यह बहुत गंदी बात है")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence}")
```

### Step 3: Start API Server

```bash
# Start server with trained model
python api/app.py
```

### Step 4: Test API

```bash
# Test endpoint
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "यह बहुत गंदी बात है"}'
```

### Step 5: Load Testing (Optional)

```bash
# Install locust
pip install locust

# Run load test
python scripts/load_test.py
```

---

## 📊 Phase 8: Compare Models

### Model Selection Matrix

| Criteria | mBERT-cased | mBERT-uncased | XLM-RoBERTa |
|----------|-------------|---------------|-------------|
| **Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Memory** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Tamil** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Hindi** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Code-mixed** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Recommendation

- **Production (Speed)**: mBERT-cased
- **Production (Accuracy)**: XLM-RoBERTa
- **Development**: mBERT-cased
- **Research**: XLM-RoBERTa

---

## 🛠️ Troubleshooting

### Issue 1: OOM During Training

```bash
# Reduce batch size
python main.py --run-dl --save-models
# Edit src/models/classifiers.py: batch_size=4 (instead of 8)
```

### Issue 2: Training Too Slow

```bash
# Use GPU if available
python -c "import torch; print(torch.cuda.is_available())"

# Reduce epochs for testing
# Edit main.py: epochs=1 (instead of 3)
```

### Issue 3: Model Not Saving

```bash
# Check disk space
df -h  # Linux/Mac
dir    # Windows

# Manually specify output directory
python main.py --run-dl --save-models
```

### Issue 4: Low Accuracy

- ✅ Check data quality
- ✅ Increase training epochs (3 → 5)
- ✅ Use larger model (mBERT → XLM-R)
- ✅ Add more training data
- ✅ Check label distribution (balanced?)

---

## 📚 Training Recipes

### Recipe 1: Quick Test (30 mins)

```bash
# Small dataset, 1 epoch, mBERT only
python main.py \
    --data data/raw/sample_dataset.csv \
    --run-dl
```

### Recipe 2: Standard Training (2-3 hours)

```bash
# Full dataset, 3 epochs, mBERT
python main.py \
    --data data/raw/sample_dataset.csv \
    --run-dl \
    --save-models \
    --generate-report
```

### Recipe 3: Maximum Accuracy (4-5 hours)

```bash
# Full dataset, 5 epochs, XLM-R
python main.py \
    --data data/raw/sample_dataset.csv \
    --run-xlm \
    --save-models \
    --generate-report
```

### Recipe 4: Complete Benchmark (6-8 hours)

```bash
# All models + traditional ML
python main.py \
    --data data/raw/sample_dataset.csv \
    --run-dl \
    --run-xlm \
    --run-uncased \
    --save-models \
    --generate-report
```

---

## 🎯 Training Checklist

### Before Training
- [ ] GPU available and working
- [ ] Dataset validated (format, labels)
- [ ] Dependencies installed
- [ ] Models downloaded
- [ ] Disk space available (20GB+)
- [ ] Configuration set (.env)

### During Training
- [ ] Monitor GPU usage
- [ ] Check training logs
- [ ] Watch validation metrics
- [ ] Ensure no errors

### After Training
- [ ] Review results report
- [ ] Compare model performance
- [ ] Test inference speed
- [ ] Deploy best model
- [ ] Document results

---

## 📖 Next Steps

After training:

1. **Deploy Models**: See `LAUNCH_GUIDE.md`
2. **Test API**: See `FRONTEND_README.md`
3. **Optimize Performance**: See `TRANSFORMER_GUIDE.md`
4. **Production Setup**: See `README.md`

---

**Training Time Estimates:**

| Configuration | Time | GPU | Dataset |
|---------------|------|-----|---------|
| Quick Test | 30 mins | Optional | 1K samples |
| Standard | 2-3 hours | Recommended | 10K samples |
| Full Training | 4-5 hours | Required | 50K samples |
| Research | 8-12 hours | Required | 100K+ samples |

---

**Last Updated:** February 4, 2026  
**Version:** 1.0.0
