# MUST-IN: Multilingual Explainable Hate Speech Detection Framework

This project implements the "MUST-IN" framework for detecting hate speech in Hindi, Tamil, and English (including Romanized scripts) with Explainable AI (XAI).

## 🚀 Features

- **Multilingual Support**: Hindi, Tamil, and English (including Romanized variants)
- **Comprehensive Preprocessing**: Noise removal, emoji handling, transliteration normalization
- **Language Identification**: Automatic detection of text language
- **Multiple Classification Models**: 
  - Traditional ML: Naive Bayes, SVM, Random Forest
  - Deep Learning: mBERT, XLM-RoBERTa (optional)
- **Explainable AI**: LIME-based explanations for model predictions
- **Evaluation Metrics**: Accuracy, F1-score, ROC-AUC, Confusion matrices
- **Visualization**: Comprehensive analysis notebook with charts and insights

## 📁 Project Structure

```
MUST-IN/
├── data/
│   ├── raw/                # Source data (sample dataset provided)
│   └── processed/          # Cleaned data
├── src/
│   ├── preprocessing/      # Cleaner, Emoji handling, Transliteration
│   ├── features/           # TF-IDF, BoW, BERT embeddings
│   ├── models/             # Language ID, Traditional ML, Transformers
│   ├── evaluation/         # Metrics, Confusion Matrices
│   ├── xai/                # LIME Explainability
│   └── utils/              # Config, Constants
├── results/                # Output plots and LIME visualizations
├── main.py                 # Main execution script
├── analysis.ipynb          # Jupyter notebook for exploration
└── requirements.txt        # Dependencies
```

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: You may need to install PyTorch specific to your CUDA version from [pytorch.org](https://pytorch.org).

### 2. Verify Installation

```bash
python -c "import torch; import transformers; import lime; print('All dependencies installed successfully!')"
```

## 🏃 Running the Project

### Basic Usage

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Train language identification model
3. Train traditional ML models (8 configurations)
4. Generate evaluation metrics and confusion matrices
5. Create LIME explanations
6. Save all results to `results/` folder

### Advanced Options

To enable deep learning models, edit [main.py](main.py) and set:
```python
RUN_DL = True
```

To enable uncased preprocessing comparison:
```python
RUN_TRADITIONAL_UNCASED = True
```

To enable BERT embeddings baseline:
```python
RUN_BERT_EMBED_BASELINES = True
```

### Using the Analysis Notebook

Open and run the Jupyter notebook for interactive analysis:

```bash
jupyter notebook analysis.ipynb
```

The notebook includes:
- Data distribution visualization
- Text length analysis
- Preprocessing comparison
- Model performance comparison
- Recommendations for improvements

## 📊 Understanding Results

### Output Files

After running `main.py`, you'll find:

- **`results/*_confusion_matrix.png`**: Confusion matrices for each model
- **`results/lime_explanation_traditional.html`**: LIME explanation (open in browser)
- **`data/processed/dataset_cleaned.csv`**: Preprocessed dataset

### Interpreting LIME Explanations

Open `results/lime_explanation_traditional.html` in a web browser to see:
- Words contributing positively/negatively to the prediction
- Feature importance scores
- Probability distribution across classes

### Model Performance

Check the terminal output for:
- Classification reports (precision, recall, F1-score)
- Accuracy scores
- ROC-AUC scores
- Confusion matrix locations

## 🔬 Implementation Details

### 1. Preprocessing (`src/preprocessing/`)

- **Noise Removal**: Removes URLs, mentions, repeated punctuation using regex
- **Emoji Handling**: Converts emojis like 😡 to `:enraged_face:` text tokens
- **Transliteration**: Dictionary-based normalization for Romanized Hindi/Tamil

### 2. Language Identification (`src/models/language_id.py`)

- Uses **Multinomial Naive Bayes** with TF-IDF character n-grams (1-3)
- Classifies text as Hindi, Tamil, or English
- Handles Romanized script variants

### 3. Feature Extraction (`src/features/`)

- **Statistical Features**: Bag-of-Words (BoW), TF-IDF
- **Deep Features**: BERT CLS embeddings (optional)
- Configurable max features (default: 5000)

### 4. Classification Models

#### Traditional ML (`src/models/classifiers.py`)
- **Multinomial Naive Bayes (MNB)**: Fast, works well with sparse features
- **Gaussian Naive Bayes (GNB)**: Handles dense features
- **Support Vector Machine (SVM)**: Linear kernel with probability estimates
- **Random Forest (RF)**: 100 trees, robust to overfitting

#### Deep Learning (Optional)
- **mBERT**: Multilingual BERT (cased/uncased variants)
- **XLM-RoBERTa**: Enhanced multilingual model
- Fine-tuned end-to-end with HuggingFace Transformers

### 5. Explainability (XAI) (`src/xai/`)

Uses **LIME (Local Interpretable Model-agnostic Explanations)** to:
- Generate local explanations for individual predictions
- Identify important words/features
- Work with both traditional and deep learning models

### 6. Evaluation (`src/evaluation/`)

Comprehensive metrics:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC (One-vs-Rest for multiclass)
- Confusion matrices with visualizations
- Per-class performance breakdown

## ⚙️ Configuration

Edit [src/utils/config.py](src/utils/config.py) to:

- Add more transliteration rules
- Change class labels
- Modify language mappings
- Adjust random seed

Example:
```python
NORMALIZATION_DICT_HINDI = {
    "kese": "kaise",
    "thik": "theek",
    # Add more rules
}

LABELS = ["Neutral", "Offensive", "Hate"]
SEED = 42
```

## 📈 Sample Results

With the provided sample dataset (50 examples):

| Model        | Vectorizer | Accuracy | Best Use Case |
|--------------|-----------|----------|---------------|
| MNB          | TF-IDF    | ~0.33    | Fast baseline |
| SVM          | TF-IDF    | ~0.33    | Interpretable |
| Random Forest| TF-IDF    | ~0.33    | Robust        |
| mBERT        | -         | TBD      | Best accuracy |

**Note**: Results improve significantly with larger, balanced datasets.

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'torch'**
```bash
pip install torch torchvision torchaudio
```

**2. CUDA Out of Memory (Deep Learning)**
- Reduce batch size in [src/models/classifiers.py](src/models/classifiers.py)
- Use CPU: Set `device = torch.device('cpu')`

**3. Warning: Precision is ill-defined**
- Expected with small datasets
- Indicates some classes have no predictions
- Solution: Add more training data

**4. LIME explanation not showing**
- Ensure you have a web browser installed
- Manually open `results/lime_explanation_traditional.html`

### Dataset Issues

If you see poor performance:
1. **Dataset too small**: Add more examples (minimum 100 per class recommended)
2. **Class imbalance**: Balance Neutral, Offensive, and Hate examples
3. **Language distribution**: Ensure all languages have sufficient representation

## 🎯 Next Steps & Improvements

### Data Enhancement
- [ ] Collect 1000+ examples per class
- [ ] Balance across all languages
- [ ] Add more code-mixed examples
- [ ] Include context from conversation threads

### Model Improvements
- [ ] Implement cross-validation
- [ ] Add ensemble methods
- [ ] Experiment with IndicBERT for better Indic language support
- [ ] Try XLM-RoBERTa Large

### Feature Engineering
- [ ] Add linguistic features (POS tags, dependency parsing)
- [ ] Incorporate user metadata (platform, time, etc.)
- [ ] Use word2vec/fastText embeddings

### Deployment
- [ ] Create REST API with FastAPI/Flask
- [ ] Build web interface for real-time predictions
- [ ] Implement batch processing pipeline
- [ ] Add model versioning and A/B testing

### Evaluation
- [ ] Add per-language metrics
- [ ] Implement error analysis
- [ ] Create comparative dashboard
- [ ] Add interpretability metrics

## 📚 References

- LIME Paper: ["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- mBERT: [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md)
- Hate Speech Detection: Various recent papers on multilingual hate speech

## 📝 License

This project is for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Happy Hate Speech Detection! 🚀**
