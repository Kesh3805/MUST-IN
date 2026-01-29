from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Deep Learning Imports
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding


def _to_dense(X):
    """Helper function to convert sparse matrix to dense (for pickling)"""
    return X.toarray()


class TraditionalClassifier:
    """
    Implements Section 5.1: Traditional ML Models
    """
    def __init__(self, model_type='nb', vectorizer=None):
        """
        model_type: 'nb' (MultinomialNB), 'gnb' (GaussianNB), 'svm' (SVC), 'rf' (RandomForest)
        """
        self.model_type = model_type
        self.vectorizer = vectorizer
        self.model = self._get_model()
        self.pipeline = None

    def _get_model(self):
        if self.model_type == 'nb':
            return MultinomialNB()
        elif self.model_type == 'gnb':
            return GaussianNB()
        elif self.model_type == 'svm':
            return SVC(probability=True, kernel='linear') # Probability needed for LIME
        elif self.model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unknown model type")

    def train(self, X_train, y_train):
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be provided")

        # GaussianNB does not accept sparse matrices; densify after vectorization.
        if self.model_type == 'gnb':
            to_dense = FunctionTransformer(_to_dense, accept_sparse=True)
            self.pipeline = Pipeline([
                ('vectorizer', self.vectorizer),
                ('to_dense', to_dense),
                ('classifier', self.model)
            ])
        else:
            self.pipeline = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.model)
            ])
        
        print(f"Training Traditional Model: {self.model_type}...")
        self.pipeline.fit(X_train, y_train)
        print("Training Complete.")

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


class DLDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class TransformerClassifier:
    """
    Implements Section 5.2: Deep Learning Models
    - mBERT (cased/uncased)
    - XLM-RoBERTa
    """
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=3, output_dir='./results'):
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
    def train(self, X_train, y_train, X_val, y_val, epochs=3, batch_size=8):
        """
        End-to-end training (Section 5.3)
        """
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=128)
        val_encodings = self.tokenizer(X_val, truncation=True, padding=True, max_length=128)
        
        train_dataset = DLDataset(train_encodings, y_train)
        val_dataset = DLDataset(val_encodings, y_val)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="no", # Save space for this demo
            # Section 5.3: Use categorical cross-entropy (default in HF for classifications)
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        print(f"Training Transformer Model: {self.model_name}...")
        self.trainer.train()
        print("Training Complete.")

    def compute_metrics(self, eval_pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def predict(self, texts):
        # For inference
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        # Move to device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        return predictions
