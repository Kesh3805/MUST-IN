from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
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


class WeightedLossTrainer(Trainer):
    """Custom Trainer with class-weighted loss for imbalanced datasets"""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            # Apply class weights to cross-entropy loss
            weight_tensor = torch.tensor(self.class_weights, dtype=torch.float32).to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


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
        With class weighting for imbalanced datasets (MANDATORY)
        """
        # Compute class weights for imbalanced dataset
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        print(f"Class weights (imbalance correction): {dict(zip(unique_classes, class_weights))}")
        
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
            # Section 5.3: Categorical cross-entropy with class weighting
        )
        
        self.trainer = WeightedLossTrainer(
            class_weights=class_weights,
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

    def predict(self, texts, batch_size=32):
        """Batch prediction to avoid OOM errors"""
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.eval()
        
        all_predictions = []
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = self.tokenizer(batch_texts, truncation=True, padding=True, 
                                      max_length=128, return_tensors='pt')
            encodings = {k: v.to(device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = self.model(**encodings)
            
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(batch_preds)
        
        return np.array(all_predictions)
    
    def save_model(self, save_dir):
        """Save trained model and tokenizer"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Model saved to: {save_dir}")
    
    def load_model(self, load_dir):
        """Load trained model and tokenizer"""
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        print(f"Model loaded from: {load_dir}")
