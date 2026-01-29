from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
import os

class LanguageIdentifier:
    """
    Implements Section 3: Language Identification Module
    """
    
    def __init__(self, method='tfidf', ngram_range=(1, 3), analyzer='char'):
        """
        Args:
            method: 'count' or 'tfidf' (Section 3.3)
            ngram_range: tuple (min, max) for n-grams
            analyzer: 'word' or 'char' (Section 3.3)
        """
        self.method = method
        self.vectorizer = None
        self.model = MultinomialNB() # Section 3: Multinomial Naive Bayes
        self.pipeline = None
        self.ngram_range = ngram_range
        self.analyzer = analyzer

    def build_pipeline(self):
        if self.method == 'count':
            self.vectorizer = CountVectorizer(ngram_range=self.ngram_range, analyzer=self.analyzer)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, analyzer=self.analyzer)
            
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])

    def train(self, X_train, y_train):
        """
        Args:
            X_train: List of preprocessed text strings
            y_train: List of language labels (Hindi, Tamil, English)
        """
        if self.pipeline is None:
            self.build_pipeline()
            
        print("Training Language Identification Model...")
        self.pipeline.fit(X_train, y_train)
        print("LID Training Complete.")

    def predict(self, X):
        """
        Returns predicted language label
        """
        if self.pipeline is None:
            raise Exception("Model not trained yet.")
        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test):
        if self.pipeline is None:
            raise Exception("Model not trained yet.")
        y_pred = self.pipeline.predict(X_test)
        return classification_report(y_test, y_pred)

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
