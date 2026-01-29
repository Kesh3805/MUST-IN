from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

class FeatureExtractor:
    """
    Implements Section 4.1: Statistical Features
    """
    
    def __init__(self):
        pass

    def get_bow_vectorizer(self, max_features=5000):
        """
        Bag-of-Words
        """
        return CountVectorizer(max_features=max_features)

    def get_tfidf_vectorizer(self, max_features=5000):
        """
        TF-IDF
        """
        return TfidfVectorizer(max_features=max_features)

    # Section 4.2: Word Embeddings
    # For a minimal implementation, we might skip loading heavy GloVe/FastText models 
    # and focus on the required BoW/TF-IDF for Traditional ML as per 5.1
