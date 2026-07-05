"""
Text Preprocessing Module for AI Chatbot NLP
Handles tokenization, cleaning, and feature extraction
"""

import re
import string
import nltk
from typing import List, Dict, Any
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TextPreprocessor:
    """
    Advanced text preprocessing for customer support queries
    Features: Tokenization, Cleaning, Lemmatization, Feature Extraction
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Email and URL patterns for entity recognition
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.order_id_pattern = r'\b[A-Z]{2,3}\d{6,10}\b'  # Common order ID pattern
        self.tracking_pattern = r'\b1Z[0-9A-Z]{16}\b|\b\d{10,22}\b'  # UPS, FedEx tracking numbers
    
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\?\!\@\-]', '', text)
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using regex patterns
        Returns: Dictionary with entity types and their values
        """
        entities = {
            'email': [],
            'url': [],
            'order_id': [],
            'tracking_number': []
        }
        
        # Extract emails
        emails = re.findall(self.email_pattern, text)
        entities['email'].extend(emails)
        
        # Extract URLs
        urls = re.findall(self.url_pattern, text)
        entities['url'].extend(urls)
        
        # Extract order IDs
        order_ids = re.findall(self.order_id_pattern, text)
        entities['order_id'].extend(order_ids)
        
        # Extract tracking numbers
        tracking_nums = re.findall(self.tracking_pattern, text)
        entities['tracking_number'].extend(tracking_nums)
        
        return entities
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize text and apply cleaning operations
        """
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords if configured
        if self.config.get('nlp', {}).get('tokenization', {}).get('remove_stopwords', True):
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Lemmatization if configured
        if self.config.get('nlp', {}).get('tokenization', {}).get('lemmatization', True):
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter out empty tokens and single characters
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def preprocess_for_training(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts for training
        """
        processed_texts = []
        
        for text in texts:
            tokens = self.tokenize_and_clean(text)
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def fit_tfidf(self, texts: List[str]):
        """
        Fit TF-IDF vectorizer on training texts
        """
        nlp_config = self.config.get('nlp', {})
        features_config = nlp_config.get('features', {})
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=features_config.get('max_features', 5000),
            ngram_range=tuple(features_config.get('ngram_range', [1, 2])),
            lowercase=True,
            stop_words='english'
        )
        
        processed_texts = self.preprocess_for_training(texts)
        self.tfidf_vectorizer.fit(processed_texts)
    
    def transform_text(self, text: str):
        """
        Transform single text using fitted TF-IDF vectorizer
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        
        tokens = self.tokenize_and_clean(text)
        processed_text = ' '.join(tokens)
        
        return self.tfidf_vectorizer.transform([processed_text])
    
    def transform_texts(self, texts: List[str]):
        """
        Transform multiple texts using fitted TF-IDF vectorizer
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        
        processed_texts = self.preprocess_for_training(texts)
        return self.tfidf_vectorizer.transform(processed_texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from TF-IDF vectorizer
        """
        if self.tfidf_vectorizer is None:
            return []
        
        return self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze text complexity and characteristics
        """
        tokens = self.tokenize_and_clean(text)
        entities = self.extract_entities(text)
        
        analysis = {
            'original_length': len(text),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'avg_word_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'entities_found': sum(len(entity_list) for entity_list in entities.values()),
            'entity_breakdown': entities,
            'has_question_mark': '?' in text,
            'has_exclamation': '!' in text,
            'is_uppercase': text.isupper()
        }
        
        return analysis

class NamedEntityRecognizer:
    """
    Advanced Named Entity Recognition for customer support
    """
    
    def __init__(self):
        self.entity_patterns = {
            'order_id': [
                r'\b[A-Z]{2,3}\d{6,10}\b',  # Standard order ID
                r'\border[:\s]+([A-Za-z0-9]+)\b',  # "order: 12345"
                r'\b#([A-Za-z0-9]+)\b'  # "#12345"
            ],
            'tracking_number': [
                r'\b1Z[0-9A-Z]{16}\b',  # UPS tracking
                r'\b\d{12}\b',  # FedEx tracking
                r'\b\d{20,22}\b',  # USPS tracking
                r'\btracking[:\s]+([A-Za-z0-9]+)\b'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'phone': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\)\s?\d{3}[-.]?\d{4}'
            ],
            'amount': [
                r'\$\d+\.?\d*',
                r'\b\d+\.?\d*\s?dollars?\b'
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using pattern matching
        """
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            entities[entity_type] = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Handle tuples from groups in regex
                    if isinstance(matches[0], tuple):
                        matches = [match[0] for match in matches if match[0]]
                    entities[entity_type].extend(matches)
        
        # Remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities