"""
Intent Classification Module for AI Chatbot
Uses Machine Learning to classify customer intents
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import os

class IntentClassifier:
    """
    Machine Learning based Intent Classification
    Supports multiple algorithms: RandomForest, SVM, Naive Bayes, Logistic Regression
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.is_trained = False
        self.feature_importance = None
        
        # Initialize model based on configuration
        self._init_model()
    
    def _init_model(self):
        """Initialize the ML model based on configuration"""
        model_config = self.config.get('model', {}).get('intent_classifier', {})
        algorithm = model_config.get('algorithm', 'RandomForest')
        
        if algorithm == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', 10),
                random_state=42
            )
        elif algorithm == 'SVM':
            self.model = SVC(
                kernel='linear',
                probability=True,
                random_state=42
            )
        elif algorithm == 'NaiveBayes':
            self.model = MultinomialNB()
        elif algorithm == 'LogisticRegression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def prepare_labels(self, labels: List[str]) -> np.ndarray:
        """
        Encode string labels to numerical format
        """
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        encoded_labels = [self.label_encoder[label] for label in labels]
        return np.array(encoded_labels)
    
    def train(self, X_train, y_train: List[str]) -> Dict[str, Any]:
        """
        Train the intent classifier
        
        Args:
            X_train: Feature matrix (from TF-IDF)
            y_train: Intent labels
            
        Returns:
            Training metrics and statistics
        """
        # Prepare labels
        y_encoded = self.prepare_labels(y_train)
        
        # Train the model
        self.model.fit(X_train, y_encoded)
        self.is_trained = True
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_encoded, train_predictions)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_encoded, cv=5)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        training_stats = {
            'train_accuracy': train_accuracy,
            'cv_mean_score': cv_scores.mean(),
            'cv_std_score': cv_scores.std(),
            'n_samples': len(y_train),
            'n_features': X_train.shape[1],
            'n_intents': len(self.label_encoder)
        }
        
        return training_stats
    
    def predict(self, X_test) -> List[str]:
        """
        Predict intents for test data
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X_test)
        return [self.reverse_label_encoder[pred] for pred in predictions]
    
    def predict_proba(self, X_test) -> List[Dict[str, float]]:
        """
        Predict intent probabilities
        
        Returns:
            List of dictionaries with intent: probability pairs
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        probabilities = self.model.predict_proba(X_test)
        
        results = []
        for prob_array in probabilities:
            intent_probs = {}
            for idx, prob in enumerate(prob_array):
                intent = self.reverse_label_encoder[idx]
                intent_probs[intent] = float(prob)
            
            # Sort by probability (highest first)
            sorted_probs = dict(sorted(intent_probs.items(), 
                                     key=lambda x: x[1], reverse=True))
            results.append(sorted_probs)
        
        return results
    
    def classify_intent(self, X_input) -> Dict[str, Any]:
        """
        Classify intent for a single input with confidence score
        
        Returns:
            Dictionary with predicted intent, confidence, and all probabilities
        """
        probabilities = self.predict_proba(X_input)[0]
        predicted_intent = list(probabilities.keys())[0]
        confidence = list(probabilities.values())[0]
        
        # Check confidence threshold
        confidence_threshold = self.config.get('model', {}).get('confidence_threshold', 0.7)
        is_confident = confidence >= confidence_threshold
        
        return {
            'intent': predicted_intent,
            'confidence': confidence,
            'is_confident': is_confident,
            'all_probabilities': probabilities,
            'threshold_used': confidence_threshold
        }
    
    def evaluate(self, X_test, y_test: List[str]) -> Dict[str, Any]:
        """
        Evaluate model performance on test data
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Encode test labels
        y_test_encoded = [self.label_encoder.get(label, -1) for label in y_test]
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, predictions)
        
        # Classification report
        target_names = [self.reverse_label_encoder[i] for i in range(len(self.reverse_label_encoder))]
        class_report = classification_report(y_test_encoded, predictions, 
                                           target_names=target_names, 
                                           output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test_encoded, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'intent_labels': target_names
        }
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top important features for intent classification
        """
        if self.feature_importance is None:
            return []
        
        # Create feature importance pairs
        feature_importance_pairs = list(zip(feature_names, self.feature_importance))
        
        # Sort by importance (descending)
        sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
        
        return sorted_features[:top_n]
    
    def save_model(self, filepath: str):
        """
        Save the trained model and encoders
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder,
            'config': self.config,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model and encoders
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.reverse_label_encoder = model_data['reverse_label_encoder']
        self.config = model_data['config']
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = True

class IntentAnalyzer:
    """
    Advanced intent analysis and insights
    """
    
    def __init__(self, classifier: IntentClassifier):
        self.classifier = classifier
    
    def analyze_intent_distribution(self, intents: List[str]) -> Dict[str, Any]:
        """
        Analyze the distribution of intents in training data
        """
        from collections import Counter
        
        intent_counts = Counter(intents)
        total_samples = len(intents)
        
        distribution = {}
        for intent, count in intent_counts.items():
            distribution[intent] = {
                'count': count,
                'percentage': (count / total_samples) * 100,
                'samples_per_intent': count
            }
        
        return {
            'total_samples': total_samples,
            'unique_intents': len(intent_counts),
            'distribution': distribution,
            'most_common': intent_counts.most_common(),
            'least_common': intent_counts.most_common()[-1] if intent_counts else None
        }
    
    def get_intent_confidence_stats(self, X_data) -> Dict[str, Any]:
        """
        Analyze confidence statistics across predictions
        """
        probabilities = self.classifier.predict_proba(X_data)
        
        confidences = [max(prob_dict.values()) for prob_dict in probabilities]
        
        return {
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'std_confidence': np.std(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.7),
            'high_confidence_count': sum(1 for c in confidences if c > 0.9)
        }
    
    def identify_ambiguous_samples(self, X_data, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Identify samples where the model is uncertain (similar probabilities for top intents)
        """
        probabilities = self.classifier.predict_proba(X_data)
        ambiguous_samples = []
        
        for idx, prob_dict in enumerate(probabilities):
            sorted_probs = sorted(prob_dict.values(), reverse=True)
            
            if len(sorted_probs) >= 2:
                top_diff = sorted_probs[0] - sorted_probs[1]
                
                if top_diff <= threshold:
                    ambiguous_samples.append({
                        'sample_index': idx,
                        'top_intent': list(prob_dict.keys())[0],
                        'top_confidence': sorted_probs[0],
                        'second_intent': list(prob_dict.keys())[1],
                        'second_confidence': sorted_probs[1],
                        'difference': top_diff,
                        'all_probabilities': prob_dict
                    })
        
        return ambiguous_samples