"""
Configuration Loading Utilities
"""

import yaml
import os
from typing import Dict, Any
from dotenv import load_dotenv

class ConfigLoader:
    """
    Load and manage configuration from multiple sources
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = {}
        self._load_config()
        self._load_env_variables()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            self.config = self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            self.config = self._get_default_config()
    
    def _load_env_variables(self):
        """Load environment variables"""
        load_dotenv()
        
        # Override config with environment variables
        env_mappings = {
            'FLASK_ENV': ['flask', 'env'],
            'API_HOST': ['api', 'host'],
            'API_PORT': ['api', 'port'],
            'CONFIDENCE_THRESHOLD': ['model', 'confidence_threshold']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_config(config_path, value)
    
    def _set_nested_config(self, path: list, value: str):
        """Set nested configuration value"""
        current = self.config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Type conversion
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            value = float(value)
        
        current[path[-1]] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model': {
                'intent_classifier': {
                    'algorithm': 'RandomForest',
                    'n_estimators': 100,
                    'max_depth': 10
                },
                'entity_extractor': {
                    'algorithm': 'spacy',
                    'model': 'en_core_web_sm'
                },
                'confidence_threshold': 0.7
            },
            'nlp': {
                'tokenization': {
                    'remove_stopwords': True,
                    'lemmatization': True,
                    'lowercase': True
                },
                'features': {
                    'use_tfidf': True,
                    'use_ngrams': True,
                    'ngram_range': [1, 2],
                    'max_features': 5000
                }
            },
            'api': {
                'host': 'localhost',
                'port': 5000
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration"""
        return self.config
    
    def get(self, path: str, default=None):
        """Get configuration value by dot-separated path"""
        keys = path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except KeyError:
            return default