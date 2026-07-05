"""
Main AI Chatbot Engine
Coordinates all NLP components for intelligent customer support
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import our NLP components
from src.nlp.preprocessor import TextPreprocessor, NamedEntityRecognizer
from src.nlp.intent_classifier import IntentClassifier, IntentAnalyzer
from src.nlp.entity_extractor import EntityExtractor, EntityLinker
from src.nlp.conversation_manager import ConversationManager, ConversationState
from src.utils.config_loader import ConfigLoader

class ChatbotEngine:
    """
    Main AI Chatbot Engine for Customer Support
    
    Features:
    - Intent Recognition with confidence scoring
    - Named Entity Recognition and extraction
    - Conversation flow management
    - Context-aware response generation
    - Real-time analytics and monitoring
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the chatbot engine with all components
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize NLP components
        self.logger.info("Initializing NLP components...")
        self._initialize_components()
        
        # Load or create models
        self._setup_models()
        
        self.logger.info("AI Chatbot Engine initialized successfully!")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'chatbot.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all NLP components"""
        try:
            # Text preprocessing
            self.preprocessor = TextPreprocessor(self.config)
            self.logger.info("✓ Text preprocessor initialized")
            
            # Intent classification
            self.intent_classifier = IntentClassifier(self.config)
            self.intent_analyzer = IntentAnalyzer(self.intent_classifier)
            self.logger.info("✓ Intent classifier initialized")
            
            # Entity extraction
            self.entity_extractor = EntityExtractor(self.config)
            self.entity_linker = EntityLinker()
            self.named_entity_recognizer = NamedEntityRecognizer()
            self.logger.info("✓ Entity extractor initialized")
            
            # Conversation management
            self.conversation_manager = ConversationManager(
                self.config, 
                self.intent_classifier,
                self.entity_extractor, 
                self.preprocessor
            )
            self.logger.info("✓ Conversation manager initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _setup_models(self):
        """Setup and load ML models"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Check if trained models exist
        intent_model_path = models_dir / "intent_classifier.joblib"
        
        if intent_model_path.exists():
            try:
                self.intent_classifier.load_model(str(intent_model_path))
                self.logger.info("✓ Loaded pre-trained intent classifier")
            except Exception as e:
                self.logger.warning(f"Could not load intent model: {e}")
                self.logger.info("Will use fallback rule-based intent detection")
        else:
            self.logger.info("No pre-trained models found. Will use fallback methods.")
    
    def chat(self, message: str, session_id: Optional[str] = None, 
             user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main chat interface - process user message and return response
        
        Args:
            message: User's input message
            session_id: Optional conversation session ID
            user_id: Optional user identifier
            
        Returns:
            Dictionary with bot response and metadata
        """
        try:
            start_time = datetime.now()
            
            # Start new conversation if no session_id provided
            if session_id is None:
                session_id = self.conversation_manager.start_conversation(user_id)
                self.logger.info(f"Started new conversation: {session_id}")
            
            # Process message through conversation manager
            result = self.conversation_manager.process_message(session_id, message)
            
            # Add additional metadata
            result.update({
                'timestamp': datetime.now().isoformat(),
                'total_processing_time': (datetime.now() - start_time).total_seconds(),
                'engine_version': '1.0.0',
                'nlp_features_used': ['intent_classification', 'entity_extraction', 'conversation_management']
            })
            
            # Log interaction
            self.logger.info(f"Processed message in session {session_id}: "
                           f"Intent={result['intent']}, Confidence={result['confidence']:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return self._generate_error_response(str(e), session_id)
    
    def _generate_error_response(self, error_msg: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate error response when something goes wrong"""
        return {
            'response': "I apologize, but I'm experiencing some technical difficulties. Please try again or contact our support team.",
            'intent': 'error',
            'confidence': 0.0,
            'entities': {},
            'session_id': session_id,
            'conversation_state': 'error',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
    
    def train_from_data(self, training_data_path: str) -> Dict[str, Any]:
        """
        Train the chatbot models from training data
        
        Args:
            training_data_path: Path to training data file (JSON format)
            
        Returns:
            Training results and statistics
        """
        try:
            self.logger.info(f"Training models from: {training_data_path}")
            
            # Load training data
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            # Extract texts and labels
            texts = [item['text'] for item in training_data['samples']]
            labels = [item['intent'] for item in training_data['samples']]
            
            self.logger.info(f"Loaded {len(texts)} training samples with {len(set(labels))} unique intents")
            
            # Preprocess texts
            self.preprocessor.fit_tfidf(texts)
            X_train = self.preprocessor.transform_texts(texts)
            
            # Train intent classifier
            training_stats = self.intent_classifier.train(X_train, labels)
            
            # Save trained model
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            model_path = models_dir / "intent_classifier.joblib"
            self.intent_classifier.save_model(str(model_path))
            
            # Save preprocessor
            import joblib
            preprocessor_path = models_dir / "preprocessor.joblib"
            joblib.dump(self.preprocessor, preprocessor_path)
            
            self.logger.info("✓ Models trained and saved successfully")
            
            return {
                'status': 'success',
                'training_stats': training_stats,
                'model_path': str(model_path),
                'preprocessor_path': str(preprocessor_path),
                'training_samples': len(texts),
                'unique_intents': len(set(labels))
            }
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Detailed text analysis for debugging and insights
        
        Args:
            text: Input text to analyze
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Text preprocessing analysis
            preprocessing_analysis = self.preprocessor.analyze_text_complexity(text)
            
            # Entity extraction
            entities = self.entity_extractor.extract_entities(text)
            entity_summary = self.entity_extractor.get_entity_summary(entities)
            
            # Intent classification (if model is trained)
            intent_analysis = {}
            if self.intent_classifier.is_trained:
                text_features = self.preprocessor.transform_text(text)
                intent_result = self.intent_classifier.classify_intent(text_features)
                intent_analysis = intent_result
            
            return {
                'input_text': text,
                'preprocessing_analysis': preprocessing_analysis,
                'entity_analysis': {
                    'entities_found': entity_summary,
                    'detailed_entities': entities
                },
                'intent_analysis': intent_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return {'error': str(e)}
    
    def get_conversation_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get conversation analytics and insights
        
        Args:
            session_id: Optional session ID for specific conversation
            
        Returns:
            Analytics data
        """
        try:
            if session_id:
                # Get specific conversation history
                return self.conversation_manager.get_conversation_history(session_id)
            else:
                # Get overall analytics summary
                return self.conversation_manager.get_analytics_summary()
                
        except Exception as e:
            self.logger.error(f"Error getting analytics: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and health check
        
        Returns:
            System status information
        """
        try:
            return {
                'status': 'healthy',
                'components': {
                    'preprocessor': 'active',
                    'intent_classifier': 'trained' if self.intent_classifier.is_trained else 'fallback',
                    'entity_extractor': 'active',
                    'conversation_manager': 'active'
                },
                'models': {
                    'intent_classifier_trained': self.intent_classifier.is_trained,
                    'tfidf_fitted': self.preprocessor.tfidf_vectorizer is not None
                },
                'analytics': self.conversation_manager.get_analytics_summary(),
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def export_conversation_data(self, output_path: str) -> Dict[str, Any]:
        """
        Export conversation data for analysis or training
        
        Args:
            output_path: Path to save exported data
            
        Returns:
            Export results
        """
        try:
            # Collect all conversation data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_conversations': len(self.conversation_manager.conversations),
                'conversations': []
            }
            
            for session_id, conversation in self.conversation_manager.conversations.items():
                conversation_data = {
                    'session_id': session_id,
                    'user_id': conversation.user_id,
                    'created_at': conversation.created_at.isoformat(),
                    'last_activity': conversation.last_activity.isoformat(),
                    'state': conversation.conversation_state.value,
                    'current_intent': conversation.current_intent,
                    'collected_entities': conversation.collected_entities,
                    'turns': [
                        {
                            'turn_id': turn.turn_id,
                            'user_message': turn.user_message,
                            'bot_response': turn.bot_response,
                            'intent': turn.intent,
                            'entities': turn.entities,
                            'confidence': turn.confidence,
                            'timestamp': turn.timestamp.isoformat(),
                            'processing_time': turn.processing_time
                        }
                        for turn in conversation.turns
                    ]
                }
                export_data['conversations'].append(conversation_data)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported conversation data to: {output_path}")
            
            return {
                'status': 'success',
                'output_path': output_path,
                'conversations_exported': len(export_data['conversations']),
                'total_turns': sum(len(conv['turns']) for conv in export_data['conversations'])
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

# Convenience functions for easy usage
def create_chatbot(config_path: str = "config.yaml") -> ChatbotEngine:
    """
    Create and initialize a chatbot instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized ChatbotEngine instance
    """
    return ChatbotEngine(config_path)

def quick_chat(message: str, chatbot: Optional[ChatbotEngine] = None) -> str:
    """
    Quick chat function for simple interactions
    
    Args:
        message: User message
        chatbot: Optional existing chatbot instance
        
    Returns:
        Bot response text
    """
    if chatbot is None:
        chatbot = create_chatbot()
    
    result = chatbot.chat(message)
    return result['response']