"""
Conversation Management System for AI Chatbot
Handles dialogue flow, context tracking, and response generation
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import random

class ConversationState(Enum):
    """Conversation states for tracking dialogue flow"""
    GREETING = "greeting"
    INTENT_COLLECTION = "intent_collection"
    ENTITY_COLLECTION = "entity_collection"
    PROCESSING = "processing"
    RESPONSE_DELIVERY = "response_delivery"
    FOLLOW_UP = "follow_up"
    ESCALATION = "escalation"
    CLOSING = "closing"
    COMPLETED = "completed"

@dataclass
class ConversationTurn:
    """Single conversation turn data structure"""
    turn_id: str
    user_message: str
    bot_response: str
    intent: str
    entities: Dict[str, List[str]]
    confidence: float
    timestamp: datetime
    processing_time: float
    state: ConversationState

@dataclass
class ConversationContext:
    """Conversation context and memory"""
    session_id: str
    user_id: Optional[str]
    current_intent: Optional[str]
    collected_entities: Dict[str, List[str]]
    conversation_state: ConversationState
    turns: List[ConversationTurn]
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]

class ResponseGenerator:
    """
    Generates contextual responses based on intents and entities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.response_templates = self._load_response_templates()
        self.follow_up_questions = self._load_follow_up_questions()
        
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates from configuration"""
        responses = self.config.get('responses', {})
        
        # Default templates if not in config
        default_templates = {
            'track_order': [
                "I can help you track your order! Please provide your order ID or tracking number.",
                "Let me help you check the status of your order. What's your order ID?",
                "I'll look up your order for you. Can you share your order number or tracking ID?"
            ],
            'reset_password': [
                "I'll help you reset your password. Please provide your email address.",
                "No problem! I can guide you through resetting your password. What's your email?",
                "Let me help you with that password reset. Which email address is associated with your account?"
            ],
            'delivery_inquiry': [
                "I can check delivery information for you. What's your order ID?",
                "Let me look up the delivery details. Please share your order number.",
                "I'll help you track your delivery. Can you provide your order or tracking number?"
            ],
            'request_refund': [
                "I understand you'd like a refund. Can you provide your order ID and the reason for the return?",
                "I'll help you with the refund process. What's your order number?",
                "Let me assist you with that refund. Please share your order ID and tell me what happened."
            ],
            'contact_support': [
                "I'm connecting you with our support team. What type of issue are you experiencing?",
                "Let me get you to the right person. Can you tell me more about your concern?",
                "I'll help escalate this to our human agents. What's the main issue you're facing?"
            ],
            'greeting': [
                "Hello! I'm your AI customer support assistant. How can I help you today?",
                "Hi there! I'm here to help with any questions or issues you have. What can I do for you?",
                "Welcome! I'm your virtual assistant. How may I assist you today?"
            ],
            'goodbye': [
                "Thank you for contacting us! Have a great day!",
                "You're welcome! Don't hesitate to reach out if you need anything else.",
                "Glad I could help! Take care and have a wonderful day!"
            ],
            'fallback': [
                "I'm sorry, I didn't quite understand that. Could you please rephrase your question?",
                "I'm not sure I follow. Can you tell me more about what you need help with?",
                "Let me make sure I understand correctly. Could you provide more details about your request?"
            ]
        }
        
        # Merge with config responses
        for intent, template_list in default_templates.items():
            if intent in responses:
                if isinstance(responses[intent], str):
                    responses[intent] = [responses[intent]]
                responses[intent].extend(template_list)
            else:
                responses[intent] = template_list
        
        return responses
    
    def _load_follow_up_questions(self) -> Dict[str, List[str]]:
        """Load follow-up questions for entity collection"""
        return {
            'order_id': [
                "What's your order ID?",
                "Can you provide your order number?",
                "I'll need your order ID to help you."
            ],
            'email': [
                "What's your email address?",
                "Can you provide the email associated with your account?",
                "I'll need your email to proceed."
            ],
            'tracking_number': [
                "Do you have a tracking number?",
                "What's your tracking number?",
                "Can you share your package tracking ID?"
            ],
            'reason': [
                "What's the reason for your request?",
                "Can you tell me more about the issue?",
                "What happened with your order?"
            ]
        }
    
    def generate_response(self, intent: str, entities: Dict[str, List[str]], 
                         context: ConversationContext) -> str:
        """
        Generate contextual response based on intent and entities
        """
        # Check if we have the required entities for this intent
        missing_entities = self._check_missing_entities(intent, entities)
        
        if missing_entities:
            return self._generate_entity_collection_response(missing_entities[0])
        
        # Generate main response
        templates = self.response_templates.get(intent, self.response_templates['fallback'])
        base_response = random.choice(templates)
        
        # Personalize response with entities
        personalized_response = self._personalize_response(base_response, entities, context)
        
        return personalized_response
    
    def _check_missing_entities(self, intent: str, entities: Dict[str, List[str]]) -> List[str]:
        """Check which entities are missing for the given intent"""
        required_entities = {
            'track_order': ['order_id'],
            'reset_password': ['email'],
            'delivery_inquiry': ['order_id'],
            'request_refund': ['order_id'],
            'contact_support': []
        }
        
        intent_requirements = required_entities.get(intent, [])
        missing = []
        
        for required_entity in intent_requirements:
            if required_entity not in entities or not entities[required_entity]:
                missing.append(required_entity)
        
        return missing
    
    def _generate_entity_collection_response(self, missing_entity: str) -> str:
        """Generate response to collect missing entity"""
        questions = self.follow_up_questions.get(missing_entity, 
                                                ["Can you provide more information?"])
        return random.choice(questions)
    
    def _personalize_response(self, response: str, entities: Dict[str, List[str]], 
                            context: ConversationContext) -> str:
        """Personalize response with entity information"""
        # Replace entity placeholders
        if 'order_id' in entities and entities['order_id']:
            response = response.replace('{order_id}', entities['order_id'][0])
        
        if 'email' in entities and entities['email']:
            response = response.replace('{email}', entities['email'][0])
        
        # Add context-aware personalization
        if len(context.turns) > 0:
            # If this is a follow-up, acknowledge previous interaction
            if context.current_intent == context.turns[-1].intent:
                response = f"Continuing with your {context.current_intent.replace('_', ' ')} request... {response}"
        
        return response

class ConversationManager:
    """
    Main conversation management system
    Coordinates NLP components and manages dialogue flow
    """
    
    def __init__(self, config: Dict[str, Any], intent_classifier, entity_extractor, preprocessor):
        self.config = config
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.preprocessor = preprocessor
        self.response_generator = ResponseGenerator(config)
        
        # Active conversations storage
        self.conversations: Dict[str, ConversationContext] = {}
        
        # Conversation analytics
        self.analytics = {
            'total_conversations': 0,
            'total_turns': 0,
            'intent_counts': defaultdict(int),
            'average_conversation_length': 0.0,
            'escalation_rate': 0.0
        }
    
    def start_conversation(self, user_id: Optional[str] = None) -> str:
        """
        Start a new conversation session
        
        Returns:
            session_id: Unique conversation identifier
        """
        session_id = str(uuid.uuid4())
        
        conversation = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            current_intent=None,
            collected_entities={},
            conversation_state=ConversationState.GREETING,
            turns=[],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata={}
        )
        
        self.conversations[session_id] = conversation
        self.analytics['total_conversations'] += 1
        
        return session_id
    
    def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Process user message and generate response
        
        Args:
            session_id: Conversation session identifier
            user_message: User's input message
            
        Returns:
            Dictionary with response and conversation metadata
        """
        start_time = datetime.now()
        
        # Get or create conversation
        if session_id not in self.conversations:
            session_id = self.start_conversation()
        
        conversation = self.conversations[session_id]
        conversation.last_activity = datetime.now()
        
        # Process the message through NLP pipeline
        nlp_results = self._process_nlp_pipeline(user_message)
        
        # Update conversation context
        self._update_conversation_context(conversation, nlp_results)
        
        # Generate response
        response = self._generate_response(conversation, nlp_results)
        
        # Create conversation turn record
        processing_time = (datetime.now() - start_time).total_seconds()
        turn = self._create_conversation_turn(
            user_message, response, nlp_results, processing_time
        )
        
        conversation.turns.append(turn)
        self.analytics['total_turns'] += 1
        
        # Update conversation state
        self._update_conversation_state(conversation, nlp_results)
        
        return {
            'response': response,
            'intent': nlp_results['intent'],
            'confidence': nlp_results['confidence'],
            'entities': nlp_results['entities'],
            'session_id': session_id,
            'conversation_state': conversation.conversation_state.value,
            'turn_id': turn.turn_id,
            'processing_time': processing_time
        }
    
    def _process_nlp_pipeline(self, user_message: str) -> Dict[str, Any]:
        """
        Process message through complete NLP pipeline
        """
        # Preprocess text
        processed_text = self.preprocessor.clean_text(user_message)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(user_message)
        entity_summary = self.entity_extractor.format_entities_for_response(entities)
        
        # Classify intent
        if self.intent_classifier.is_trained:
            text_features = self.preprocessor.transform_text(processed_text)
            intent_result = self.intent_classifier.classify_intent(text_features)
            
            intent = intent_result['intent']
            confidence = intent_result['confidence']
            all_probabilities = intent_result['all_probabilities']
        else:
            # Fallback to rule-based intent detection
            intent, confidence = self._fallback_intent_detection(user_message)
            all_probabilities = {intent: confidence}
        
        return {
            'original_message': user_message,
            'processed_text': processed_text,
            'intent': intent,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'entities': entity_summary,
            'entity_details': entities
        }
    
    def _fallback_intent_detection(self, message: str) -> Tuple[str, float]:
        """
        Fallback rule-based intent detection when ML model is not available
        """
        message_lower = message.lower()
        
        # Simple keyword-based intent detection
        intent_keywords = {
            'track_order': ['track', 'order', 'status', 'where is', 'delivery'],
            'reset_password': ['password', 'reset', 'forgot', 'login'],
            'delivery_inquiry': ['delivery', 'shipping', 'arrive', 'when'],
            'request_refund': ['refund', 'return', 'money back', 'cancel'],
            'contact_support': ['human', 'agent', 'support', 'help'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'goodbye': ['bye', 'goodbye', 'thanks', 'thank you']
        }
        
        best_match = 'fallback'
        best_score = 0.0
        
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > best_score:
                best_score = score
                best_match = intent
        
        confidence = min(0.8, best_score / 3.0) if best_score > 0 else 0.3
        
        return best_match, confidence
    
    def _update_conversation_context(self, conversation: ConversationContext, 
                                   nlp_results: Dict[str, Any]):
        """
        Update conversation context with new information
        """
        # Update current intent
        conversation.current_intent = nlp_results['intent']
        
        # Merge entities with existing collected entities
        for entity_type, entity_values in nlp_results['entities'].items():
            if entity_type not in conversation.collected_entities:
                conversation.collected_entities[entity_type] = []
            
            # Add new entities (avoid duplicates)
            for value in entity_values:
                if value not in conversation.collected_entities[entity_type]:
                    conversation.collected_entities[entity_type].append(value)
    
    def _generate_response(self, conversation: ConversationContext, 
                          nlp_results: Dict[str, Any]) -> str:
        """
        Generate appropriate response based on conversation context
        """
        intent = nlp_results['intent']
        entities = conversation.collected_entities
        
        # Update analytics
        self.analytics['intent_counts'][intent] += 1
        
        # Generate response using response generator
        response = self.response_generator.generate_response(intent, entities, conversation)
        
        return response
    
    def _create_conversation_turn(self, user_message: str, bot_response: str,
                                nlp_results: Dict[str, Any], processing_time: float) -> ConversationTurn:
        """
        Create a conversation turn record
        """
        return ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_message=user_message,
            bot_response=bot_response,
            intent=nlp_results['intent'],
            entities=nlp_results['entities'],
            confidence=nlp_results['confidence'],
            timestamp=datetime.now(),
            processing_time=processing_time,
            state=self.conversations[list(self.conversations.keys())[-1]].conversation_state
        )
    
    def _update_conversation_state(self, conversation: ConversationContext,
                                 nlp_results: Dict[str, Any]):
        """
        Update conversation state based on current interaction
        """
        intent = nlp_results['intent']
        confidence = nlp_results['confidence']
        
        # State transition logic
        if intent == 'greeting':
            conversation.conversation_state = ConversationState.INTENT_COLLECTION
        elif intent == 'goodbye':
            conversation.conversation_state = ConversationState.CLOSING
        elif confidence < 0.5:
            conversation.conversation_state = ConversationState.ESCALATION
        elif self._has_required_entities(conversation, intent):
            conversation.conversation_state = ConversationState.PROCESSING
        else:
            conversation.conversation_state = ConversationState.ENTITY_COLLECTION
    
    def _has_required_entities(self, conversation: ConversationContext, intent: str) -> bool:
        """
        Check if conversation has all required entities for the given intent
        """
        required_entities = {
            'track_order': ['order_id'],
            'reset_password': ['email'],
            'delivery_inquiry': ['order_id'],
            'request_refund': ['order_id']
        }
        
        intent_requirements = required_entities.get(intent, [])
        
        for required_entity in intent_requirements:
            if (required_entity not in conversation.collected_entities or 
                not conversation.collected_entities[required_entity]):
                return False
        
        return True
    
    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete conversation history
        """
        if session_id not in self.conversations:
            return {'error': 'Conversation not found'}
        
        conversation = self.conversations[session_id]
        
        return {
            'session_id': session_id,
            'user_id': conversation.user_id,
            'state': conversation.conversation_state.value,
            'created_at': conversation.created_at.isoformat(),
            'last_activity': conversation.last_activity.isoformat(),
            'current_intent': conversation.current_intent,
            'collected_entities': conversation.collected_entities,
            'turn_count': len(conversation.turns),
            'turns': [asdict(turn) for turn in conversation.turns]
        }
    
    def end_conversation(self, session_id: str) -> bool:
        """
        End a conversation session
        """
        if session_id in self.conversations:
            conversation = self.conversations[session_id]
            conversation.conversation_state = ConversationState.COMPLETED
            
            # Update analytics
            conversation_length = len(conversation.turns)
            total_conversations = self.analytics['total_conversations']
            current_avg = self.analytics['average_conversation_length']
            
            # Update running average
            self.analytics['average_conversation_length'] = (
                (current_avg * (total_conversations - 1) + conversation_length) / total_conversations
            )
            
            # Remove from active conversations (optional - you might want to keep for analytics)
            # del self.conversations[session_id]
            
            return True
        
        return False
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get conversation analytics summary
        """
        active_conversations = sum(1 for conv in self.conversations.values() 
                                 if conv.conversation_state != ConversationState.COMPLETED)
        
        escalated_conversations = sum(1 for conv in self.conversations.values() 
                                    if conv.conversation_state == ConversationState.ESCALATION)
        
        escalation_rate = (escalated_conversations / self.analytics['total_conversations'] 
                          if self.analytics['total_conversations'] > 0 else 0.0)
        
        return {
            'total_conversations': self.analytics['total_conversations'],
            'active_conversations': active_conversations,
            'completed_conversations': len(self.conversations) - active_conversations,
            'total_turns': self.analytics['total_turns'],
            'average_conversation_length': self.analytics['average_conversation_length'],
            'escalation_rate': escalation_rate,
            'top_intents': dict(self.analytics['intent_counts'].most_common(5)),
            'current_time': datetime.now().isoformat()
        }