"""
Entity Extraction Module for AI Chatbot
Combines rule-based patterns with NLP techniques for entity recognition
"""

import re
import spacy
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class Entity:
    """Entity data structure"""
    text: str
    label: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    context: str = ""

class EntityExtractor:
    """
    Advanced Entity Extraction combining multiple approaches:
    1. Rule-based pattern matching
    2. SpaCy NLP model
    3. Custom business logic for customer support
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp = None
        self._load_spacy_model()
        
        # Define patterns for customer support entities
        self.patterns = {
            'order_id': [
                r'\b[A-Z]{2,3}[-]?\d{6,12}\b',  # ORD123456, AB-123456789
                r'\border\s*[:#]?\s*([A-Za-z0-9\-]+)\b',  # "order: 123456"
                r'\b#([A-Za-z0-9\-]+)\b',  # "#123456"
                r'\bid\s*[:#]?\s*([A-Za-z0-9\-]+)\b'  # "id: 123456"
            ],
            'tracking_number': [
                r'\b1Z[0-9A-Z]{16}\b',  # UPS tracking
                r'\b\d{12}\b',  # FedEx 12-digit
                r'\b\d{20,22}\b',  # USPS 20-22 digits
                r'\btracking\s*[:#]?\s*([A-Za-z0-9]+)\b',  # "tracking: 123456"
                r'\b94[0-9]{20}\b'  # USPS Priority Mail
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
            ],
            'phone': [
                r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            ],
            'amount': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,234.56
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?\b',  # 1234 dollars
                r'\b\d+\.\d{2}\b'  # 19.99
            ],
            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
                r'\b\d{4}[-]\d{1,2}[-]\d{1,2}\b',  # YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
            ],
            'product_name': [
                r'\b[A-Z][a-zA-Z0-9\s]{2,30}\b(?=\s(?:model|version|size))',  # Product names before model/version
                r'\bmodel\s+([A-Za-z0-9\-]+)\b',  # "model XYZ-123"
                r'\bsku\s*[:#]?\s*([A-Za-z0-9\-]+)\b'  # "sku: ABC123"
            ],
            'address': [
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
                r'\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'  # City, State ZIP
            ]
        }
        
        # Priority order for entity types (higher priority entities are extracted first)
        self.entity_priority = [
            'email', 'phone', 'tracking_number', 'order_id', 
            'amount', 'date', 'product_name', 'address'
        ]
    
    def _load_spacy_model(self):
        """Load SpaCy NLP model"""
        try:
            model_name = self.config.get('model', {}).get('entity_extractor', {}).get('model', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
        except IOError:
            print(f"Warning: SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_with_patterns(self, text: str) -> Dict[str, List[Entity]]:
        """
        Extract entities using regex patterns
        """
        entities = defaultdict(list)
        text_lower = text.lower()
        
        for entity_type in self.entity_priority:
            if entity_type in self.patterns:
                for pattern in self.patterns[entity_type]:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        entity_text = match.group(1) if match.groups() else match.group(0)
                        start_pos = match.start(1) if match.groups() else match.start()
                        end_pos = match.end(1) if match.groups() else match.end()
                        
                        # Get context (20 chars before and after)
                        context_start = max(0, start_pos - 20)
                        context_end = min(len(text), end_pos + 20)
                        context = text[context_start:context_end]
                        
                        entity = Entity(
                            text=entity_text.strip(),
                            label=entity_type,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=0.9,  # High confidence for pattern matching
                            context=context
                        )
                        
                        entities[entity_type].append(entity)
        
        return dict(entities)
    
    def extract_with_spacy(self, text: str) -> Dict[str, List[Entity]]:
        """
        Extract entities using SpaCy NLP model
        """
        entities = defaultdict(list)
        
        if self.nlp is None:
            return dict(entities)
        
        doc = self.nlp(text)
        
        # Map SpaCy labels to our entity types
        spacy_label_mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'MONEY': 'amount',
            'DATE': 'date',
            'TIME': 'time',
            'PRODUCT': 'product_name',
            'CARDINAL': 'number',
            'ORDINAL': 'number'
        }
        
        for ent in doc.ents:
            entity_type = spacy_label_mapping.get(ent.label_, ent.label_.lower())
            
            # Get context
            context_start = max(0, ent.start_char - 20)
            context_end = min(len(text), ent.end_char + 20)
            context = text[context_start:context_end]
            
            entity = Entity(
                text=ent.text.strip(),
                label=entity_type,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                confidence=0.7,  # Medium confidence for SpaCy
                context=context
            )
            
            entities[entity_type].append(entity)
        
        return dict(entities)
    
    def extract_business_entities(self, text: str) -> Dict[str, List[Entity]]:
        """
        Extract customer support specific entities using business logic
        """
        entities = defaultdict(list)
        
        # Extract issue types based on keywords
        issue_keywords = {
            'delivery': ['delivery', 'shipping', 'arrive', 'delivered', 'package'],
            'payment': ['payment', 'charge', 'bill', 'invoice', 'refund'],
            'account': ['account', 'login', 'password', 'profile', 'settings'],
            'product': ['product', 'item', 'broken', 'defective', 'quality'],
            'return': ['return', 'exchange', 'refund', 'send back']
        }
        
        text_lower = text.lower()
        for issue_type, keywords in issue_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    start_pos = text_lower.find(keyword)
                    end_pos = start_pos + len(keyword)
                    
                    entity = Entity(
                        text=keyword,
                        label='issue_type',
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=0.8,
                        context=text[max(0, start_pos-10):end_pos+10]
                    )
                    entities['issue_type'].append(entity)
        
        # Extract urgency indicators
        urgency_patterns = [
            (r'\burgent\b|\bemergency\b|\basap\b', 'high'),
            (r'\bquickly\b|\bfast\b|\bhurry\b', 'medium'),
            (r'\bwhen\s+possible\b|\bno\s+rush\b', 'low')
        ]
        
        for pattern, urgency_level in urgency_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = Entity(
                    text=match.group(0),
                    label='urgency',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7,
                    context=text[max(0, match.start()-15):match.end()+15]
                )
                entities['urgency'].append(entity)
        
        return dict(entities)
    
    def extract_entities(self, text: str) -> Dict[str, List[Entity]]:
        """
        Main entity extraction method combining all approaches
        """
        all_entities = defaultdict(list)
        
        # Extract using different methods
        pattern_entities = self.extract_with_patterns(text)
        spacy_entities = self.extract_with_spacy(text)
        business_entities = self.extract_business_entities(text)
        
        # Combine all entities
        for entities_dict in [pattern_entities, spacy_entities, business_entities]:
            for entity_type, entity_list in entities_dict.items():
                all_entities[entity_type].extend(entity_list)
        
        # Remove duplicates and overlapping entities
        cleaned_entities = self._remove_duplicates(dict(all_entities))
        
        return cleaned_entities
    
    def _remove_duplicates(self, entities: Dict[str, List[Entity]]) -> Dict[str, List[Entity]]:
        """
        Remove duplicate and overlapping entities, keeping the highest confidence ones
        """
        cleaned_entities = defaultdict(list)
        
        for entity_type, entity_list in entities.items():
            # Sort by confidence (highest first)
            sorted_entities = sorted(entity_list, key=lambda x: x.confidence, reverse=True)
            
            for entity in sorted_entities:
                # Check for overlaps with already added entities
                is_duplicate = False
                
                for existing_entity in cleaned_entities[entity_type]:
                    # Check if entities overlap significantly
                    overlap_start = max(entity.start_pos, existing_entity.start_pos)
                    overlap_end = min(entity.end_pos, existing_entity.end_pos)
                    
                    if overlap_end > overlap_start:
                        overlap_length = overlap_end - overlap_start
                        entity_length = entity.end_pos - entity.start_pos
                        
                        # If overlap is more than 50% of entity length, consider duplicate
                        if overlap_length / entity_length > 0.5:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    cleaned_entities[entity_type].append(entity)
        
        return dict(cleaned_entities)
    
    def get_entity_summary(self, entities: Dict[str, List[Entity]]) -> Dict[str, Any]:
        """
        Generate a summary of extracted entities
        """
        summary = {
            'total_entities': sum(len(entity_list) for entity_list in entities.values()),
            'entity_types': list(entities.keys()),
            'entity_counts': {entity_type: len(entity_list) for entity_type, entity_list in entities.items()},
            'high_confidence_entities': 0,
            'medium_confidence_entities': 0,
            'low_confidence_entities': 0
        }
        
        # Count entities by confidence level
        for entity_list in entities.values():
            for entity in entity_list:
                if entity.confidence >= 0.8:
                    summary['high_confidence_entities'] += 1
                elif entity.confidence >= 0.6:
                    summary['medium_confidence_entities'] += 1
                else:
                    summary['low_confidence_entities'] += 1
        
        return summary
    
    def format_entities_for_response(self, entities: Dict[str, List[Entity]]) -> Dict[str, List[str]]:
        """
        Format entities for API response (simplified format)
        """
        formatted = {}
        
        for entity_type, entity_list in entities.items():
            formatted[entity_type] = [entity.text for entity in entity_list]
        
        return formatted
    
    def validate_entities(self, entities: Dict[str, List[Entity]]) -> Dict[str, List[Entity]]:
        """
        Validate extracted entities using business rules
        """
        validated_entities = defaultdict(list)
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                is_valid = True
                
                # Validation rules
                if entity_type == 'email':
                    # Basic email validation
                    if '@' not in entity.text or '.' not in entity.text.split('@')[-1]:
                        is_valid = False
                
                elif entity_type == 'phone':
                    # Phone number validation (must have 10 digits)
                    digits = re.sub(r'\D', '', entity.text)
                    if len(digits) != 10 and len(digits) != 11:
                        is_valid = False
                
                elif entity_type == 'order_id':
                    # Order ID should be alphanumeric and have minimum length
                    if len(entity.text) < 4 or not re.match(r'^[A-Za-z0-9\-]+$', entity.text):
                        is_valid = False
                
                elif entity_type == 'amount':
                    # Amount should be positive
                    try:
                        amount_str = re.sub(r'[^\d.]', '', entity.text)
                        amount = float(amount_str)
                        if amount <= 0:
                            is_valid = False
                    except ValueError:
                        is_valid = False
                
                if is_valid:
                    validated_entities[entity_type].append(entity)
        
        return dict(validated_entities)

class EntityLinker:
    """
    Links extracted entities to create structured information
    """
    
    def __init__(self):
        self.entity_relationships = {
            'order_inquiry': ['order_id', 'tracking_number', 'email'],
            'refund_request': ['order_id', 'amount', 'email'],
            'delivery_issue': ['tracking_number', 'address', 'date'],
            'account_issue': ['email', 'phone']
        }
    
    def link_entities(self, entities: Dict[str, List[Entity]], intent: str) -> Dict[str, Any]:
        """
        Link entities based on the detected intent
        """
        linked_data = {
            'intent': intent,
            'primary_entities': {},
            'supporting_entities': {},
            'completeness_score': 0.0
        }
        
        # Get expected entities for this intent
        expected_entities = self.entity_relationships.get(intent, [])
        
        if not expected_entities:
            # If no specific rules, include all entities
            linked_data['primary_entities'] = entities
            linked_data['completeness_score'] = 1.0
            return linked_data
        
        # Separate primary and supporting entities
        found_primary = 0
        for expected_entity in expected_entities:
            if expected_entity in entities and entities[expected_entity]:
                linked_data['primary_entities'][expected_entity] = entities[expected_entity]
                found_primary += 1
            else:
                linked_data['primary_entities'][expected_entity] = []
        
        # Add supporting entities
        for entity_type, entity_list in entities.items():
            if entity_type not in expected_entities:
                linked_data['supporting_entities'][entity_type] = entity_list
        
        # Calculate completeness score
        linked_data['completeness_score'] = found_primary / len(expected_entities) if expected_entities else 1.0
        
        return linked_data