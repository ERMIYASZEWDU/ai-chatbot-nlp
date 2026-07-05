"""
Dialogue State Tracking for Advanced Conversation Management
Tracks user goals, context, and conversation progress
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

class DialogueAction(Enum):
    """Possible dialogue actions"""
    INFORM = "inform"
    REQUEST = "request"
    CONFIRM = "confirm"
    DENY = "deny"
    GREET = "greet"
    BYE = "bye"
    THANK = "thank"
    AFFIRM = "affirm"
    NEGATE = "negate"
    CLARIFY = "clarify"

class SlotStatus(Enum):
    """Status of information slots"""
    EMPTY = "empty"
    PARTIAL = "partial"
    FILLED = "filled"
    CONFIRMED = "confirmed"

@dataclass
class InformationSlot:
    """Represents a piece of information needed for task completion"""
    name: str
    value: Optional[str] = None
    status: SlotStatus = SlotStatus.EMPTY
    confidence: float = 0.0
    confirmed: bool = False
    attempts: int = 0
    last_updated: Optional[datetime] = None

@dataclass
class DialogueGoal:
    """Represents user's current goal"""
    intent: str
    required_slots: Set[str]
    optional_slots: Set[str]
    filled_slots: Dict[str, InformationSlot] = field(default_factory=dict)
    completion_percentage: float = 0.0
    is_complete: bool = False

class DialogueStateTracker:
    """
    Advanced dialogue state tracking for multi-turn conversations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Define slot requirements for each intent
        self.intent_slot_requirements = {
            'track_order': {
                'required': {'order_id'},
                'optional': {'email', 'phone'}
            },
            'reset_password': {
                'required': {'email'},
                'optional': {'phone', 'username'}
            },
            'delivery_inquiry': {
                'required': {'order_id'},
                'optional': {'address', 'tracking_number'}
            },
            'request_refund': {
                'required': {'order_id', 'reason'},
                'optional': {'email', 'amount'}
            },
            'contact_support': {
                'required': {'issue_type'},
                'optional': {'priority', 'description'}
            }
        }
        
        # Current dialogue state
        self.current_goal: Optional[DialogueGoal] = None
        self.dialogue_history: List[Dict[str, Any]] = []
        self.context_stack: List[Dict[str, Any]] = []
        
    def initialize_goal(self, intent: str) -> DialogueGoal:
        """
        Initialize a new dialogue goal
        """
        slot_requirements = self.intent_slot_requirements.get(intent, {
            'required': set(),
            'optional': set()
        })
        
        goal = DialogueGoal(
            intent=intent,
            required_slots=slot_requirements['required'],
            optional_slots=slot_requirements['optional']
        )
        
        # Initialize slots
        all_slots = goal.required_slots.union(goal.optional_slots)
        for slot_name in all_slots:
            goal.filled_slots[slot_name] = InformationSlot(name=slot_name)
        
        self.current_goal = goal
        return goal
    
    def update_state(self, intent: str, entities: Dict[str, List[str]], 
                    confidence: float, user_message: str) -> Dict[str, Any]:
        """
        Update dialogue state with new information
        
        Args:
            intent: Detected intent
            entities: Extracted entities
            confidence: Intent confidence score
            user_message: Original user message
            
        Returns:
            State update summary
        """
        timestamp = datetime.now()
        
        # Initialize goal if needed or if intent changed significantly
        if (self.current_goal is None or 
            (intent != self.current_goal.intent and confidence > 0.7)):
            self.initialize_goal(intent)
        
        # Update slots with extracted entities
        slots_updated = self._update_slots(entities, confidence)
        
        # Calculate completion
        self._calculate_completion()
        
        # Record dialogue turn
        dialogue_turn = {
            'timestamp': timestamp,
            'user_message': user_message,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'slots_updated': slots_updated,
            'goal_completion': self.current_goal.completion_percentage if self.current_goal else 0.0
        }
        
        self.dialogue_history.append(dialogue_turn)
        
        return {
            'goal_updated': True,
            'slots_updated': slots_updated,
            'completion_percentage': self.current_goal.completion_percentage if self.current_goal else 0.0,
            'is_complete': self.current_goal.is_complete if self.current_goal else False,
            'next_required_slot': self._get_next_required_slot(),
            'missing_slots': self._get_missing_slots()
        }
    
    def _update_slots(self, entities: Dict[str, List[str]], confidence: float) -> List[str]:
        """
        Update information slots with extracted entities
        """
        if not self.current_goal:
            return []
        
        updated_slots = []
        
        for entity_type, entity_values in entities.items():
            if entity_type in self.current_goal.filled_slots and entity_values:
                slot = self.current_goal.filled_slots[entity_type]
                
                # Take the first/best entity value
                new_value = entity_values[0]
                
                # Update slot if it's empty or new confidence is higher
                if (slot.status == SlotStatus.EMPTY or 
                    confidence > slot.confidence):
                    
                    slot.value = new_value
                    slot.confidence = confidence
                    slot.status = SlotStatus.FILLED
                    slot.last_updated = datetime.now()
                    slot.attempts += 1
                    
                    updated_slots.append(entity_type)
        
        return updated_slots
    
    def _calculate_completion(self):
        """
        Calculate goal completion percentage
        """
        if not self.current_goal:
            return
        
        required_filled = sum(1 for slot_name in self.current_goal.required_slots 
                             if (slot_name in self.current_goal.filled_slots and 
                                 self.current_goal.filled_slots[slot_name].status != SlotStatus.EMPTY))
        
        total_required = len(self.current_goal.required_slots)
        
        if total_required > 0:
            self.current_goal.completion_percentage = (required_filled / total_required) * 100
            self.current_goal.is_complete = (required_filled == total_required)
        else:
            self.current_goal.completion_percentage = 100.0
            self.current_goal.is_complete = True
    
    def _get_next_required_slot(self) -> Optional[str]:
        """
        Get the next required slot that needs to be filled
        """
        if not self.current_goal:
            return None
        
        for slot_name in self.current_goal.required_slots:
            if (slot_name not in self.current_goal.filled_slots or 
                self.current_goal.filled_slots[slot_name].status == SlotStatus.EMPTY):
                return slot_name
        
        return None
    
    def _get_missing_slots(self) -> List[str]:
        """
        Get all missing required slots
        """
        if not self.current_goal:
            return []
        
        missing_slots = []
        for slot_name in self.current_goal.required_slots:
            if (slot_name not in self.current_goal.filled_slots or 
                self.current_goal.filled_slots[slot_name].status == SlotStatus.EMPTY):
                missing_slots.append(slot_name)
        
        return missing_slots
    
    def get_slot_value(self, slot_name: str) -> Optional[str]:
        """
        Get the value of a specific slot
        """
        if (self.current_goal and 
            slot_name in self.current_goal.filled_slots):
            return self.current_goal.filled_slots[slot_name].value
        
        return None
    
    def confirm_slot(self, slot_name: str) -> bool:
        """
        Mark a slot as confirmed by the user
        """
        if (self.current_goal and 
            slot_name in self.current_goal.filled_slots):
            slot = self.current_goal.filled_slots[slot_name]
            slot.confirmed = True
            slot.status = SlotStatus.CONFIRMED
            return True
        
        return False
    
    def clear_slot(self, slot_name: str) -> bool:
        """
        Clear a slot value (user correction)
        """
        if (self.current_goal and 
            slot_name in self.current_goal.filled_slots):
            slot = self.current_goal.filled_slots[slot_name]
            slot.value = None
            slot.status = SlotStatus.EMPTY
            slot.confirmed = False
            slot.confidence = 0.0
            return True
        
        return False
    
    def get_dialogue_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current dialogue state
        """
        if not self.current_goal:
            return {
                'status': 'no_active_goal',
                'completion_percentage': 0.0
            }
        
        filled_slots = {
            slot_name: {
                'value': slot.value,
                'status': slot.status.value,
                'confidence': slot.confidence,
                'confirmed': slot.confirmed
            }
            for slot_name, slot in self.current_goal.filled_slots.items()
            if slot.status != SlotStatus.EMPTY
        }
        
        return {
            'current_intent': self.current_goal.intent,
            'completion_percentage': self.current_goal.completion_percentage,
            'is_complete': self.current_goal.is_complete,
            'required_slots': list(self.current_goal.required_slots),
            'optional_slots': list(self.current_goal.optional_slots),
            'filled_slots': filled_slots,
            'missing_slots': self._get_missing_slots(),
            'next_required_slot': self._get_next_required_slot(),
            'total_turns': len(self.dialogue_history)
        }
    
    def generate_clarification_question(self) -> Optional[str]:
        """
        Generate a clarification question for the next missing slot
        """
        next_slot = self._get_next_required_slot()
        
        if not next_slot:
            return None
        
        # Slot-specific questions
        slot_questions = {
            'order_id': [
                "What's your order ID?",
                "Can you provide your order number?",
                "I'll need your order ID to help you with that."
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
            ],
            'issue_type': [
                "What type of issue are you experiencing?",
                "Can you describe the problem?",
                "What kind of help do you need?"
            ]
        }
        
        questions = slot_questions.get(next_slot, [f"Can you provide your {next_slot.replace('_', ' ')}?"])
        
        # Return the first question (you could randomize this)
        return questions[0]
    
    def reset_dialogue(self):
        """
        Reset dialogue state for new conversation
        """
        self.current_goal = None
        self.dialogue_history = []
        self.context_stack = []
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export current dialogue state for persistence
        """
        return {
            'current_goal': {
                'intent': self.current_goal.intent,
                'required_slots': list(self.current_goal.required_slots),
                'optional_slots': list(self.current_goal.optional_slots),
                'filled_slots': {
                    name: {
                        'value': slot.value,
                        'status': slot.status.value,
                        'confidence': slot.confidence,
                        'confirmed': slot.confirmed,
                        'attempts': slot.attempts,
                        'last_updated': slot.last_updated.isoformat() if slot.last_updated else None
                    }
                    for name, slot in self.current_goal.filled_slots.items()
                },
                'completion_percentage': self.current_goal.completion_percentage,
                'is_complete': self.current_goal.is_complete
            } if self.current_goal else None,
            'dialogue_history': self.dialogue_history,
            'context_stack': self.context_stack
        }
    
    def import_state(self, state_data: Dict[str, Any]):
        """
        Import dialogue state from saved data
        """
        if state_data.get('current_goal'):
            goal_data = state_data['current_goal']
            
            goal = DialogueGoal(
                intent=goal_data['intent'],
                required_slots=set(goal_data['required_slots']),
                optional_slots=set(goal_data['optional_slots']),
                completion_percentage=goal_data['completion_percentage'],
                is_complete=goal_data['is_complete']
            )
            
            # Restore slots
            for slot_name, slot_data in goal_data['filled_slots'].items():
                slot = InformationSlot(
                    name=slot_name,
                    value=slot_data['value'],
                    status=SlotStatus(slot_data['status']),
                    confidence=slot_data['confidence'],
                    confirmed=slot_data['confirmed'],
                    attempts=slot_data['attempts']
                )
                
                if slot_data['last_updated']:
                    slot.last_updated = datetime.fromisoformat(slot_data['last_updated'])
                
                goal.filled_slots[slot_name] = slot
            
            self.current_goal = goal
        
        self.dialogue_history = state_data.get('dialogue_history', [])
        self.context_stack = state_data.get('context_stack', [])