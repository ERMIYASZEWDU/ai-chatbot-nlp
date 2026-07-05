"""
Pytest configuration and fixtures for AI Chatbot NLP tests.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_training_data():
    """Sample training data for testing."""
    return {
        "metadata": {"total_samples": 2},
        "samples": [
            {
                "text": "Track my order #ORD123456",
                "intent": "track_order",
                "entities": {"order_id": ["ORD123456"]}
            },
            {
                "text": "Hello, I need help",
                "intent": "greeting",
                "entities": {}
            }
        ]
    }


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        "Hello, how are you?",
        "Track my order #ORD123456",
        "I forgot my password for user@example.com",
        "When will my delivery arrive?",
        "I want a refund for my order",
        "Connect me to a human agent",
        "Goodbye, thanks!"
    ]
