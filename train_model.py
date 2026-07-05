#!/usr/bin/env python3
"""
AI Chatbot (NLP) - Model Training Script
Trains the intent classification model with sample data
"""

import json
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.chatbot import create_chatbot

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_training_data(data):
    """Validate the training data format"""
    required_keys = ['samples']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    
    if not isinstance(data['samples'], list):
        raise ValueError("'samples' must be a list")
    
    for i, sample in enumerate(data['samples']):
        if 'text' not in sample or 'intent' not in sample:
            raise ValueError(f"Sample {i} missing 'text' or 'intent'")
        
        if not isinstance(sample['text'], str) or not sample['text'].strip():
            raise ValueError(f"Sample {i} has invalid text")
        
        if not isinstance(sample['intent'], str) or not sample['intent'].strip():
            raise ValueError(f"Sample {i} has invalid intent")

def print_training_stats(data):
    """Print statistics about the training data"""
    samples = data['samples']
    intents = {}
    entities = {}
    
    for sample in samples:
        intent = sample['intent']
        intents[intent] = intents.get(intent, 0) + 1
        
        if 'entities' in sample and sample['entities']:
            for entity_type, entity_values in sample['entities'].items():
                entities[entity_type] = entities.get(entity_type, 0) + len(entity_values)
    
    print("\n" + "="*60)
    print("📊 TRAINING DATA STATISTICS")
    print("="*60)
    print(f"Total samples: {len(samples)}")
    print(f"Unique intents: {len(intents)}")
    print(f"Entity types: {len(entities)}")
    
    print(f"\n📋 Intent Distribution:")
    for intent, count in sorted(intents.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(samples)) * 100
        print(f"  {intent:20} | {count:3} samples ({percentage:5.1f}%)")
    
    if entities:
        print(f"\n🏷️  Entity Types:")
        for entity_type, count in sorted(entities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity_type:20} | {count:3} instances")
    
    print("="*60)

def main():
    """Main training function"""
    logger = setup_logging()
    
    print("""
    🤖 AI Chatbot (NLP) - Model Training
    ===================================
    
    This script will train the intent classification model
    using the provided training data.
    """)
    
    # Check for training data file
    training_data_path = Path("data/training_samples.json")
    
    if not training_data_path.exists():
        logger.error(f"Training data file not found: {training_data_path}")
        print(f"\n❌ Error: Training data file not found at {training_data_path}")
        print("\nPlease ensure you have a training data file in JSON format with the following structure:")
        print("""
        {
            "samples": [
                {
                    "text": "Hello, I need help",
                    "intent": "greeting",
                    "entities": {}
                },
                ...
            ]
        }
        """)
        return 1
    
    try:
        # Load training data
        logger.info(f"Loading training data from: {training_data_path}")
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # Validate data
        validate_training_data(training_data)
        logger.info("Training data validation passed")
        
        # Print statistics
        print_training_stats(training_data)
        
        # Confirm before training
        response = input(f"\n🚀 Ready to train the model? (y/N): ").strip().lower()
        if response != 'y':
            print("Training cancelled.")
            return 0
        
        # Create chatbot instance
        logger.info("Initializing AI Chatbot...")
        chatbot = create_chatbot()
        
        # Train the model
        print(f"\n🔥 Starting model training...")
        start_time = datetime.now()
        
        result = chatbot.train_from_data(str(training_data_path))
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Check results
        if result['status'] == 'success':
            print(f"\n✅ Training completed successfully!")
            print(f"⏱️  Training time: {training_duration:.2f} seconds")
            print(f"📁 Model saved to: {result['model_path']}")
            print(f"📁 Preprocessor saved to: {result['preprocessor_path']}")
            print(f"📊 Training samples: {result['training_samples']}")
            print(f"🎯 Unique intents: {result['unique_intents']}")
            
            if 'training_stats' in result:
                stats = result['training_stats']
                if 'accuracy' in stats:
                    print(f"🎯 Model accuracy: {stats['accuracy']:.3f}")
                if 'cross_val_mean' in stats:
                    print(f"📊 Cross-validation score: {stats['cross_val_mean']:.3f} ± {stats['cross_val_std']:.3f}")
            
            print(f"\n🎉 Your AI chatbot is now ready to use!")
            print(f"💡 Start the web interface with: python app.py")
            
            # Test the model
            test_model = input(f"\n🧪 Test the trained model? (y/N): ").strip().lower()
            if test_model == 'y':
                test_chatbot_model(chatbot)
            
        else:
            logger.error(f"Training failed: {result.get('error', 'Unknown error')}")
            print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except FileNotFoundError:
        logger.error(f"Training data file not found: {training_data_path}")
        print(f"\n❌ Error: Could not find training data file")
        return 1
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in training data: {e}")
        print(f"\n❌ Error: Invalid JSON format in training data file")
        print(f"Details: {e}")
        return 1
        
    except ValueError as e:
        logger.error(f"Training data validation error: {e}")
        print(f"\n❌ Error: Invalid training data format")
        print(f"Details: {e}")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

def test_chatbot_model(chatbot):
    """Interactive testing of the trained model"""
    print(f"\n🧪 MODEL TESTING MODE")
    print("="*40)
    print("Enter messages to test the AI chatbot")
    print("Type 'quit' to exit testing mode")
    print("="*40)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
            
            # Get chatbot response
            response = chatbot.chat(user_input)
            
            print(f"\n🤖 Bot: {response['response']}")
            print(f"📊 Analysis:")
            print(f"   Intent: {response.get('intent', 'unknown')}")
            print(f"   Confidence: {response.get('confidence', 0):.3f}")
            
            if response.get('entities'):
                print(f"   Entities: {response['entities']}")
            
            if response.get('total_processing_time'):
                print(f"   Processing time: {response['total_processing_time']*1000:.1f}ms")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
    
    print(f"\n👋 Testing completed!")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)