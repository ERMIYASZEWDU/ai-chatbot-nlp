"""
NLP Utilities for AI Chatbot
Additional helper functions and testing utilities
"""

import json
import re
from typing import Dict, List, Tuple
from collections import Counter

def validate_training_data(file_path: str) -> Dict[str, any]:
    """Validate training data integrity"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {
            'valid': True,
            'total_samples': 0,
            'intents': set(),
            'entity_types': set(),
            'issues': []
        }
        
        if 'samples' not in data:
            results['issues'].append("No 'samples' key found")
            results['valid'] = False
            return results
        
        for i, sample in enumerate(data['samples']):
            results['total_samples'] += 1
            
            # Check required fields
            if 'text' not in sample:
                results['issues'].append(f"Sample {i}: Missing 'text' field")
                results['valid'] = False
            
            if 'intent' not in sample:
                results['issues'].append(f"Sample {i}: Missing 'intent' field")
                results['valid'] = False
            else:
                results['intents'].add(sample['intent'])
            
            # Check entities
            if 'entities' in sample:
                for entity_type in sample['entities'].keys():
                    results['entity_types'].add(entity_type)
        
        results['intents'] = list(results['intents'])
        results['entity_types'] = list(results['entity_types'])
        
        return results
        
    except FileNotFoundError:
        return {'valid': False, 'issues': ['Training data file not found']}
    except json.JSONDecodeError:
        return {'valid': False, 'issues': ['Invalid JSON format']}
    except Exception as e:
        return {'valid': False, 'issues': [f'Error: {str(e)}']}

def test_entity_extraction():
    """Test entity extraction with various patterns"""
    test_cases = [
        {
            'text': 'Track my order #ORD-123456',
            'expected': {'order_id': 'ORD-123456'}
        },
        {
            'text': 'Reset password for john.doe@email.com',
            'expected': {'email': 'john.doe@email.com'}
        },
        {
            'text': 'My tracking number is 1Z999AA1234567890',
            'expected': {'tracking_number': '1Z999AA1234567890'}
        },
        {
            'text': 'Call me at 555-123-4567',
            'expected': {'phone': '555-123-4567'}
        },
        {
            'text': 'Return my iPhone 13 Pro',
            'expected': {'product_name': 'Iphone 13 Pro'}
        }
    ]
    
    # Import from main module (would need to be adjusted based on structure)
    from chatbot import extract_entities
    
    results = []
    for test in test_cases:
        extracted = extract_entities(test['text'])
        success = all(
            entity_type in extracted and 
            extracted[entity_type].lower() in test['expected'][entity_type].lower()
            for entity_type in test['expected']
        )
        
        results.append({
            'text': test['text'],
            'expected': test['expected'],
            'extracted': extracted,
            'success': success
        })
    
    return results

def analyze_intent_keywords(training_data: Dict) -> Dict[str, List[str]]:
    """Analyze and extract keywords from training data for each intent"""
    intent_words = {}
    
    if 'samples' not in training_data:
        return intent_words
    
    for sample in training_data['samples']:
        intent = sample.get('intent')
        text = sample.get('text', '').lower()
        
        if intent not in intent_words:
            intent_words[intent] = []
        
        # Extract meaningful words (filter out common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        words = re.findall(r'\b\w+\b', text)
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        intent_words[intent].extend(meaningful_words)
    
    # Count frequency and return top keywords for each intent
    intent_keywords = {}
    for intent, words in intent_words.items():
        word_counts = Counter(words)
        top_words = [word for word, count in word_counts.most_common(10)]
        intent_keywords[intent] = top_words
    
    return intent_keywords

def generate_test_report(chatbot_instance=None) -> str:
    """Generate a comprehensive test report"""
    report = []
    report.append("=== AI CHATBOT NLP TEST REPORT ===\n")
    
    # Training data validation
    report.append("1. TRAINING DATA VALIDATION")
    report.append("-" * 30)
    
    validation = validate_training_data("data/training_samples.json")
    if validation['valid']:
        report.append(f"✅ Training data is valid")
        report.append(f"📊 Total samples: {validation['total_samples']}")
        report.append(f"🎯 Intents: {len(validation['intents'])}")
        report.append(f"🏷️ Entity types: {len(validation['entity_types'])}")
        report.append(f"📝 Intents list: {', '.join(validation['intents'])}")
        report.append(f"🔖 Entity types: {', '.join(validation['entity_types'])}")
    else:
        report.append(f"❌ Training data validation failed")
        for issue in validation['issues']:
            report.append(f"   - {issue}")
    
    report.append("")
    
    # Entity extraction tests
    report.append("2. ENTITY EXTRACTION TESTS")
    report.append("-" * 30)
    
    try:
        entity_tests = test_entity_extraction()
        passed = sum(1 for test in entity_tests if test['success'])
        total = len(entity_tests)
        
        report.append(f"✅ Entity extraction tests: {passed}/{total} passed")
        
        for test in entity_tests:
            status = "✅" if test['success'] else "❌"
            report.append(f"   {status} '{test['text']}'")
            if not test['success']:
                report.append(f"      Expected: {test['expected']}")
                report.append(f"      Got: {test['extracted']}")
    
    except Exception as e:
        report.append(f"❌ Entity extraction test failed: {str(e)}")
    
    report.append("")
    
    # Intent analysis
    report.append("3. INTENT KEYWORD ANALYSIS")
    report.append("-" * 30)
    
    try:
        with open("data/training_samples.json", 'r') as f:
            training_data = json.load(f)
        
        keywords = analyze_intent_keywords(training_data)
        for intent, words in keywords.items():
            report.append(f"🎯 {intent}: {', '.join(words[:5])}")  # Top 5 keywords
    
    except Exception as e:
        report.append(f"❌ Intent analysis failed: {str(e)}")
    
    report.append("")
    report.append("=== END OF REPORT ===")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Generate and print test report
    print(generate_test_report())