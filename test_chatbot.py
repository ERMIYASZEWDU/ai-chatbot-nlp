"""
Test Runner for AI Chatbot
Run comprehensive tests to validate NLP functionality
"""

import sys
import json
from chatbot import (
    classify_intent, 
    extract_entities, 
    analyze_sentiment, 
    preprocess_text,
    generate_response
)

def test_intent_classification():
    """Test intent classification with various inputs"""
    print("🎯 Testing Intent Classification...")
    
    test_cases = [
        ("Track my order #ORD-123", "track_order"),
        ("I forgot my password", "reset_password"), 
        ("When will my package arrive?", "delivery_inquiry"),
        ("I want a refund", "request_refund"),
        ("I need to speak with someone", "contact_support"),
        ("Hello there", "greeting"),
        ("Thank you, goodbye", "goodbye"),
        ("What's the price of MacBook?", "product_inquiry")
    ]
    
    correct = 0
    total = len(test_cases)
    
    for text, expected_intent in test_cases:
        predicted_intent, confidence = classify_intent(text)
        is_correct = predicted_intent == expected_intent
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"   {status} '{text}' -> {predicted_intent} ({confidence:.1f}%)")
        if not is_correct:
            print(f"      Expected: {expected_intent}")
    
    accuracy = (correct / total) * 100
    print(f"\n📊 Intent Classification Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy

def test_entity_extraction():
    """Test entity extraction functionality"""
    print("\n🏷️ Testing Entity Extraction...")
    
    test_cases = [
        ("Track order #ORD-123456", ["order_id"]),
        ("Reset password for john@email.com", ["email"]),
        ("My tracking is 1Z999AA1234567890", ["tracking_number"]),
        ("Call me at 555-123-4567", ["phone"]),
        ("Return my iPhone 13 Pro", ["product_name"]),
        ("Deliver to 123 Main Street tomorrow", ["address", "date"])
    ]
    
    total_expected = 0
    total_found = 0
    
    for text, expected_types in test_cases:
        entities = extract_entities(text)
        found_types = list(entities.keys())
        
        total_expected += len(expected_types)
        matched = len(set(expected_types) & set(found_types))
        total_found += matched
        
        print(f"   📝 '{text}'")
        print(f"      Expected: {expected_types}")
        print(f"      Found: {found_types}")
        print(f"      Entities: {entities}")
        print()
    
    extraction_rate = (total_found / total_expected) * 100 if total_expected > 0 else 0
    print(f"📊 Entity Extraction Rate: {extraction_rate:.1f}% ({total_found}/{total_expected})")
    return extraction_rate

def test_sentiment_analysis():
    """Test sentiment analysis"""
    print("\n😊 Testing Sentiment Analysis...")
    
    test_cases = [
        ("This is excellent service!", "positive"),
        ("I'm very frustrated with this", "negative"),
        ("Track my order please", "neutral"),
        ("Thank you so much for helping", "positive"),
        ("This is terrible and awful", "negative"),
        ("The delivery was okay", "neutral")
    ]
    
    correct = 0
    total = len(test_cases)
    
    for text, expected_sentiment in test_cases:
        predicted_sentiment, emoji = analyze_sentiment(text)
        is_correct = predicted_sentiment == expected_sentiment
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"   {status} '{text}' -> {predicted_sentiment} {emoji}")
        if not is_correct:
            print(f"      Expected: {expected_sentiment}")
    
    accuracy = (correct / total) * 100
    print(f"\n📊 Sentiment Analysis Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy

def test_preprocessing():
    """Test text preprocessing"""
    print("\n🔧 Testing Text Preprocessing...")
    
    test_cases = [
        ("HELLO WORLD!!!", "hello world"),
        ("  Multiple   spaces  ", "multiple spaces"),
        ("Email: user@test.com", "email user@test.com"),
        ("Order #12345", "order #12345")
    ]
    
    for original, expected in test_cases:
        processed = preprocess_text(original)
        is_correct = processed == expected
        status = "✅" if is_correct else "❌"
        
        print(f"   {status} '{original}' -> '{processed}'")
        if not is_correct:
            print(f"      Expected: '{expected}'")

def test_end_to_end():
    """Test complete conversation flow"""
    print("\n🔄 Testing End-to-End Conversation Flow...")
    
    test_messages = [
        "Hello, I need help",
        "Track my order #ORD-555777", 
        "I want to return my defective laptop",
        "Thank you for your help"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n   Step {i}: User says '{message}'")
        
        # Process message
        intent, confidence = classify_intent(message)
        entities = extract_entities(message)
        sentiment, emoji = analyze_sentiment(message)
        response = generate_response(message, intent, entities, emoji)
        
        print(f"      Intent: {intent} ({confidence:.1f}%)")
        print(f"      Entities: {entities}")
        print(f"      Sentiment: {sentiment} {emoji}")
        print(f"      Response: {response[:100]}{'...' if len(response) > 100 else ''}")

def main():
    """Run all tests"""
    print("🤖 AI CHATBOT NLP TESTING SUITE")
    print("=" * 50)
    
    try:
        # Run all tests
        intent_accuracy = test_intent_classification()
        entity_rate = test_entity_extraction() 
        sentiment_accuracy = test_sentiment_analysis()
        test_preprocessing()
        test_end_to_end()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        print(f"🎯 Intent Classification: {intent_accuracy:.1f}%")
        print(f"🏷️ Entity Extraction: {entity_rate:.1f}%")
        print(f"😊 Sentiment Analysis: {sentiment_accuracy:.1f}%")
        
        # Overall score
        overall_score = (intent_accuracy + entity_rate + sentiment_accuracy) / 3
        print(f"\n🏆 Overall NLP Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("✅ Chatbot is performing well!")
        elif overall_score >= 60:
            print("⚠️ Chatbot needs some improvements")
        else:
            print("❌ Chatbot needs significant improvements")
            
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())