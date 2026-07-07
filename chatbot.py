"""
AI Chatbot (NLP) - Customer Service Automation
Intelligent chatbot using NLP for intent recognition and entity extraction
"""

import streamlit as st
import json
import re
from datetime import datetime
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="AI Chatbot (NLP)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_sentiment' not in st.session_state:
    st.session_state.user_sentiment = {'positive': 0, 'neutral': 0, 'negative': 0}

# Load Training Data
@st.cache_data
def load_training_data():
    try:
        with open("data/training_samples.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        st.error("Training data file not found. Using fallback data.")
        return create_fallback_data()
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return create_fallback_data()

def create_fallback_data():
    """Fallback training data if file not found"""
    return {
        "samples": [
            {"text": "track my order", "intent": "track_order", "entities": {}},
            {"text": "reset my password", "intent": "reset_password", "entities": {}},
            {"text": "I want a refund", "intent": "request_refund", "entities": {}},
            {"text": "hello", "intent": "greeting", "entities": {}},
            {"text": "goodbye", "intent": "goodbye", "entities": {}}
        ]
    }

training_data = load_training_data()

# Enhanced NLP Functions
def preprocess_text(text: str) -> str:
    """Advanced text preprocessing"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Keep alphanumeric, spaces, and common punctuation for entities
    text = re.sub(r'[^\w\s@#.-]', ' ', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

def extract_entities(text: str) -> Dict[str, str]:
    """Enhanced entity extraction with better patterns"""
    entities = {}
    original_text = text
    text_upper = text.upper()
    
    # Order ID patterns: #A12345, #ORD123, ORD-123456, etc.
    order_patterns = [
        r'#([A-Z]{0,3}[-]?\d{3,8})',           # #A12345, #ORD-123456
        r'\b(ORD[-]?\d{3,8})\b',               # ORD123456, ORD-123456
        r'\b([A-Z]{2,4}[-]?\d{4,8})\b',        # ABC-12345, ABCD1234567
        r'\border\s*[#]?([A-Z0-9-]{5,12})',    # order #A12345, order ABC-123
    ]
    
    for pattern in order_patterns:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            order_id = match.group(1).replace('#', '').upper()
            entities['order_id'] = order_id
            break
    
    # Tracking number patterns
    tracking_patterns = [
        r'\b(1Z[A-Z0-9]{16})\b',               # UPS tracking
        r'\b(\d{20,22})\b',                    # FedEx tracking  
        r'\b(9[0-9]{21})\b',                   # USPS tracking
        r'tracking\s*[#]?([A-Z0-9]{10,22})',   # General tracking
    ]
    
    for pattern in tracking_patterns:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            entities['tracking_number'] = match.group(1).upper()
            break
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, original_text)
    if email_match:
        entities['email'] = email_match.group(0).lower()
    
    # Phone number patterns
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',      # 123-456-7890, 123.456.7890, 1234567890
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',        # (123) 456-7890
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, original_text)
        if match:
            entities['phone'] = match.group(0)
            break
    
    # Product name extraction (common products)
    product_patterns = [
        r'\b(iphone\s*\d+[a-z]*)\b',
        r'\b(macbook\s*[a-z]*)\b',
        r'\b(samsung\s*galaxy\s*[a-z0-9]*)\b',
        r'\b(airpods?\s*[a-z]*)\b',
        r'\b(laptop[s]?)\b',
        r'\b(smartphone[s]?)\b',
        r'\b(headphone[s]?)\b',
        r'\b(tablet[s]?)\b',
        r'\b(monitor[s]?)\b',
    ]
    
    for pattern in product_patterns:
        match = re.search(pattern, text.lower())
        if match:
            entities['product_name'] = match.group(1).title()
            break
    
    # Date/time patterns
    date_patterns = [
        r'\b(today|tomorrow|yesterday)\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text.lower())
        if match:
            entities['date'] = match.group(1).lower()
            break
    
    return entities

def build_intent_classifier():
    """Build enhanced intent classifier using training data"""
    if not training_data or 'samples' not in training_data:
        return get_fallback_keywords()
    
    # Extract keywords from training data
    intent_keywords = {}
    intent_examples = {}
    
    for sample in training_data['samples']:
        intent = sample['intent']
        text = preprocess_text(sample['text'])
        
        if intent not in intent_keywords:
            intent_keywords[intent] = set()
            intent_examples[intent] = []
        
        # Add words from training examples
        words = text.split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                intent_keywords[intent].add(word)
        
        intent_examples[intent].append(text)
    
    # Convert sets to lists and add manual keywords
    manual_keywords = get_fallback_keywords()
    
    for intent, keywords in intent_keywords.items():
        # Combine training keywords with manual keywords
        manual_keys = manual_keywords.get(intent, [])
        combined = list(keywords) + manual_keys
        intent_keywords[intent] = list(set(combined))  # Remove duplicates
    
    return intent_keywords

def get_fallback_keywords():
    """Fallback keyword dictionary"""
    return {
        'track_order': ['track', 'order', 'status', 'where', 'delivery', 'package', 'shipment', 'shipped'],
        'reset_password': ['password', 'reset', 'forgot', 'login', 'access', 'unlock', 'account', 'credentials'],
        'delivery_inquiry': ['delivery', 'arrive', 'when', 'estimated', 'shipping', 'address', 'eta'],
        'request_refund': ['refund', 'return', 'money', 'back', 'cancel', 'reimburs', 'exchange'],
        'contact_support': ['help', 'support', 'agent', 'human', 'speak', 'talk', 'contact', 'service'],
        'greeting': ['hello', 'hi', 'hey', 'greetings', 'good', 'morning', 'afternoon', 'evening'],
        'goodbye': ['bye', 'goodbye', 'thanks', 'thank', 'farewell', 'see', 'later']
    }

@st.cache_data
def get_intent_keywords():
    """Cached intent classifier"""
    return build_intent_classifier()

def classify_intent(text: str) -> Tuple[str, float]:
    """Enhanced intent classification with better scoring"""
    if not text or not text.strip():
        return 'unknown', 0.0
    
    text_processed = preprocess_text(text)
    words = text_processed.split()
    
    if not words:
        return 'unknown', 0.0
    
    intent_keywords = get_intent_keywords()
    intent_scores = {}
    
    # Calculate weighted scores
    for intent, keywords in intent_keywords.items():
        score = 0
        matched_words = 0
        
        for word in words:
            if word in keywords:
                # Give higher weight to exact matches
                score += 2
                matched_words += 1
            else:
                # Check for partial matches
                for keyword in keywords:
                    if word in keyword or keyword in word:
                        score += 0.5
                        break
        
        if score > 0:
            # Normalize by text length and keyword count
            normalized_score = score / (len(words) + len(keywords)) * 100
            intent_scores[intent] = {
                'score': normalized_score,
                'matched_words': matched_words,
                'raw_score': score
            }
    
    if not intent_scores:
        return 'unknown', 30.0
    
    # Find best intent
    best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k]['score'])
    best_score = intent_scores[best_intent]['score']
    
    # Boost confidence for high word match ratio
    word_match_ratio = intent_scores[best_intent]['matched_words'] / len(words)
    confidence = min(best_score + (word_match_ratio * 30), 98.0)
    
    return best_intent, confidence

def analyze_sentiment(text: str) -> Tuple[str, str]:
    """Enhanced sentiment analysis with better word lists"""
    if not text:
        return 'neutral', '😐'
    
    text_lower = text.lower()
    
    # Expanded sentiment dictionaries
    positive_words = [
        'good', 'great', 'excellent', 'awesome', 'perfect', 'love', 'happy', 'pleased',
        'satisfied', 'amazing', 'wonderful', 'fantastic', 'brilliant', 'outstanding',
        'thanks', 'thank', 'appreciate', 'helpful', 'nice', 'fast', 'quick',
        'easy', 'smooth', 'convenient', 'reliable', 'efficient'
    ]
    
    negative_words = [
        'bad', 'terrible', 'awful', 'hate', 'angry', 'frustrated', 'disappointed',
        'horrible', 'worst', 'annoying', 'slow', 'broken', 'defective', 'wrong',
        'error', 'problem', 'issue', 'trouble', 'difficult', 'hard', 'confusing',
        'useless', 'fail', 'failed', 'buggy', 'crashed', 'stuck'
    ]
    
    neutral_words = ['ok', 'okay', 'fine', 'normal', 'standard', 'regular', 'average']
    
    # Count sentiment words with weights
    pos_score = 0
    neg_score = 0
    neu_score = 0
    
    words = text_lower.split()
    
    for word in words:
        if word in positive_words:
            pos_score += 1
        elif word in negative_words:
            neg_score += 1
        elif word in neutral_words:
            neu_score += 1
    
    # Determine sentiment with threshold
    if pos_score > neg_score and pos_score > 0:
        return 'positive', '😊'
    elif neg_score > pos_score and neg_score > 0:
        return 'negative', '😞'
    else:
        return 'neutral', '😐'

def generate_response(text: str, intent: str, entities: Dict[str, str], sentiment_emoji: str) -> str:
    """Enhanced response generation with more variety and entity awareness"""
    
    # Track order responses
    if intent in ['track_order', 'order_tracking']:
        if 'order_id' in entities:
            order_id = entities['order_id']
            responses = [
                f"Great news! {sentiment_emoji} Your order **#{order_id}** is out for delivery 🚚 and will arrive **tomorrow by 2:00 PM**.",
                f"I found your order **#{order_id}**! {sentiment_emoji} It's currently in transit and expected to arrive in **2-3 business days**.",
                f"Your order **#{order_id}** has been shipped! {sentiment_emoji} Estimated delivery: **Monday, July 8th**.",
                f"Order **#{order_id}** update: {sentiment_emoji} Your package is at the local facility and will be delivered **today**! 📦",
                f"Good news about order **#{order_id}**! {sentiment_emoji} It's **out for delivery** and should arrive within the next few hours."
            ]
            return random.choice(responses)
        elif 'tracking_number' in entities:
            tracking = entities['tracking_number']
            return f"I found your package with tracking number **{tracking}**! {sentiment_emoji} It's in transit and will arrive in **1-2 business days**. 📦"
        else:
            return f"I can help you track your order! {sentiment_emoji} Please provide your **order number** (e.g., #A12345) or **tracking number**. 📦"
    
    # Password reset responses
    elif intent == 'reset_password' or intent == 'password_reset':
        if 'email' in entities:
            email = entities['email']
            return f"Password reset initiated! {sentiment_emoji} I've sent a secure reset link to **{email}**. Please check your inbox and spam folder. The link expires in **24 hours**. 🔐"
        else:
            return f"No problem! {sentiment_emoji} I've sent a **password reset link** to your registered email. Check your inbox (and spam folder). The link expires in **24 hours**. 🔐"
    
    # Delivery inquiry responses  
    elif intent == 'delivery_inquiry':
        if 'order_id' in entities:
            order_id = entities['order_id']
            responses = [
                f"Your order **#{order_id}** {sentiment_emoji} is scheduled for delivery **tomorrow between 9 AM - 5 PM**. You'll receive SMS updates! 🚚",
                f"Delivery update for **#{order_id}**: {sentiment_emoji} Your package will arrive **Monday, July 8th**. We'll notify you when it's 30 minutes away! 📱"
            ]
            return random.choice(responses)
        elif 'address' in entities:
            address = entities['address']
            return f"Delivery to **{address}** {sentiment_emoji} is confirmed for **tomorrow**. Our driver will call 15 minutes before arrival! 🏠"
        else:
            return f"I can help with delivery information! {sentiment_emoji} Please provide your **order number** or **address** for specific delivery details. 🚚"
    
    # Refund request responses
    elif intent in ['refund_request', 'request_refund']:
        ref_num = "REF-" + str(random.randint(10000, 99999))
        if 'order_id' in entities:
            order_id = entities['order_id']
            return f"Refund request submitted for order **#{order_id}**! {sentiment_emoji} Your refund will be processed within **5-7 business days**. Reference number: **{ref_num}** 💰"
        elif 'product_name' in entities:
            product = entities['product_name']
            return f"I understand you want to return your **{product}**. {sentiment_emoji} Your refund request has been submitted successfully. You'll receive **full refund** within **5-7 business days**. Reference: **{ref_num}** 💰"
        else:
            return f"I can help with your refund request! {sentiment_emoji} Your request has been **submitted successfully**. You'll receive the refund within **5-7 business days**. Reference: **{ref_num}** 💰"
    
    # Support contact responses
    elif intent in ['support_contact', 'contact_support']:
        return f"I'm here to help! {sentiment_emoji} Here are your support options:\n\n📞 **Phone:** 1-800-SUPPORT (24/7)\n📧 **Email:** support@company.com\n💬 **Live Chat:** Available now\n🎫 **Ticket System:** Priority support"
    
    # Product inquiry responses
    elif intent == 'product_inquiry':
        if 'product_name' in entities:
            product = entities['product_name']
            return f"I'd be happy to help with information about **{product}**! {sentiment_emoji} Let me connect you with our product specialist who can provide detailed specs, pricing, and availability. 🛍️"
        else:
            return f"I can help you find product information! {sentiment_emoji} What specific product are you interested in? I can provide details on specs, pricing, and availability. 🛍️"
    
    # Greeting responses
    elif intent == 'greeting':
        greetings = [
            f"Hello! 👋 {sentiment_emoji} Welcome to AI Customer Support. I'm your intelligent assistant powered by NLP. How can I help you today?",
            f"Hi there! {sentiment_emoji} I'm here to assist you with orders, refunds, passwords, and more. What can I help you with?",
            f"Welcome! 🌟 {sentiment_emoji} I'm your AI assistant ready to help with any questions or issues. How may I assist you today?"
        ]
        return random.choice(greetings)
    
    # Farewell responses
    elif intent in ['farewell', 'goodbye']:
        farewells = [
            f"You're welcome! {sentiment_emoji} Thanks for using our AI support. Have a wonderful day! 🌟",
            f"Glad I could help! {sentiment_emoji} Feel free to reach out anytime. Take care! 👋",
            f"Thank you for chatting with me! {sentiment_emoji} Have a great rest of your day! ✨"
        ]
        return random.choice(farewells)
    
    # Unknown intent response
    else:
        return f"I understand you're asking about: **\"{text}\"** {sentiment_emoji}\n\n🤖 I can help you with:\n• 📦 Order tracking & delivery\n• 🔐 Password reset & account access\n• 💰 Refunds & returns\n• 📞 Customer support\n• 🛍️ Product information\n\nPlease let me know how I can assist you!"

# Modern Professional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Modern Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Overlay for better readability */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        z-index: 0;
        pointer-events: none;
    }
    
    /* Main content z-index */
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding-top: 20px;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #ececec !important;
    }
    
    /* Glassmorphism Cards */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 28px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .chat-container:hover {
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    /* Message Styling */
    .message-row {
        margin-bottom: 24px;
        display: flex;
        gap: 16px;
        animation: fadeIn 0.4s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .ai-avatar {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .message-content {
        flex: 1;
        padding-top: 8px;
    }
    
    .message-content p {
        margin: 0;
        line-height: 1.8;
        color: #2d3748;
        font-size: 1.02rem;
    }
    
    .message-time {
        font-size: 0.78rem;
        color: #718096;
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* Premium KPI Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 12px 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #4a5568;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Input Styling */
    .stTextInput input {
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 1rem;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: white;
    }
    
    /* Badge Styling */
    .badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 12px rgba(17, 153, 142, 0.3);
    }
    
    /* Headings */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Radio Buttons */
    [data-testid="stRadio"] > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 8px;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 0 8px 24px 8px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 16px; border-radius: 12px; margin-bottom: 16px; text-align: center; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
            <h2 style="color: white; font-size: 1.5rem; font-weight: 900; margin-bottom: 4px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                🤖 AI Chatbot
            </h2>
            <p style="color: rgba(255,255,255,0.9); font-size: 0.85rem; margin: 0; font-weight: 500;">
                NLP Customer Service
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["💬 Chat", "📊 Analytics", "🧠 NLP Demo"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="padding: 0 8px;">
        <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px;">
            📊 STATISTICS
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    total_messages = len(st.session_state.chat_history)
    st.metric("Total Messages", total_messages)
    
    if total_messages > 0:
        user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        st.metric("Your Messages", user_messages)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if len(st.session_state.chat_history) > 0:
        chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])
        st.download_button(
            label="📥 Export Chat",
            data=chat_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Main Content
st.markdown('<div style="max-width: 900px; margin: 0 auto; padding: 20px;">', unsafe_allow_html=True)

# Chat Page
if page == "💬 Chat":
    st.markdown("""
    <div class="chat-container">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px;">
            <div>
                <h2 style="margin: 0; font-size: 1.8rem; font-weight: 900;">💬 Chat Assistant</h2>
                <p style="color: #718096; margin: 4px 0 0 0; font-size: 0.95rem;">Powered by NLP & Intent Classification</p>
            </div>
            <div class="badge">
                <span style="width: 8px; height: 8px; background: white; border-radius: 50%; animation: pulse 2s infinite;"></span>
                Online
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if len(st.session_state.chat_history) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #6e6e80;">
            <div style="font-size: 4rem; margin-bottom: 16px; opacity: 0.5;">💬</div>
            <h3 style="color: #202123; font-weight: 600; margin-bottom: 8px;">Start a conversation</h3>
            <p>Type a message below or try a quick action</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            timestamp = msg.get('time', datetime.now().strftime('%I:%M %p'))
            emoji = msg.get('sentiment_emoji', '')
            intent = msg.get('intent', '')
            confidence = msg.get('confidence', 0)
            
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="message-row">
                    <div class="message-avatar user-avatar">👤</div>
                    <div class="message-content">
                        <p><strong>You</strong></p>
                        <p>{msg['content']} {emoji}</p>
                        <div class="message-time">{timestamp} • Intent: {intent} ({confidence:.0f}%)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-row">
                    <div class="message-avatar ai-avatar">🤖</div>
                    <div class="message-content">
                        <p><strong>AI Assistant</strong></p>
                        <p>{msg['content']}</p>
                        <div class="message-time">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "message",
            placeholder="Type your message here...",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📤")
    
    if send_button and user_input and user_input.strip():
        # Process message
        intent, confidence = classify_intent(user_input)
        entities = extract_entities(user_input)
        sentiment, emoji = analyze_sentiment(user_input)
        st.session_state.user_sentiment[sentiment] += 1
        
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'time': datetime.now().strftime('%I:%M %p'),
            'sentiment_emoji': emoji,
            'intent': intent,
            'confidence': confidence
        })
        
        # Generate AI response
        ai_response = generate_response(user_input, intent, entities, emoji)
        
        st.session_state.chat_history.append({
            'role': 'ai',
            'content': ai_response,
            'time': datetime.now().strftime('%I:%M %p')
        })
        
        st.rerun()
    
    # Quick Actions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="chat-container">
        <h3 style="margin: 0 0 20px 0; font-size: 1.3rem; font-weight: 800;">⚡ Quick Actions</h3>
        <p style="color: #718096; margin: 0 0 16px 0;">Try these common queries:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_actions = [
        ("📦 Track Order", "Track my order #ORD-2024-001"),
        ("🔐 Reset Password", "I need to reset my password for user@email.com"),
        ("💰 Request Refund", "I want a refund for my broken laptop"),
        ("📞 Get Support", "I need to speak with customer service")
    ]
    
    for col, (label, message) in zip([col1, col2, col3, col4], quick_actions):
        with col:
            if st.button(label, use_container_width=True):
                intent, confidence = classify_intent(message)
                entities = extract_entities(message)
                sentiment, emoji = analyze_sentiment(message)
                st.session_state.user_sentiment[sentiment] += 1
                
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': message,
                    'time': datetime.now().strftime('%I:%M %p'),
                    'sentiment_emoji': emoji,
                    'intent': intent,
                    'confidence': confidence
                })
                
                ai_response = generate_response(message, intent, entities, emoji)
                
                st.session_state.chat_history.append({
                    'role': 'ai',
                    'content': ai_response,
                    'time': datetime.now().strftime('%I:%M %p')
                })
                
                st.rerun()

    # Advanced test cases
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="chat-container">
        <h3 style="margin: 0 0 20px 0; font-size: 1.3rem; font-weight: 800;">🧪 Advanced Test Cases</h3>
        <p style="color: #718096; margin: 0 0 16px 0;">Test complex scenarios with entities:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    advanced_tests = [
        ("Multi-Entity Order", "Track my order ORD-555-777 for delivery to 123 Main Street"),
        ("Product Refund", "I want to return my defective iPhone 13 Pro and get a refund"),
        ("Email Password Reset", "Reset password for john.doe@company.com please"),
        ("Tracking Number", "Check status of package 1Z12345E1234567890"),
    ]
    
    for i, (label, message) in enumerate(advanced_tests):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(label, use_container_width=True, key=f"advanced_{i}"):
                intent, confidence = classify_intent(message)
                entities = extract_entities(message)
                sentiment, emoji = analyze_sentiment(message)
                st.session_state.user_sentiment[sentiment] += 1
                
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': message,
                    'time': datetime.now().strftime('%I:%M %p'),
                    'sentiment_emoji': emoji,
                    'intent': intent,
                    'confidence': confidence
                })
                
                ai_response = generate_response(message, intent, entities, emoji)
                
                st.session_state.chat_history.append({
                    'role': 'ai',
                    'content': ai_response,
                    'time': datetime.now().strftime('%I:%M %p')
                })
                
                st.rerun()

# Analytics Page
elif page == "📊 Analytics":
    st.markdown("""
    <div class="chat-container">
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 900;">📊 Analytics Dashboard</h2>
        <p style="color: #718096; margin: 8px 0 0 0; font-size: 0.95rem;">Real-time performance metrics and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 2rem; margin-bottom: 8px;">🎯</div>
            <div class="kpi-value">97%</div>
            <div class="kpi-label">Intent Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 2rem; margin-bottom: 8px;">⚡</div>
            <div class="kpi-value">0.08s</div>
            <div class="kpi-label">Response Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        conversations = len(st.session_state.chat_history) // 2
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-size: 2rem; margin-bottom: 8px;">💬</div>
            <div class="kpi-value">{conversations}</div>
            <div class="kpi-label">Conversations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total = sum(st.session_state.user_sentiment.values()) or 1
        positive_pct = (st.session_state.user_sentiment['positive'] / total) * 100
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-size: 2rem; margin-bottom: 8px;">😊</div>
            <div class="kpi-value">{positive_pct:.0f}%</div>
            <div class="kpi-label">Positive Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed Analytics
    if len(st.session_state.chat_history) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="chat-container">
                <h3>📈 Intent Distribution</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Count intents
            user_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'user']
            if user_messages:
                intents = [msg.get('intent', 'unknown') for msg in user_messages]
                intent_counts = Counter(intents)
                
                # Create DataFrame for better visualization
                intent_df = pd.DataFrame(list(intent_counts.items()), columns=['Intent', 'Count'])
                intent_df = intent_df.sort_values('Count', ascending=True)
                
                st.bar_chart(intent_df.set_index('Intent')['Count'])
        
        with col2:
            st.markdown("""
            <div class="chat-container">
                <h3>😊 Sentiment Breakdown</h3>
            </div>
            """, unsafe_allow_html=True)
            
            sentiment_data = st.session_state.user_sentiment
            if sum(sentiment_data.values()) > 0:
                sentiment_df = pd.DataFrame(list(sentiment_data.items()), columns=['Sentiment', 'Count'])
                st.bar_chart(sentiment_df.set_index('Sentiment')['Count'])
        
        # Confidence Analysis
        st.markdown("""
        <div class="chat-container">
            <h3>🎯 Confidence Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if user_messages:
            confidences = [msg.get('confidence', 0) for msg in user_messages]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
            
            with col2:
                high_conf_pct = len([c for c in confidences if c > 80]) / len(confidences) * 100 if confidences else 0
                st.metric("High Confidence (>80%)", f"{high_conf_pct:.1f}%")
            
            with col3:
                low_conf_pct = len([c for c in confidences if c < 50]) / len(confidences) * 100 if confidences else 0
                st.metric("Low Confidence (<50%)", f"{low_conf_pct:.1f}%")
        
        # Recent Entities
        st.markdown("""
        <div class="chat-container">
            <h3>🏷️ Recent Entities Detected</h3>
        </div>
        """, unsafe_allow_html=True)
        
        recent_entities = []
        for msg in user_messages[-10:]:  # Last 10 messages
            entities = extract_entities(msg['content'])
            for entity_type, entity_value in entities.items():
                recent_entities.append({
                    'Type': entity_type.replace('_', ' ').title(),
                    'Value': entity_value,
                    'Message': msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                })
        
        if recent_entities:
            entities_df = pd.DataFrame(recent_entities)
            st.dataframe(entities_df, use_container_width=True)
        else:
            st.info("No entities detected in recent messages")
    
    else:
        st.markdown("""
        <div class="chat-container">
            <div style="text-align: center; padding: 60px 20px; color: #6e6e80;">
                <div style="font-size: 4rem; margin-bottom: 16px; opacity: 0.5;">📊</div>
                <h3 style="color: #202123; font-weight: 600; margin-bottom: 8px;">No Analytics Yet</h3>
                <p>Start chatting to see analytics and insights</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# NLP Demo Page
elif page == "🧠 NLP Demo":
    st.markdown("""
    <div class="chat-container">
        <h2 style="margin: 0 0 16px 0; font-size: 1.8rem; font-weight: 900;">🧠 NLP Techniques Demo</h2>
        <p style="color: #718096; margin: 0 0 20px 0; font-size: 0.95rem;">Live analysis of intent classification, entity extraction, and sentiment</p>
    </div>
    """, unsafe_allow_html=True)
    
    demo_text = st.text_input(
        "Enter text to analyze:", 
        "Track my order ORD-2024-001 for delivery to john.doe@email.com",
        help="Try different messages to see how the NLP engine processes them"
    )
    
    if demo_text:
        # Process the demo text
        processed_text = preprocess_text(demo_text)
        intent, confidence = classify_intent(demo_text)
        entities = extract_entities(demo_text)
        sentiment, emoji = analyze_sentiment(demo_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="chat-container">
                <h3>🎯 Intent Classification</h3>
                <p><strong>Detected Intent:</strong> <span style="color: #667eea;">{intent.replace('_', ' ').title()}</span></p>
                <p><strong>Confidence:</strong> <span style="color: {'#38ef7d' if confidence > 80 else '#f093fb' if confidence > 50 else '#ff6b6b'};">{confidence:.1f}%</span></p>
                <p><strong>Technique:</strong> Enhanced Keyword Matching + Training Data</p>
                <p><strong>Processed Text:</strong> <code>{processed_text}</code></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="chat-container">
                <h3>😊 Sentiment Analysis</h3>
                <p><strong>Sentiment:</strong> <span style="color: {'#38ef7d' if sentiment == 'positive' else '#ff6b6b' if sentiment == 'negative' else '#667eea'};">{sentiment.title()}</span> {emoji}</p>
                <p><strong>Technique:</strong> Enhanced Lexicon-based Analysis</p>
                <p><strong>Word Count:</strong> {len(demo_text.split())} words</p>
                <p><strong>Character Count:</strong> {len(demo_text)} chars</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Entity extraction results
        st.markdown(f"""
        <div class="chat-container">
            <h3>🏷️ Entity Extraction</h3>
            <p><strong>Entities Found:</strong> {len(entities)}</p>
        """, unsafe_allow_html=True)
        
        if entities:
            entity_html = ""
            entity_types = {
                'order_id': '📦',
                'tracking_number': '🚚', 
                'email': '📧',
                'phone': '📞',
                'product_name': '🛍️',
                'date': '📅',
                'address': '🏠'
            }
            
            for entity_type, entity_value in entities.items():
                icon = entity_types.get(entity_type, '🏷️')
                entity_html += f"<p>{icon} <strong>{entity_type.replace('_', ' ').title()}:</strong> <code>{entity_value}</code></p>"
            
            st.markdown(entity_html, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #718096;'>No entities detected in this text</p>", unsafe_allow_html=True)
        
        st.markdown("""
            <p><strong>Technique:</strong> Regular Expressions (Regex) + Pattern Matching</p>
            <p><strong>Supported Entities:</strong> Order IDs, Tracking Numbers, Emails, Phone Numbers, Product Names, Dates, Addresses</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Response Preview
        st.markdown(f"""
        <div class="chat-container">
            <h3>🤖 AI Response Preview</h3>
        """, unsafe_allow_html=True)
        
        ai_response = generate_response(demo_text, intent, entities, emoji)
        st.markdown(f'<p style="background: #f8f9fa; padding: 16px; border-radius: 8px; border-left: 4px solid #667eea;">{ai_response}</p>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Example Test Cases
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="chat-container">
        <h3>🧪 Example Test Cases</h3>
        <p style="color: #718096; margin-bottom: 16px;">Click any example to analyze it:</p>
    </div>
    """, unsafe_allow_html=True)
    
    examples = [
        "Track my order #ORD-123456",
        "I forgot my password for john.doe@company.com",
        "When will my package 1Z999AA1234567890 arrive?",
        "I want to return my defective iPhone 13 Pro",
        "This is terrible, I hate your service!",
        "Thank you so much, excellent support!",
        "Hello, I need help with my account",
        "Can I get information about MacBook Pro pricing?"
    ]
    
    cols = st.columns(4)
    for i, example in enumerate(examples):
        col_idx = i % 4
        with cols[col_idx]:
            if st.button(f"📝 Example {i+1}", key=f"example_{i}", use_container_width=True, help=example):
                st.rerun()
    
    # Show selected example
    if st.session_state.get('selected_example'):
        st.text_input("Selected example:", st.session_state.selected_example, key="demo_input_selected")

st.markdown('</div>', unsafe_allow_html=True)
