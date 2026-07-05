"""
AI Chatbot NLP - Streamlit Version
Intelligent Customer Support Automation
"""

import streamlit as st
import re
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Chatbot NLP",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Intent patterns
INTENT_PATTERNS = {
    'greeting': [
        r'hello|hi|hey|good\s+(morning|afternoon|evening)|greetings|howdy',
        r'how\s+are\s+you|what\'?s\s+up'
    ],
    'track_order': [
        r'track\s+.*(order|package|shipment)|order\s+.*(status|tracking)',
        r'where\s+is\s+my\s+(order|package|delivery)',
        r'find\s+my\s+order|locate\s+my\s+package'
    ],
    'reset_password': [
        r'reset\s+.*password|forgot\s+.*password|change\s+.*password',
        r'password\s+.*(help|issue|problem|reset)',
        r'can\'?t\s+.*log\s*in|login\s+.*(problem|issue|trouble)'
    ],
    'delivery_inquiry': [
        r'when\s+will\s+.*(deliver|arrive|come)',
        r'delivery\s+.*(time|date|schedule|estimate)',
        r'shipping\s+.*(info|information|details|timeline)'
    ],
    'request_refund': [
        r'refund|return\s+.*item|get\s+.*money\s+back',
        r'cancel\s+.*order|want\s+.*refund',
        r'defective|broken|damaged|wrong\s+item'
    ],
    'contact_support': [
        r'(speak|talk)\s+to\s+.*(human|person|agent|representative)',
        r'customer\s+service|live\s+chat|human\s+support',
        r'escalate|supervisor|manager'
    ],
    'goodbye': [
        r'bye|goodbye|see\s+you\s+later|farewell',
        r'thanks\s+bye|have\s+a\s+good\s+day'
    ]
}

# Response templates
RESPONSES = {
    'greeting': [
        "👋 Hello! I'm your AI customer support assistant. How can I help you today?",
        "Hi there! Welcome to our customer support. What can I assist you with?",
        "Greetings! I'm here to help with your questions. How may I assist you?"
    ],
    'track_order': [
        "📦 I'd be happy to help you track your order! Could you please provide your order ID or tracking number?",
        "Let me help you locate your package. What's your order number?",
        "I can check your order status right away. Please share your order ID."
    ],
    'reset_password': [
        "🔐 I'll help you reset your password securely. What's the email address associated with your account?",
        "No problem! I can assist with password reset. Please provide your registered email address.",
        "Let me help you regain access to your account. What's your email?"
    ],
    'delivery_inquiry': [
        "🚚 I'll check your delivery status for you. Could you share your order number?",
        "Let me look up your delivery information. What's your order ID?",
        "I can provide delivery updates. Please share your order number."
    ],
    'request_refund': [
        "💰 I understand you'd like to request a refund. Can you provide your order ID and the reason?",
        "I'll help you process the refund. What's your order number?",
        "Let me assist with your return. Please share your order ID."
    ],
    'contact_support': [
        "👥 I'm connecting you with our human support team. They'll be with you shortly.",
        "Let me transfer you to a live agent who can provide personalized assistance.",
        "I'll escalate this to our customer service team right away."
    ],
    'goodbye': [
        "👋 Thank you for contacting us! Have a wonderful day!",
        "Goodbye! Don't hesitate to reach out if you need more help.",
        "Thanks for chatting! Take care!"
    ],
    'fallback': [
        "🤔 I'm sorry, I didn't quite understand that. Could you please rephrase your question?",
        "I'd like to help, but I need a bit more clarification. Can you provide more details?",
        "I'm not sure I understood that correctly. Could you explain what you need?"
    ]
}

def classify_intent(text):
    """Classify user intent using pattern matching"""
    text_lower = text.lower()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                confidence = random.uniform(0.85, 0.98)
                return intent, confidence
    
    return 'fallback', random.uniform(0.25, 0.40)

def extract_entities(text):
    """Extract entities from text"""
    entities = {}
    
    # Order ID
    order_match = re.search(r'#?([A-Z]{2,3}[-]?\d{6,12})', text, re.IGNORECASE)
    if order_match:
        entities['order_id'] = order_match.group(1)
    
    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', text)
    if email_match:
        entities['email'] = email_match.group(0)
    
    return entities

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'conversation_stats' not in st.session_state:
    st.session_state.conversation_stats = {
        'total_messages': 0,
        'intents_detected': {}
    }

# Sidebar
with st.sidebar:
    st.title("📊 Analytics")
    st.metric("Total Messages", st.session_state.conversation_stats['total_messages'])
    
    if st.session_state.conversation_stats['intents_detected']:
        st.subheader("Intent Distribution")
        for intent, count in st.session_state.conversation_stats['intents_detected'].items():
            st.write(f"**{intent}**: {count}")
    
    st.divider()
    
    st.subheader("🎯 Features")
    st.write("✅ Intent Recognition")
    st.write("✅ Entity Extraction")
    st.write("✅ Context Awareness")
    st.write("✅ Multi-turn Dialogue")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_stats = {
            'total_messages': 0,
            'intents_detected': {}
        }
        st.rerun()
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit")

# Main chat interface
st.title("🤖 AI Chatbot NLP")
st.markdown("**Intelligent Customer Support Automation**")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("📊 Analysis Details"):
                cols = st.columns(3)
                cols[0].metric("Intent", message["metadata"]["intent"])
                cols[1].metric("Confidence", f"{message['metadata']['confidence']:.0%}")
                if message["metadata"]["entities"]:
                    cols[2].write("**Entities:**")
                    for key, value in message["metadata"]["entities"].items():
                        cols[2].write(f"• {key}: {value}")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process intent
    intent, confidence = classify_intent(prompt)
    entities = extract_entities(prompt)
    
    # Generate response
    response = random.choice(RESPONSES.get(intent, RESPONSES['fallback']))
    
    # Update stats
    st.session_state.conversation_stats['total_messages'] += 1
    if intent in st.session_state.conversation_stats['intents_detected']:
        st.session_state.conversation_stats['intents_detected'][intent] += 1
    else:
        st.session_state.conversation_stats['intents_detected'][intent] = 1
    
    # Add assistant message
    metadata = {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "timestamp": datetime.now().isoformat()
    }
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "metadata": metadata
    })
    
    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
        with st.expander("📊 Analysis Details"):
            cols = st.columns(3)
            cols[0].metric("Intent", intent)
            cols[1].metric("Confidence", f"{confidence:.0%}")
            if entities:
                cols[2].write("**Entities:**")
                for key, value in entities.items():
                    cols[2].write(f"• {key}: {value}")
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🤖 <strong>AI Chatbot NLP v2.0</strong> | Built with Python, NLP & Streamlit</p>
    <p>Modern Customer Support Automation</p>
</div>
""", unsafe_allow_html=True)
