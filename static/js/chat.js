/**
 * AI Chatbot (NLP) - Interactive Chat Interface
 * Real-time messaging with NLP features visualization
 */

class ChatInterface {
    constructor() {
        this.sessionId = null;
        this.messageCount = 0;
        this.isTyping = false;
        this.analyticsData = {};
        
        // Initialize interface
        this.init();
        this.loadAnalytics();
        
        // Auto-refresh analytics every 30 seconds
        setInterval(() => this.loadAnalytics(), 30000);
    }
    
    init() {
        // Get DOM elements
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.messagesContainer = document.getElementById('chatMessages');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // Bind event listeners
        this.bindEvents();
        
        // Add welcome message
        this.addWelcomeMessage();
        
        // Focus input
        this.messageInput.focus();
    }
    
    bindEvents() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Enter key in input
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.autoResizeInput();
        });
        
        // Example buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('example-button')) {
                this.sendExampleMessage(e.target.textContent);
            }
        });
        
        // Refresh analytics
        const refreshButton = document.getElementById('refreshAnalytics');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => this.loadAnalytics());
        }
    }
    
    autoResizeInput() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
    
    addWelcomeMessage() {
        const welcomeMsg = {
            text: "👋 Hello! I'm your AI customer support assistant. I can help you with:",
            sender: 'bot',
            intent: 'greeting',
            confidence: 1.0,
            entities: {},
            timestamp: new Date()
        };
        
        this.displayMessage(welcomeMsg);
        
        // Add capability list
        setTimeout(() => {
            const capabilitiesMsg = {
                text: "• 📦 Track your orders\\n• 🔑 Reset passwords\\n• 🚚 Check delivery status\\n• 💰 Process refunds\\n• 🆘 Connect you with support\\n\\nTry asking me something like 'Track my order' or click one of the examples below!",
                sender: 'bot',
                intent: 'greeting',
                confidence: 1.0,
                entities: {},
                timestamp: new Date()
            };
            this.displayMessage(capabilitiesMsg);
        }, 1000);
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message || this.isTyping) {
            return;
        }
        
        // Display user message
        this.displayMessage({
            text: message,
            sender: 'user',
            timestamp: new Date()
        });
        
        // Clear input
        this.messageInput.value = '';
        this.autoResizeInput();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Send to API
            const response = await this.callChatAPI(message);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Display bot response
            this.displayMessage({
                text: response.response,
                sender: 'bot',
                intent: response.intent,
                confidence: response.confidence,
                entities: response.entities,
                processingTime: response.processing_time,
                timestamp: new Date()
            });
            
            // Update session ID
            this.sessionId = response.session_id;
            
            // Update analytics
            this.updateAnalytics(response);
            
        } catch (error) {
            console.error('Chat API Error:', error);
            this.hideTypingIndicator();
            
            // Display error message
            this.displayMessage({
                text: "I'm sorry, I'm experiencing some technical difficulties. Please try again in a moment.",
                sender: 'bot',
                intent: 'error',
                confidence: 0.0,
                entities: {},
                timestamp: new Date(),
                isError: true
            });
        }
    }
    
    async callChatAPI(message) {
        const requestBody = {
            message: message
        };
        
        if (this.sessionId) {
            requestBody.session_id = this.sessionId;
        }
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    displayMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${message.sender}`;
        
        // Create message content
        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';
        
        // Format message text (handle line breaks)
        const formattedText = message.text.replace(/\\n/g, '<br>');
        contentElement.innerHTML = formattedText;
        
        // Add error styling
        if (message.isError) {
            contentElement.style.borderLeft = '3px solid var(--danger-color)';
        }
        
        messageElement.appendChild(contentElement);
        
        // Add metadata for bot messages
        if (message.sender === 'bot' && !message.isError) {
            const metaElement = this.createMessageMeta(message);
            messageElement.appendChild(metaElement);
        }
        
        // Add to container
        this.messagesContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Increment message count
        this.messageCount++;
    }
    
    createMessageMeta(message) {
        const metaElement = document.createElement('div');
        metaElement.className = 'message-meta';
        
        const metaInfo = [];
        
        // Intent
        if (message.intent) {
            const intentSpan = document.createElement('span');
            intentSpan.textContent = `Intent: ${message.intent.replace('_', ' ')}`;
            metaInfo.push(intentSpan);
        }
        
        // Confidence
        if (message.confidence !== undefined) {
            const confidenceSpan = document.createElement('span');
            confidenceSpan.className = `confidence-badge ${this.getConfidenceClass(message.confidence)}`;
            confidenceSpan.textContent = `${Math.round(message.confidence * 100)}% confidence`;
            metaInfo.push(confidenceSpan);
        }
        
        // Entities
        if (message.entities && Object.keys(message.entities).length > 0) {
            const entitiesSpan = document.createElement('span');
            const entityCount = Object.values(message.entities).flat().length;
            entitiesSpan.textContent = `${entityCount} entities found`;
            entitiesSpan.title = JSON.stringify(message.entities, null, 2);
            metaInfo.push(entitiesSpan);
        }
        
        // Processing time
        if (message.processingTime) {
            const timeSpan = document.createElement('span');
            timeSpan.textContent = `${Math.round(message.processingTime * 1000)}ms`;
            metaInfo.push(timeSpan);
        }
        
        // Add all info to meta element
        metaInfo.forEach((info, index) => {
            metaElement.appendChild(info);
            if (index < metaInfo.length - 1) {
                metaElement.appendChild(document.createTextNode(' • '));
            }
        });
        
        return metaElement;
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'high';
        if (confidence >= 0.6) return 'medium';
        return 'low';
    }
    
    showTypingIndicator() {
        this.isTyping = true;
        this.sendButton.disabled = true;
        this.sendButton.innerHTML = '<span class="spinner"></span>';
        
        this.typingIndicator.style.display = 'block';
        this.typingIndicator.innerHTML = 'AI is thinking<span class="typing-dots">...</span>';
        
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        this.sendButton.disabled = false;
        this.sendButton.innerHTML = 'Send';
        
        this.typingIndicator.style.display = 'none';
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }
    
    sendExampleMessage(message) {
        this.messageInput.value = message;
        this.sendMessage();
    }
    
    async loadAnalytics() {
        try {
            const response = await fetch('/api/analytics');
            if (response.ok) {
                this.analyticsData = await response.json();
                this.updateAnalyticsDisplay();
            }
        } catch (error) {
            console.error('Analytics loading error:', error);
        }
    }
    
    updateAnalytics(chatResponse) {
        // Update local analytics data
        this.analyticsData.total_messages = (this.analyticsData.total_messages || 0) + 1;
        this.analyticsData.average_confidence = this.calculateAverageConfidence(chatResponse.confidence);
        
        // Update intent counts
        if (!this.analyticsData.intent_distribution) {
            this.analyticsData.intent_distribution = {};
        }
        
        const intent = chatResponse.intent;
        this.analyticsData.intent_distribution[intent] = (this.analyticsData.intent_distribution[intent] || 0) + 1;
        
        // Update display
        this.updateAnalyticsDisplay();
    }
    
    calculateAverageConfidence(newConfidence) {
        const currentAvg = this.analyticsData.average_confidence || 0;
        const currentCount = this.analyticsData.total_messages || 1;
        
        return ((currentAvg * (currentCount - 1)) + newConfidence) / currentCount;
    }
    
    updateAnalyticsDisplay() {
        // Update message count
        const messageCountElement = document.getElementById('messageCount');
        if (messageCountElement) {
            messageCountElement.textContent = this.analyticsData.total_messages || this.messageCount;
        }
        
        // Update confidence
        const confidenceElement = document.getElementById('averageConfidence');
        if (confidenceElement && this.analyticsData.average_confidence) {
            const confidence = Math.round(this.analyticsData.average_confidence * 100);
            confidenceElement.textContent = `${confidence}%`;
        }
        
        // Update response time
        const responseTimeElement = document.getElementById('responseTime');
        if (responseTimeElement && this.analyticsData.average_response_time) {
            responseTimeElement.textContent = `${Math.round(this.analyticsData.average_response_time * 1000)}ms`;
        }
        
        // Update active sessions
        const sessionsElement = document.getElementById('activeSessions');
        if (sessionsElement) {
            sessionsElement.textContent = this.analyticsData.active_conversations || (this.sessionId ? 1 : 0);
        }
        
        // Update intent chart
        this.updateIntentChart();
    }
    
    updateIntentChart() {
        const chartContainer = document.getElementById('intentChart');
        if (!chartContainer || !this.analyticsData.intent_distribution) return;
        
        const intents = this.analyticsData.intent_distribution;
        const maxCount = Math.max(...Object.values(intents));
        
        chartContainer.innerHTML = '';
        
        // Sort intents by count (descending)
        const sortedIntents = Object.entries(intents).sort((a, b) => b[1] - a[1]);
        
        sortedIntents.forEach(([intent, count]) => {
            const item = document.createElement('div');
            item.className = 'intent-item';
            
            const percentage = (count / maxCount) * 100;
            
            item.innerHTML = `
                <span class="intent-name">${intent.replace('_', ' ')}</span>
                <span class="intent-count">${count}</span>
                <div class="intent-bar">
                    <div class="intent-bar-fill" style="width: ${percentage}%"></div>
                </div>
            `;
            
            chartContainer.appendChild(item);
        });
    }
    
    // Public methods for external use
    clearChat() {
        this.messagesContainer.innerHTML = '';
        this.sessionId = null;
        this.messageCount = 0;
        this.addWelcomeMessage();
    }
    
    exportChatHistory() {
        const messages = Array.from(this.messagesContainer.children).map(messageElement => {
            const isUser = messageElement.classList.contains('user');
            const content = messageElement.querySelector('.message-content').textContent;
            
            return {
                sender: isUser ? 'user' : 'bot',
                message: content,
                timestamp: new Date().toISOString()
            };
        });
        
        const dataStr = JSON.stringify(messages, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `chat-history-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
    }
}

// Analytics Dashboard
class AnalyticsDashboard {
    constructor() {
        this.metricsUpdateInterval = null;
        this.init();
    }
    
    init() {
        // Start metrics updates
        this.startMetricsUpdates();
        
        // Bind refresh button
        const refreshBtn = document.getElementById('refreshAnalytics');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshMetrics());
        }
    }
    
    startMetricsUpdates() {
        this.refreshMetrics();
        
        // Update every 30 seconds
        this.metricsUpdateInterval = setInterval(() => {
            this.refreshMetrics();
        }, 30000);
    }
    
    async refreshMetrics() {
        try {
            const response = await fetch('/api/analytics');
            if (response.ok) {
                const data = await response.json();
                this.updateMetricsDisplay(data);
            }
        } catch (error) {
            console.error('Metrics refresh error:', error);
        }
    }
    
    updateMetricsDisplay(data) {
        // Update all metric cards
        this.updateMetric('totalConversations', data.total_conversations || 0);
        this.updateMetric('activeConversations', data.active_conversations || 0);
        this.updateMetric('averageConfidence', data.average_confidence ? `${Math.round(data.average_confidence * 100)}%` : 'N/A');
        this.updateMetric('responseTime', data.average_response_time ? `${Math.round(data.average_response_time * 1000)}ms` : 'N/A');
        
        // Update intent distribution
        if (data.intent_distribution) {
            this.updateIntentDistribution(data.intent_distribution);
        }
    }
    
    updateMetric(metricId, value) {
        const element = document.getElementById(metricId);
        if (element) {
            element.textContent = value;
        }
    }
    
    updateIntentDistribution(intentData) {
        const container = document.getElementById('intentDistribution');
        if (!container) return;
        
        const maxCount = Math.max(...Object.values(intentData));
        container.innerHTML = '';
        
        // Sort by count
        const sorted = Object.entries(intentData).sort((a, b) => b[1] - a[1]);
        
        sorted.slice(0, 5).forEach(([intent, count]) => {
            const percentage = (count / maxCount) * 100;
            
            const item = document.createElement('div');
            item.className = 'intent-item';
            item.innerHTML = `
                <span class="intent-name">${intent.replace('_', ' ')}</span>
                <span class="intent-count">${count}</span>
                <div class="intent-bar">
                    <div class="intent-bar-fill" style="width: ${percentage}%"></div>
                </div>
            `;
            
            container.appendChild(item);
        });
    }
    
    destroy() {
        if (this.metricsUpdateInterval) {
            clearInterval(this.metricsUpdateInterval);
        }
    }
}

// Utility Functions
const ChatUtils = {
    formatTimestamp(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },
    
    getConfidenceColor(confidence) {
        if (confidence >= 0.8) return '#10b981'; // green
        if (confidence >= 0.6) return '#f59e0b'; // yellow
        return '#ef4444'; // red
    },
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize chat interface
    window.chatInterface = new ChatInterface();
    
    // Initialize analytics dashboard
    window.analyticsDashboard = new AnalyticsDashboard();
    
    // Add global functions for debugging
    window.clearChat = () => window.chatInterface.clearChat();
    window.exportChat = () => window.chatInterface.exportChatHistory();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, pause updates
        if (window.analyticsDashboard) {
            window.analyticsDashboard.destroy();
        }
    } else {
        // Page is visible, resume updates
        if (window.analyticsDashboard) {
            window.analyticsDashboard.init();
        }
    }
});