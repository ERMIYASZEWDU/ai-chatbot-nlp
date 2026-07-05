// Enhanced AI Chatbot Interface with PWA Support

// Global variables
let sessionStartTime = null;
let sessionTimer = null;
let currentSessionId = null;
let messageCount = 0;
let isOnline = navigator.onLine;
let installPrompt = null;

// PWA and Mobile specific variables
let touchStartY = 0;
let touchEndY = 0;
let isPullToRefresh = false;
let swipeThreshold = 50;
let offlineMessageQueue = [];
let recognition = null;
let isListening = false;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeSession();
    setupEventListeners();
    updateSessionTimer();
    initializePWA();
    setupMobileInteractions();
    checkNetworkStatus();
});

// PWA Initialization
function initializePWA() {
    // Handle app install banner
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        installPrompt = e;
        showInstallBanner();
    });
    
    // Handle app installed
    window.addEventListener('appinstalled', () => {
        console.log('PWA was installed');
        hideInstallBanner();
        showNotification('App installed successfully!', 'success');
    });
    
    // Handle visibility change (app focus/blur)
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            syncPendingMessages();
        }
    });
}

// Mobile interaction setup
function setupMobileInteractions() {
    const chatContainer = document.getElementById('chat-messages');
    if (chatContainer) {
        // Pull to refresh
        chatContainer.addEventListener('touchstart', handleTouchStart, { passive: true });
        chatContainer.addEventListener('touchmove', handleTouchMove, { passive: false });
        chatContainer.addEventListener('touchend', handleTouchEnd, { passive: true });
        
        // Prevent zoom on double tap
        let lastTouchTime = 0;
        chatContainer.addEventListener('touchend', (e) => {
            const currentTime = new Date().getTime();
            const tapLength = currentTime - lastTouchTime;
            if (tapLength < 500 && tapLength > 0) {
                e.preventDefault();
            }
            lastTouchTime = currentTime;
        });
    }
}
// Voice input setup
function setupVoiceInput() {
    const voiceButton = document.createElement('button');
    voiceButton.type = 'button';
    voiceButton.className = 'btn btn-outline-primary voice-button ms-2';
    voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
    voiceButton.title = 'Voice input';
    
    const sendButton = document.getElementById('send-button');
    if (sendButton && 'webkitSpeechRecognition' in window) {
        sendButton.parentNode.insertBefore(voiceButton, sendButton);
        voiceButton.addEventListener('click', toggleVoiceInput);
    }
}

function toggleVoiceInput() {
    if (isListening) {
        stopVoiceInput();
    } else {
        startVoiceInput();
    }
}

function startVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onstart = () => {
        isListening = true;
        const voiceButton = document.querySelector('.voice-button');
        if (voiceButton) {
            voiceButton.classList.add('recording');
            voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
        }
        vibrate(100);
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        const messageInput = document.getElementById('message-input');
        if (messageInput) {
            messageInput.value = transcript;
            messageInput.focus();
        }
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        stopVoiceInput();
        showNotification('Voice input error. Please try again.', 'error');
    };
    
    recognition.onend = () => {
        stopVoiceInput();
    };
    
    try {
        recognition.start();
    } catch (error) {
        console.error('Failed to start voice recognition:', error);
        stopVoiceInput();
    }
}
function stopVoiceInput() {
    if (recognition) {
        recognition.stop();
        recognition = null;
    }
    
    isListening = false;
    const voiceButton = document.querySelector('.voice-button');
    if (voiceButton) {
        voiceButton.classList.remove('recording');
        voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
    }
}

// Haptic feedback
function setupHapticFeedback() {
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', () => vibrate(50));
    });
}

function vibrate(duration) {
    if ('vibrate' in navigator) {
        navigator.vibrate(duration);
    }
}

// Touch gesture handling for pull-to-refresh
function handleTouchStart(e) {
    touchStartY = e.touches[0].clientY;
    const chatContainer = e.currentTarget;
    if (chatContainer.scrollTop === 0) {
        isPullToRefresh = true;
    }
}

function handleTouchMove(e) {
    if (!isPullToRefresh) return;
    
    touchEndY = e.touches[0].clientY;
    const pullDistance = touchEndY - touchStartY;
    
    if (pullDistance > swipeThreshold) {
        e.preventDefault();
        showPullToRefreshIndicator(pullDistance);
    }
}

function handleTouchEnd(e) {
    if (!isPullToRefresh) return;
    
    const pullDistance = touchEndY - touchStartY;
    hidePullToRefreshIndicator();
    
    if (pullDistance > swipeThreshold * 2) {
        refreshChatData();
        vibrate(200);
    }
    
    isPullToRefresh = false;
    touchStartY = 0;
    touchEndY = 0;
}
function showPullToRefreshIndicator(distance) {
    let indicator = document.querySelector('.pull-to-refresh');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.className = 'pull-to-refresh';
        indicator.innerHTML = '<i class="fas fa-arrow-down"></i> Pull to refresh';
        document.body.appendChild(indicator);
    }
    
    const progress = Math.min(distance / (swipeThreshold * 2), 1);
    indicator.style.transform = `translateX(-50%) translateY(${Math.min(distance * 0.5, 100)}px)`;
    indicator.style.opacity = progress;
}

function hidePullToRefreshIndicator() {
    const indicator = document.querySelector('.pull-to-refresh');
    if (indicator) {
        indicator.style.transform = 'translateX(-50%) translateY(-60px)';
        indicator.style.opacity = '0';
    }
}

function refreshChatData() {
    showNotification('Refreshing...', 'info');
    setTimeout(() => {
        showNotification('Chat refreshed!', 'success');
    }, 1000);
}

// Network status management
function checkNetworkStatus() {
    updateOnlineStatus(navigator.onLine);
    
    window.addEventListener('online', () => {
        updateOnlineStatus(true);
        syncPendingMessages();
        showNotification('Back online!', 'success');
    });
    
    window.addEventListener('offline', () => {
        updateOnlineStatus(false);
        showNotification('You\'re offline. Messages will be queued.', 'warning');
    });
}

function updateOnlineStatus(online) {
    isOnline = online;
    const statusBadge = document.querySelector('.status-indicator');
    if (statusBadge) {
        statusBadge.className = `status-indicator ${online ? 'status-online' : 'status-offline'}`;
        statusBadge.innerHTML = `<i class="fas fa-${online ? 'wifi' : 'wifi-slash'} me-1"></i>${online ? 'Online' : 'Offline'}`;
    }
}
// Offline message queue
function queueMessageForLater(message) {
    offlineMessageQueue.push({
        message,
        timestamp: new Date(),
        sessionId: currentSessionId
    });
    
    localStorage.setItem('offlineMessages', JSON.stringify(offlineMessageQueue));
    showNotification('Message queued for when you\'re back online', 'info');
}

function syncPendingMessages() {
    const storedMessages = localStorage.getItem('offlineMessages');
    if (storedMessages) {
        const messages = JSON.parse(storedMessages);
        
        messages.forEach(async (item) => {
            try {
                await sendChatMessage(item.message);
                showNotification('Synced offline message', 'success');
            } catch (error) {
                console.error('Failed to sync message:', error);
            }
        });
        
        offlineMessageQueue = [];
        localStorage.removeItem('offlineMessages');
    }
}

// Session management
function initializeSession() {
    sessionStartTime = new Date();
    messageCount = 0;
    
    sessionTimer = setInterval(updateSessionTimer, 1000);
    updateMessageCount();
    
    const storedMessages = localStorage.getItem('offlineMessages');
    if (storedMessages) {
        offlineMessageQueue = JSON.parse(storedMessages);
    }
}

function updateSessionTimer() {
    if (!sessionStartTime) return;
    
    const now = new Date();
    const elapsed = Math.floor((now - sessionStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    
    const timerElement = document.getElementById('session-time');
    if (timerElement) {
        timerElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
}

function updateMessageCount() {
    const countElement = document.getElementById('message-count');
    if (countElement) {
        countElement.textContent = messageCount;
    }
}