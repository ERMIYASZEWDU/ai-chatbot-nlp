"""
AI Chatbot NLP - Modern Flask Application
Professional customer support automation with NLP capabilities.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def create_app(config_path: str = "config.yaml") -> Flask:
    """
    Application factory pattern for Flask app creation.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", 
        "ai-chatbot-secret-key-change-in-production"
    )
    app.config["CONFIG_PATH"] = config_path
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize SocketIO (will be attached to app later)
    app.config["SOCKETIO"] = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    
    # Register blueprints and routes
    _register_routes(app)
    _register_error_handlers(app)
    _register_socketio_handlers(app)
    
    # Initialize chatbot lazily
    app.config["CHATBOT"] = None
    
    logger.info("Flask application created successfully")
    
    return app


def _get_chatbot(app: Flask) -> Any:
    """Lazy initialization of chatbot engine."""
    if app.config["CHATBOT"] is None:
        with suppress(Exception):
            from src.chatbot import create_chatbot
            app.config["CHATBOT"] = create_chatbot(app.config["CONFIG_PATH"])
            logger.info("Chatbot engine initialized")
    return app.config["CHATBOT"]


def _register_routes(app: Flask) -> None:
    """Register all application routes."""
    
    @app.route("/")
    def index() -> str:
        """Main chat interface."""
        return render_template("index.html")
    
    @app.route("/dashboard")
    def dashboard() -> str:
        """Analytics dashboard."""
        return render_template("dashboard.html")
    
    @app.route("/demo")
    def demo() -> str:
        """Demo page with sample interactions."""
        return render_template("demo.html")
    
    @app.route("/api/status")
    def get_status() -> tuple[Any, int]:
        """Get system status and health check."""
        try:
            chatbot = _get_chatbot(app)
            if chatbot is None:
                return jsonify({
                    "status": "degraded",
                    "message": "Chatbot not initialized - running in fallback mode",
                    "mode": "fallback",
                    "version": "2.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "api": "healthy",
                        "chatbot": "not_initialized",
                        "nlp": "fallback"
                    }
                }), 200
            
            status = chatbot.get_system_status()
            status["mode"] = "full"
            return jsonify(status), 200
            
        except Exception as e:
            logger.error("Error getting status", error=str(e))
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route("/api/chat", methods=["POST"])
    def chat_api() -> tuple[Any, int]:
        """
        REST API endpoint for chat interactions.
        
        Request body:
            {
                "message": "user message",
                "session_id": "optional session id"
            }
        """
        try:
            data = request.get_json()
            
            if not data or "message" not in data:
                return jsonify({
                    "error": "Missing 'message' in request body",
                    "status": "error"
                }), 400
            
            message = data["message"].strip()
            if not message:
                return jsonify({
                    "error": "Empty message not allowed",
                    "status": "error"
                }), 400
            
            session_id = data.get("session_id") or str(uuid.uuid4())
            
            chatbot = _get_chatbot(app)
            
            if chatbot is None:
                # Fallback response when chatbot is not initialized
                return jsonify({
                    "response": _fallback_response(message),
                    "intent": "fallback",
                    "confidence": 0.5,
                    "entities": {},
                    "session_id": session_id,
                    "status": "success",
                    "mode": "fallback",
                    "timestamp": datetime.now().isoformat()
                }), 200
            
            result = chatbot.chat(message, session_id)
            result["session_id"] = session_id
            result["status"] = "success"
            
            logger.info(
                "Chat API processed",
                session_id=session_id[:8],
                intent=result.get("intent", "unknown"),
                confidence=result.get("confidence", 0)
            )
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error("Error in chat API", error=str(e))
            return jsonify({
                "error": "Internal server error",
                "status": "error",
                "message": "Sorry, I encountered an error processing your request."
            }), 500
    
    @app.route("/api/analyze", methods=["POST"])
    def analyze_api() -> tuple[Any, int]:
        """
        Text analysis API endpoint for detailed NLP analysis.
        
        Request body:
            {
                "text": "text to analyze"
            }
        """
        try:
            data = request.get_json()
            
            if not data or "text" not in data:
                return jsonify({
                    "error": "Missing 'text' in request body",
                    "status": "error"
                }), 400
            
            text = data["text"].strip()
            if not text:
                return jsonify({
                    "error": "Empty text not allowed",
                    "status": "error"
                }), 400
            
            chatbot = _get_chatbot(app)
            
            if chatbot is None:
                return jsonify({
                    "error": "Chatbot not available",
                    "status": "error"
                }), 503
            
            result = chatbot.analyze_text(text)
            result["status"] = "success"
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error("Error in analyze API", error=str(e))
            return jsonify({
                "error": "Internal server error",
                "status": "error"
            }), 500
    
    @app.route("/api/analytics")
    def get_overall_analytics() -> tuple[Any, int]:
        """Get overall system analytics."""
        try:
            chatbot = _get_chatbot(app)
            
            if chatbot is None:
                return jsonify({
                    "analytics": _default_analytics(),
                    "status": "success",
                    "mode": "fallback"
                }), 200
            
            analytics = chatbot.get_conversation_analytics()
            
            return jsonify({
                "analytics": analytics,
                "status": "success"
            }), 200
            
        except Exception as e:
            logger.error("Error getting analytics", error=str(e))
            return jsonify({
                "error": "Internal server error",
                "status": "error"
            }), 500
    
    @app.route("/api/analytics/<session_id>")
    def get_session_analytics(session_id: str) -> tuple[Any, int]:
        """Get conversation analytics for a specific session."""
        try:
            chatbot = _get_chatbot(app)
            
            if chatbot is None:
                return jsonify({
                    "error": "Chatbot not available",
                    "status": "error"
                }), 503
            
            analytics = chatbot.get_conversation_analytics(session_id)
            
            return jsonify({
                "analytics": analytics,
                "status": "success"
            }), 200
            
        except Exception as e:
            logger.error("Error getting session analytics", error=str(e))
            return jsonify({
                "error": "Internal server error",
                "status": "error"
            }), 500


def _register_error_handlers(app: Flask) -> None:
    """Register error handlers."""
    
    @app.errorhandler(404)
    def not_found_error(error: Any) -> tuple[str, int]:
        return render_template("error.html", 
                             error_code=404, 
                             error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error: Any) -> tuple[str, int]:
        logger.error("Internal server error", error=str(error))
        return render_template("error.html", 
                             error_code=500, 
                             error_message="Internal server error"), 500


def _register_socketio_handlers(app: Flask) -> None:
    """Register Socket.IO event handlers."""
    socketio = app.config["SOCKETIO"]
    
    @socketio.on("connect")
    def handle_connect() -> None:
        """Handle client connection."""
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
        
        emit("connected", {
            "session_id": session_id,
            "message": "Connected to AI Chatbot",
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info("Client connected", session_id=session_id[:8])
    
    @socketio.on("disconnect")
    def handle_disconnect() -> None:
        """Handle client disconnection."""
        session_id = session.get("session_id", "unknown")
        logger.info("Client disconnected", session_id=session_id[:8] if len(session_id) >= 8 else session_id)
    
    @socketio.on("send_message")
    def handle_message(data: dict) -> None:
        """Handle real-time chat messages."""
        try:
            message = data.get("message", "").strip()
            session_id = session.get("session_id", str(uuid.uuid4()))
            
            if not message:
                emit("bot_response", {
                    "error": "Empty message not allowed",
                    "status": "error"
                })
                return
            
            chatbot = _get_chatbot(app)
            
            if chatbot is None:
                emit("bot_response", {
                    "response": _fallback_response(message),
                    "intent": "fallback",
                    "confidence": 0.5,
                    "session_id": session_id,
                    "status": "success",
                    "mode": "fallback"
                })
                return
            
            result = chatbot.chat(message, session_id)
            
            emit("bot_response", {
                "response": result["response"],
                "intent": result.get("intent"),
                "confidence": result.get("confidence"),
                "entities": result.get("entities", {}),
                "session_id": session_id,
                "timestamp": result.get("timestamp"),
                "processing_time": result.get("total_processing_time"),
                "status": "success"
            })
            
            logger.info(
                "SocketIO message processed",
                session_id=session_id[:8],
                intent=result.get("intent", "unknown")
            )
            
        except Exception as e:
            logger.error("Error handling SocketIO message", error=str(e))
            emit("bot_response", {
                "error": "Internal server error",
                "status": "error"
            })


def _fallback_response(message: str) -> str:
    """Generate a fallback response when chatbot is not available."""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
        return "Hello! I'm your AI assistant. How can I help you today?"
    elif any(word in message_lower for word in ["track", "order", "where"]):
        return "I can help you track your order. Please provide your order ID."
    elif any(word in message_lower for word in ["password", "reset", "forgot"]):
        return "I can help you reset your password. Please provide your email address."
    elif any(word in message_lower for word in ["delivery", "shipping", "arrive"]):
        return "I can check your delivery status. What's your order ID?"
    elif any(word in message_lower for word in ["refund", "return", "money"]):
        return "I understand you'd like a refund. Can you provide your order ID?"
    elif any(word in message_lower for word in ["human", "agent", "support"]):
        return "I'm connecting you with our support team. Please hold on."
    elif any(word in message_lower for word in ["bye", "goodbye", "thanks"]):
        return "Thank you for contacting us! Have a great day!"
    else:
        return "I'm here to help with orders, deliveries, passwords, and refunds. How can I assist you?"


def _default_analytics() -> dict:
    """Return default analytics when chatbot is not available."""
    return {
        "total_conversations": 0,
        "active_sessions": 0,
        "avg_response_time": 0,
        "intent_distribution": {},
        "mode": "fallback"
    }


# Create application instance
app = create_app()
socketio = app.config["SOCKETIO"]


def main() -> None:
    """Main entry point for the application."""
    # Ensure required directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Get configuration from environment
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║           🤖 AI Chatbot NLP - v2.0.0                     ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  🌐 Server:   http://{host}:{port}                        ║
    ║  📊 Dashboard: http://{host}:{port}/dashboard             ║
    ║  🎮 Demo:     http://{host}:{port}/demo                   ║
    ║  📡 API:      http://{host}:{port}/api/status             ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Features:                                                ║
    ║  ✅ Intent Classification  ✅ Entity Extraction          ║
    ║  ✅ Real-time Chat         ✅ Analytics Dashboard        ║
    ║  ✅ REST API               ✅ WebSocket Support          ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=debug
    )


if __name__ == "__main__":
    main()
