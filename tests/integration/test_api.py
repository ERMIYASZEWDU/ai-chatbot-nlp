"""
Integration tests for API endpoints.
"""

import pytest
from app import create_app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    def test_status_endpoint_returns_200(self, client):
        """Test that /api/status returns 200 status code."""
        response = client.get("/api/status")
        assert response.status_code == 200
    
    def test_status_endpoint_returns_json(self, client):
        """Test that /api/status returns JSON."""
        response = client.get("/api/status")
        assert response.content_type == "application/json"
    
    def test_chat_endpoint_missing_message_returns_400(self, client):
        """Test that /api/chat returns 400 when message is missing."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 400
    
    def test_chat_endpoint_empty_message_returns_400(self, client):
        """Test that /api/chat returns 400 when message is empty."""
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code == 400
    
    def test_chat_endpoint_valid_message_returns_200(self, client):
        """Test that /api/chat returns 200 for valid message."""
        response = client.post("/api/chat", json={"message": "Hello"})
        assert response.status_code == 200
    
    def test_analytics_endpoint_returns_200(self, client):
        """Test that /api/analytics returns 200."""
        response = client.get("/api/analytics")
        assert response.status_code == 200
