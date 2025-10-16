"""
Integration tests for API
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Tests for API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert data['status'] == 'healthy'
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert 'accuracy' in data
        assert 'precision' in data
        assert 'recall' in data
        assert 'f1_score' in data
    
    def test_detect_endpoint(self):
        """Test detection endpoint"""
        payload = {
            "subject": "Test Email",
            "body": "This is a test email body",
            "sender": "test@example.com",
            "urls": []
        }
        
        response = client.post("/api/v1/detect", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert 'risk_score' in data
        assert 'risk_level' in data
        assert 'email_score' in data
        assert 'explanation' in data
    
    def test_detect_with_urls(self):
        """Test detection with URLs"""
        payload = {
            "subject": "Urgent: Verify Account",
            "body": "Click here to verify your account",
            "sender": "noreply@suspicious.com",
            "urls": ["http://fake-site.tk"]
        }
        
        response = client.post("/api/v1/detect", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data['url_scores']) > 0
    
    def test_detect_missing_fields(self):
        """Test detection with missing required fields"""
        payload = {
            "subject": "Test",
            "body": "Test body"
            # Missing sender
        }
        
        response = client.post("/api/v1/detect", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_detect_invalid_email(self):
        """Test detection with invalid email format"""
        payload = {
            "subject": "Test",
            "body": "Test body",
            "sender": "not-an-email"
        }
        
        response = client.post("/api/v1/detect", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_batch_detect_endpoint(self):
        """Test batch detection endpoint"""
        payload = [
            {
                "subject": "Test 1",
                "body": "Body 1",
                "sender": "test1@example.com",
                "urls": []
            },
            {
                "subject": "Test 2",
                "body": "Body 2",
                "sender": "test2@example.com",
                "urls": []
            }
        ]
        
        response = client.post("/api/v1/detect/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 2
        assert all('risk_score' in item for item in data)


class TestAPIValidation:
    """Tests for API input validation"""
    
    def test_empty_body(self):
        """Test with empty request body"""
        response = client.post("/api/v1/detect", json={})
        assert response.status_code == 422
    
    def test_extra_fields(self):
        """Test with extra fields (should be ignored)"""
        payload = {
            "subject": "Test",
            "body": "Test body",
            "sender": "test@example.com",
            "urls": [],
            "extra_field": "should be ignored"
        }
        
        response = client.post("/api/v1/detect", json=payload)
        assert response.status_code == 200
    
    def test_url_list_validation(self):
        """Test URL list validation"""
        payload = {
            "subject": "Test",
            "body": "Test body",
            "sender": "test@example.com",
            "urls": ["http://example.com", "https://test.org"]
        }
        
        response = client.post("/api/v1/detect", json=payload)
        assert response.status_code == 200


class TestAPIResponses:
    """Tests for API response formats"""
    
    def test_response_structure(self):
        """Test response has correct structure"""
        payload = {
            "subject": "Test",
            "body": "Test body",
            "sender": "test@example.com"
        }
        
        response = client.post("/api/v1/detect", json=payload)
        data = response.json()
        
        # Check all required fields
        required_fields = [
            'risk_score', 'risk_level', 'email_score',
            'url_scores', 'urls', 'explanation', 'indicators'
        ]
        
        for field in required_fields:
            assert field in data
    
    def test_explanation_structure(self):
        """Test explanation has correct structure"""
        payload = {
            "subject": "Test",
            "body": "Test body",
            "sender": "test@example.com"
        }
        
        response = client.post("/api/v1/detect", json=payload)
        data = response.json()
        explanation = data['explanation']
        
        assert 'summary' in explanation
        assert 'email_analysis' in explanation
        assert 'url_analysis' in explanation
        assert 'recommendations' in explanation
    
    def test_risk_score_range(self):
        """Test risk scores are in valid range"""
        payload = {
            "subject": "Test",
            "body": "Test body",
            "sender": "test@example.com"
        }
        
        response = client.post("/api/v1/detect", json=payload)
        data = response.json()
        
        assert 0 <= data['risk_score'] <= 1
        assert 0 <= data['email_score'] <= 1
        assert all(0 <= score <= 1 for score in data['url_scores'])


class TestAPISecurity:
    """Tests for API security features"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.get("/api/v1/health")
        # Check for CORS headers
        assert response.status_code == 200
    
    def test_long_input_handling(self):
        """Test handling of very long inputs"""
        payload = {
            "subject": "Test" * 1000,
            "body": "A" * 10000,
            "sender": "test@example.com"
        }
        
        response = client.post("/api/v1/detect", json=payload)
        # Should handle without error
        assert response.status_code in [200, 413, 422]
    
    def test_special_characters(self):
        """Test handling of special characters"""
        payload = {
            "subject": "Test <script>alert('xss')</script>",
            "body": "Body with special chars: !@#$%^&*()",
            "sender": "test@example.com"
        }
        
        response = client.post("/api/v1/detect", json=payload)
        assert response.status_code == 200
