"""
Unit tests for models
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.url_features import URLFeatureExtractor
from src.models.email_classifier import EmailClassifier
from src.models.url_classifier import URLClassifier
from src.models.detector import PhishingDetector


class TestURLFeatureExtractor:
    """Tests for URL feature extraction"""
    
    @pytest.fixture
    def extractor(self):
        return URLFeatureExtractor()
    
    def test_extract_features_basic(self, extractor):
        """Test basic feature extraction"""
        url = "http://example.com"
        features = extractor.extract_features(url)
        
        assert 'length' in features
        assert 'num_dots' in features
        assert 'is_https' in features
        assert features['length'] == len(url)
        assert features['is_https'] == 0
    
    def test_extract_features_https(self, extractor):
        """Test HTTPS detection"""
        url = "https://example.com"
        features = extractor.extract_features(url)
        assert features['is_https'] == 1
    
    def test_extract_features_ip(self, extractor):
        """Test IP address detection"""
        url = "http://192.168.1.1"
        features = extractor.extract_features(url)
        assert features['has_ip'] == 1
    
    def test_extract_features_suspicious_tld(self, extractor):
        """Test suspicious TLD detection"""
        url = "http://example.tk"
        features = extractor.extract_features(url)
        assert features['suspicious_tld'] == 1
    
    def test_extract_features_suspicious_keyword(self, extractor):
        """Test suspicious keyword detection"""
        url = "http://example.com/login/verify/account"
        features = extractor.extract_features(url)
        assert features['has_suspicious_keyword'] == 1
    
    def test_calculate_entropy(self, extractor):
        """Test entropy calculation"""
        # High entropy (random)
        entropy1 = extractor._calculate_entropy("abc123xyz789")
        # Low entropy (repetitive)
        entropy2 = extractor._calculate_entropy("aaaaaaaaaa")
        
        assert entropy1 > entropy2
    
    def test_extract_features_batch(self, extractor):
        """Test batch feature extraction"""
        urls = [
            "http://example.com",
            "https://test.org",
            "http://192.168.1.1"
        ]
        
        features_list = extractor.extract_features_batch(urls)
        assert len(features_list) == 3
        assert all('length' in f for f in features_list)
    
    def test_get_feature_names(self, extractor):
        """Test feature names retrieval"""
        names = extractor.get_feature_names()
        assert len(names) > 0
        assert 'length' in names
        assert 'is_https' in names
    
    def test_empty_url(self, extractor):
        """Test handling of empty URL"""
        features = extractor.extract_features("")
        assert all(v == 0 for v in features.values())


class TestEmailClassifier:
    """Tests for email classifier"""
    
    @pytest.fixture
    def classifier(self):
        # Use pretrained model for testing
        return EmailClassifier(use_pretrained=True)
    
    def test_predict_proba(self, classifier):
        """Test probability prediction"""
        text = "This is a test email"
        proba = classifier.predict_proba(text)
        
        assert len(proba) == 2
        assert 0 <= proba[0] <= 1
        assert 0 <= proba[1] <= 1
        assert abs(sum(proba) - 1.0) < 0.01  # Sum should be ~1
    
    def test_predict(self, classifier):
        """Test class prediction"""
        text = "This is a test email"
        prediction = classifier.predict(text)
        
        assert prediction in [0, 1]
    
    def test_predict_empty(self, classifier):
        """Test prediction with empty text"""
        proba = classifier.predict_proba("")
        assert len(proba) == 2
    
    def test_predict_batch(self, classifier):
        """Test batch prediction"""
        texts = [
            "Test email 1",
            "Test email 2",
            "Test email 3"
        ]
        
        predictions = classifier.predict_batch(texts)
        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)


class TestURLClassifier:
    """Tests for URL classifier"""
    
    @pytest.fixture
    def classifier(self):
        return URLClassifier()
    
    def test_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier.feature_extractor is not None
        assert classifier.model is not None
        assert classifier.is_trained is False
    
    def test_predict_untrained(self, classifier):
        """Test prediction with untrained model"""
        url = "http://example.com"
        prediction = classifier.predict(url)
        # Should return default value
        assert prediction in [0, 1]
    
    def test_predict_proba_untrained(self, classifier):
        """Test probability prediction with untrained model"""
        url = "http://example.com"
        proba = classifier.predict_proba(url)
        
        assert len(proba) == 2
        # Untrained should return neutral probabilities
        assert proba[0] == 0.5
        assert proba[1] == 0.5
    
    def test_train(self, classifier):
        """Test model training"""
        urls = [
            "http://legitimate.com",
            "http://phishing.tk",
            "http://192.168.1.1",
            "http://safe-site.org"
        ]
        labels = [0, 1, 1, 0]
        
        classifier.train(urls, labels)
        assert classifier.is_trained is True


class TestPhishingDetector:
    """Tests for integrated phishing detector"""
    
    @pytest.fixture
    def detector(self):
        return PhishingDetector()
    
    def test_detect_basic(self, detector):
        """Test basic detection"""
        result = detector.detect(
            subject="Test Subject",
            body="This is a test email",
            sender="test@example.com"
        )
        
        assert 'risk_score' in result
        assert 'risk_level' in result
        assert 'email_score' in result
        assert 'explanation' in result
        assert 'indicators' in result
        
        assert 0 <= result['risk_score'] <= 1
        assert 0 <= result['email_score'] <= 1
    
    def test_detect_with_urls(self, detector):
        """Test detection with URLs"""
        result = detector.detect(
            subject="Urgent: Verify Account",
            body="Click here to verify: http://fake-site.tk",
            sender="noreply@suspicious.com",
            urls=["http://fake-site.tk"]
        )
        
        assert len(result['url_scores']) > 0
        assert len(result['urls']) > 0
    
    def test_detect_auto_extract_urls(self, detector):
        """Test automatic URL extraction"""
        result = detector.detect(
            subject="Test",
            body="Visit http://example.com for more info",
            sender="test@example.com"
        )
        
        # URLs should be extracted automatically
        assert 'urls' in result
    
    def test_risk_categorization(self, detector):
        """Test risk level categorization"""
        assert "HIGH" in detector._categorize_risk(0.9)
        assert "MEDIUM" in detector._categorize_risk(0.6)
        assert "LOW" in detector._categorize_risk(0.4)
        assert "SAFE" in detector._categorize_risk(0.2)
    
    def test_identify_indicators(self, detector):
        """Test indicator identification"""
        features = {
            'urgency_score': 2,
            'sensitive_info_request': 1,
            'url_count': 5
        }
        
        indicators = detector._identify_indicators(features, 0.8, [0.9])
        
        assert len(indicators) > 0
        assert any('urgency' in ind.lower() for ind in indicators)
    
    def test_generate_explanation(self, detector):
        """Test explanation generation"""
        features = {
            'text_length': 100,
            'urgency_score': 1,
            'sensitive_info_request': 1
        }
        
        explanation = detector._generate_explanation(0.8, [0.7], features)
        
        assert 'summary' in explanation
        assert 'email_analysis' in explanation
        assert 'url_analysis' in explanation
        assert 'recommendations' in explanation
    
    def test_batch_detect(self, detector):
        """Test batch detection"""
        emails = [
            {
                'subject': 'Test 1',
                'body': 'Body 1',
                'sender': 'test1@example.com'
            },
            {
                'subject': 'Test 2',
                'body': 'Body 2',
                'sender': 'test2@example.com'
            }
        ]
        
        results = detector.batch_detect(emails)
        
        assert len(results) == 2
        assert all('risk_score' in r for r in results)
