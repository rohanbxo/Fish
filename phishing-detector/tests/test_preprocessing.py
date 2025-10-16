"""
Unit tests for preprocessing modules
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.email_processor import EmailProcessor
from src.preprocessing.url_extractor import URLExtractor


class TestEmailProcessor:
    """Tests for EmailProcessor class"""
    
    @pytest.fixture
    def processor(self):
        return EmailProcessor()
    
    def test_preprocess_basic(self, processor):
        """Test basic text preprocessing"""
        text = "Hello World"
        result = processor.preprocess(text)
        assert result == "hello world"
    
    def test_remove_html(self, processor):
        """Test HTML tag removal"""
        html_text = "<p>Hello <b>World</b></p>"
        result = processor.preprocess(html_text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "hello" in result
        assert "world" in result
    
    def test_normalize_whitespace(self, processor):
        """Test whitespace normalization"""
        text = "Hello    World\n\n\nTest"
        result = processor.normalize_whitespace(text)
        assert "    " not in result
        assert "\n\n\n" not in result
    
    def test_extract_urls(self, processor):
        """Test URL extraction"""
        text = "Visit http://example.com and https://test.com"
        urls = processor.extract_urls(text)
        assert len(urls) == 2
        assert "http://example.com" in urls
        assert "https://test.com" in urls
    
    def test_extract_urls_no_urls(self, processor):
        """Test URL extraction with no URLs"""
        text = "No URLs here"
        urls = processor.extract_urls(text)
        assert len(urls) == 0
    
    def test_extract_features(self, processor):
        """Test feature extraction"""
        subject = "Urgent: Verify Account"
        body = "Please verify your password at http://fake-site.com"
        sender = "test@example.com"
        
        features = processor.extract_features(subject, body, sender)
        
        assert 'text_length' in features
        assert 'word_count' in features
        assert 'url_count' in features
        assert 'urgency_score' in features
        assert 'sensitive_info_request' in features
        assert features['urgency_score'] > 0  # "urgent" and "verify"
        assert features['sensitive_info_request'] > 0  # "password"
        assert features['url_count'] == 1
    
    def test_combine_email_text(self, processor):
        """Test email text combination"""
        subject = "Test Subject"
        body = "Test Body"
        result = processor.combine_email_text(subject, body)
        assert "Test Subject" in result
        assert "Test Body" in result
    
    def test_empty_text(self, processor):
        """Test handling of empty text"""
        result = processor.preprocess("")
        assert result == ""
        
        result = processor.preprocess(None)
        assert result == ""


class TestURLExtractor:
    """Tests for URLExtractor class"""
    
    @pytest.fixture
    def extractor(self):
        return URLExtractor()
    
    def test_extract_urls(self, extractor):
        """Test URL extraction"""
        text = "Check http://example.com and https://test.org"
        urls = extractor.extract_urls(text)
        assert len(urls) == 2
    
    def test_parse_url(self, extractor):
        """Test URL parsing"""
        url = "https://subdomain.example.com/path?query=value"
        parsed = extractor.parse_url(url)
        
        assert parsed['scheme'] == 'https'
        assert parsed['domain'] == 'example'
        assert parsed['subdomain'] == 'subdomain'
        assert parsed['suffix'] == 'com'
        assert '/path' in parsed['path']
    
    def test_is_ip_address(self, extractor):
        """Test IP address detection"""
        url_with_ip = "http://192.168.1.1/path"
        url_with_domain = "http://example.com/path"
        
        assert extractor.is_ip_address(url_with_ip) is True
        assert extractor.is_ip_address(url_with_domain) is False
    
    def test_has_suspicious_tld(self, extractor):
        """Test suspicious TLD detection"""
        suspicious_url = "http://example.tk"
        normal_url = "http://example.com"
        
        assert extractor.has_suspicious_tld(suspicious_url) is True
        assert extractor.has_suspicious_tld(normal_url) is False
    
    def test_count_special_chars(self, extractor):
        """Test special character counting"""
        url = "http://ex-ample.com/path?key=value&test=1"
        counts = extractor.count_special_chars(url)
        
        assert counts['dots'] > 0
        assert counts['hyphens'] > 0
        assert counts['question_marks'] == 1
        assert counts['equals'] == 2
        assert counts['ampersands'] == 1
    
    def test_extract_urls_empty(self, extractor):
        """Test URL extraction from empty text"""
        urls = extractor.extract_urls("")
        assert len(urls) == 0
        
        urls = extractor.extract_urls(None)
        assert len(urls) == 0
