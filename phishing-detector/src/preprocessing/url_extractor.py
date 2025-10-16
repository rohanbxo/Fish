"""
URL extraction and parsing utilities
"""

import re
from typing import List, Dict
from urllib.parse import urlparse
import tldextract


class URLExtractor:
    """Extract and parse URLs from text"""
    
    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.ip_pattern = re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        )
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click'
        }
        
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract all URLs from text
        
        Args:
            text: Text containing URLs
            
        Returns:
            List of URLs
        """
        if not text:
            return []
        return self.url_pattern.findall(text)
    
    def parse_url(self, url: str) -> Dict:
        """
        Parse URL into components
        
        Args:
            url: URL string
            
        Returns:
            Dictionary of URL components
        """
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        
        return {
            'scheme': parsed.scheme,
            'domain': ext.domain,
            'subdomain': ext.subdomain,
            'suffix': ext.suffix,
            'path': parsed.path,
            'query': parsed.query,
            'full_domain': ext.fqdn
        }
    
    def is_ip_address(self, url: str) -> bool:
        """Check if URL contains IP address instead of domain"""
        parsed = urlparse(url)
        return bool(self.ip_pattern.search(parsed.netloc))
    
    def has_suspicious_tld(self, url: str) -> bool:
        """Check if URL has suspicious top-level domain"""
        ext = tldextract.extract(url)
        return ext.suffix.lower() in self.suspicious_tlds
    
    def count_special_chars(self, url: str) -> Dict[str, int]:
        """Count special characters in URL"""
        return {
            'dots': url.count('.'),
            'hyphens': url.count('-'),
            'underscores': url.count('_'),
            'slashes': url.count('/'),
            'at_signs': url.count('@'),
            'question_marks': url.count('?'),
            'equals': url.count('='),
            'ampersands': url.count('&')
        }
