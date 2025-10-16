"""
URL feature extraction for classification
"""

import math
import re
from typing import Dict, List
from urllib.parse import urlparse
import tldextract


class URLFeatureExtractor:
    """Extract features from URLs for classification"""
    
    def __init__(self):
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click',
            'loan', 'download', 'racing'
        }
        self.suspicious_keywords = {
            'login', 'signin', 'account', 'verify', 'secure', 'update',
            'confirm', 'banking', 'paypal', 'amazon', 'ebay', 'apple'
        }
        
    def extract_features(self, url: str) -> Dict:
        """
        Extract all features from URL
        
        Args:
            url: URL string
            
        Returns:
            Dictionary of features
        """
        if not url or not isinstance(url, str):
            return self._get_empty_features()
        
        features = {}
        
        try:
            parsed = urlparse(url)
            ext = tldextract.extract(url)
            
            # Lexical features
            features['length'] = len(url)
            features['num_dots'] = url.count('.')
            features['num_hyphens'] = url.count('-')
            features['num_underscores'] = url.count('_')
            features['num_slashes'] = url.count('/')
            features['num_at'] = url.count('@')
            features['num_question'] = url.count('?')
            features['num_equals'] = url.count('=')
            features['num_ampersand'] = url.count('&')
            features['num_percent'] = url.count('%')
            
            # Domain features
            features['domain_length'] = len(ext.domain) if ext.domain else 0
            features['subdomain_count'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
            features['path_length'] = len(parsed.path)
            features['query_length'] = len(parsed.query)
            
            # Security features
            features['is_https'] = 1 if parsed.scheme == 'https' else 0
            features['has_ip'] = 1 if self._has_ip_address(url) else 0
            features['suspicious_tld'] = 1 if self._check_suspicious_tld(ext.suffix) else 0
            
            # Content features
            features['has_suspicious_keyword'] = 1 if self._has_suspicious_keyword(url) else 0
            features['entropy'] = self._calculate_entropy(url)
            
            # Digit ratio
            digit_count = sum(c.isdigit() for c in url)
            features['digit_ratio'] = digit_count / len(url) if len(url) > 0 else 0
            
        except Exception as e:
            print(f"Error extracting features from URL {url}: {e}")
            return self._get_empty_features()
        
        return features
    
    def extract_features_batch(self, urls: List[str]) -> List[Dict]:
        """Extract features from multiple URLs"""
        return [self.extract_features(url) for url in urls]
    
    def _has_ip_address(self, url: str) -> bool:
        """Check if URL contains IP address"""
        parsed = urlparse(url)
        return bool(self.ip_pattern.search(parsed.netloc))
    
    def _check_suspicious_tld(self, tld: str) -> bool:
        """Check if TLD is suspicious"""
        return tld.lower() in self.suspicious_tlds
    
    def _has_suspicious_keyword(self, url: str) -> bool:
        """Check if URL contains suspicious keywords"""
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in self.suspicious_keywords)
    
    def _calculate_entropy(self, string: str) -> float:
        """
        Calculate Shannon entropy of string
        
        Args:
            string: Input string
            
        Returns:
            Entropy value
        """
        if not string:
            return 0.0
        
        # Calculate character frequency
        prob = [float(string.count(c)) / len(string) 
                for c in dict.fromkeys(list(string))]
        
        # Calculate entropy
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob if p > 0])
        
        return entropy
    
    def _get_empty_features(self) -> Dict:
        """Return dictionary with zero values for all features"""
        return {
            'length': 0,
            'num_dots': 0,
            'num_hyphens': 0,
            'num_underscores': 0,
            'num_slashes': 0,
            'num_at': 0,
            'num_question': 0,
            'num_equals': 0,
            'num_ampersand': 0,
            'num_percent': 0,
            'domain_length': 0,
            'subdomain_count': 0,
            'path_length': 0,
            'query_length': 0,
            'is_https': 0,
            'has_ip': 0,
            'suspicious_tld': 0,
            'has_suspicious_keyword': 0,
            'entropy': 0,
            'digit_ratio': 0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return list(self._get_empty_features().keys())
