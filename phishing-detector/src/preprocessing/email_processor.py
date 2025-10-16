"""
Email preprocessing utilities
"""

import re
from typing import Dict, List
from bs4 import BeautifulSoup
import html


class EmailProcessor:
    """Processes and cleans email content for analysis"""
    
    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
    def preprocess(self, email_text: str) -> str:
        """
        Clean and normalize email text
        
        Args:
            email_text: Raw email content
            
        Returns:
            Cleaned email text
        """
        if not email_text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(email_text)
        
        # Remove HTML tags
        text = self.remove_html(text)
        
        # Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def remove_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        return text
    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text
        
        Args:
            text: Text containing URLs
            
        Returns:
            List of extracted URLs
        """
        return self.url_pattern.findall(text)
    
    def extract_features(self, subject: str, body: str, sender: str) -> Dict:
        """
        Extract features from email components
        
        Args:
            subject: Email subject line
            body: Email body content
            sender: Sender email address
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Text features
        combined_text = f"{subject} {body}"
        features['text_length'] = len(combined_text)
        features['word_count'] = len(combined_text.split())
        
        # URL features
        urls = self.extract_urls(body)
        features['url_count'] = len(urls)
        features['has_urls'] = len(urls) > 0
        
        # Urgency indicators
        urgency_words = [
            'urgent', 'immediate', 'important', 'action required',
            'verify', 'confirm', 'suspend', 'expire', 'limited time'
        ]
        text_lower = combined_text.lower()
        features['urgency_score'] = sum(
            word in text_lower for word in urgency_words
        )
        
        # Sensitive information requests
        sensitive_words = [
            'password', 'credit card', 'social security', 'ssn',
            'account number', 'pin', 'verify account', 'update payment'
        ]
        features['sensitive_info_request'] = sum(
            word in text_lower for word in sensitive_words
        )
        
        # Sender domain
        if '@' in sender:
            domain = sender.split('@')[-1]
            features['sender_domain'] = domain
        else:
            features['sender_domain'] = ""
        
        return features
    
    def combine_email_text(self, subject: str, body: str) -> str:
        """
        Combine subject and body for classification
        
        Args:
            subject: Email subject
            body: Email body
            
        Returns:
            Combined text
        """
        return f"{subject}\n\n{body}"
