"""
Models module for phishing detection
"""

from .email_classifier import EmailClassifier
from .url_classifier import URLClassifier
from .detector import PhishingDetector

__all__ = ['EmailClassifier', 'URLClassifier', 'PhishingDetector']
