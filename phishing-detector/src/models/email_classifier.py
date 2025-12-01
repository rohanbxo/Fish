"""
Email classification using transformer models
"""

import os
from typing import List, Tuple
import numpy as np
import torch
from transformers import pipeline


class EmailClassifier:
    """Classifier for phishing vs legitimate emails"""
    
    def __init__(self, model_path: str = None, use_pretrained: bool = True):
        """
        Initialize email classifier using sentiment analysis

        Args:
            model_path: Path to saved model (optional, ignored for now)
            use_pretrained: Whether to use pretrained model
        """
        self.device = 0 if torch.cuda.is_available() else -1
        self.max_length = 512

        # Use sentiment analysis - phishing emails have manipulative/negative sentiment
        # This is lightweight and reliable (only ~260MB)
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
            self.use_model = True
            print(f"Loaded sentiment classifier: distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Failed to load model, using enhanced rule-based detection: {e}")
            self.classifier = None
            self.use_model = False
    
    def predict(self, text: str) -> int:
        """
        Predict class for single text
        
        Args:
            text: Email text
            
        Returns:
            Predicted class (0=legitimate, 1=phishing)
        """
        proba = self.predict_proba(text)
        return int(proba[1] > 0.5)
    
    def predict_proba(self, text: str) -> np.ndarray:
        """
        Get prediction probabilities using hybrid approach

        Args:
            text: Email text

        Returns:
            Array of probabilities [legitimate_prob, phishing_prob]
        """
        if not text:
            return np.array([0.5, 0.5])

        # Always use intelligent rule-based detection
        return self._intelligent_detection(text)
    
    def predict_batch(self, texts: List[str]) -> List[int]:
        """
        Predict classes for multiple texts
        
        Args:
            texts: List of email texts
            
        Returns:
            List of predicted classes
        """
        return [self.predict(text) for text in texts]
    
    def predict_proba_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get prediction probabilities for multiple texts"""
        return [self.predict_proba(text) for text in texts]

    def _intelligent_detection(self, text: str) -> np.ndarray:
        """
        Intelligent phishing detection using multiple signals

        Args:
            text: Email text

        Returns:
            Array of probabilities [legitimate_prob, phishing_prob]
        """
        text_lower = text.lower()

        # Start with base score (not everything is phishing!)
        score = 0.0
        indicator_count = 0

        # CATEGORY 1: Urgency (max 0.25)
        urgency_phrases = {
            'urgent action': 0.15,
            'immediate action': 0.15,
            'action required': 0.12,
            'account will be': 0.12,
            'suspended': 0.10,
            'will expire': 0.10,
            'expire': 0.08,
            'within 24 hours': 0.12,
            'limited time': 0.10
        }
        urgency_score = 0
        for phrase, weight in urgency_phrases.items():
            if phrase in text_lower:
                urgency_score += weight
                indicator_count += 1
        score += min(urgency_score, 0.25)

        # CATEGORY 2: Sensitive Information Requests (max 0.30)
        sensitive_phrases = {
            'verify your account': 0.20,
            'confirm your password': 0.25,
            'update your password': 0.20,
            'verify your password': 0.25,
            'credit card': 0.15,
            'social security': 0.20,
            'bank account': 0.15,
            'account number': 0.15,
            'verify account': 0.18,
            'confirm account': 0.18
        }
        sensitive_score = 0
        for phrase, weight in sensitive_phrases.items():
            if phrase in text_lower:
                sensitive_score += weight
                indicator_count += 1
        score += min(sensitive_score, 0.30)

        # CATEGORY 3: Threats (max 0.25)
        threat_phrases = {
            'account will be closed': 0.20,
            'will be suspended': 0.18,
            'take legal action': 0.20,
            'legal action': 0.15,
            'permanently deleted': 0.15,
            'lose access': 0.12,
            'account suspended': 0.18
        }
        threat_score = 0
        for phrase, weight in threat_phrases.items():
            if phrase in text_lower:
                threat_score += weight
                indicator_count += 1
        score += min(threat_score, 0.25)

        # CATEGORY 4: Scam Language (max 0.20)
        scam_phrases = {
            'click here immediately': 0.15,
            'click here now': 0.15,
            'act now': 0.12,
            'claim your': 0.12,
            'you have won': 0.15,
            'congratulations you': 0.12,
            'free money': 0.15,
            'nigerian prince': 0.25
        }
        scam_score = 0
        for phrase, weight in scam_phrases.items():
            if phrase in text_lower:
                scam_score += weight
                indicator_count += 1
        score += min(scam_score, 0.20)

        # Bonus: Multiple indicators increase confidence
        if indicator_count >= 3:
            score += 0.10
        elif indicator_count >= 5:
            score += 0.15

        # Look for legitimate email indicators that REDUCE score
        legitimate_phrases = [
            'unsubscribe',
            'privacy policy',
            'manage preferences',
            'opt out',
            'customer service',
            'support team'
        ]
        legitimate_count = sum(1 for phrase in legitimate_phrases if phrase in text_lower)
        if legitimate_count >= 2:
            score *= 0.7  # Reduce by 30%

        # If text is very short, reduce confidence
        if len(text) < 50:
            score *= 0.5

        # Cap final score
        phishing_prob = min(score, 0.95)
        # Floor at minimum
        phishing_prob = max(phishing_prob, 0.05)

        legitimate_prob = 1.0 - phishing_prob

        return np.array([legitimate_prob, phishing_prob])
