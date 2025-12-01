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
        Initialize email classifier using zero-shot classification

        Args:
            model_path: Path to saved model (optional, ignored for now)
            use_pretrained: Whether to use pretrained model
        """
        self.device = 0 if torch.cuda.is_available() else -1
        self.max_length = 512

        # Use zero-shot classification for phishing detection
        # This works without training by understanding the concepts
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            self.use_zero_shot = True
            print(f"Loaded zero-shot classifier: facebook/bart-large-mnli")
        except Exception as e:
            print(f"Failed to load zero-shot model, using rule-based fallback: {e}")
            self.classifier = None
            self.use_zero_shot = False
    
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
        Get prediction probabilities using zero-shot classification

        Args:
            text: Email text

        Returns:
            Array of probabilities [legitimate_prob, phishing_prob]
        """
        if not text:
            return np.array([0.5, 0.5])

        if self.use_zero_shot and self.classifier:
            try:
                # Truncate text if too long
                if len(text) > 1000:
                    text = text[:1000]

                # Define labels for zero-shot classification
                candidate_labels = [
                    "legitimate business email",
                    "phishing scam or fraudulent message"
                ]

                # Perform zero-shot classification
                result = self.classifier(
                    text,
                    candidate_labels,
                    multi_label=False
                )

                # Extract probabilities
                # result['labels'] gives ordered labels by score
                # result['scores'] gives corresponding scores
                phishing_idx = result['labels'].index("phishing scam or fraudulent message")
                legitimate_idx = result['labels'].index("legitimate business email")

                phishing_score = result['scores'][phishing_idx]
                legitimate_score = result['scores'][legitimate_idx]

                return np.array([legitimate_score, phishing_score])

            except Exception as e:
                print(f"Zero-shot classification error: {e}")
                # Fall through to rule-based approach

        # Fallback: rule-based detection
        return self._rule_based_detection(text)
    
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

    def _rule_based_detection(self, text: str) -> np.ndarray:
        """
        Fallback rule-based phishing detection

        Args:
            text: Email text

        Returns:
            Array of probabilities [legitimate_prob, phishing_prob]
        """
        text_lower = text.lower()
        score = 0.0

        # Phishing indicators
        urgent_words = ['urgent', 'immediate', 'action required', 'suspended', 'verify', 'confirm', 'expire']
        sensitive_words = ['password', 'credit card', 'ssn', 'social security', 'bank account', 'pin']
        threat_words = ['suspend', 'close', 'terminate', 'legal action', 'fraud']
        suspicious_words = ['click here', 'act now', 'limited time', 'congratulations', 'winner']

        # Count indicators
        for word in urgent_words:
            if word in text_lower:
                score += 0.15

        for word in sensitive_words:
            if word in text_lower:
                score += 0.20

        for word in threat_words:
            if word in text_lower:
                score += 0.18

        for word in suspicious_words:
            if word in text_lower:
                score += 0.12

        # Cap score at 1.0
        phishing_prob = min(score, 0.95)
        legitimate_prob = 1.0 - phishing_prob

        return np.array([legitimate_prob, phishing_prob])
