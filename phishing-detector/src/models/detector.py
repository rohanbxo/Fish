"""
Integrated phishing detection system
"""

from typing import List, Dict, Optional
import numpy as np

from .email_classifier import EmailClassifier
from .url_classifier import URLClassifier
from ..preprocessing.email_processor import EmailProcessor


class PhishingDetector:
    """Integrated phishing detection system combining email and URL analysis"""
    
    def __init__(
        self,
        email_model_path: Optional[str] = None,
        url_model_path: Optional[str] = None
    ):
        """
        Initialize detector
        
        Args:
            email_model_path: Path to email classifier model
            url_model_path: Path to URL classifier model
        """
        self.email_processor = EmailProcessor()
        self.email_classifier = EmailClassifier(
            model_path=email_model_path,
            use_pretrained=True
        )
        self.url_classifier = URLClassifier(model_path=url_model_path)
        
        # Weights for combining scores
        self.email_weight = 0.6
        self.url_weight = 0.4
    
    def detect(
        self,
        subject: str,
        body: str,
        sender: str,
        urls: Optional[List[str]] = None
    ) -> Dict:
        """
        Detect phishing in email
        
        Args:
            subject: Email subject
            body: Email body
            sender: Sender email
            urls: List of URLs (optional, will be extracted if not provided)
            
        Returns:
            Detection result dictionary
        """
        # Preprocess email
        email_text = self.email_processor.combine_email_text(subject, body)
        processed_text = self.email_processor.preprocess(email_text)
        
        # Extract URLs if not provided
        if urls is None:
            urls = self.email_processor.extract_urls(body)
        
        # Email analysis
        email_proba = self.email_classifier.predict_proba(processed_text)
        email_score = float(email_proba[1])  # Probability of phishing
        
        # URL analysis
        url_scores = []
        if urls:
            for url in urls:
                try:
                    url_proba = self.url_classifier.predict_proba(url)
                    url_scores.append(float(url_proba[1]))
                except Exception as e:
                    print(f"Error analyzing URL {url}: {e}")
                    url_scores.append(0.5)  # Neutral score on error
        
        # Combined scoring
        if url_scores:
            max_url_score = max(url_scores)
            final_score = (self.email_weight * email_score + 
                          self.url_weight * max_url_score)
        else:
            final_score = email_score
        
        # Risk categorization
        risk_level = self._categorize_risk(final_score)
        
        # Extract features and indicators
        features = self.email_processor.extract_features(subject, body, sender)
        indicators = self._identify_indicators(features, email_score, url_scores)
        
        # Generate explanation
        explanation = self._generate_explanation(
            email_score, url_scores, features
        )
        
        return {
            'risk_score': float(final_score),
            'risk_level': risk_level,
            'email_score': float(email_score),
            'url_scores': url_scores,
            'urls': urls,
            'explanation': explanation,
            'indicators': indicators,
            'features': features
        }
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level based on score"""
        if score >= 0.8:
            return "HIGH - Likely Phishing"
        elif score >= 0.5:
            return "MEDIUM - Suspicious"
        elif score >= 0.3:
            return "LOW - Potentially Suspicious"
        else:
            return "SAFE - Likely Legitimate"
    
    def _identify_indicators(
        self,
        features: Dict,
        email_score: float,
        url_scores: List[float]
    ) -> List[str]:
        """Identify specific phishing indicators"""
        indicators = []
        
        # Check urgency
        if features.get('urgency_score', 0) > 0:
            indicators.append(
                f"Contains {features['urgency_score']} urgency keyword(s)"
            )
        
        # Check sensitive info requests
        if features.get('sensitive_info_request', 0) > 0:
            indicators.append(
                f"Requests sensitive information ({features['sensitive_info_request']} keyword(s))"
            )
        
        # Check URLs
        if features.get('url_count', 0) > 3:
            indicators.append(f"Contains {features['url_count']} URLs")
        
        # Check suspicious URLs
        if url_scores:
            high_risk_urls = sum(1 for score in url_scores if score > 0.7)
            if high_risk_urls > 0:
                indicators.append(
                    f"{high_risk_urls} suspicious URL(s) detected"
                )
        
        # Check email score
        if email_score > 0.7:
            indicators.append("Email text shows phishing patterns")
        
        return indicators
    
    def _generate_explanation(
        self,
        email_score: float,
        url_scores: List[float],
        features: Dict
    ) -> Dict:
        """Generate detailed explanation of results"""
        explanation = {
            'summary': '',
            'email_analysis': {},
            'url_analysis': {},
            'recommendations': []
        }
        
        # Email analysis
        explanation['email_analysis'] = {
            'phishing_probability': f"{email_score * 100:.1f}%",
            'text_length': features.get('text_length', 0),
            'urgency_indicators': features.get('urgency_score', 0),
            'sensitive_info_requests': features.get('sensitive_info_request', 0)
        }
        
        # URL analysis
        if url_scores:
            explanation['url_analysis'] = {
                'total_urls': len(url_scores),
                'max_risk_score': f"{max(url_scores) * 100:.1f}%",
                'high_risk_urls': sum(1 for s in url_scores if s > 0.7),
                'medium_risk_urls': sum(1 for s in url_scores if 0.4 <= s <= 0.7)
            }
        else:
            explanation['url_analysis'] = {'total_urls': 0}
        
        # Generate summary
        if email_score >= 0.8:
            explanation['summary'] = "High confidence phishing detection"
        elif email_score >= 0.5:
            explanation['summary'] = "Email shows suspicious characteristics"
        else:
            explanation['summary'] = "Email appears legitimate"
        
        # Recommendations
        if email_score > 0.5 or (url_scores and max(url_scores) > 0.5):
            explanation['recommendations'] = [
                "Do not click on any links in this email",
                "Do not provide any personal information",
                "Verify sender identity through alternative means",
                "Report this email to your security team"
            ]
        else:
            explanation['recommendations'] = [
                "Email appears safe, but always verify sender",
                "Hover over links before clicking to check destination"
            ]
        
        return explanation
    
    def batch_detect(self, emails: List[Dict]) -> List[Dict]:
        """
        Detect phishing in multiple emails
        
        Args:
            emails: List of email dictionaries with keys: subject, body, sender, urls
            
        Returns:
            List of detection results
        """
        results = []
        for email in emails:
            result = self.detect(
                subject=email.get('subject', ''),
                body=email.get('body', ''),
                sender=email.get('sender', ''),
                urls=email.get('urls')
            )
            results.append(result)
        
        return results
