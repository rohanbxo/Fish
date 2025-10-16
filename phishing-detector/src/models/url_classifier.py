"""
URL classification using traditional ML
"""

import os
import pickle
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from ..features.url_features import URLFeatureExtractor


class URLClassifier:
    """Classifier for malicious vs legitimate URLs"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize URL classifier
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.feature_extractor = URLFeatureExtractor()
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            # Initialize default model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
            )
            self.is_trained = False
    
    def train(self, urls: List[str], labels: List[int]):
        """
        Train the classifier
        
        Args:
            urls: List of URLs
            labels: List of labels (0=legitimate, 1=malicious)
        """
        # Extract features
        features = self.feature_extractor.extract_features_batch(urls)
        X = pd.DataFrame(features)
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, labels)
        self.is_trained = True
        print(f"Model trained on {len(urls)} samples")
    
    def predict(self, url: str) -> int:
        """
        Predict class for single URL
        
        Args:
            url: URL string
            
        Returns:
            Predicted class (0=legitimate, 1=malicious)
        """
        if not self.is_trained:
            # Return random prediction if not trained
            return 0
        
        features = self.feature_extractor.extract_features(url)
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        
        return int(self.model.predict(X_scaled)[0])
    
    def predict_proba(self, url: str) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            url: URL string
            
        Returns:
            Array of probabilities [legitimate_prob, malicious_prob]
        """
        if not self.is_trained:
            # Return neutral probabilities if not trained
            return np.array([0.5, 0.5])
        
        features = self.feature_extractor.extract_features(url)
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)[0]
    
    def predict_batch(self, urls: List[str]) -> List[int]:
        """Predict classes for multiple URLs"""
        return [self.predict(url) for url in urls]
    
    def predict_proba_batch(self, urls: List[str]) -> List[np.ndarray]:
        """Get prediction probabilities for multiple URLs"""
        return [self.predict_proba(url) for url in urls]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        
        feature_names = self.feature_extractor.get_feature_names()
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))
    
    def save(self, save_path: str):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, model_path: str):
        """Load model and scaler"""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data.get('is_trained', True)
        
        print(f"Model loaded from {model_path}")
