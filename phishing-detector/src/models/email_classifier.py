"""
Email classification using transformer models
"""

import os
from typing import List, Tuple
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline
)


class EmailClassifier:
    """Classifier for phishing vs legitimate emails"""
    
    def __init__(self, model_path: str = None, use_pretrained: bool = True):
        """
        Initialize email classifier
        
        Args:
            model_path: Path to saved model (optional)
            use_pretrained: Whether to use pretrained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 512
        
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded model from {model_path}")
        elif use_pretrained:
            # Use base pretrained model
            model_name = "distilbert-base-uncased"
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded base model: {model_name}")
        else:
            raise ValueError("Either provide model_path or set use_pretrained=True")
    
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
        Get prediction probabilities
        
        Args:
            text: Email text
            
        Returns:
            Array of probabilities [legitimate_prob, phishing_prob]
        """
        if not text:
            return np.array([0.5, 0.5])
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        return probs[0].cpu().numpy()
    
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
    
    def save(self, save_path: str):
        """Save model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
