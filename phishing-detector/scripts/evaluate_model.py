"""
Evaluation script for phishing detector
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.detector import PhishingDetector


def evaluate_detector(test_file: str, output_dir: str = "results"):
    """
    Evaluate phishing detector on test set
    
    Args:
        test_file: Path to test data CSV
        output_dir: Directory for saving results
    """
    print(f"Loading test data from {test_file}...")
    test_df = pd.read_csv(test_file)
    
    print(f"Test samples: {len(test_df)}")
    
    # Initialize detector
    print("Initializing detector...")
    detector = PhishingDetector()
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    risk_scores = []
    
    for idx, row in test_df.iterrows():
        result = detector.detect(
            subject=row['subject'],
            body=row['body'],
            sender=row['sender']
        )
        
        risk_scores.append(result['risk_score'])
        predictions.append(1 if result['risk_score'] >= 0.5 else 0)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} emails")
    
    y_true = test_df['label'].values
    y_pred = np.array(predictions)
    y_scores = np.array(risk_scores)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred,
        target_names=['Legitimate', 'Phishing']
    )
    print(report)
    
    # Save report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Saved confusion matrix to {output_dir}/confusion_matrix.png")
    
    # ROC curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        print(f"Saved ROC curve to {output_dir}/roc_curve.png")
        print(f"AUC-ROC: {auc:.4f}")
    
    # Error analysis
    false_positives = test_df[(y_true == 0) & (y_pred == 1)]
    false_negatives = test_df[(y_true == 1) & (y_pred == 0)]
    
    print(f"\nFalse Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    
    # Save errors
    if len(false_positives) > 0:
        false_positives.to_csv(
            os.path.join(output_dir, 'false_positives.csv'),
            index=False
        )
    
    if len(false_negatives) > 0:
        false_negatives.to_csv(
            os.path.join(output_dir, 'false_negatives.csv'),
            index=False
        )
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    test_file = "data/processed/test.csv"
    
    if os.path.exists(test_file):
        evaluate_detector(test_file)
    else:
        print(f"Test file not found: {test_file}")
        print("Please run prepare_data.py first.")
