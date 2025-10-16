"""
Training script for email classifier
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_email_classifier(
    train_file: str,
    val_file: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16
):
    """
    Train email classifier
    
    Args:
        train_file: Path to training data CSV
        val_file: Path to validation data CSV
        output_dir: Output directory for model
        model_name: Pretrained model name
        num_epochs: Number of training epochs
        batch_size: Batch size
    """
    print(f"Loading data...")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['processed_text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['processed_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['processed_text', 'label']])
    
    # Tokenize
    print("Tokenizing data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Initialize model
    print(f"Loading model: {model_name}")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final')
    print(f"Saving model to {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Evaluate
    print("Evaluating model...")
    results = trainer.evaluate()
    print("Validation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    return model, tokenizer


if __name__ == "__main__":
    train_file = "data/processed/train.csv"
    val_file = "data/processed/val.csv"
    output_dir = "models/email_classifier"
    
    if os.path.exists(train_file) and os.path.exists(val_file):
        train_email_classifier(train_file, val_file, output_dir)
    else:
        print("Training data not found. Please run prepare_data.py first.")
