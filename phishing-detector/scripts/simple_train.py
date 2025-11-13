"""
Simple training script that ACTUALLY WORKS on Windows
Pre-tokenizes data to avoid runtime issues
"""

import os
import sys
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

print("\n" + "="*70)
print("PHISHING EMAIL CLASSIFIER - FULL TRAINING")
print("="*70 + "\n")

# Configuration
QUICK_MODE = False  # Use FULL dataset for production training
EPOCHS = 3  # More epochs for better accuracy
BATCH_SIZE = 16  # Larger batch for efficiency
MAX_LENGTH = 256  # Shorter for speed

print(f"Mode: {'QUICK (500 samples)' if QUICK_MODE else 'FULL (56k+ samples)'}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max length: {MAX_LENGTH} tokens\n")

# Load data
print("Loading data...")
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')

# Filter out extremely long texts
print("Filtering long emails...")
train_df = train_df[train_df['text'].str.len() < 5000].copy()
val_df = val_df[val_df['text'].str.len() < 5000].copy()

if QUICK_MODE:
    train_df = train_df.sample(n=500, random_state=42)
    val_df = val_df.sample(n=100, random_state=42)

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}\n")

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, local_files_only=True)

device = torch.device('cpu')
model.to(device)
print(f"Device: {device}\n")

# PRE-TOKENIZE ALL DATA (this is the key!)
print("Pre-tokenizing training data...")
train_encodings = tokenizer(
    train_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='pt'
)

print("Pre-tokenizing validation data...")
val_encodings = tokenizer(
    val_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='pt'
)

train_labels = torch.tensor(train_df['label'].tolist())
val_labels = torch.tensor(val_df['label'].tolist())

print("Tokenization complete!\n")

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
print("="*70)
print("TRAINING")
print("="*70 + "\n")

best_f1 = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)
    
    model.train()
    total_loss = 0
    num_batches = (len(train_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    start_time = time.time()
    
    for i in range(0, len(train_df), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(train_df))
        
        # Get batch
        input_ids = train_encodings['input_ids'][i:batch_end].to(device)
        attention_mask = train_encodings['attention_mask'][i:batch_end].to(device)
        labels = train_labels[i:batch_end].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress
        batch_num = i // BATCH_SIZE + 1
        if batch_num % 10 == 0 or batch_num == num_batches:
            elapsed = time.time() - start_time
            print(f"  Batch {batch_num}/{num_batches} - Loss: {loss.item():.4f} - Time: {elapsed:.1f}s")
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    print(f"\nTrain Loss: {avg_loss:.4f} (Time: {epoch_time:.1f}s)")
    
    # Validation
    print("\nValidating...")
    model.eval()
    
    val_preds = []
    val_true = []
    val_loss = 0
    
    with torch.no_grad():
        for i in range(0, len(val_df), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(val_df))
            
            input_ids = val_encodings['input_ids'][i:batch_end].to(device)
            attention_mask = val_encodings['attention_mask'][i:batch_end].to(device)
            labels = val_labels[i:batch_end].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(val_true, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='binary')
    
    print(f"Val Loss:      {val_loss / (len(val_df) // BATCH_SIZE):.4f}")
    print(f"Val Accuracy:  {accuracy:.4f}")
    print(f"Val Precision: {precision:.4f}")
    print(f"Val Recall:    {recall:.4f}")
    print(f"Val F1:        {f1:.4f}")
    
    # Save best model
    if f1 > best_f1:
        best_f1 = f1
        output_dir = 'models/email_classifier/best_model'
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\nNew best F1: {best_f1:.4f} - Model saved!")
    
    print("\n" + "="*70 + "\n")

print("TRAINING COMPLETE!")
print(f"Best F1 Score: {best_f1:.4f}")
print(f"Model saved to: models/email_classifier/best_model/")
print("\nNext steps:")
print("  1. python api/main.py")
print("  2. streamlit run web/app.py\n")
